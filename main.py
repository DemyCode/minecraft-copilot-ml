import os
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from improved_diffusion.script_util import create_gaussian_diffusion
from improved_diffusion.unet import UNetModel
from improved_diffusion.train_util import TrainLoop
from improved_diffusion import dist_util, logger
from minecraft_dataset import MinecraftSchematicDataset
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )


def warmup_lr(step, warmup_steps=5000):
    """Learning rate warmup function.
    
    Args:
        step: Current training step
        warmup_steps: Number of steps for warmup
        
    Returns:
        Learning rate multiplier between 0 and 1
    """
    return min(step, warmup_steps) / warmup_steps


if __name__ == "__main__":
    # Set up error handling
    try:
        # Create necessary directories
        os.makedirs("cache", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("output/unet", exist_ok=True)

        # Configuration parameters
        batch_size = 32
        num_workers = 4
        embedding_dim = 32
        learning_rate = 2e-4
        
        # Create the dataset with sentence transformer embeddings
        dataset = MinecraftSchematicDataset(
            schematics_dir="minecraft-schematics-raw",
            chunk_size=16,
            cache_file="cache/block_mappings.pkl",
            embedding_cache_file="cache/block_embeddings.pt",
            max_files=None,  # Use all files
            embedding_dim=embedding_dim,  # Dimension for embeddings after PCA reduction
        )

        # Split into train and validation sets
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        # Create DataLoaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        # Create the model using embeddings
        num_block_types = len(dataset.block_to_idx)
        print(f"Number of block types: {num_block_types}")
        print(f"Embedding dimension: {embedding_dim}")

        model = UNetModel(
            in_channels=embedding_dim,  # Using embedding dimension instead of one-hot
            model_channels=64,
            out_channels=embedding_dim,  # Output will be in embedding space
            num_res_blocks=2,
            attention_resolutions=(4,),
            dropout=0.1,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=3,
            num_classes=None,
            use_checkpoint=False,
            num_heads=4,
        )

        # Show model size
        model_size = 0
        for param in model.parameters():
            model_size += param.data.nelement()
        print(f"Model params: {model_size / 1e6:.2f} M")

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Move model to device
        model = model.to(device)
        ema_model = copy.deepcopy(model)  # This will also be on the same device

        # Set up optimizer with AdamW (better weight decay handling than Adam)
        optim = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
        
        # Set up flow matcher
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
        
        # Set up training parameters
        num_epochs = 10000
        savedir = "./output/unet/"
        os.makedirs(savedir, exist_ok=True)
        
        # Enable automatic mixed precision for faster training if available
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        for num_epoch in tqdm(range(num_epochs), desc="Epochs"):
            # Initialize epoch metrics
            epoch_losses = []
            epoch_accuracies = []

            for step_i, data in tqdm(
                enumerate(train_dataloader), desc=f"Epoch {num_epoch}", leave=False
            ):
                # Get block data, embeddings, and mask
                blocks = data["blocks"].to(device)
                block_embeddings = data["block_embeddings"].to(device)
                mask = data["mask"].to(device)

                # Data already moved to device in the previous step

                # Prepare embeddings for the model
                # block_embeddings shape: [batch_size, chunk_size, chunk_size, chunk_size, embedding_dim]
                # We need to permute to get channels as the second dimension [batch, channels, depth, height, width]
                batch_size, depth, height, width = blocks.shape
                x = block_embeddings.permute(0, 4, 1, 2, 3)

                # Generate random noise and sample flow
                x0 = torch.randn_like(x)
                t, xt, ut = FM.sample_location_and_conditional_flow(x0, x)

                # Use mixed precision training if available
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        # Forward pass through the model
                        vt = model(t, xt)
                        mse = (vt - ut) ** 2
                else:
                    # Forward pass through the model
                    vt = model(t, xt)
                    mse = (vt - ut) ** 2

                # Reshape mask for proper broadcasting
                # Mask shape: [batch_size, 16, 16, 16]
                # MSE shape: [batch_size, channels, 16, 16, 16]
                # We need to add a channel dimension to the mask
                mask_expanded = mask.unsqueeze(1)  # Shape: [batch_size, 1, 16, 16, 16]

                # Now the broadcasting will work correctly
                loss = mse * mask_expanded  # Broadcasting will apply mask to all channels
                loss = loss.mean()

                # Calculate accuracy (using cosine similarity for embeddings)
                with torch.no_grad():
                    # For embeddings, we use cosine similarity instead of exact matching
                    # Normalize the vectors for cosine similarity
                    vt_norm = F.normalize(vt, p=2, dim=1)
                    ut_norm = F.normalize(ut, p=2, dim=1)
                    
                    # Calculate cosine similarity (dot product of normalized vectors)
                    # This will be in range [-1, 1], where 1 means perfect match
                    similarity = (vt_norm * ut_norm).sum(dim=1, keepdim=True)
                    
                    # Consider prediction correct if similarity is above threshold (e.g., 0.8)
                    similarity_threshold = 0.8
                    correct = (similarity > similarity_threshold).float() * mask_expanded
                    
                    # Avoid division by zero
                    mask_sum = mask_expanded.sum()
                    accuracy = correct.sum() / mask_sum if mask_sum > 0 else torch.tensor(0.0, device=device)

                # Store metrics for epoch averaging
                epoch_losses.append(loss.item())
                epoch_accuracies.append(accuracy.item())

                # Update tqdm description with current metrics (less frequently to reduce overhead)
                if step_i % 10 == 0:
                    tqdm.write(
                        f"Step {step_i}: Loss = {loss.item():.6f}, Accuracy = {accuracy.item():.4f}"
                    )

                # Zero gradients before backward pass
                optim.zero_grad()
                
                # Backward pass and optimization with mixed precision if available
                if scaler is not None:
                    # Scale loss and perform backward pass
                    scaler.scale(loss).backward()
                    
                    # Unscale gradients for clipping
                    scaler.unscale_(optim)
                    
                    # Clip gradients to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    # Step optimizer and update scaler
                    scaler.step(optim)
                    scaler.update()
                else:
                    # Standard backward pass and optimization
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step()
                
                # Update learning rate scheduler
                sched.step()
                
                # Update EMA model with adaptive decay based on training progress
                # Start with lower decay value and increase it as training progresses
                ema_decay = min(0.9999, 0.99 + step_i / 10000)
                ema(model, ema_model, ema_decay)

            # Save checkpoint periodically (less frequently to reduce I/O overhead)
            if step_i % 500 == 0:  # Reduced frequency to save storage and I/O
                checkpoint_path = os.path.join(savedir, f"unet_minecraft_weights_step_{step_i}.pt")
                torch.save(
                    {
                        "net_model": model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "sched": sched.state_dict(),
                        "optim": optim.state_dict(),
                        "step": step_i,
                        "epoch": num_epoch,
                    },
                    checkpoint_path,
                )
                print(f"Saved checkpoint to {checkpoint_path}")

        # Calculate and display epoch summary
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        avg_epoch_accuracy = (
            sum(epoch_accuracies) / len(epoch_accuracies) if epoch_accuracies else 0
        )

        # Print epoch summary
        print(f"\n{'=' * 50}")
        print(f"Epoch {num_epoch} Summary:")
        print(f"Average Loss: {avg_epoch_loss:.6f}")
        print(f"Average Accuracy: {avg_epoch_accuracy:.4f}")
        print(f"Learning Rate: {sched.get_last_lr()[0]:.6f}")
        print(f"{'=' * 50}\n")

        # Save epoch checkpoint and evaluate on validation set
        if num_epoch % 10 == 0:
            # Save checkpoint
            torch.save(
                {
                    "net_model": model.state_dict(),
                    "ema_model": ema_model.state_dict(),
                    "sched": sched.state_dict(),
                    "optim": optim.state_dict(),
                    "epoch": num_epoch,
                    "avg_loss": avg_epoch_loss,
                    "avg_accuracy": avg_epoch_accuracy,
                },
                savedir + f"unet_minecraft_epoch_{num_epoch}_checkpoint.pt",
            )
            
            # Evaluate on validation set
            model.eval()
            val_losses = []
            val_accuracies = []
            
            print(f"Evaluating on validation set...")
            with torch.no_grad():
                for val_data in tqdm(val_dataloader, desc="Validation", leave=False):
                    # Get data
                    val_blocks = val_data["blocks"].to(device)
                    val_block_embeddings = val_data["block_embeddings"].to(device)
                    val_mask = val_data["mask"].to(device)
                    
                    # Prepare embeddings
                    val_x = val_block_embeddings.permute(0, 4, 1, 2, 3)
                    
                    # Sample and get flow
                    val_x0 = torch.randn_like(val_x)
                    val_t, val_xt, val_ut = FM.sample_location_and_conditional_flow(val_x0, val_x)
                    # Data already on device from previous steps
                    
                    # Forward pass
                    val_vt = model(val_t, val_xt)
                    
                    # Calculate loss
                    val_mask_expanded = val_mask.unsqueeze(1)
                    val_mse = ((val_vt - val_ut) ** 2) * val_mask_expanded
                    val_loss = val_mse.mean()
                    
                    # Calculate accuracy
                    val_vt_norm = F.normalize(val_vt, p=2, dim=1)
                    val_ut_norm = F.normalize(val_ut, p=2, dim=1)
                    val_similarity = (val_vt_norm * val_ut_norm).sum(dim=1, keepdim=True)
                    val_correct = (val_similarity > 0.8).float() * val_mask_expanded
                    
                    val_mask_sum = val_mask_expanded.sum()
                    val_accuracy = val_correct.sum() / val_mask_sum if val_mask_sum > 0 else torch.tensor(0.0, device=device)
                    
                    val_losses.append(val_loss.item())
                    val_accuracies.append(val_accuracy.item())
            
            # Calculate average validation metrics
            avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0
            avg_val_accuracy = sum(val_accuracies) / len(val_accuracies) if val_accuracies else 0
            
            print(f"Validation Loss: {avg_val_loss:.6f}, Validation Accuracy: {avg_val_accuracy:.4f}")
            
            # Set model back to training mode
            model.train()
