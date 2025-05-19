import torch
from torch.utils.data import DataLoader
from improved_diffusion.script_util import create_gaussian_diffusion
from minecraft_dataset import MinecraftSchematicDataset
from improved_diffusion.unet import UNetModel
from improved_diffusion.train_util import TrainLoop
import os
from improved_diffusion import dist_util, logger
import copy
from tqdm import tqdm


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )


def warmup_lr(step):
    return min(step, 5000) / 5000


if __name__ == "__main__":
    # Create cache directory if it doesn't exist
    os.makedirs("cache", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Create the dataset with embeddings
    dataset = MinecraftSchematicDataset(
        schematics_dir="minecraft-schematics-raw",
        chunk_size=16,
        cache_file="cache/block_mappings.pkl",
        embedding_cache_file="cache/block_embeddings.pt",
        max_files=None,  # Use all files
        embedding_dim=32,  # Dimension for embeddings after PCA reduction
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
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
    )

    # Create the model using embeddings
    num_block_types = len(dataset.block_to_idx)
    embedding_dim = dataset.embedding_dim
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
    from torchcfm.conditional_flow_matching import (
        ExactOptimalTransportConditionalFlowMatcher,
    )

    # show model size
    model_size = 0
    for param in model.parameters():
        model_size += param.data.nelement()
    print("Model params: %.2f M" % (model_size / 1024 / 1024))

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Move model to device
    model = model.to(device)
    ema_model = copy.deepcopy(model)  # This will also be on the same device

    optim = torch.optim.Adam(model.parameters(), lr=2e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    EPOCHS = 10000
    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
    savedir = "./output/unet/"
    os.makedirs(savedir, exist_ok=True)
    for num_epoch in tqdm(range(EPOCHS), desc="Epochs"):
        # Initialize epoch metrics
        epoch_losses = []
        epoch_accuracies = []

        for step_i, data in tqdm(
            enumerate(train_dataloader), desc=f"Epoch {num_epoch}", leave=False
        ):
            # Get block data, embeddings, and mask
            blocks = data["blocks"]
            block_embeddings = data["block_embeddings"]
            mask = data["mask"]

            # Move to device
            blocks = blocks.to(device)
            block_embeddings = block_embeddings.to(device)
            mask = mask.to(device)

            # Prepare embeddings for the model
            # block_embeddings shape: [batch_size, chunk_size, chunk_size, chunk_size, embedding_dim]
            # We need to permute to get channels as the second dimension [batch, channels, depth, height, width]
            batch_size, depth, height, width = blocks.shape
            x = block_embeddings.permute(0, 4, 1, 2, 3)

            x0 = torch.randn_like(x)  # This will be on the same device as x
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x)
            # Ensure all tensors are on the same device
            t = t.to(device)
            xt = xt.to(device)
            ut = ut.to(device)

            # Debug tensor shapes

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
                vt_norm = torch.nn.functional.normalize(vt, p=2, dim=1)
                ut_norm = torch.nn.functional.normalize(ut, p=2, dim=1)
                
                # Calculate cosine similarity (dot product of normalized vectors)
                # This will be in range [-1, 1], where 1 means perfect match
                similarity = (vt_norm * ut_norm).sum(dim=1, keepdim=True)
                
                # Consider prediction correct if similarity is above threshold (e.g., 0.8)
                similarity_threshold = 0.8
                correct = (similarity > similarity_threshold).float() * mask_expanded
                accuracy = correct.sum() / (
                    mask_expanded.sum()
                )  # Avoid division by zero

                # Store metrics for epoch averaging
                epoch_losses.append(loss.item())
                epoch_accuracies.append(accuracy.item())

                # Update tqdm description with current metrics
                tqdm.write(
                    f"Step {step_i}: Loss = {loss.item():.6f}, Accuracy = {accuracy.item():.4f}"
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # new
            optim.step()
            sched.step()
            ema(model, ema_model, 0.9999)  # new

            # sample and Saving the weights
            if step_i % 5 == 0:
                torch.save(
                    {
                        "net_model": model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "sched": sched.state_dict(),
                        "optim": optim.state_dict(),
                        "step": step_i,
                    },
                    savedir + f"unet_cifar10_weights_step_{step_i}.pt",
                )

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

        # Save epoch checkpoint
        if num_epoch % 10 == 0:
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
                savedir + f"unet_epoch_{num_epoch}_checkpoint.pt",
            )
