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

    # Create the dataset
    dataset = MinecraftSchematicDataset(
        schematics_dir="minecraft-schematics-raw",
        chunk_size=16,
        cache_file="cache/block_mappings.pkl",
        max_files=None,  # Use all files
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
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
    )

    # Create the model
    num_block_types = len(dataset.block_to_idx)
    print(f"Number of block types (channels): {num_block_types}")

    model = UNetModel(
        in_channels=num_block_types,
        model_channels=64,
        out_channels=num_block_types,
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
    for num_epoch in tqdm(range(EPOCHS)):
        for step_i, data in tqdm(enumerate(train_dataloader)):
            # Get block data and mask
            blocks = data["blocks"]
            mask = data["mask"]
            
            # Move to device
            blocks = blocks.to(device)
            mask = mask.to(device)
            
            # One-hot encode the blocks
            # blocks is expected to contain integer indices
            num_classes = len(dataset.block_to_idx)
            print(f"One-hot encoding blocks with {num_classes} classes")
            print(f"Original blocks shape: {blocks.shape}, min: {blocks.min()}, max: {blocks.max()}")
            
            # Ensure block indices are within valid range
            blocks_clamped = torch.clamp(blocks.long(), 0, num_classes - 1)
            
            # Create one-hot encoded tensor using PyTorch's built-in function
            # First reshape blocks to be a 1D tensor of indices
            batch_size, depth, height, width = blocks.shape
            blocks_flat = blocks_clamped.reshape(-1)
            
            # Create one-hot encoding
            one_hot_flat = torch.zeros(blocks_flat.size(0), num_classes, device=device)
            one_hot_flat.scatter_(1, blocks_flat.unsqueeze(1), 1.0)
            
            # Reshape back to original dimensions with channels
            x = one_hot_flat.reshape(batch_size, depth, height, width, num_classes)
            # Permute to get channels as the second dimension [batch, channels, depth, height, width]
            x = x.permute(0, 4, 1, 2, 3)
            
            print(f"One-hot encoded x shape: {x.shape}")
            x0 = torch.randn_like(x)  # This will be on the same device as x
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x)
            # Ensure all tensors are on the same device
            t = t.to(device)
            xt = xt.to(device)
            ut = ut.to(device)
            
            # Debug tensor shapes
            print(f"x shape: {x.shape}, x0 shape: {x0.shape}")
            print(f"xt shape: {xt.shape}, t shape: {t.shape}, ut shape: {ut.shape}")
            print(f"Model in_channels: {model.in_channels}, model_channels: {model.model_channels}")
            
            # With one-hot encoding, xt and ut should already have the correct number of channels
            # But let's check and fix if needed
            if xt.shape[1] != model.in_channels:
                print(f"Warning: xt has {xt.shape[1]} channels but model expects {model.in_channels}")
                if xt.shape[1] < model.in_channels:
                    # Pad with zeros if we have fewer channels than needed
                    padding = torch.zeros(xt.shape[0], model.in_channels - xt.shape[1], 
                                         xt.shape[2], xt.shape[3], xt.shape[4], 
                                         device=device)
                    xt = torch.cat([xt, padding], dim=1)
                else:
                    # Truncate if we have more channels than needed
                    xt = xt[:, :model.in_channels]
                print(f"Adjusted xt shape: {xt.shape}")
                
            # Similarly, ensure ut has the correct number of channels
            if ut.shape[1] != model.out_channels:
                print(f"Warning: ut has {ut.shape[1]} channels but model outputs {model.out_channels}")
                if ut.shape[1] < model.out_channels:
                    # Pad with zeros if we have fewer channels than needed
                    padding = torch.zeros(ut.shape[0], model.out_channels - ut.shape[1], 
                                         ut.shape[2], ut.shape[3], ut.shape[4], 
                                         device=device)
                    ut = torch.cat([ut, padding], dim=1)
                else:
                    # Truncate if we have more channels than needed
                    ut = ut[:, :model.out_channels]
                print(f"Adjusted ut shape: {ut.shape}")
                
            vt = model(t, xt)
            print(f"Model output vt shape: {vt.shape}")
            
            # Double-check that vt and ut have the same shape for the loss calculation
            if vt.shape != ut.shape:
                print(f"Warning: vt shape {vt.shape} doesn't match ut shape {ut.shape}")
                # Make them compatible for the loss calculation
                if vt.shape[1] != ut.shape[1]:
                    min_channels = min(vt.shape[1], ut.shape[1])
                    vt = vt[:, :min_channels]
                    ut = ut[:, :min_channels]
                    print(f"Adjusted to common shape: vt {vt.shape}, ut {ut.shape}")
            
            mse = (vt - ut) ** 2
            # apply mask
            print(f"MSE shape: {mse.shape}, Mask shape: {mask.shape}")

            # Ensure mask has the right shape for broadcasting
            if len(mask.shape) < len(mse.shape):
                # Add channel dimension if needed
                mask_expanded = mask.unsqueeze(1)
                # Repeat mask across all channels if needed
                if mask_expanded.shape[1] != mse.shape[1]:
                    mask_expanded = mask_expanded.repeat(1, mse.shape[1], 1, 1, 1)
                print(f"Expanded mask shape: {mask_expanded.shape}")
                mask = mask_expanded

            loss = mse * mask
            loss = loss.mean()
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
