import torch
from torch.utils.data import DataLoader
from improved_diffusion.script_util import create_model
from minecraft_dataset import MinecraftSchematicDataset
from improved_diffusion.unet import UNetModel
import os

if __name__ == "__main__":
    # Create cache directory if it doesn't exist
    os.makedirs("cache", exist_ok=True)
    os.makedirs("models", exist_ok=True)

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

    # Example usage
    model = UNetModel(
        in_channels=len(dataset.block_to_idx),
        model_channels=64,
        out_channels=len(dataset.block_to_idx),
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
