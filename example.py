#!/usr/bin/env python3
"""
Example script for using the Minecraft VAE.
This script demonstrates how to use the VAE for encoding and decoding Minecraft maps.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from minecraft_dataset import MinecraftSchematicDataset
from minecraft_vae import MinecraftVAE, train_vae, evaluate_vae, save_vae, load_vae

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create cache directory if it doesn't exist
os.makedirs('cache', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Create a synthetic dataset for demonstration
print("Creating a synthetic dataset with random blocks for demonstration purposes.")

# Number of blocks in our synthetic dataset
num_blocks = 12
chunk_size = 16
num_samples = 20

# Create synthetic block mappings
block_to_idx = {'<pad>': 0, '<unk>': 1}
idx_to_block = {0: '<pad>', 1: '<unk>'}

for i in range(10):
    block_name = f"minecraft:block_{i}"
    block_to_idx[block_name] = i + 2
    idx_to_block[i + 2] = block_name

# Create synthetic data
synthetic_blocks = []
synthetic_masks = []

for _ in range(num_samples):
    # Create random blocks
    blocks = torch.randint(0, num_blocks, (chunk_size, chunk_size, chunk_size))
    
    # Create a random mask (1 for valid positions, 0 for padding)
    mask = torch.ones((chunk_size, chunk_size, chunk_size))
    
    # Randomly mask out some regions
    if np.random.random() < 0.5:
        # Randomly choose a corner to mask
        x_start = np.random.randint(0, chunk_size // 2)
        y_start = np.random.randint(0, chunk_size // 2)
        z_start = np.random.randint(0, chunk_size // 2)
        
        x_end = np.random.randint(chunk_size // 2, chunk_size)
        y_end = np.random.randint(chunk_size // 2, chunk_size)
        z_end = np.random.randint(chunk_size // 2, chunk_size)
        
        mask[x_start:x_end, y_start:y_end, z_start:z_end] = 0
    
    synthetic_blocks.append(blocks)
    synthetic_masks.append(mask)

# Create a custom dataset
class SyntheticMinecraftDataset(torch.utils.data.Dataset):
    def __init__(self, blocks, masks):
        self.blocks = blocks
        self.masks = masks
        self.block_to_idx = block_to_idx
        self.idx_to_block = idx_to_block
    
    def __len__(self):
        return len(self.blocks)
    
    def __getitem__(self, idx):
        return {
            'blocks': self.blocks[idx],
            'mask': self.masks[idx],
            'file_path': f"synthetic_sample_{idx}"
        }

# Create the dataset
dataset = SyntheticMinecraftDataset(synthetic_blocks, synthetic_masks)

# Split into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

# Create the VAE
vae = MinecraftVAE(
    num_blocks=len(dataset.block_to_idx),
    chunk_size=16,
    embedding_dim=32,
    latent_dim=64,
    hidden_dims=[32, 64, 128]
).to(device)

# Print model summary
print(f"\nVAE Model:")
print(f"  Number of blocks: {len(dataset.block_to_idx)}")
print(f"  Latent dimension: {vae.latent_dim}")
print(f"  Parameter count: {sum(p.numel() for p in vae.parameters())}")

# Create optimizer
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

# Train for a few epochs (just for demonstration)
print("\nTraining VAE for 2 epochs (demonstration only)...")
losses = train_vae(
    vae=vae,
    dataloader=train_dataloader,
    optimizer=optimizer,
    device=device,
    epochs=2,
    kld_weight=0.01
)

# Save the VAE
save_vae(
    vae=vae,
    optimizer=optimizer,
    losses=losses,
    filename='models/minecraft_vae_demo.pth'
)

# Load the VAE (just to demonstrate loading)
vae, optimizer, _ = load_vae(
    filename='models/minecraft_vae_demo.pth',
    vae=vae,
    optimizer=optimizer
)

# Generate some samples
print("\nGenerating samples...")
with torch.no_grad():
    samples = vae.sample(num_samples=2, device=device)
    print(f"Generated {len(samples)} samples of shape {samples.shape}")

# Print the most common blocks in the dataset
print("\nMost common blocks in the dataset:")
for i in range(min(5, len(dataset.idx_to_block))):
    print(f"  {i}: {dataset.idx_to_block.get(i, '<unknown>')}")

print("\nExample complete! You can now use the VAE for your own projects.")
print("Next steps:")
print("1. Train the VAE on your own Minecraft schematic files")
print("2. Use the VAE with the Conditional Flow Matching network")
print("3. Generate new Minecraft maps on the fly")