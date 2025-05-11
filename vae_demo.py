#!/usr/bin/env python3
"""
Demo script for the Minecraft VAE.
This script demonstrates how to use the VAE for encoding and decoding Minecraft maps,
and how it can be integrated with a Conditional Flow Matching network.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from minecraft_dataset import MinecraftSchematicDataset
from minecraft_vae import MinecraftVAE, train_vae, evaluate_vae, save_vae, load_vae, calculate_class_weights

def visualize_3d_blocks(blocks, mask=None, block_to_color=None, title=None, ax=None):
    """
    Visualize a 3D array of blocks.
    
    Args:
        blocks (torch.Tensor or np.ndarray): 3D array of block indices
        mask (torch.Tensor or np.ndarray, optional): Mask tensor
        block_to_color (dict, optional): Mapping from block indices to colors
        title (str, optional): Title for the plot
        ax (matplotlib.axes.Axes, optional): Axes to plot on
    """
    if isinstance(blocks, torch.Tensor):
        blocks = blocks.cpu().numpy()
    
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    # Default color mapping
    if block_to_color is None:
        # Create a colormap
        import matplotlib.colors as mcolors
        try:
            # For matplotlib >= 3.7
            cmap = plt.colormaps['tab20']
        except:
            # For older matplotlib versions
            cmap = plt.cm.get_cmap('tab20', 20)
        
        # Create a mapping from block indices to colors
        unique_blocks = np.unique(blocks)
        block_to_color = {}
        for i, block_idx in enumerate(unique_blocks):
            block_to_color[block_idx] = cmap(i % 20)
    
    # Create a figure if not provided
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
    
    # Plot the blocks as scatter points instead of voxels
    # This is more compatible across matplotlib versions
    points = []
    colors = []
    
    for i in range(blocks.shape[0]):
        for j in range(blocks.shape[1]):
            for k in range(blocks.shape[2]):
                block_idx = blocks[i, j, k]
                
                # Skip air blocks (assuming 0 is air/padding)
                if block_idx == 0:
                    continue
                
                # Skip masked blocks
                if mask is not None and mask[i, j, k] == 0:
                    continue
                
                points.append((i, j, k))
                colors.append(block_to_color.get(block_idx, [0, 0, 0, 1]))
    
    if points:
        points = np.array(points)
        colors = np.array(colors)
        
        # Plot as scatter
        ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            c=colors, marker='s', s=100, alpha=0.7
        )
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    
    # Set title
    if title:
        ax.set_title(title)
    
    return ax


def visualize_latent_space(vae, dataloader, device="cuda", n_samples=1000, perplexity=30):
    """
    Visualize the latent space using t-SNE.
    
    Args:
        vae (MinecraftVAE): The VAE model
        dataloader (DataLoader): DataLoader for the data
        device (str): Device to use
        n_samples (int): Number of samples to use
        perplexity (int): Perplexity for t-SNE
    """
    from sklearn.manifold import TSNE
    
    vae.eval()
    
    # Collect latent vectors
    latent_vectors = []
    block_counts = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Collecting latent vectors")):
            if len(latent_vectors) >= n_samples:
                break
            
            # Get data
            blocks = batch['blocks'].to(device)
            mask = batch['mask'].to(device)
            
            # Encode
            mu, _ = vae.encode(blocks, mask)
            
            # Store latent vectors
            latent_vectors.append(mu.cpu().numpy())
            
            # Count block types
            for j in range(blocks.shape[0]):
                # Get the most common block type (excluding air/padding)
                block_counts_j = {}
                for b, m in zip(blocks[j].flatten(), mask[j].flatten()):
                    if m > 0 and b > 0:  # Skip padding and air
                        if b.item() not in block_counts_j:
                            block_counts_j[b.item()] = 0
                        block_counts_j[b.item()] += 1
                
                # Get the most common block
                if block_counts_j:
                    most_common = max(block_counts_j.items(), key=lambda x: x[1])[0]
                    block_counts.append(most_common)
                else:
                    block_counts.append(0)
    
    # Concatenate latent vectors
    latent_vectors = np.concatenate(latent_vectors, axis=0)[:n_samples]
    block_counts = np.array(block_counts)[:n_samples]
    
    # Apply t-SNE
    print("Applying t-SNE...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    latent_2d = tsne.fit_transform(latent_vectors)
    
    # Plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=block_counts, cmap='tab20', alpha=0.7)
    plt.colorbar(scatter, label='Most common block type')
    plt.title(f't-SNE visualization of the latent space (perplexity={perplexity})')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.savefig('latent_space_tsne.png')
    plt.close()
    
    print(f"t-SNE visualization saved to latent_space_tsne.png")


def generate_samples(vae, dataset, device="cuda", num_samples=5):
    """
    Generate samples from the VAE.
    
    Args:
        vae (MinecraftVAE): The VAE model
        dataset (MinecraftSchematicDataset): The dataset (for block mapping)
        device (str): Device to use
        num_samples (int): Number of samples to generate
    """
    vae.eval()
    
    # Generate samples
    with torch.no_grad():
        samples = vae.sample(num_samples=num_samples, device=device)
    
    # Create a figure
    fig = plt.figure(figsize=(20, 4 * num_samples))
    
    # Create a mapping from block indices to colors
    import matplotlib.colors as mcolors
    cmap = plt.cm.get_cmap('tab20', 20)
    
    block_to_color = {}
    for i in range(len(dataset.idx_to_block)):
        block_to_color[i] = cmap(i % 20)
    
    # Plot each sample
    for i in range(num_samples):
        ax = fig.add_subplot(num_samples, 1, i + 1, projection='3d')
        visualize_3d_blocks(
            samples[i],
            block_to_color=block_to_color,
            title=f"Generated Sample {i+1}",
            ax=ax
        )
    
    plt.tight_layout()
    plt.savefig('generated_samples.png')
    plt.close()
    
    print(f"Generated samples saved to generated_samples.png")


def analyze_class_distribution(dataloader, num_classes, device="cuda", top_n=10):
    """
    Analyze and visualize the class distribution in the dataset.
    
    Args:
        dataloader (DataLoader): DataLoader for the data
        num_classes (int): Number of block types
        device (str): Device to use
        top_n (int): Number of top classes to display
    """
    # Count occurrences of each class
    class_counts = torch.zeros(num_classes, device=device)
    total_blocks = 0
    valid_blocks = 0
    
    for batch in tqdm(dataloader, desc="Analyzing class distribution"):
        blocks = batch['blocks'].to(device)
        mask = batch['mask'].to(device)
        
        # Count total blocks
        total_blocks += blocks.numel()
        
        # Count valid blocks
        valid_blocks += mask.sum().item()
        
        # Only count blocks where mask is 1
        valid_block_indices = blocks[mask.bool()]
        
        # Update counts using histogram
        class_histogram = torch.histc(valid_block_indices.float(), bins=num_classes, min=0, max=num_classes-1)
        class_counts += class_histogram
    
    # Convert to percentages
    class_percentages = class_counts / class_counts.sum() * 100
    
    # Get the top N most common classes
    top_indices = torch.argsort(class_counts, descending=True)[:top_n]
    
    # Create a bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(
        range(top_n), 
        [class_percentages[i].item() for i in top_indices],
        tick_label=[f"Block {i.item()}" for i in top_indices]
    )
    plt.title(f"Top {top_n} Most Common Block Types")
    plt.xlabel("Block Type")
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=45)
    
    # Add percentage labels on top of each bar
    for i, idx in enumerate(top_indices):
        plt.text(
            i, 
            class_percentages[idx].item() + 0.5, 
            f"{class_percentages[idx].item():.1f}%\n({class_counts[idx].item():.0f})",
            ha='center'
        )
    
    plt.tight_layout()
    plt.savefig("class_distribution.png")
    plt.close()
    
    print(f"Class distribution analysis:")
    print(f"  Total blocks: {total_blocks}")
    print(f"  Valid blocks: {valid_blocks} ({valid_blocks/total_blocks*100:.1f}%)")
    print(f"  Number of unique block types: {(class_counts > 0).sum().item()}")
    print(f"  Most common block: Block {top_indices[0].item()} ({class_percentages[top_indices[0]].item():.1f}%)")
    print(f"  Class distribution saved to class_distribution.png")


def interpolate_samples(vae, dataloader, device="cuda", num_pairs=3, steps=10):
    """
    Interpolate between pairs of samples.
    
    Args:
        vae (MinecraftVAE): The VAE model
        dataloader (DataLoader): DataLoader for the data
        device (str): Device to use
        num_pairs (int): Number of pairs to interpolate
        steps (int): Number of interpolation steps
    """
    vae.eval()
    
    # Get some samples
    samples = []
    masks = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Get data
            blocks = batch['blocks']
            mask = batch['mask']
            
            # Store samples
            for i in range(blocks.shape[0]):
                samples.append(blocks[i])
                masks.append(mask[i])
                
                if len(samples) >= num_pairs * 2:
                    break
            
            if len(samples) >= num_pairs * 2:
                break
    
    # Create a figure
    fig = plt.figure(figsize=(20, 4 * num_pairs))
    
    # Create a mapping from block indices to colors
    import matplotlib.colors as mcolors
    cmap = plt.cm.get_cmap('tab20', 20)
    
    block_to_color = {}
    for i in range(vae.num_blocks):
        block_to_color[i] = cmap(i % 20)
    
    # Interpolate between pairs
    for pair in range(num_pairs):
        # Get a pair of samples
        x1 = samples[pair * 2].to(device)
        x2 = samples[pair * 2 + 1].to(device)
        mask1 = masks[pair * 2].to(device)
        mask2 = masks[pair * 2 + 1].to(device)
        
        # Interpolate
        interpolated = vae.interpolate(x1, x2, mask1, mask2, steps=steps, device=device)
        
        # Plot the interpolation
        for i, interp in enumerate([x1] + list(interpolated) + [x2]):
            if i == 0 or i == steps + 1:
                # Original samples
                title = "Original" if i == 0 else "Target"
                mask = mask1 if i == 0 else mask2
            else:
                # Interpolated
                title = f"Interpolation {i}/{steps}"
                mask = None
            
            # Plot
            ax = fig.add_subplot(num_pairs, steps + 2, pair * (steps + 2) + i + 1, projection='3d')
            visualize_3d_blocks(
                interp,
                mask=mask,
                block_to_color=block_to_color,
                title=title,
                ax=ax
            )
    
    plt.tight_layout()
    plt.savefig('interpolated_samples.png')
    plt.close()
    
    print(f"Interpolated samples saved to interpolated_samples.png")


def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    
    # Create cache directory if it doesn't exist
    os.makedirs('cache', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Create the dataset
    dataset = MinecraftSchematicDataset(
        schematics_dir=args.data_dir,
        chunk_size=16,
        cache_file='cache/block_mappings.pkl',
        max_files=args.max_files
    )
    
    # Split into train and validation sets
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    
    # Create or load the VAE
    vae = MinecraftVAE(
        num_blocks=len(dataset.block_to_idx),
        chunk_size=16,
        embedding_dim=args.embedding_dim,
        latent_dim=args.latent_dim,
        hidden_dims=args.hidden_dims
    ).to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)
    
    # Load model if specified
    if args.load_model:
        vae, optimizer, _ = load_vae(args.load_model, vae, optimizer)
    
    # Train if specified
    if args.train:
        losses = train_vae(
            vae=vae,
            dataloader=train_dataloader,
            optimizer=optimizer,
            val_dataloader=val_dataloader,
            device=device,
            epochs=args.epochs,
            kld_weight=args.kld_weight,
            use_class_weights=args.use_class_weights
        )
        
        # Save the VAE
        save_vae(
            vae=vae,
            optimizer=optimizer,
            losses=losses,
            filename=args.save_model
        )
    
    # Evaluate if specified
    if args.evaluate:
        # Calculate class weights if needed
        class_weights = None
        if args.use_class_weights:
            class_weights = calculate_class_weights(train_dataloader, len(dataset.block_to_idx), device)
            
        eval_results = evaluate_vae(
            vae=vae,
            dataloader=val_dataloader,
            device=device,
            kld_weight=args.kld_weight,
            class_weights=class_weights
        )
    
    # Visualize latent space if specified
    if args.visualize_latent:
        visualize_latent_space(
            vae=vae,
            dataloader=val_dataloader,
            device=device,
            n_samples=args.n_samples,
            perplexity=args.perplexity
        )
    
    # Generate samples if specified
    if args.generate:
        generate_samples(
            vae=vae,
            dataset=dataset,
            device=device,
            num_samples=args.n_samples
        )
    
    # Interpolate samples if specified
    if args.interpolate:
        interpolate_samples(
            vae=vae,
            dataloader=val_dataloader,
            device=device,
            num_pairs=args.n_pairs,
            steps=args.steps
        )
    
    # Analyze class distribution if specified
    if args.analyze_distribution:
        analyze_class_distribution(
            dataloader=train_dataloader,
            num_classes=len(dataset.block_to_idx),
            device=device,
            top_n=10
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minecraft VAE Demo")
    
    # Data options
    parser.add_argument("--data_dir", type=str, default="minecraft-schematics-raw",
                        help="Directory containing schematic files")
    parser.add_argument("--max_files", type=int, default=None,
                        help="Maximum number of files to load")
    
    # Model options
    parser.add_argument("--embedding_dim", type=int, default=32,
                        help="Dimension of the block embeddings")
    parser.add_argument("--latent_dim", type=int, default=128,
                        help="Dimension of the latent space")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[64, 128, 256],
                        help="Dimensions of the hidden layers")
    
    # Training options
    parser.add_argument("--train", action="store_true",
                        help="Train the VAE")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--kld_weight", type=float, default=0.01,
                        help="Weight for the KL divergence term")
    parser.add_argument("--use_class_weights", action="store_true",
                        help="Use class weights to handle imbalanced data")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    
    # Model loading/saving options
    parser.add_argument("--load_model", type=str, default=None,
                        help="Path to load model from")
    parser.add_argument("--save_model", type=str, default="models/minecraft_vae.pth",
                        help="Path to save model to")
    
    # Evaluation options
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate the VAE")
    parser.add_argument("--visualize_latent", action="store_true",
                        help="Visualize the latent space")
    parser.add_argument("--generate", action="store_true",
                        help="Generate samples from the VAE")
    parser.add_argument("--interpolate", action="store_true",
                        help="Interpolate between samples")
    parser.add_argument("--analyze_distribution", action="store_true",
                        help="Analyze and visualize the class distribution")
    
    # Visualization options
    parser.add_argument("--n_samples", type=int, default=5,
                        help="Number of samples to generate or visualize")
    parser.add_argument("--n_pairs", type=int, default=3,
                        help="Number of pairs to interpolate between")
    parser.add_argument("--steps", type=int, default=10,
                        help="Number of interpolation steps")
    parser.add_argument("--perplexity", type=int, default=30,
                        help="Perplexity for t-SNE")
    
    # Misc options
    parser.add_argument("--cpu", action="store_true",
                        help="Use CPU instead of GPU")
    
    args = parser.parse_args()
    
    main(args)