#!/usr/bin/env python3
"""
Variational Autoencoder (VAE) for Minecraft maps.
This VAE can encode and decode 3D Minecraft maps, taking into account the mask.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CyclicLR
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from minecraft_dataset import MinecraftSchematicDataset


class MinecraftVAE(nn.Module):
    """
    Variational Autoencoder for Minecraft maps.

    This VAE encodes 3D Minecraft maps into a latent space and can decode them back.
    It takes into account the mask to handle variable-sized inputs.
    """

    def __init__(
        self,
        num_blocks,
        chunk_size=16,
        embedding_dim=32,
        latent_dim=128,
        hidden_dims=[64, 128, 256],
    ):
        """
        Initialize the VAE.

        Args:
            num_blocks (int): Number of unique block types
            chunk_size (int): Size of the chunks (default: 16)
            embedding_dim (int): Dimension of the block embeddings
            latent_dim (int): Dimension of the latent space
            hidden_dims (list): Dimensions of the hidden layers
        """
        super(MinecraftVAE, self).__init__()

        self.num_blocks = num_blocks
        self.chunk_size = chunk_size
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        # Block embedding layer
        self.block_embedding = nn.Embedding(num_blocks, embedding_dim)

        # Encoder
        encoder_layers = []

        # Input shape: (batch_size, embedding_dim, chunk_size, chunk_size, chunk_size)
        in_channels = embedding_dim

        for h_dim in hidden_dims:
            encoder_layers.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm3d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Calculate the size of the flattened features after the encoder
        # For chunk_size=16, after 3 layers with stride 2, the size is 2x2x2
        self.flatten_size = (
            hidden_dims[-1] * (chunk_size // (2 ** len(hidden_dims))) ** 3
        )

        # Latent space
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_var = nn.Linear(self.flatten_size, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.flatten_size)

        # Decoder layers
        decoder_layers = []

        # Reverse the hidden dimensions
        hidden_dims_reversed = list(reversed(hidden_dims))

        for i in range(len(hidden_dims_reversed) - 1):
            decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose3d(
                        hidden_dims_reversed[i],
                        hidden_dims_reversed[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm3d(hidden_dims_reversed[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        # Final layer
        decoder_layers.append(
            nn.Sequential(
                nn.ConvTranspose3d(
                    hidden_dims_reversed[-1],
                    hidden_dims_reversed[-1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.BatchNorm3d(hidden_dims_reversed[-1]),
                nn.LeakyReLU(),
                nn.Conv3d(
                    hidden_dims_reversed[-1], embedding_dim, kernel_size=3, padding=1
                ),
            )
        )

        self.decoder = nn.Sequential(*decoder_layers)

        # Output layer to predict block types
        self.block_predictor = nn.Conv3d(embedding_dim, num_blocks, kernel_size=1)

    def encode(self, x, mask=None):
        """
        Encode the input into the latent space.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, chunk_size, chunk_size, chunk_size)
            mask (torch.Tensor, optional): Mask tensor of shape (batch_size, chunk_size, chunk_size, chunk_size)

        Returns:
            tuple: (mu, log_var) of the latent space
        """
        # Embed the blocks
        # x shape: (batch_size, chunk_size, chunk_size, chunk_size)
        # embedded shape: (batch_size, chunk_size, chunk_size, chunk_size, embedding_dim)
        embedded = self.block_embedding(x)

        # Permute to get the embedding dimension as channels
        # embedded shape: (batch_size, embedding_dim, chunk_size, chunk_size, chunk_size)
        embedded = embedded.permute(0, 4, 1, 2, 3)

        # Apply mask if provided
        if mask is not None:
            # Expand mask to match embedding dimension
            # mask shape: (batch_size, 1, chunk_size, chunk_size, chunk_size)
            mask_expanded = mask.unsqueeze(1)

            # Apply mask
            # embedded shape: (batch_size, embedding_dim, chunk_size, chunk_size, chunk_size)
            embedded = embedded * mask_expanded

        # Encode
        # encoded shape: (batch_size, hidden_dims[-1], chunk_size/8, chunk_size/8, chunk_size/8)
        encoded = self.encoder(embedded)

        # Flatten
        # flattened shape: (batch_size, flatten_size)
        flattened = encoded.reshape(encoded.size(0), -1)

        # Get latent parameters
        mu = self.fc_mu(flattened)
        log_var = self.fc_var(flattened)

        return mu, log_var

    def decode(self, z):
        """
        Decode from the latent space.

        Args:
            z (torch.Tensor): Latent vector of shape (batch_size, latent_dim)

        Returns:
            torch.Tensor: Reconstructed blocks of shape (batch_size, chunk_size, chunk_size, chunk_size)
        """
        # Decode from latent space
        # z shape: (batch_size, latent_dim)
        # decoded shape: (batch_size, flatten_size)
        decoded = self.decoder_input(z)

        # Reshape to 3D volume
        # reshaped shape: (batch_size, hidden_dims[-1], chunk_size/8, chunk_size/8, chunk_size/8)
        last_h_dim = self.hidden_dims[-1]
        spatial_dim = self.chunk_size // (2 ** len(self.hidden_dims))
        reshaped = decoded.reshape(
            -1, last_h_dim, spatial_dim, spatial_dim, spatial_dim
        )

        # Decode
        # decoded shape: (batch_size, embedding_dim, chunk_size, chunk_size, chunk_size)
        decoded = self.decoder(reshaped)

        # Predict block types
        # logits shape: (batch_size, num_blocks, chunk_size, chunk_size, chunk_size)
        logits = self.block_predictor(decoded)

        # Permute to get the block dimension at the end
        # logits shape: (batch_size, chunk_size, chunk_size, chunk_size, num_blocks)
        logits = logits.permute(0, 2, 3, 4, 1)

        return logits

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from the latent space.

        Args:
            mu (torch.Tensor): Mean of the latent Gaussian
            log_var (torch.Tensor): Log variance of the latent Gaussian

        Returns:
            torch.Tensor: Sampled latent vector
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, mask=None):
        """
        Forward pass through the VAE.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, chunk_size, chunk_size, chunk_size)
            mask (torch.Tensor, optional): Mask tensor of shape (batch_size, chunk_size, chunk_size, chunk_size)

        Returns:
            tuple: (reconstructed, mu, log_var)
        """
        # Encode
        mu, log_var = self.encode(x, mask)

        # Sample from the latent space
        z = self.reparameterize(mu, log_var)

        # Decode
        reconstructed = self.decode(z)

        return reconstructed, mu, log_var

    def sample(self, num_samples=1, device="cuda"):
        """
        Sample from the latent space and decode.

        Args:
            num_samples (int): Number of samples to generate
            device (str): Device to use

        Returns:
            torch.Tensor: Generated samples
        """
        # Sample from the latent space
        z = torch.randn(num_samples, self.latent_dim).to(device)

        # Decode
        samples = self.decode(z)

        # Get the most likely block for each position
        # samples shape: (num_samples, chunk_size, chunk_size, chunk_size, num_blocks)
        # blocks shape: (num_samples, chunk_size, chunk_size, chunk_size)
        blocks = torch.argmax(samples, dim=-1)

        return blocks

    def interpolate(self, x1, x2, mask1=None, mask2=None, steps=10, device="cuda"):
        """
        Interpolate between two inputs in the latent space.

        Args:
            x1 (torch.Tensor): First input tensor
            x2 (torch.Tensor): Second input tensor
            mask1 (torch.Tensor, optional): Mask for the first input
            mask2 (torch.Tensor, optional): Mask for the second input
            steps (int): Number of interpolation steps
            device (str): Device to use

        Returns:
            torch.Tensor: Interpolated samples
        """
        # Encode both inputs
        mu1, log_var1 = self.encode(
            x1.unsqueeze(0), mask1.unsqueeze(0) if mask1 is not None else None
        )
        mu2, log_var2 = self.encode(
            x2.unsqueeze(0), mask2.unsqueeze(0) if mask2 is not None else None
        )

        # Interpolate in the latent space
        interpolated = []
        for alpha in np.linspace(0, 1, steps):
            mu_interp = mu1 * (1 - alpha) + mu2 * alpha
            z = mu_interp  # No sampling, just use the mean

            # Decode
            logits = self.decode(z)
            blocks = torch.argmax(logits, dim=-1)
            interpolated.append(blocks[0])  # Remove batch dimension

        return torch.stack(interpolated)


def count_blocks_in_batch(batch_data):
    """
    Count block occurrences in a single batch.
    This function is designed to be used with multiprocessing.
    
    Args:
        batch_data (tuple): Tuple containing (blocks, mask, num_classes)
        
    Returns:
        torch.Tensor: Histogram of block counts
    """
    blocks, mask, num_classes = batch_data
    # Only count blocks where mask is 1
    valid_blocks = blocks[mask.bool()]
    
    # Update counts using histogram
    return torch.histc(
        valid_blocks.float(), bins=num_classes, min=0, max=num_classes - 1
    )


def count_blocks_parallel(dataloader, num_classes, device="cuda", cache_file=None, force_recount=False):
    """
    Count block occurrences in parallel using multiprocessing.
    Also supports caching results to avoid recounting.
    
    Args:
        dataloader (DataLoader): DataLoader for the training data
        num_classes (int): Number of block types
        device (str): Device to use
        cache_file (str, optional): Path to cache file
        force_recount (bool): Whether to force recounting even if cache exists
        
    Returns:
        torch.Tensor: Tensor of class counts
    """
    import os
    import multiprocessing as mp
    from functools import partial
    
    # Check if cache exists and use it if available
    if cache_file and os.path.exists(cache_file) and not force_recount:
        print(f"Loading block counts from cache: {cache_file}")
        try:
            class_counts = torch.load(cache_file)
            print(f"Successfully loaded cached block counts with shape: {class_counts.shape}")
            return class_counts.to(device)
        except Exception as e:
            print(f"Error loading cache: {e}. Will recount blocks.")
    
    print("Counting block occurrences in parallel...")
    
    # Prepare batches for parallel processing
    # Move data to CPU for multiprocessing
    batches = []
    for batch in dataloader:
        blocks = batch["blocks"].cpu()
        mask = batch["mask"].cpu()
        batches.append((blocks, mask, num_classes))
    
    # Use multiprocessing to count blocks in parallel
    num_processes = min(mp.cpu_count(), 8)  # Limit to 8 processes max
    print(f"Using {num_processes} processes for parallel counting")
    
    with mp.Pool(processes=num_processes) as pool:
        # Process batches in parallel
        results = list(tqdm(
            pool.imap(count_blocks_in_batch, batches),
            total=len(batches),
            desc="Counting block types"
        ))
    
    # Combine results
    class_counts = torch.zeros(num_classes)
    for result in results:
        class_counts += result
    
    # Move to specified device
    class_counts = class_counts.to(device)
    
    # Cache results if cache_file is provided
    if cache_file:
        print(f"Saving block counts to cache: {cache_file}")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        torch.save(class_counts.cpu(), cache_file)
    
    return class_counts


def calculate_class_weights(dataloader, num_classes, device="cuda", method="effective_samples", beta=0.9999, 
                           cache_file="cache/block_counts.pt", force_recount=False):
    """
    Calculate class weights based on frequency in the dataset.
    Rare classes get higher weights, common classes get lower weights.
    
    Several methods are available to prevent extremely low weights for common classes:
    - 'inverse': Standard inverse frequency (1/count)
    - 'log': Log-based weighting (log(N/count))
    - 'sqrt': Square root weighting (sqrt(N/count))
    - 'effective_samples': Based on effective number of samples (1-beta)/(1-beta^count)
    - 'balanced': Balanced approach that ensures no class gets zero weight

    Args:
        dataloader (DataLoader): DataLoader for the training data
        num_classes (int): Number of block types
        device (str): Device to use
        method (str): Method to calculate weights ('inverse', 'log', 'sqrt', 'effective_samples', 'balanced')
        beta (float): Parameter for effective number of samples method (0.9-0.999)
        cache_file (str): Path to cache file for block counts
        force_recount (bool): Whether to force recounting even if cache exists

    Returns:
        torch.Tensor: Class weights tensor
    """
    print(f"Calculating class weights using '{method}' method...")
    
    # Count blocks using parallel processing and caching
    class_counts = count_blocks_parallel(
        dataloader, 
        num_classes, 
        device=device,
        cache_file=cache_file,
        force_recount=force_recount
    )

    # Calculate weights based on the selected method
    epsilon = 1e-6  # Small value to avoid division by zero
    total_samples = class_counts.sum()
    
    if method == 'inverse':
        # Standard inverse frequency (with minimum threshold to avoid too small weights)
        min_weight_threshold = 0.01
        class_weights = 1.0 / (class_counts + epsilon)
        class_weights = torch.clamp(class_weights, min=min_weight_threshold)
    
    elif method == 'log':
        # Log-based weighting
        class_weights = torch.log(total_samples / (class_counts + epsilon) + 1.0)
    
    elif method == 'sqrt':
        # Square root weighting
        class_weights = torch.sqrt(total_samples / (class_counts + epsilon))
    
    elif method == 'effective_samples':
        # Based on "Class-Balanced Loss Based on Effective Number of Samples"
        # Formula: (1-beta)/(1-beta^n)
        class_weights = (1.0 - beta) / (1.0 - torch.pow(beta, class_counts + epsilon))
    
    else:  # 'balanced' (default fallback)
        # Balanced approach that ensures no class gets zero weight
        # Uses a combination of inverse frequency and minimum threshold
        class_weights = total_samples / (class_counts * num_classes + epsilon)
        # Ensure minimum weight is at least 10% of the maximum weight
        max_weight = class_weights.max()
        min_weight = max_weight * 0.1
        class_weights = torch.clamp(class_weights, min=min_weight)
    
    # Normalize weights to sum to num_classes
    class_weights = class_weights * (num_classes / class_weights.sum())

    # Print the top 5 most common and top 5 rarest blocks
    sorted_indices = torch.argsort(class_counts, descending=True)
    print("\nTop 5 most common blocks (lower weight):")
    for i in range(min(5, num_classes)):
        idx = sorted_indices[i].item()
        print(
            f"  Block ID {idx}: count={class_counts[idx]:.0f}, weight={class_weights[idx]:.4f}"
        )

    print("\nTop 5 rarest blocks (higher weight):")
    for i in range(min(5, num_classes)):
        idx = sorted_indices[-(i + 1)].item()
        print(
            f"  Block ID {idx}: count={class_counts[idx]:.0f}, weight={class_weights[idx]:.4f}"
        )
    
    # Visualize class weights distribution
    plot_class_weights(class_weights, class_counts)
    
    return class_weights


def vae_loss_function(
    reconstructed, target, mu, log_var, mask=None, kld_weight=0.01, class_weights=None
):
    """
    VAE loss function.

    Args:
        reconstructed (torch.Tensor): Reconstructed output logits
        target (torch.Tensor): Target tensor
        mu (torch.Tensor): Mean of the latent Gaussian
        log_var (torch.Tensor): Log variance of the latent Gaussian
        mask (torch.Tensor, optional): Mask tensor
        kld_weight (float): Weight for the KL divergence term
        class_weights (torch.Tensor, optional): Weights for each class to handle imbalance

    Returns:
        tuple: (total_loss, reconstruction_loss, kld_loss)
    """
    # Reconstruction loss (cross-entropy)
    # reconstructed shape: (batch_size, chunk_size, chunk_size, chunk_size, num_blocks)
    # target shape: (batch_size, chunk_size, chunk_size, chunk_size)
    if class_weights is not None:
        # Use weighted cross-entropy loss
        recon_loss = F.cross_entropy(
            reconstructed.reshape(-1, reconstructed.size(-1)),
            target.reshape(-1),
            weight=class_weights,
            reduction="none",
        )
    else:
        # Use standard cross-entropy loss
        recon_loss = F.cross_entropy(
            reconstructed.reshape(-1, reconstructed.size(-1)),
            target.reshape(-1),
            reduction="none",
        )

    # Apply mask if provided
    if mask is not None:
        # Reshape mask to match recon_loss
        mask_flat = mask.reshape(-1)

        # Apply mask
        recon_loss = recon_loss * mask_flat

        # Take mean over valid positions
        recon_loss = recon_loss.sum() / mask_flat.sum()
    else:
        recon_loss = recon_loss.mean()

    # KL divergence
    kld_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    # Total loss
    total_loss = recon_loss + kld_weight * kld_loss

    return total_loss, recon_loss, kld_loss


def compare_weighting_methods(dataloader, num_classes, device="cuda", cache_file="cache/block_counts.pt", force_recount=False):
    """
    Compare different class weighting methods and visualize their effects.
    
    Args:
        dataloader (DataLoader): DataLoader for the training data
        num_classes (int): Number of block types
        device (str): Device to use
        cache_file (str): Path to cache file for block counts
        force_recount (bool): Whether to force recounting even if cache exists
    
    Returns:
        dict: Dictionary of class weights for each method
    """
    print("Comparing different class weighting methods...")
    
    # Count blocks using parallel processing and caching
    class_counts = count_blocks_parallel(
        dataloader, 
        num_classes, 
        device=device,
        cache_file=cache_file,
        force_recount=force_recount
    )
    
    # Calculate weights using different methods
    methods = ['inverse', 'log', 'sqrt', 'effective_samples', 'balanced']
    weights_dict = {}
    
    epsilon = 1e-6
    total_samples = class_counts.sum()
    
    # Standard inverse frequency
    weights = 1.0 / (class_counts + epsilon)
    weights = weights * (num_classes / weights.sum())
    weights_dict['inverse'] = weights
    
    # Log-based weighting
    weights = torch.log(total_samples / (class_counts + epsilon) + 1.0)
    weights = weights * (num_classes / weights.sum())
    weights_dict['log'] = weights
    
    # Square root weighting
    weights = torch.sqrt(total_samples / (class_counts + epsilon))
    weights = weights * (num_classes / weights.sum())
    weights_dict['sqrt'] = weights
    
    # Effective number of samples (beta=0.9999)
    beta = 0.9999
    weights = (1.0 - beta) / (1.0 - torch.pow(beta, class_counts + epsilon))
    weights = weights * (num_classes / weights.sum())
    weights_dict['effective_samples'] = weights
    
    # Balanced approach
    weights = total_samples / (class_counts * num_classes + epsilon)
    max_weight = weights.max()
    min_weight = max_weight * 0.1
    weights = torch.clamp(weights, min=min_weight)
    weights = weights * (num_classes / weights.sum())
    weights_dict['balanced'] = weights
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Get indices of top classes by count
    top_indices = torch.argsort(class_counts, descending=True)[:20].cpu().numpy()
    
    # Plot class counts
    plt.subplot(2, 1, 1)
    plt.bar(range(len(top_indices)), class_counts[top_indices].cpu().numpy())
    plt.yscale('log')
    plt.title('Class Distribution (Top 20 Classes)')
    plt.xlabel('Class Index')
    plt.ylabel('Count (log scale)')
    plt.xticks(range(len(top_indices)), [str(idx) for idx in top_indices], rotation=90)
    
    # Plot weights from different methods
    plt.subplot(2, 1, 2)
    x = np.arange(len(top_indices))
    width = 0.15
    offsets = [-2, -1, 0, 1, 2]
    
    for i, (method, offset) in enumerate(zip(methods, offsets)):
        plt.bar(x + offset * width, weights_dict[method][top_indices].cpu().numpy(), 
                width=width, label=method)
    
    plt.title('Class Weights Comparison')
    plt.xlabel('Class Index')
    plt.ylabel('Weight')
    plt.xticks(range(len(top_indices)), [str(idx) for idx in top_indices], rotation=90)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('class_weights_comparison.png')
    plt.close()
    print("Class weights comparison saved to 'class_weights_comparison.png'")
    
    return weights_dict


def plot_class_weights(class_weights, class_counts, num_to_show=20):
    """
    Visualize the class weights distribution.
    
    Args:
        class_weights (torch.Tensor): Class weights tensor
        class_counts (torch.Tensor): Class counts tensor
        num_to_show (int): Number of classes to show in the plot
    """
    # Move tensors to CPU for plotting
    weights = class_weights.cpu().numpy()
    counts = class_counts.cpu().numpy()
    
    # Get indices of top classes by count
    top_indices = np.argsort(counts)[::-1][:num_to_show]
    
    # Extract data for these classes
    selected_weights = weights[top_indices]
    selected_counts = counts[top_indices]
    selected_indices = top_indices
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot class counts (log scale)
    ax1.bar(range(num_to_show), selected_counts)
    ax1.set_yscale('log')
    ax1.set_xlabel('Block ID Index')
    ax1.set_ylabel('Count (log scale)')
    ax1.set_title('Class Distribution (Top Classes)')
    ax1.set_xticks(range(num_to_show))
    ax1.set_xticklabels([str(idx) for idx in selected_indices], rotation=90)
    
    # Plot class weights
    ax2.bar(range(num_to_show), selected_weights)
    ax2.set_xlabel('Block ID Index')
    ax2.set_ylabel('Weight')
    ax2.set_title('Class Weights')
    ax2.set_xticks(range(num_to_show))
    ax2.set_xticklabels([str(idx) for idx in selected_indices], rotation=90)
    
    plt.tight_layout()
    plt.savefig('class_weights_distribution.png')
    plt.close()
    print("Class weights visualization saved to 'class_weights_distribution.png'")


def plot_lr_schedule(scheduler, num_steps):
    """
    Visualize the learning rate schedule.
    
    Args:
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler
        num_steps (int): Number of steps to visualize
    """
    # Store original LR
    original_lrs = []
    for param_group in scheduler.optimizer.param_groups:
        original_lrs.append(param_group['lr'])
    
    # Get learning rates for each step
    lrs = []
    for i in range(num_steps):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()
    
    # Reset scheduler and optimizer to original state
    for i, param_group in enumerate(scheduler.optimizer.param_groups):
        param_group['lr'] = original_lrs[i]
    scheduler.base_lrs = [original_lrs[0]]  # Reset base_lrs
    
    # Plot learning rates
    plt.figure(figsize=(10, 5))
    plt.plot(lrs)
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('CyclicLR Schedule')
    plt.grid(True)
    plt.savefig('cyclic_lr_schedule.png')
    plt.close()


def train_vae(
    vae,
    dataloader,
    optimizer,
    val_dataloader=None,
    device="cuda",
    epochs=10,
    kld_weight=0.01,
    use_class_weights=False,
    scheduler=None,
    args=None,  # Command-line arguments
):
    """
    Train the VAE.

    Args:
        vae (MinecraftVAE): The VAE model
        dataloader (DataLoader): DataLoader for the training data
        optimizer (torch.optim.Optimizer): Optimizer
        val_dataloader (DataLoader, optional): DataLoader for validation data
        device (str): Device to use
        epochs (int): Number of epochs
        kld_weight (float): Weight for the KL divergence term
        use_class_weights (bool): Whether to use class weights to handle imbalance
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler
        args (argparse.Namespace, optional): Command-line arguments containing weight_method and beta

    Returns:
        list: Training losses
    """
    losses = []

    # Calculate class weights if needed
    class_weights = None
    if use_class_weights:
        # Use the method specified by command-line arguments or the default
        cache_file = None if (args and args.no_cache) else (args.cache_file if args else "cache/block_counts.pt")
        force_recount = args.force_recount if args else False
        
        class_weights = calculate_class_weights(
            dataloader, 
            vae.num_blocks, 
            device,
            method=args.weight_method if args else "effective_samples",
            beta=args.beta if args else 0.9999,
            cache_file=cache_file,
            force_recount=force_recount
        )

    for epoch in range(epochs):
        # Training phase
        vae.train()
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kld_loss = 0
        epoch_correct = 0
        epoch_valid_positions = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in progress_bar:
            # Get data
            blocks = batch["blocks"].to(device)
            mask = batch["mask"].to(device)

            # Forward pass
            reconstructed, mu, log_var = vae(blocks, mask)

            # Calculate loss
            loss, recon_loss, kld_loss = vae_loss_function(
                reconstructed, blocks, mu, log_var, mask, kld_weight, class_weights
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            optimizer.step()
            
            # Step the scheduler if provided
            if scheduler is not None:
                scheduler.step()
                # Update progress bar to show current learning rate
                current_lr = scheduler.get_last_lr()[0]
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.6f}"})

            # Calculate accuracy
            with torch.no_grad():
                pred_blocks = torch.argmax(reconstructed, dim=-1)
                correct = (pred_blocks == blocks) & (mask.bool())
                epoch_correct += correct.sum().item()
                epoch_valid_positions += mask.sum().item()

            # Update progress bar
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kld_loss += kld_loss.item()

            # Calculate current accuracy
            current_accuracy = (
                epoch_correct / epoch_valid_positions
                if epoch_valid_positions > 0
                else 0
            )

            progress_bar.set_postfix(
                {
                    "loss": epoch_loss / (progress_bar.n + 1),
                    "recon_loss": epoch_recon_loss / (progress_bar.n + 1),
                    "kld_loss": epoch_kld_loss / (progress_bar.n + 1),
                    "acc": current_accuracy,
                }
            )

        # Calculate average losses and accuracy
        avg_loss = epoch_loss / len(dataloader)
        avg_recon_loss = epoch_recon_loss / len(dataloader)
        avg_kld_loss = epoch_kld_loss / len(dataloader)
        avg_accuracy = (
            epoch_correct / epoch_valid_positions if epoch_valid_positions > 0 else 0
        )

        print(f"Epoch {epoch + 1}/{epochs}:")
        print(f"  Train Loss: {avg_loss:.4f}")
        print(f"  Train Reconstruction Loss: {avg_recon_loss:.4f}")
        print(f"  Train KLD Loss: {avg_kld_loss:.4f}")
        print(
            f"  Train Accuracy: {avg_accuracy:.4f} ({epoch_correct}/{epoch_valid_positions})"
        )

        # Validation phase
        if val_dataloader is not None:
            # Use the evaluate_vae function with verbose=False to avoid duplicate printing
            val_results = evaluate_vae(
                vae=vae,
                dataloader=val_dataloader,
                device=device,
                kld_weight=kld_weight,
                num_samples=0,  # Don't need samples during training
                verbose=False,
                class_weights=class_weights,
            )

            # Print validation results
            print(f"  Val Loss: {val_results['loss']:.4f}")
            print(f"  Val Reconstruction Loss: {val_results['recon_loss']:.4f}")
            print(f"  Val KLD Loss: {val_results['kld_loss']:.4f}")
            print(
                f"  Val Accuracy: {val_results['accuracy']:.4f} ({val_results['correct']}/{val_results['total']})"
            )

            # Save loss with validation
            losses.append(
                {
                    "total": avg_loss,
                    "reconstruction": avg_recon_loss,
                    "kld": avg_kld_loss,
                    "accuracy": avg_accuracy,
                    "val_total": val_results["loss"],
                    "val_reconstruction": val_results["recon_loss"],
                    "val_kld": val_results["kld_loss"],
                    "val_accuracy": val_results["accuracy"],
                }
            )
        else:
            # Save loss without validation
            losses.append(
                {
                    "total": avg_loss,
                    "reconstruction": avg_recon_loss,
                    "kld": avg_kld_loss,
                    "accuracy": avg_accuracy,
                }
            )

    return losses


def evaluate_vae(
    vae,
    dataloader,
    device="cuda",
    kld_weight=0.01,
    num_samples=5,
    verbose=True,
    class_weights=None,
):
    """
    Evaluate the VAE on a validation set.

    Args:
        vae (MinecraftVAE): The VAE model
        dataloader (DataLoader): DataLoader for the validation data
        device (str): Device to use
        kld_weight (float): Weight for the KL divergence term
        num_samples (int): Number of samples to visualize
        verbose (bool): Whether to print evaluation results
        class_weights (torch.Tensor, optional): Weights for each class to handle imbalance
    Returns:
        dict: Evaluation metrics
    """
    vae.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kld_loss = 0
    total_correct = 0
    total_valid_positions = 0

    # Sample some examples for visualization
    samples = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # Get data
            blocks = batch["blocks"].to(device)
            mask = batch["mask"].to(device)

            # Forward pass
            reconstructed, mu, log_var = vae(blocks, mask)

            # Calculate loss
            loss, recon_loss, kld_loss = vae_loss_function(
                reconstructed, blocks, mu, log_var, mask, kld_weight, class_weights
            )

            # Update totals
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kld_loss += kld_loss.item()

            # Calculate accuracy
            pred_blocks = torch.argmax(reconstructed, dim=-1)
            correct = (pred_blocks == blocks) & (mask.bool())
            total_correct += correct.sum().item()
            total_valid_positions += mask.sum().item()

            # Store samples for visualization
            if i < num_samples:
                samples.append(
                    {
                        "input": blocks[0].cpu(),
                        "mask": mask[0].cpu(),
                        "reconstructed": pred_blocks[0].cpu(),
                    }
                )

    # Calculate average losses and accuracy
    avg_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_kld_loss = total_kld_loss / len(dataloader)
    accuracy = total_correct / total_valid_positions if total_valid_positions > 0 else 0

    if verbose:
        print(f"Evaluation:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Reconstruction Loss: {avg_recon_loss:.4f}")
        print(f"  KLD Loss: {avg_kld_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f} ({total_correct}/{total_valid_positions})")

    return {
        "loss": avg_loss,
        "recon_loss": avg_recon_loss,
        "kld_loss": avg_kld_loss,
        "accuracy": accuracy,
        "correct": total_correct,
        "total": total_valid_positions,
        "samples": samples,
    }


def save_vae(vae, optimizer, losses, filename):
    """
    Save the VAE model and training state.

    Args:
        vae (MinecraftVAE): The VAE model
        optimizer (torch.optim.Optimizer): Optimizer
        losses (list): Training losses
        filename (str): Filename to save to
    """
    torch.save(
        {
            "model_state_dict": vae.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "losses": losses,
        },
        filename,
    )
    print(f"Model saved to {filename}")


def load_vae(filename, vae, optimizer=None):
    """
    Load a saved VAE model.

    Args:
        filename (str): Filename to load from
        vae (MinecraftVAE): The VAE model
        optimizer (torch.optim.Optimizer, optional): Optimizer

    Returns:
        tuple: (vae, optimizer, losses)
    """
    checkpoint = torch.load(filename)
    vae.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    losses = checkpoint.get("losses", [])

    print(f"Model loaded from {filename}")

    return vae, optimizer, losses


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a VAE for Minecraft structures")
    parser.add_argument(
        "--weight-method", 
        type=str, 
        default="effective_samples",
        choices=["inverse", "log", "sqrt", "effective_samples", "balanced"],
        help="Method to calculate class weights (default: effective_samples)"
    )
    parser.add_argument(
        "--beta", 
        type=float, 
        default=0.9999,
        help="Beta parameter for effective_samples method (default: 0.9999)"
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        default="cache/block_counts.pt",
        help="Path to cache file for block counts (default: cache/block_counts.pt)"
    )
    parser.add_argument(
        "--force-recount",
        action="store_true",
        help="Force recounting blocks even if cache exists"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching of block counts"
    )
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create cache directory if it doesn't exist
    os.makedirs("cache", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Create the dataset
    dataset = MinecraftSchematicDataset(
        schematics_dir="minecraft-schematics-raw",
        chunk_size=16,
        cache_file="cache/block_mappings.pkl",
        max_files=None,  # Use all files
        # preload=True,
    )

    # Split into train and validation sets
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True
    )

    # Create the VAE
    vae = MinecraftVAE(
        num_blocks=len(dataset.block_to_idx),
        chunk_size=16,
        embedding_dim=32,
        latent_dim=128,
        hidden_dims=[64, 128, 256],
    ).to(device)

    # Create optimizer
    optimizer = torch.optim.AdamW(vae.parameters(), lr=1e-3)

    # Compare different class weighting methods to help choose the best one
    print("\nComparing different class weighting methods...")
    compare_weighting_methods(train_dataloader, vae.num_blocks, device)
    print("You can examine 'class_weights_comparison.png' to choose the best method")
    
    # Create CyclicLR scheduler
    # Using triangular2 mode which reduces the amplitude by half each cycle
    # This is often more performant for neural networks
    step_size_up = len(train_dataloader) * 2  # Size of one cycle = 2 epochs
    scheduler = CyclicLR(
        optimizer,
        base_lr=1e-4,  # Lower learning rate boundary
        max_lr=5e-3,   # Upper learning rate boundary
        step_size_up=step_size_up,
        mode='triangular2',  # Triangular cycle with amplitude halved each time
        cycle_momentum=False  # AdamW doesn't use momentum
    )

    # Visualize the learning rate schedule for one cycle
    print("\nGenerating learning rate schedule visualization...")
    plot_lr_schedule(scheduler, step_size_up * 2)  # Plot for one complete cycle
    print("Learning rate schedule visualization saved to 'cyclic_lr_schedule.png'")

    # Train the VAE
    losses = train_vae(
        vae=vae,
        use_class_weights=True,  # Enable class weights with our improved weighting method
        dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        device=device,
        epochs=100,
        kld_weight=0.01,
        scheduler=scheduler,
        args=args,  # Pass command-line arguments
    )

    # Evaluate the VAE
    eval_results = evaluate_vae(vae=vae, dataloader=val_dataloader, device=device)

    # Save the VAE
    save_vae(
        vae=vae, optimizer=optimizer, losses=losses, filename="models/minecraft_vae.pth"
    )

    # Generate some samples
    print("Generating samples...")
    samples = vae.sample(num_samples=5, device=device)
    print(f"Generated {len(samples)} samples of shape {samples.shape}")
