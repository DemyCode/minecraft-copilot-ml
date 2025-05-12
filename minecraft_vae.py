#!/usr/bin/env python3
"""
Variational Autoencoder (VAE) for Minecraft maps.
This VAE can encode and decode 3D Minecraft maps, taking into account the mask.
"""

from typing import Optional
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
import gc
from torch.cuda.amp import autocast, GradScaler
import time
from functools import partial
import torch.multiprocessing as mp


class MinecraftVAE(nn.Module):
    """
    Variational Autoencoder for Minecraft maps.

    This VAE encodes 3D Minecraft maps into a latent space and can decode them back.
    It takes into account the mask to handle variable-sized inputs.

    Optimized with:
    - Efficient memory management
    - Optional JIT compilation
    - Gradient checkpointing support
    - Optimized layer configurations
    """

    def __init__(
        self,
        num_blocks,
        chunk_size=16,
        embedding_dim=32,
        latent_dim=128,
        hidden_dims=[64, 128, 256],
        use_jit=True,  # Use JIT compilation for faster inference
        use_checkpointing=True,  # Use gradient checkpointing to save memory
        use_efficient_attention=False,  # Use efficient attention mechanism (experimental)
    ):
        """
        Initialize the VAE with performance optimizations.

        Args:
            num_blocks (int): Number of unique block types
            chunk_size (int): Size of the chunks (default: 16)
            embedding_dim (int): Dimension of the block embeddings
            latent_dim (int): Dimension of the latent space
            hidden_dims (list): Dimensions of the hidden layers
            use_jit (bool): Whether to use JIT compilation for faster inference
            use_checkpointing (bool): Whether to use gradient checkpointing to save memory
            use_efficient_attention (bool): Whether to use efficient attention mechanism
        """
        super(MinecraftVAE, self).__init__()

        self.num_blocks = num_blocks
        self.chunk_size = chunk_size
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.use_jit = use_jit
        self.use_checkpointing = use_checkpointing
        self.use_efficient_attention = use_efficient_attention

        # Block embedding layer
        self.block_embedding = nn.Embedding(num_blocks, embedding_dim)

        # Encoder
        encoder_layers = []

        # Input shape: (batch_size, embedding_dim, chunk_size, chunk_size, chunk_size)
        in_channels = embedding_dim

        for h_dim in hidden_dims:
            # Use more efficient layer configuration
            encoder_layers.append(
                nn.Sequential(
                    nn.Conv3d(
                        in_channels,
                        h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm3d(h_dim),
                    nn.LeakyReLU(inplace=True),  # Use inplace operations where possible
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
                        bias=False,  # Remove bias for better performance with BatchNorm
                    ),
                    nn.BatchNorm3d(hidden_dims_reversed[i + 1]),
                    nn.LeakyReLU(inplace=True),  # Use inplace operations
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
                    bias=False,
                ),
                nn.BatchNorm3d(hidden_dims_reversed[-1]),
                nn.LeakyReLU(inplace=True),
                nn.Conv3d(
                    hidden_dims_reversed[-1], embedding_dim, kernel_size=3, padding=1
                ),
            )
        )

        self.decoder = nn.Sequential(*decoder_layers)

        # Output layer to predict block types
        self.block_predictor = nn.Sequential(
            nn.Conv3d(embedding_dim, num_blocks, kernel_size=1), nn.Sigmoid()
        )

        # JIT compile critical functions if requested
        if use_jit:
            try:
                # Only compile if torch.jit is available
                self._encode_jit = torch.jit.script(self._encode_impl)
                self._decode_jit = torch.jit.script(self._decode_impl)
                self._reparameterize_jit = torch.jit.script(self._reparameterize_impl)
            except Exception as e:
                print(
                    f"Warning: JIT compilation failed, falling back to regular functions: {e}"
                )
                self.use_jit = False

    def _encode_impl(self, x, mask=None):
        """
        Internal implementation of encode for JIT compilation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, chunk_size, chunk_size, chunk_size)
            mask (torch.Tensor, optional): Mask tensor of shape (batch_size, chunk_size, chunk_size, chunk_size)

        Returns:
            tuple: (mu, log_var) of the latent space
        """
        # Embed the blocks
        embedded = self.block_embedding(x)

        # Permute to get the embedding dimension as channels
        embedded = embedded.permute(0, 4, 1, 2, 3)

        # Apply mask if provided
        if mask is not None:
            mask_expanded = mask.unsqueeze(1)
            embedded = embedded * mask_expanded

        # Encode
        encoded = self.encoder(embedded)

        # Flatten
        flattened = encoded.reshape(encoded.size(0), -1)

        # Get latent parameters
        mu = self.fc_mu(flattened)
        log_var = self.fc_var(flattened)

        return mu, log_var

    def _decode_impl(self, z):
        """
        Internal implementation of decode for JIT compilation.

        Args:
            z (torch.Tensor): Latent vector of shape (batch_size, latent_dim)

        Returns:
            torch.Tensor: Reconstructed blocks of shape (batch_size, chunk_size, chunk_size, chunk_size, num_blocks)
        """
        # Decode from latent space
        decoded = self.decoder_input(z)

        # Reshape to 3D volume
        last_h_dim = self.hidden_dims[-1]
        spatial_dim = self.chunk_size // (2 ** len(self.hidden_dims))
        reshaped = decoded.reshape(
            -1, last_h_dim, spatial_dim, spatial_dim, spatial_dim
        )

        # Decode
        decoded = self.decoder(reshaped)

        # Predict block types
        logits = self.block_predictor(decoded)

        # Permute to get the block dimension at the end
        logits = logits.permute(0, 2, 3, 4, 1)

        return logits

    def _reparameterize_impl(self, mu, log_var):
        """
        Internal implementation of reparameterize for JIT compilation.

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

    def encode(self, x, mask=None):
        """
        Encode the input into the latent space.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, chunk_size, chunk_size, chunk_size)
            mask (torch.Tensor, optional): Mask tensor of shape (batch_size, chunk_size, chunk_size, chunk_size)

        Returns:
            tuple: (mu, log_var) of the latent space
        """
        # Use JIT compiled version if available
        if self.use_jit and hasattr(self, "_encode_jit"):
            return self._encode_jit(x, mask)

        # Use gradient checkpointing if requested
        if self.use_checkpointing and self.training and torch.is_grad_enabled():
            # Embed the blocks
            embedded = self.block_embedding(x)

            # Permute to get the embedding dimension as channels
            embedded = embedded.permute(0, 4, 1, 2, 3)

            # Apply mask if provided
            if mask is not None:
                mask_expanded = mask.unsqueeze(1)
                embedded = embedded * mask_expanded
                del mask_expanded

            # Use checkpointing for the encoder to save memory during training
            if hasattr(torch.utils.checkpoint, "checkpoint"):
                encoded = torch.utils.checkpoint.checkpoint(self.encoder, embedded)
            else:
                encoded = self.encoder(embedded)

            del embedded

            # Flatten
            flattened = encoded.reshape(encoded.size(0), -1)
            del encoded

            # Get latent parameters
            mu = self.fc_mu(flattened)
            log_var = self.fc_var(flattened)
            del flattened

            return mu, log_var
        else:
            # Standard implementation
            # Embed the blocks
            embedded = self.block_embedding(x)

            # Permute to get the embedding dimension as channels
            embedded = embedded.permute(0, 4, 1, 2, 3)

            # Apply mask if provided
            if mask is not None:
                mask_expanded = mask.unsqueeze(1)
                embedded = embedded * mask_expanded
                del mask_expanded

            # Encode
            encoded = self.encoder(embedded)
            del embedded

            # Flatten
            flattened = encoded.reshape(encoded.size(0), -1)
            del encoded

            # Get latent parameters
            mu = self.fc_mu(flattened)
            log_var = self.fc_var(flattened)
            del flattened

            return mu, log_var

    def decode(self, z):
        """
        Decode from the latent space.

        Args:
            z (torch.Tensor): Latent vector of shape (batch_size, latent_dim)

        Returns:
            torch.Tensor: Reconstructed blocks of shape (batch_size, chunk_size, chunk_size, chunk_size, num_blocks)
        """
        # Use JIT compiled version if available
        if self.use_jit and hasattr(self, "_decode_jit"):
            return self._decode_jit(z)

        # Use gradient checkpointing if requested
        if self.use_checkpointing and self.training and torch.is_grad_enabled():
            # Decode from latent space
            decoded = self.decoder_input(z)

            # Reshape to 3D volume
            last_h_dim = self.hidden_dims[-1]
            spatial_dim = self.chunk_size // (2 ** len(self.hidden_dims))
            reshaped = decoded.reshape(
                -1, last_h_dim, spatial_dim, spatial_dim, spatial_dim
            )
            del decoded

            # Use checkpointing for the decoder to save memory during training
            if hasattr(torch.utils.checkpoint, "checkpoint"):
                decoded = torch.utils.checkpoint.checkpoint(self.decoder, reshaped)
            else:
                decoded = self.decoder(reshaped)

            del reshaped

            # Predict block types
            logits = self.block_predictor(decoded)
            del decoded

            # Permute to get the block dimension at the end
            logits = logits.permute(0, 2, 3, 4, 1)

            return logits
        else:
            # Standard implementation
            # Decode from latent space
            decoded = self.decoder_input(z)

            # Reshape to 3D volume
            last_h_dim = self.hidden_dims[-1]
            spatial_dim = self.chunk_size // (2 ** len(self.hidden_dims))
            reshaped = decoded.reshape(
                -1, last_h_dim, spatial_dim, spatial_dim, spatial_dim
            )
            del decoded

            # Decode
            decoded = self.decoder(reshaped)
            del reshaped

            # Predict block types
            logits = self.block_predictor(decoded)
            del decoded

            # Permute to get the block dimension at the end
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
        # Use JIT compiled version if available
        if self.use_jit and hasattr(self, "_reparameterize_jit"):
            return self._reparameterize_jit(mu, log_var)

        # Standard implementation
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # Free memory
        del std, eps

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

        # Free memory
        del z

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

        # Free memory
        del z

        # Get the most likely block for each position
        # samples shape: (num_samples, chunk_size, chunk_size, chunk_size, num_blocks)
        # blocks shape: (num_samples, chunk_size, chunk_size, chunk_size)
        blocks = torch.argmax(samples, dim=-1)

        # Free memory
        del samples

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


def count_blocks_parallel(
    dataloader, num_classes, device="cuda", cache_file=None, force_recount=False
):
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
            print(
                f"Successfully loaded cached block counts with shape: {class_counts.shape}"
            )
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

    # Create a progress bar for the batches
    pbar = tqdm(total=len(batches), desc="Counting block types")

    def update_pbar(*args):
        """Callback function to update progress bar"""
        pbar.update()

    # Process batches in parallel with callback for progress updates
    results = []
    with mp.Pool(processes=num_processes) as pool:
        for batch in batches:
            # Use apply_async with callback to update progress bar
            results.append(
                pool.apply_async(
                    count_blocks_in_batch, args=(batch,), callback=update_pbar
                )
            )

        # Wait for all processes to complete
        pool.close()
        pool.join()

    # Close progress bar
    pbar.close()

    # Get actual results from AsyncResult objects
    results = [r.get() for r in results]

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


def calculate_class_weights(
    dataloader,
    num_classes,
    device="cuda",
    method="effective_samples",
    beta=0.9999,
    cache_file="cache/block_counts.pt",
    force_recount=False,
):
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
        force_recount=force_recount,
    )

    # Calculate weights based on the selected method
    epsilon = 1e-6  # Small value to avoid division by zero
    total_samples = class_counts.sum()

    if method == "inverse":
        # Standard inverse frequency (with minimum threshold to avoid too small weights)
        min_weight_threshold = 0.01
        class_weights = 1.0 / (class_counts + epsilon)
        class_weights = torch.clamp(class_weights, min=min_weight_threshold)

    elif method == "log":
        # Log-based weighting
        class_weights = torch.log(total_samples / (class_counts + epsilon) + 1.0)

    elif method == "sqrt":
        # Square root weighting
        class_weights = torch.sqrt(total_samples / (class_counts + epsilon))

    elif method == "effective_samples":
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


@torch.jit.script
def _focal_loss_impl(
    log_probs: torch.Tensor,
    target_flat: torch.Tensor,
    gamma: float,
    alpha: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Optimized implementation of focal loss for JIT compilation.

    Args:
        log_probs: Log probabilities from softmax
        target_flat: Flattened target indices
        gamma: Focusing parameter
        alpha: Optional class weights

    Returns:
        Focal loss tensor
    """
    # Get the probabilities for the target class
    probs = torch.exp(log_probs)
    target_probs = probs.gather(1, target_flat.unsqueeze(1)).squeeze(1)

    # Calculate focal weight: (1 - p_t)^gamma
    focal_weight = (1 - target_probs).pow(gamma)

    # Apply alpha weighting if provided
    if alpha is not None:
        alpha_t = alpha.gather(0, target_flat)
        focal_weight = alpha_t * focal_weight

    # Calculate focal loss: -alpha_t * (1 - p_t)^gamma * log(p_t)
    loss = -focal_weight * log_probs.gather(1, target_flat.unsqueeze(1)).squeeze(1)

    return loss


def vae_loss_function(
    reconstructed,
    target,
    mu,
    log_var,
    mask=None,
    kld_weight=0.01,
    class_weights=None,
    use_focal_loss=False,
    gamma=2.0,
    alpha=None,
    reduction="mean",
):
    """
    Optimized VAE loss function with optional focal loss.

    Args:
        reconstructed (torch.Tensor): Reconstructed output logits
        target (torch.Tensor): Target tensor
        mu (torch.Tensor): Mean of the latent Gaussian
        log_var (torch.Tensor): Log variance of the latent Gaussian
        mask (torch.Tensor, optional): Mask tensor
        kld_weight (float): Weight for the KL divergence term
        class_weights (torch.Tensor, optional): Weights for each class to handle imbalance
        use_focal_loss (bool): Whether to use focal loss instead of cross-entropy
        gamma (float): Focusing parameter for focal loss (higher values focus more on hard examples)
        alpha (torch.Tensor, optional): Alpha weighting for focal loss (can be same as class_weights)
        reduction (str): Reduction method for the loss ('mean', 'sum', 'none')

    Returns:
        tuple: (total_loss, reconstruction_loss, kld_loss)
    """
    # Reshape inputs for loss calculation - use view instead of reshape for better performance
    batch_size = reconstructed.size(0)
    reconstructed_flat = reconstructed.contiguous().view(-1, reconstructed.size(-1))
    target_flat = target.contiguous().view(-1)

    # Calculate reconstruction loss
    if use_focal_loss:
        # Compute log softmax once for efficiency
        log_probs = F.log_softmax(reconstructed_flat, dim=-1)

        # Use optimized focal loss implementation
        if alpha is None and class_weights is not None:
            alpha = class_weights

        recon_loss = _focal_loss_impl(log_probs, target_flat, gamma, alpha)
    else:
        # Use standard cross-entropy loss with optimized implementation
        if class_weights is not None:
            # Use weighted cross-entropy loss
            recon_loss = F.cross_entropy(
                reconstructed_flat,
                target_flat,
                weight=class_weights,
                reduction="none",
            )
        else:
            # Use standard cross-entropy loss
            recon_loss = F.cross_entropy(
                reconstructed_flat,
                target_flat,
                reduction="none",
            )

    # Apply mask if provided - use vectorized operations for better performance
    if mask is not None:
        # Reshape mask to match recon_loss - use view for better performance
        mask_flat = mask.contiguous().view(-1)

        # Apply mask
        recon_loss = recon_loss * mask_flat

        # Take mean over valid positions
        valid_positions = mask_flat.sum()
        if valid_positions > 0:
            recon_loss = recon_loss.sum() / valid_positions
        else:
            recon_loss = torch.tensor(0.0, device=recon_loss.device)
    else:
        # Apply requested reduction
        if reduction == "mean":
            recon_loss = recon_loss.mean()
        elif reduction == "sum":
            recon_loss = recon_loss.sum()
        # If reduction is "none", keep as is

    # KL divergence - optimized implementation
    # Use vectorized operations and avoid unnecessary computations
    kld_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    # Total loss
    total_loss = recon_loss + kld_weight * kld_loss

    return total_loss, recon_loss, kld_loss


def compare_weighting_methods(
    dataloader,
    num_classes,
    device="cuda",
    cache_file="cache/block_counts.pt",
    force_recount=False,
):
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
        force_recount=force_recount,
    )

    # Calculate weights using different methods
    methods = ["inverse", "log", "sqrt", "effective_samples", "balanced"]
    weights_dict = {}

    epsilon = 1e-6
    total_samples = class_counts.sum()

    # Standard inverse frequency
    weights = 1.0 / (class_counts + epsilon)
    weights = weights * (num_classes / weights.sum())
    weights_dict["inverse"] = weights

    # Log-based weighting
    weights = torch.log(total_samples / (class_counts + epsilon) + 1.0)
    weights = weights * (num_classes / weights.sum())
    weights_dict["log"] = weights

    # Square root weighting
    weights = torch.sqrt(total_samples / (class_counts + epsilon))
    weights = weights * (num_classes / weights.sum())
    weights_dict["sqrt"] = weights

    # Effective number of samples (beta=0.9999)
    beta = 0.9999
    weights = (1.0 - beta) / (1.0 - torch.pow(beta, class_counts + epsilon))
    weights = weights * (num_classes / weights.sum())
    weights_dict["effective_samples"] = weights

    # Balanced approach
    weights = total_samples / (class_counts * num_classes + epsilon)
    max_weight = weights.max()
    min_weight = max_weight * 0.1
    weights = torch.clamp(weights, min=min_weight)
    weights = weights * (num_classes / weights.sum())
    weights_dict["balanced"] = weights

    # Create visualization
    plt.figure(figsize=(15, 10))

    # Get indices of top classes by count
    top_indices = torch.argsort(class_counts, descending=True)[:20].cpu().numpy()

    # Plot class counts
    plt.subplot(2, 1, 1)
    plt.bar(range(len(top_indices)), class_counts[top_indices].cpu().numpy())
    plt.yscale("log")
    plt.title("Class Distribution (Top 20 Classes)")
    plt.xlabel("Class Index")
    plt.ylabel("Count (log scale)")
    plt.xticks(range(len(top_indices)), [str(idx) for idx in top_indices], rotation=90)

    # Plot weights from different methods
    plt.subplot(2, 1, 2)
    x = np.arange(len(top_indices))
    width = 0.15
    offsets = [-2, -1, 0, 1, 2]

    for i, (method, offset) in enumerate(zip(methods, offsets)):
        plt.bar(
            x + offset * width,
            weights_dict[method][top_indices].cpu().numpy(),
            width=width,
            label=method,
        )

    plt.title("Class Weights Comparison")
    plt.xlabel("Class Index")
    plt.ylabel("Weight")
    plt.xticks(range(len(top_indices)), [str(idx) for idx in top_indices], rotation=90)
    plt.legend()

    plt.tight_layout()
    plt.savefig("class_weights_comparison.png")
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
    ax1.set_yscale("log")
    ax1.set_xlabel("Block ID Index")
    ax1.set_ylabel("Count (log scale)")
    ax1.set_title("Class Distribution (Top Classes)")
    ax1.set_xticks(range(num_to_show))
    ax1.set_xticklabels([str(idx) for idx in selected_indices], rotation=90)

    # Plot class weights
    ax2.bar(range(num_to_show), selected_weights)
    ax2.set_xlabel("Block ID Index")
    ax2.set_ylabel("Weight")
    ax2.set_title("Class Weights")
    ax2.set_xticks(range(num_to_show))
    ax2.set_xticklabels([str(idx) for idx in selected_indices], rotation=90)

    plt.tight_layout()
    plt.savefig("class_weights_distribution.png")
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
        original_lrs.append(param_group["lr"])

    # Get learning rates for each step
    lrs = []
    for i in range(num_steps):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()

    # Reset scheduler and optimizer to original state
    for i, param_group in enumerate(scheduler.optimizer.param_groups):
        param_group["lr"] = original_lrs[i]
    scheduler.base_lrs = [original_lrs[0]]  # Reset base_lrs

    # Plot learning rates
    plt.figure(figsize=(10, 5))
    plt.plot(lrs)
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title("CyclicLR Schedule")
    plt.grid(True)
    plt.savefig("cyclic_lr_schedule.png")
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
    use_focal_loss=False,
    focal_gamma=2.0,
    focal_alpha=None,
    save_interval=5,  # Save model every N epochs
    checkpoint_dir="models/checkpoints",  # Directory to save checkpoints
    use_amp=True,  # Use Automatic Mixed Precision
    gradient_accumulation_steps=4,  # Accumulate gradients over multiple batches
    num_workers=4,  # Number of workers for data loading
    pin_memory=True,  # Pin memory for faster data transfer to GPU
    prefetch_factor=2,  # Prefetch batches
    checkpoint_activation=True,  # Use gradient checkpointing to save memory
):
    """
    Train the VAE with performance optimizations.

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
        use_focal_loss (bool): Whether to use focal loss instead of cross-entropy
        focal_gamma (float): Focusing parameter for focal loss (higher values focus more on hard examples)
        focal_alpha (torch.Tensor, optional): Alpha weighting for focal loss (can be same as class_weights)
        save_interval (int): Save model checkpoint every N epochs
        checkpoint_dir (str): Directory to save checkpoints
        use_amp (bool): Whether to use automatic mixed precision
        gradient_accumulation_steps (int): Number of steps to accumulate gradients
        num_workers (int): Number of workers for data loading
        pin_memory (bool): Whether to pin memory for faster data transfer to GPU
        prefetch_factor (int): Number of batches to prefetch
        checkpoint_activation (bool): Whether to use gradient checkpointing to save memory

    Returns:
        list: Training losses
    """
    losses = []
    start_epoch = 0

    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Enable gradient checkpointing if requested (saves memory)
    if checkpoint_activation and hasattr(vae, "encoder"):
        vae.encoder.apply(
            lambda m: m.register_forward_hook(
                lambda m, _, output: output.requires_grad_(True)
            )
        )

    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler() if use_amp and torch.cuda.is_available() else None

    # Optimize dataloader if it's not already optimized
    if hasattr(dataloader, "num_workers") and dataloader.num_workers < num_workers:
        # Create a new dataloader with optimized settings
        optimized_dataloader = DataLoader(
            dataloader.dataset,
            batch_size=dataloader.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False,
        )
        dataloader = optimized_dataloader

    # Calculate class weights if needed
    class_weights = None
    if use_class_weights:
        # Use the method specified by command-line arguments or the default
        cache_file = (
            None
            if (args and args.no_cache)
            else (args.cache_file if args else "cache/block_counts.pt")
        )
        force_recount = args.force_recount if args else False

        class_weights = calculate_class_weights(
            dataloader,
            vae.num_blocks,
            device,
            method=args.weight_method if args else "effective_samples",
            beta=args.beta if args else 0.9999,
            cache_file=cache_file,
            force_recount=force_recount,
        )

    try:
        for epoch in range(start_epoch, epochs):
            # Training phase
            vae.train()
            epoch_loss = 0
            epoch_recon_loss = 0
            epoch_kld_loss = 0
            epoch_correct = 0
            epoch_valid_positions = 0

            # Track time for performance monitoring
            epoch_start_time = time.time()

            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
            optimizer.zero_grad()  # Zero gradients once at the beginning of epoch

            for batch_idx, batch in enumerate(progress_bar):
                # Get data
                blocks = batch["blocks"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)

                # Forward pass with mixed precision
                if use_amp and torch.cuda.is_available():
                    with autocast():
                        reconstructed, mu, log_var = vae(blocks, mask)

                        # Calculate loss
                        loss, recon_loss, kld_loss = vae_loss_function(
                            reconstructed,
                            blocks,
                            mu,
                            log_var,
                            mask,
                            kld_weight,
                            class_weights,
                            use_focal_loss=use_focal_loss,
                            gamma=focal_gamma,
                            alpha=focal_alpha,
                        )

                        # Scale loss for gradient accumulation
                        loss = loss / gradient_accumulation_steps

                    # Backward pass with scaler
                    scaler.scale(loss).backward()

                    # Only step optimizer and scaler after accumulating gradients
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        # Clip gradients
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)

                        # Update weights
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()

                        # Step the scheduler if provided
                        if scheduler is not None:
                            scheduler.step()
                else:
                    # Standard precision training
                    reconstructed, mu, log_var = vae(blocks, mask)

                    # Calculate loss
                    loss, recon_loss, kld_loss = vae_loss_function(
                        reconstructed,
                        blocks,
                        mu,
                        log_var,
                        mask,
                        kld_weight,
                        class_weights,
                        use_focal_loss=use_focal_loss,
                        gamma=focal_gamma,
                        alpha=focal_alpha,
                    )

                    # Scale loss for gradient accumulation
                    loss = loss / gradient_accumulation_steps

                    # Backward pass
                    loss.backward()

                    # Only step optimizer after accumulating gradients
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        # Clip gradients
                        torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)

                        # Update weights
                        optimizer.step()
                        optimizer.zero_grad()

                        # Step the scheduler if provided
                        if scheduler is not None:
                            scheduler.step()

                # Extract scalar values to prevent memory leaks
                loss_value = (
                    loss.item() * gradient_accumulation_steps
                )  # Scale back for reporting
                recon_loss_value = recon_loss.item()
                kld_loss_value = kld_loss.item()

                # Free memory
                del loss, recon_loss, kld_loss

                # Calculate accuracy (use torch operations instead of item() where possible)
                with torch.no_grad():
                    pred_blocks = torch.argmax(reconstructed, dim=-1)
                    correct = (pred_blocks == blocks) & (mask.bool())
                    correct_sum = correct.sum().item()
                    mask_sum = mask.sum().item()
                    epoch_correct += correct_sum
                    epoch_valid_positions += mask_sum
                    
                    # Calculate current accuracy for progress bar
                    current_accuracy = correct_sum / mask_sum if mask_sum > 0 else 0

                    # Free memory
                    del pred_blocks, correct
                
                # Update progress bar with detailed metrics
                if scheduler is not None:
                    current_lr = scheduler.get_last_lr()[0]
                    progress_bar.set_postfix(
                        {
                            "loss": f"{loss_value:.4f}",
                            "recon_loss": f"{recon_loss_value:.4f}",
                            "kld_loss": f"{kld_loss_value:.4f}",
                            "acc": f"{current_accuracy:.4f}",
                            "lr": f"{current_lr:.6f}",
                        }
                    )
                else:
                    progress_bar.set_postfix(
                        {
                            "loss": f"{loss_value:.4f}",
                            "recon_loss": f"{recon_loss_value:.4f}",
                            "kld_loss": f"{kld_loss_value:.4f}",
                            "acc": f"{current_accuracy:.4f}",
                        }
                    )

                # Free memory from forward pass
                del reconstructed, mu, log_var, blocks, mask

                # Update epoch metrics
                epoch_loss += loss_value
                epoch_recon_loss += recon_loss_value
                epoch_kld_loss += kld_loss_value

                # Explicitly clear CUDA cache periodically
                if batch_idx % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Force garbage collection at the end of each epoch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
                epoch_correct / epoch_valid_positions
                if epoch_valid_positions > 0
                else 0
            )

            # Calculate elapsed time for the epoch
            epoch_elapsed_time = time.time() - epoch_start_time

            print(
                f"Epoch {epoch + 1}/{epochs} completed in {epoch_elapsed_time:.2f} seconds:"
            )
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
                    use_focal_loss=False,  # Don't use focal loss for evaluation to keep metrics comparable
                )

                # Print validation results with elapsed time
                print(
                    f"  Val Loss: {val_results['loss']:.4f} (completed in {val_results['elapsed_time']:.2f} seconds)"
                )
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

            # Save checkpoint if needed
            if (epoch + 1) % save_interval == 0:
                checkpoint_path = os.path.join(
                    checkpoint_dir, f"vae_checkpoint_epoch_{epoch + 1}.pth"
                )
                save_vae(
                    vae=vae,
                    optimizer=optimizer,
                    losses=losses,
                    filename=checkpoint_path,
                )
                print(f"Checkpoint saved at epoch {epoch + 1}")

            # Clear memory at the end of each epoch
            if device == "cuda":
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving model...")
        # Save the model at the current state
        interrupted_path = os.path.join(checkpoint_dir, "vae_interrupted.pth")
        save_vae(vae=vae, optimizer=optimizer, losses=losses, filename=interrupted_path)
        print(f"Model saved to {interrupted_path}")

        # Also save to the default location
        save_vae(
            vae=vae,
            optimizer=optimizer,
            losses=losses,
            filename="models/minecraft_vae.pth",
        )
        print("Model also saved to models/minecraft_vae.pth")

    return losses


def evaluate_vae(
    vae,
    dataloader,
    device="cuda",
    kld_weight=0.01,
    num_samples=5,
    verbose=True,
    class_weights=None,
    use_focal_loss=False,
    focal_gamma=2.0,
    focal_alpha=None,
    use_amp=True,  # Use Automatic Mixed Precision
    num_workers=4,  # Number of workers for data loading
    pin_memory=True,  # Pin memory for faster data transfer to GPU
    batch_size=None,  # Override batch size for evaluation
):
    """
    Evaluate the VAE on a validation set with performance optimizations.

    Args:
        vae (MinecraftVAE): The VAE model
        dataloader (DataLoader): DataLoader for the validation data
        device (str): Device to use
        kld_weight (float): Weight for the KL divergence term
        num_samples (int): Number of samples to visualize
        verbose (bool): Whether to print evaluation results
        class_weights (torch.Tensor, optional): Weights for each class to handle imbalance
        use_focal_loss (bool): Whether to use focal loss instead of cross-entropy
        focal_gamma (float): Focusing parameter for focal loss (higher values focus more on hard examples)
        focal_alpha (torch.Tensor, optional): Alpha weighting for focal loss (can be same as class_weights)
        use_amp (bool): Whether to use automatic mixed precision
        num_workers (int): Number of workers for data loading
        pin_memory (bool): Whether to pin memory for faster data transfer to GPU
        batch_size (int, optional): Override batch size for evaluation
    Returns:
        dict: Evaluation metrics
    """
    # Start timing
    start_time = time.time()

    # Optimize dataloader if needed
    if batch_size is not None or (
        hasattr(dataloader, "num_workers") and dataloader.num_workers < num_workers
    ):
        # Create a new dataloader with optimized settings
        optimized_dataloader = DataLoader(
            dataloader.dataset,
            batch_size=batch_size or dataloader.batch_size,
            shuffle=False,  # No need to shuffle for evaluation
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False,
        )
        dataloader = optimized_dataloader

    vae.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kld_loss = 0
    total_correct = 0
    total_valid_positions = 0

    # Sample some examples for visualization
    samples = []

    # Use tqdm for progress tracking
    progress_bar = tqdm(dataloader, desc="Evaluating", disable=not verbose)

    with torch.no_grad():
        for i, batch in enumerate(progress_bar):
            # Get data with non-blocking transfer
            blocks = batch["blocks"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            # Forward pass with mixed precision if available
            if use_amp and torch.cuda.is_available():
                with autocast():
                    reconstructed, mu, log_var = vae(blocks, mask)

                    # Calculate loss
                    loss, recon_loss, kld_loss = vae_loss_function(
                        reconstructed,
                        blocks,
                        mu,
                        log_var,
                        mask,
                        kld_weight,
                        class_weights,
                        use_focal_loss=False,  # Don't use focal loss for evaluation to keep metrics comparable
                    )
            else:
                # Standard precision evaluation
                reconstructed, mu, log_var = vae(blocks, mask)

                # Calculate loss
                loss, recon_loss, kld_loss = vae_loss_function(
                    reconstructed,
                    blocks,
                    mu,
                    log_var,
                    mask,
                    kld_weight,
                    class_weights,
                    use_focal_loss=False,  # Don't use focal loss for evaluation to keep metrics comparable
                )

            # Extract scalar values to prevent memory leaks
            loss_value = loss.item()
            recon_loss_value = recon_loss.item()
            kld_loss_value = kld_loss.item()

            # Update totals
            total_loss += loss_value
            total_recon_loss += recon_loss_value
            total_kld_loss += kld_loss_value

            # Free memory
            del loss, recon_loss, kld_loss

            # Calculate accuracy using vectorized operations
            pred_blocks = torch.argmax(reconstructed, dim=-1)
            correct = (pred_blocks == blocks) & (mask.bool())
            correct_sum = correct.sum().item()
            mask_sum = mask.sum().item()

            # Update metrics
            total_correct += correct_sum
            total_valid_positions += mask_sum
            
            # Update progress bar with detailed metrics
            if verbose:
                current_accuracy = correct_sum / mask_sum if mask_sum > 0 else 0
                progress_bar.set_postfix({
                    "loss": f"{loss_value:.4f}",
                    "recon_loss": f"{recon_loss_value:.4f}",
                    "kld_loss": f"{kld_loss_value:.4f}",
                    "acc": f"{current_accuracy:.4f}"
                })

            # Store samples for visualization (only if needed)
            if i < num_samples and num_samples > 0:
                # Store only what's needed and immediately detach and move to CPU
                samples.append(
                    {
                        "input": blocks[0].cpu().detach(),
                        "mask": mask[0].cpu().detach(),
                        "reconstructed": pred_blocks[0].cpu().detach(),
                    }
                )

            # Free memory
            del blocks, mask, reconstructed, mu, log_var, pred_blocks, correct

            # Clear CUDA cache periodically (every 5 batches)
            if i % 5 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Calculate average losses and accuracy
    avg_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_kld_loss = total_kld_loss / len(dataloader)
    accuracy = total_correct / total_valid_positions if total_valid_positions > 0 else 0

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    if verbose:
        print(f"Evaluation completed in {elapsed_time:.2f} seconds:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Reconstruction Loss: {avg_recon_loss:.4f}")
        print(f"  KLD Loss: {avg_kld_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f} ({total_correct}/{total_valid_positions})")

    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "loss": avg_loss,
        "recon_loss": avg_recon_loss,
        "kld_loss": avg_kld_loss,
        "accuracy": accuracy,
        "correct": total_correct,
        "total": total_valid_positions,
        "samples": samples,
        "elapsed_time": elapsed_time,
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
        help="Method to calculate class weights (default: effective_samples)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.9999,
        help="Beta parameter for effective_samples method (default: 0.9999)",
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        default="cache/block_counts.pt",
        help="Path to cache file for block counts (default: cache/block_counts.pt)",
    )
    parser.add_argument(
        "--force-recount",
        action="store_true",
        help="Force recounting blocks even if cache exists",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Disable caching of block counts"
    )
    parser.add_argument(
        "--use-focal-loss",
        action="store_true",
        help="Use focal loss instead of cross-entropy loss",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Gamma parameter for focal loss (default: 2.0)",
    )
    parser.add_argument(
        "--no-focal-loss",
        action="store_true",
        help="Disable focal loss (overrides --use-focal-loss)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=5,
        help="Save checkpoint every N epochs (default: 5)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="models/checkpoints",
        help="Directory to save checkpoints (default: models/checkpoints)",
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

    # Create CyclicLR scheduler
    # Using triangular2 mode which reduces the amplitude by half each cycle
    # This is often more performant for neural networks
    step_size_up = len(train_dataloader) * 2  # Size of one cycle = 2 epochs
    scheduler = CyclicLR(
        optimizer,
        base_lr=1e-4,  # Lower learning rate boundary
        max_lr=5e-3,  # Upper learning rate boundary
        step_size_up=step_size_up,
        mode="triangular2",  # Triangular cycle with amplitude halved each time
        cycle_momentum=False,  # AdamW doesn't use momentum
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
        use_focal_loss=args.use_focal_loss
        and not args.no_focal_loss,  # Use focal loss if enabled and not disabled
        focal_gamma=args.focal_gamma,  # Use gamma value from command-line arguments
        focal_alpha=None,  # Use class_weights as alpha
        save_interval=args.save_interval,  # Save checkpoint every N epochs from command-line
        checkpoint_dir=args.checkpoint_dir,  # Directory to save checkpoints from command-line
    )

    # Evaluate the VAE
    eval_results = evaluate_vae(
        vae=vae,
        dataloader=val_dataloader,
        device=device,
        use_focal_loss=False,  # Don't use focal loss for evaluation to keep metrics comparable
    )

    # Save the VAE
    save_vae(
        vae=vae, optimizer=optimizer, losses=losses, filename="models/minecraft_vae.pth"
    )

    # Generate some samples
    print("Generating samples...")
    samples = vae.sample(num_samples=5, device=device)
    print(f"Generated {len(samples)} samples of shape {samples.shape}")
