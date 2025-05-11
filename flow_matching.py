#!/usr/bin/env python3
"""
Conditional Flow Matching (CFM) for generating Minecraft maps.
This script demonstrates how to use the VAE with a CFM network for generating Minecraft maps.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from minecraft_dataset import MinecraftSchematicDataset
from minecraft_vae import MinecraftVAE, load_vae


class FlowMatchingNetwork(nn.Module):
    """
    Conditional Flow Matching Network for generating Minecraft maps.
    
    This network learns to model the vector field of a continuous normalizing flow
    conditioned on a context vector.
    """
    
    def __init__(self, latent_dim, context_dim=0, hidden_dims=[256, 512, 256]):
        """
        Initialize the Flow Matching Network.
        
        Args:
            latent_dim (int): Dimension of the latent space
            context_dim (int): Dimension of the context vector (0 for unconditional)
            hidden_dims (list): Dimensions of the hidden layers
        """
        super(FlowMatchingNetwork, self).__init__()
        
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        
        # Input dimension is latent_dim (z) + latent_dim (v_t) + 1 (t) + context_dim
        input_dim = 2 * latent_dim + 1 + context_dim
        
        # Build the network
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.SiLU())
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.SiLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], latent_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, z_t, v_t, t, context=None):
        """
        Forward pass through the network.
        
        Args:
            z_t (torch.Tensor): Current point in the latent space
            v_t (torch.Tensor): Current velocity in the latent space
            t (torch.Tensor): Current time
            context (torch.Tensor, optional): Context vector
            
        Returns:
            torch.Tensor: Predicted vector field
        """
        # Reshape t to match batch size
        t = t.expand(z_t.shape[0], 1)
        
        # Concatenate inputs
        if context is not None:
            inputs = torch.cat([z_t, v_t, t, context], dim=1)
        else:
            inputs = torch.cat([z_t, v_t, t], dim=1)
        
        # Forward pass
        return self.net(inputs)


class LatentSpaceDataset(Dataset):
    """
    Dataset for training the Flow Matching Network.
    
    This dataset provides pairs of latent vectors for training the flow matching network.
    """
    
    def __init__(self, vae, dataloader, device="cuda", context_dim=0):
        """
        Initialize the dataset.
        
        Args:
            vae (MinecraftVAE): The VAE model
            dataloader (DataLoader): DataLoader for the data
            device (str): Device to use
            context_dim (int): Dimension of the context vector (0 for unconditional)
        """
        self.vae = vae
        self.device = device
        self.context_dim = context_dim
        
        # Collect latent vectors
        self.latent_vectors = []
        self.contexts = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Collecting latent vectors"):
                # Get data
                blocks = batch['blocks'].to(device)
                mask = batch['mask'].to(device)
                
                # Encode
                mu, _ = vae.encode(blocks, mask)
                
                # Store latent vectors
                self.latent_vectors.append(mu.cpu())
                
                # Generate random context vectors if needed
                if context_dim > 0:
                    context = torch.randn(blocks.shape[0], context_dim)
                    self.contexts.append(context)
        
        # Concatenate latent vectors
        self.latent_vectors = torch.cat(self.latent_vectors, dim=0)
        
        # Concatenate context vectors if needed
        if context_dim > 0:
            self.contexts = torch.cat(self.contexts, dim=0)
    
    def __len__(self):
        return len(self.latent_vectors)
    
    def __getitem__(self, idx):
        # Get latent vector
        z_1 = self.latent_vectors[idx]
        
        # Sample random latent vector for z_0
        z_0 = torch.randn_like(z_1)
        
        # Get context if needed
        if self.context_dim > 0:
            context = self.contexts[idx]
            return z_0, z_1, context
        else:
            return z_0, z_1, None


def train_flow_matching(flow_model, dataset, optimizer, device="cuda", epochs=10, batch_size=64, num_workers=4):
    """
    Train the Flow Matching Network.
    
    Args:
        flow_model (FlowMatchingNetwork): The Flow Matching Network
        dataset (LatentSpaceDataset): The dataset
        optimizer (torch.optim.Optimizer): Optimizer
        device (str): Device to use
        epochs (int): Number of epochs
        batch_size (int): Batch size
        num_workers (int): Number of workers for data loading
        
    Returns:
        list: Training losses
    """
    flow_model.train()
    losses = []
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    for epoch in range(epochs):
        epoch_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            # Get data
            z_0, z_1, context = batch
            z_0 = z_0.to(device)
            z_1 = z_1.to(device)
            if context is not None:
                context = context.to(device)
            
            # Sample random time
            t = torch.rand(1, device=device)
            
            # Compute z_t and v_t
            z_t = (1 - t) * z_0 + t * z_1
            v_t = z_1 - z_0
            
            # Forward pass
            v_pred = flow_model(z_t, v_t, t, context)
            
            # Compute loss (L2 loss)
            loss = F.mse_loss(v_pred, v_t)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': epoch_loss / (progress_bar.n + 1)})
        
        # Calculate average loss
        avg_loss = epoch_loss / len(dataloader)
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Loss: {avg_loss:.6f}")
        
        losses.append(avg_loss)
    
    return losses


def generate_samples_with_flow(vae, flow_model, device="cuda", num_samples=5, steps=100, context=None):
    """
    Generate samples using the Flow Matching Network.
    
    Args:
        vae (MinecraftVAE): The VAE model
        flow_model (FlowMatchingNetwork): The Flow Matching Network
        device (str): Device to use
        num_samples (int): Number of samples to generate
        steps (int): Number of steps for the ODE solver
        context (torch.Tensor, optional): Context vector
        
    Returns:
        torch.Tensor: Generated samples in the original space
    """
    vae.eval()
    flow_model.eval()
    
    # Sample from the prior
    z_0 = torch.randn(num_samples, vae.latent_dim, device=device)
    
    # Solve the ODE
    z_t = z_0.clone()
    
    with torch.no_grad():
        # Euler method
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.tensor([i * dt], device=device)
            
            # Compute velocity
            v_t = flow_model(z_t, torch.zeros_like(z_t), t, context)
            
            # Update z_t
            z_t = z_t + v_t * dt
        
        # Decode
        logits = vae.decode(z_t)
        samples = torch.argmax(logits, dim=-1)
    
    return samples


def save_flow_model(flow_model, optimizer, losses, filename):
    """
    Save the Flow Matching Network.
    
    Args:
        flow_model (FlowMatchingNetwork): The Flow Matching Network
        optimizer (torch.optim.Optimizer): Optimizer
        losses (list): Training losses
        filename (str): Filename to save to
    """
    torch.save({
        'model_state_dict': flow_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses
    }, filename)
    print(f"Flow model saved to {filename}")


def load_flow_model(filename, flow_model, optimizer=None):
    """
    Load a saved Flow Matching Network.
    
    Args:
        filename (str): Filename to load from
        flow_model (FlowMatchingNetwork): The Flow Matching Network
        optimizer (torch.optim.Optimizer, optional): Optimizer
        
    Returns:
        tuple: (flow_model, optimizer, losses)
    """
    checkpoint = torch.load(filename)
    flow_model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    losses = checkpoint.get('losses', [])
    
    print(f"Flow model loaded from {filename}")
    
    return flow_model, optimizer, losses


def visualize_flow_samples(samples, dataset, filename="flow_samples.png"):
    """
    Visualize samples generated by the Flow Matching Network.
    
    Args:
        samples (torch.Tensor): Generated samples
        dataset (MinecraftSchematicDataset): The dataset (for block mapping)
        filename (str): Filename to save to
    """
    # Create a figure
    fig = plt.figure(figsize=(20, 4 * len(samples)))
    
    # Create a mapping from block indices to colors
    import matplotlib.colors as mcolors
    cmap = plt.cm.get_cmap('tab20', 20)
    
    block_to_color = {}
    for i in range(len(dataset.idx_to_block)):
        block_to_color[i] = cmap(i % 20)
    
    # Plot each sample
    for i in range(len(samples)):
        ax = fig.add_subplot(len(samples), 1, i + 1, projection='3d')
        
        # Get sample
        sample = samples[i].cpu()
        
        # Visualize
        x, y, z = np.indices(sample.shape)
        
        # Filter out air blocks (assuming 0 is air)
        non_air = sample > 0
        x = x[non_air]
        y = y[non_air]
        z = z[non_air]
        
        # Create colors
        colors = np.zeros((len(x), 4), dtype=np.float32)
        for j in range(len(x)):
            block_idx = sample[x[j], y[j], z[j]].item()
            colors[j] = block_to_color.get(block_idx, [0, 0, 0, 0])
        
        # Plot
        ax.scatter(x, y, z, c=colors, marker='s', alpha=0.7)
        
        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        
        # Set title
        ax.set_title(f"Generated Sample {i+1}")
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    print(f"Flow samples saved to {filename}")


def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    
    # Create directories if they don't exist
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
        hidden_dims=args.vae_hidden_dims
    ).to(device)
    
    # Load VAE
    if args.vae_model:
        vae, _, _ = load_vae(args.vae_model, vae)
    else:
        print("Error: VAE model must be provided")
        return
    
    # Create the Flow Matching Network
    flow_model = FlowMatchingNetwork(
        latent_dim=args.latent_dim,
        context_dim=args.context_dim,
        hidden_dims=args.flow_hidden_dims
    ).to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(flow_model.parameters(), lr=args.learning_rate)
    
    # Load flow model if specified
    if args.load_flow_model:
        flow_model, optimizer, _ = load_flow_model(args.load_flow_model, flow_model, optimizer)
    
    # Train if specified
    if args.train:
        # Create latent space dataset
        latent_dataset = LatentSpaceDataset(
            vae=vae,
            dataloader=train_dataloader,
            device=device,
            context_dim=args.context_dim
        )
        
        # Train the flow model
        losses = train_flow_matching(
            flow_model=flow_model,
            dataset=latent_dataset,
            optimizer=optimizer,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        # Save the flow model
        save_flow_model(
            flow_model=flow_model,
            optimizer=optimizer,
            losses=losses,
            filename=args.save_flow_model
        )
    
    # Generate samples if specified
    if args.generate:
        # Generate context if needed
        context = None
        if args.context_dim > 0:
            context = torch.randn(args.n_samples, args.context_dim, device=device)
        
        # Generate samples
        samples = generate_samples_with_flow(
            vae=vae,
            flow_model=flow_model,
            device=device,
            num_samples=args.n_samples,
            steps=args.steps,
            context=context
        )
        
        # Visualize samples
        visualize_flow_samples(
            samples=samples,
            dataset=dataset,
            filename="flow_samples.png"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conditional Flow Matching for Minecraft Maps")
    
    # Data options
    parser.add_argument("--data_dir", type=str, default="minecraft-schematics-raw",
                        help="Directory containing schematic files")
    parser.add_argument("--max_files", type=int, default=None,
                        help="Maximum number of files to load")
    
    # VAE options
    parser.add_argument("--vae_model", type=str, required=True,
                        help="Path to VAE model")
    parser.add_argument("--embedding_dim", type=int, default=32,
                        help="Dimension of the block embeddings")
    parser.add_argument("--latent_dim", type=int, default=128,
                        help="Dimension of the latent space")
    parser.add_argument("--vae_hidden_dims", type=int, nargs="+", default=[64, 128, 256],
                        help="Dimensions of the VAE hidden layers")
    
    # Flow model options
    parser.add_argument("--context_dim", type=int, default=0,
                        help="Dimension of the context vector (0 for unconditional)")
    parser.add_argument("--flow_hidden_dims", type=int, nargs="+", default=[256, 512, 256],
                        help="Dimensions of the flow model hidden layers")
    
    # Training options
    parser.add_argument("--train", action="store_true",
                        help="Train the flow model")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    
    # Model loading/saving options
    parser.add_argument("--load_flow_model", type=str, default=None,
                        help="Path to load flow model from")
    parser.add_argument("--save_flow_model", type=str, default="models/minecraft_flow.pth",
                        help="Path to save flow model to")
    
    # Generation options
    parser.add_argument("--generate", action="store_true",
                        help="Generate samples from the flow model")
    parser.add_argument("--n_samples", type=int, default=5,
                        help="Number of samples to generate")
    parser.add_argument("--steps", type=int, default=100,
                        help="Number of steps for the ODE solver")
    
    # Misc options
    parser.add_argument("--cpu", action="store_true",
                        help="Use CPU instead of GPU")
    
    args = parser.parse_args()
    
    main(args)