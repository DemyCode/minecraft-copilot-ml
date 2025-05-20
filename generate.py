#!/usr/bin/env python3
"""
Generate samples from a trained flow matching model for Minecraft structures.
This script loads a trained model and generates Minecraft structures,
mapping the continuous embedding values back to Minecraft blocks.
"""

import os
import copy
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image
from improved_diffusion.unet import UNetModel
from torchcfm.conditional_flow_matching import (
    ExactOptimalTransportConditionalFlowMatcher,
)
from torchdyn.core import NeuralODE
from minecraft_dataset import MinecraftSchematicDataset
from schematic_loader import BLOCK_ID_TO_NAME
import pickle
from sklearn.metrics.pairwise import cosine_similarity


def parse_args():
    parser = argparse.ArgumentParser(description="Generate samples from a trained flow matching model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--cache_file", type=str, default="cache/block_mappings.pkl", help="Path to the block mappings cache")
    parser.add_argument("--embedding_cache", type=str, default="cache/block_embeddings.pt", help="Path to the block embeddings cache")
    parser.add_argument("--output_dir", type=str, default="generated_samples", help="Directory to save generated samples")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of samples to generate")
    parser.add_argument("--chunk_size", type=int, default=16, help="Size of the generated chunks")
    parser.add_argument("--embedding_dim", type=int, default=32, help="Dimension of block embeddings")
    parser.add_argument("--save_npy", action="store_true", help="Save raw numpy arrays of block indices")
    parser.add_argument("--save_schematic", action="store_true", help="Save as Minecraft schematic files")
    return parser.parse_args()


def generate_samples(model, savedir, step, num_samples=64, embedding_dim=32, device="cuda"):
    """Generate samples from the model and save them as images.

    Parameters
    ----------
    model: torch.nn.Module
        The neural network model to generate samples from
    savedir: str
        Directory to save the generated images
    step: int
        Current step identifier for the saved files
    num_samples: int
        Number of samples to generate
    embedding_dim: int
        Dimension of block embeddings
    device: str
        Device to use for generation
    """
    model.eval()
    
    # Create a copy of the model for inference
    model_ = copy.deepcopy(model)
    
    # Create Neural ODE for trajectory generation
    node_ = NeuralODE(model_, solver="euler", sensitivity="adjoint")
    
    with torch.no_grad():
        # Generate random noise as starting point with correct dimensions
        # Shape: [num_samples, embedding_dim, 16, 16, 16]
        noise = torch.randn(num_samples, embedding_dim, 16, 16, 16, device=device)
        
        # Generate trajectory from noise to samples
        traj = node_.trajectory(
            noise,
            t_span=torch.linspace(0, 1, 100, device=device),
        )
        
        # Get the final state of the trajectory
        # Shape: [num_samples, embedding_dim, 16, 16, 16]
        traj = traj[-1, :]
    
    # No need to save as image since these are 3D embeddings, not 2D images
    # But we'll keep a record of the generation
    print(f"Generated {num_samples} samples with shape {traj.shape}")
    
    model.train()
    return traj


def map_embeddings_to_blocks(embeddings, block_embeddings, idx_to_block):
    """Map continuous embedding values to discrete Minecraft blocks.
    
    Parameters
    ----------
    embeddings: torch.Tensor
        Tensor of shape [batch_size, embedding_dim, depth, height, width] with continuous embeddings
    block_embeddings: torch.Tensor
        Tensor of shape [num_blocks, embedding_dim] with embeddings for each block type
    idx_to_block: dict
        Mapping from block indices to block names/IDs
    
    Returns
    -------
    blocks: numpy.ndarray
        Array of shape [batch_size, depth, height, width] with block indices
    block_names: numpy.ndarray
        Array of shape [batch_size, depth, height, width] with block names
    """
    # Reshape embeddings to [batch_size * depth * height * width, embedding_dim]
    batch_size, embedding_dim, depth, height, width = embeddings.shape
    embeddings_reshaped = embeddings.permute(0, 2, 3, 4, 1).reshape(-1, embedding_dim)
    
    # Compute cosine similarity between generated embeddings and block embeddings
    block_embeddings_np = block_embeddings.cpu().numpy()
    embeddings_np = embeddings_reshaped.cpu().numpy()
    
    # Handle NaN values that might occur in the embeddings
    embeddings_np = np.nan_to_num(embeddings_np)
    
    # Normalize embeddings for cosine similarity
    # Add small epsilon to avoid division by zero
    epsilon = 1e-8
    block_embeddings_norm = block_embeddings_np / (np.linalg.norm(block_embeddings_np, axis=1, keepdims=True) + epsilon)
    embeddings_norm = embeddings_np / (np.linalg.norm(embeddings_np, axis=1, keepdims=True) + epsilon)
    
    # Compute similarity
    similarities = cosine_similarity(embeddings_norm, block_embeddings_norm)
    
    # Get the most similar block for each position
    block_indices = np.argmax(similarities, axis=1)
    
    # Reshape back to [batch_size, depth, height, width]
    block_indices = block_indices.reshape(batch_size, depth, height, width)
    
    # Map indices to block names
    block_names = np.zeros_like(block_indices, dtype=object)
    for i in range(batch_size):
        for d in range(depth):
            for h in range(height):
                for w in range(width):
                    idx = block_indices[i, d, h, w]
                    block = idx_to_block.get(idx)
                    if isinstance(block, str):
                        # Special tokens like <pad> or <unk>
                        if block == "<pad>":
                            block_names[i, d, h, w] = "minecraft:air"
                        elif block == "<unk>":
                            block_names[i, d, h, w] = "minecraft:stone"
                        else:
                            block_names[i, d, h, w] = block
                    else:
                        # Numeric block ID, get name from BLOCK_ID_TO_NAME
                        block_name = BLOCK_ID_TO_NAME.get(block, "minecraft:stone")
                        block_names[i, d, h, w] = block_name
    
    return block_indices, block_names


def save_as_npy(block_indices, output_dir, prefix="sample"):
    """Save block indices as numpy arrays.
    
    Parameters
    ----------
    block_indices: numpy.ndarray
        Array of shape [batch_size, depth, height, width] with block indices
    output_dir: str
        Directory to save the numpy arrays
    prefix: str
        Prefix for the saved files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(block_indices.shape[0]):
        output_path = os.path.join(output_dir, f"{prefix}_{i:04d}.npy")
        np.save(output_path, block_indices[i])
        print(f"Saved block indices to {output_path}")


def save_as_schematic(block_names, output_dir, prefix="sample"):
    """Save block names as Minecraft schematic files.
    
    Parameters
    ----------
    block_names: numpy.ndarray
        Array of shape [batch_size, depth, height, width] with block names
    output_dir: str
        Directory to save the schematic files
    prefix: str
        Prefix for the saved files
    """
    try:
        from nbtlib import File, Compound, List, ByteArray, IntArray, Int, String
        
        os.makedirs(output_dir, exist_ok=True)
        
        for i in range(block_names.shape[0]):
            # Create a new schematic file
            height, length, width = block_names[i].shape
            
            # Create a palette mapping block names to indices
            palette = {}
            block_data = []
            
            # Fill the palette and block data
            for y in range(height):
                for z in range(length):
                    for x in range(width):
                        block_name = block_names[i, y, z, x]
                        if block_name not in palette:
                            palette[block_name] = len(palette)
                        block_data.append(palette[block_name])
            
            # Create the NBT structure
            schematic = {
                "Version": Int(2),
                "DataVersion": Int(2975),  # Minecraft 1.19 data version
                "Width": Int(width),
                "Height": Int(height),
                "Length": Int(length),
                "Palette": Compound({String(k): Int(v) for k, v in palette.items()}),
                "BlockData": IntArray(block_data),
            }
            
            # Save the schematic file
            output_path = os.path.join(output_dir, f"{prefix}_{i:04d}.schem")
            File({"Schematic": Compound(schematic)}).save(output_path)
            print(f"Saved schematic to {output_path}")
    except ImportError:
        print("nbtlib not installed. Cannot save as schematic files.")
        print("Install with: pip install nbtlib")


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load block mappings
    print(f"Loading block mappings from {args.cache_file}")
    with open(args.cache_file, "rb") as f:
        cache_data = pickle.load(f)
        block_to_idx = cache_data["block_to_idx"]
        idx_to_block = cache_data["idx_to_block"]
    
    # Load block embeddings
    print(f"Loading block embeddings from {args.embedding_cache}")
    block_embeddings = torch.load(args.embedding_cache)
    
    # Create model
    print("Creating model...")
    model = UNetModel(
        in_channels=args.embedding_dim,
        model_channels=64,
        out_channels=args.embedding_dim,
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
    
    # Load model checkpoint
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Check if the checkpoint contains 'net_model' or 'ema_model' keys
    if "net_model" in checkpoint:
        model.load_state_dict(checkpoint["net_model"])
    elif "ema_model" in checkpoint:
        model.load_state_dict(checkpoint["ema_model"])
    else:
        # Try loading directly
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    
    # Generate samples
    print(f"Generating {args.num_samples} samples...")
    
    # Set up flow matcher
    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
    
    # Generate samples using Neural ODE
    embeddings = generate_samples(
        model, 
        args.output_dir, 
        step=0, 
        num_samples=args.num_samples, 
        embedding_dim=args.embedding_dim,
        device=device
    )
    
    # Map embeddings back to Minecraft blocks
    print("Mapping embeddings to Minecraft blocks...")
    block_indices, block_names = map_embeddings_to_blocks(
        embeddings, 
        block_embeddings, 
        idx_to_block
    )
    
    # Save as numpy arrays if requested
    if args.save_npy:
        save_as_npy(block_indices, os.path.join(args.output_dir, "npy"))
    
    # Save as schematic files if requested
    if args.save_schematic:
        save_as_schematic(block_names, os.path.join(args.output_dir, "schematic"))
    
    print(f"Generation complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()