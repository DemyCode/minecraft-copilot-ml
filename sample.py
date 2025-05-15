#!/usr/bin/env python3
"""
Sample from a trained diffusion model.
"""

import argparse
import os
import torch
import numpy as np
from improved_diffusion.script_util import create_gaussian_diffusion
from improved_diffusion.unet import UNetModel
from improved_diffusion import dist_util, logger
import pickle
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--cache_file", type=str, default="cache/block_mappings.pkl", help="Path to the block mappings cache file")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for sampling")
    parser.add_argument("--diffusion_steps", type=int, default=1000, help="Number of diffusion steps")
    parser.add_argument("--noise_schedule", type=str, default="linear", help="Noise schedule")
    parser.add_argument("--output_dir", type=str, default="samples", help="Directory to save samples")
    parser.add_argument("--use_ddim", action="store_true", help="Use DDIM sampling")
    parser.add_argument("--timestep_respacing", type=str, default="", help="Timestep respacing")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load block mappings
    print(f"Loading block mappings from {args.cache_file}")
    with open(args.cache_file, "rb") as f:
        cache_data = pickle.load(f)
        block_to_idx = cache_data["block_to_idx"]
        idx_to_block = cache_data["idx_to_block"]
    
    num_blocks = len(block_to_idx)
    print(f"Found {num_blocks} unique block types")
    
    # Create model
    print("Creating model...")
    model = UNetModel(
        in_channels=num_blocks,
        model_channels=64,
        out_channels=num_blocks,
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
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location=dist_util.dev())
    )
    model.to(dist_util.dev())
    model.eval()
    
    # Create diffusion
    print("Creating diffusion...")
    diffusion = create_gaussian_diffusion(
        steps=args.diffusion_steps,
        noise_schedule=args.noise_schedule,
        learn_sigma=False,
        sigma_small=False,
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=False,
        timestep_respacing=args.timestep_respacing,
    )
    
    # Sample from the model
    print(f"Generating {args.num_samples} samples...")
    all_samples = []
    
    for i in tqdm(range(0, args.num_samples, args.batch_size)):
        batch_size = min(args.batch_size, args.num_samples - i)
        
        # Create a mask (all ones for full generation)
        mask = torch.ones((batch_size, 16, 16, 16), device=dist_util.dev())
        
        # Sample from the model
        sample_fn = diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        samples = sample_fn(
            model,
            (batch_size, num_blocks, 16, 16, 16),
            clip_denoised=True,
            model_kwargs={"mask": mask},
            device=dist_util.dev(),
        )
        
        # Convert samples to block indices
        samples = samples.detach().cpu().numpy()
        all_samples.append(samples)
    
    # Concatenate all samples
    all_samples = np.concatenate(all_samples, axis=0)
    
    # Save samples
    for i, sample in enumerate(all_samples):
        # Convert one-hot to block indices
        block_indices = np.argmax(sample, axis=0)
        
        # Save as numpy array
        output_path = os.path.join(args.output_dir, f"sample_{i:04d}.npy")
        np.save(output_path, block_indices)
        print(f"Saved sample to {output_path}")
        
        # Optionally, you could convert the block indices back to block names
        # and save in a format that can be imported into Minecraft

if __name__ == "__main__":
    main()