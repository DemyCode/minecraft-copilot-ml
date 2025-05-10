#!/usr/bin/env python3
"""
Test script for the MinecraftSchematicDataset.
"""

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time

from minecraft_dataset import MinecraftSchematicDataset

def test_dataset_loading():
    """Test loading the dataset and accessing items."""
    print("Testing dataset loading...")
    
    # Create cache directory if it doesn't exist
    os.makedirs('cache', exist_ok=True)
    
    # Create the dataset with a small sample
    start_time = time.time()
    dataset = MinecraftSchematicDataset(
        schematics_dir='minecraft-schematics-raw',
        chunk_size=16,
        cache_file='cache/block_mappings.pkl',
        max_files=100  # Limit to 100 files for testing
    )
    load_time = time.time() - start_time
    print(f"Dataset loaded in {load_time:.2f} seconds")
    
    # Test accessing a few items
    print("\nTesting item access...")
    for i in range(min(5, len(dataset))):
        item = dataset[i]
        print(f"Item {i}:")
        print(f"  Blocks shape: {item['blocks'].shape}")
        print(f"  Mask shape: {item['mask'].shape}")
        print(f"  Valid positions: {item['mask'].sum().item()}/{16*16*16} ({item['mask'].float().mean().item()*100:.2f}%)")
        print(f"  Source file: {os.path.basename(item['file_path'])}")
    
    return dataset

def test_dataloader(dataset):
    """Test creating and using a DataLoader."""
    print("\nTesting DataLoader...")
    
    # Create a DataLoader
    batch_size = 8
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Get a batch
    start_time = time.time()
    batch = next(iter(dataloader))
    batch_time = time.time() - start_time
    
    # Print batch info
    print(f"Batch loaded in {batch_time:.4f} seconds")
    print(f"Batch size: {batch['blocks'].shape}")
    print(f"Mask size: {batch['mask'].shape}")
    
    # Calculate the percentage of valid (non-padded) positions in the batch
    valid_percentage = batch['mask'].float().mean().item() * 100
    print(f"Percentage of valid positions in batch: {valid_percentage:.2f}%")
    
    return batch

def analyze_block_distribution(dataset):
    """Analyze the distribution of blocks in the dataset."""
    print("\nAnalyzing block distribution...")
    
    # Count occurrences of each block type in a sample of items
    block_counts = {}
    sample_size = min(20, len(dataset))
    
    for i in tqdm(range(sample_size), desc="Sampling items"):
        item = dataset[i]
        blocks = item['blocks'].numpy()
        mask = item['mask'].numpy()
        
        # Only count blocks in valid positions
        valid_blocks = blocks[mask > 0]
        unique, counts = np.unique(valid_blocks, return_counts=True)
        
        for block_idx, count in zip(unique, counts):
            if block_idx in block_counts:
                block_counts[block_idx] += count
            else:
                block_counts[block_idx] = count
    
    # Sort blocks by frequency
    sorted_blocks = sorted(block_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Print the most common blocks
    print("\nMost common blocks:")
    total_blocks = sum(block_counts.values())
    for i, (block_idx, count) in enumerate(sorted_blocks[:10]):
        block_name = dataset.idx_to_block.get(block_idx, '<unknown>')
        percentage = (count / total_blocks) * 100
        print(f"{i+1}. {block_name}: {count} occurrences ({percentage:.2f}%)")
    
    return block_counts

if __name__ == "__main__":
    # Run tests
    dataset = test_dataset_loading()
    batch = test_dataloader(dataset)
    block_counts = analyze_block_distribution(dataset)
    
    print("\nAll tests completed successfully!")