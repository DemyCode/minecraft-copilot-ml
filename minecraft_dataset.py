#!/usr/bin/env python3
"""
PyTorch dataset for Minecraft schematic files.
This dataset loads schematic files and provides 16×16×16 chunks with masks.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle
from collections import defaultdict

from schematic_loader import load_schematic_to_numpy


class MinecraftSchematicDataset(Dataset):
    """
    PyTorch dataset for Minecraft schematic files.
    Provides 16×16×16 chunks with masks for areas where dimensions are smaller than 16.
    """

    def __init__(
        self,
        schematics_dir,
        chunk_size=16,
        transform=None,
        preload=False,
        cache_file=None,
        max_files=None,
        min_dimension=16,
    ):
        """
        Initialize the dataset.

        Args:
            schematics_dir (str): Directory containing schematic files
            chunk_size (int): Size of the chunks to extract (default: 16)
            transform (callable, optional): Optional transform to be applied on a sample
            preload (bool): Whether to preload all data into memory (default: False)
            cache_file (str, optional): Path to cache file for block mappings
            max_files (int, optional): Maximum number of files to load
            min_dimension (int): Minimum dimension required for a schematic to be included
        """
        self.schematics_dir = schematics_dir
        self.chunk_size = chunk_size
        self.transform = transform
        self.preload = preload
        self.min_dimension = min_dimension

        # Find all schematic files
        self.schematic_files = []
        for root, _, files in os.walk(schematics_dir):
            for file in files:
                if file.endswith(".schematic"):
                    self.schematic_files.append(os.path.join(root, file))

        # Limit the number of files if specified
        if max_files is not None:
            self.schematic_files = self.schematic_files[:max_files]

        # Create a mapping of block names to indices
        self.block_to_idx = {}
        self.idx_to_block = {}

        # Try to load block mappings from cache
        if cache_file and os.path.exists(cache_file):
            print(f"Loading block mappings from cache: {cache_file}")
            with open(cache_file, "rb") as f:
                cache_data = pickle.load(f)
                self.block_to_idx = cache_data["block_to_idx"]
                self.idx_to_block = cache_data["idx_to_block"]
                self.file_info = cache_data.get("file_info", [])
        else:
            # Scan files to build block mappings and collect file info
            self._scan_files()

            # Save block mappings to cache if specified
            if cache_file:
                print(f"Saving block mappings to cache: {cache_file}")
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                with open(cache_file, "wb") as f:
                    pickle.dump(
                        {
                            "block_to_idx": self.block_to_idx,
                            "idx_to_block": self.idx_to_block,
                            "file_info": self.file_info,
                        },
                        f,
                    )

        # Preload data if requested
        self.preloaded_data = None
        if self.preload:
            self._preload_data()

    def _scan_files(self):
        """Scan all schematic files to build block mappings and collect file info."""
        print("Scanning schematic files to build block mappings...")

        # Add special tokens
        self.block_to_idx["<pad>"] = 0
        self.block_to_idx["<unk>"] = 1
        self.idx_to_block[0] = "<pad>"
        self.idx_to_block[1] = "<unk>"

        # Collect block types and file info
        block_counts = defaultdict(int)
        self.file_info = []

        for file_path in tqdm(self.schematic_files):
            try:
                # Load the schematic
                blocks, dimensions = load_schematic_to_numpy(file_path)

                # Check if dimensions meet the minimum requirement
                height, length, width = blocks.shape
                if (
                    height < self.min_dimension
                    or length < self.min_dimension
                    or width < self.min_dimension
                ):
                    continue

                # Count unique blocks
                unique_blocks = np.unique(blocks)
                for block in unique_blocks:
                    block_counts[block] += 1

                # Store file info
                self.file_info.append(
                    {"path": file_path, "dimensions": dimensions, "shape": blocks.shape}
                )
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        # Create block mappings (starting from 2 because 0 and 1 are reserved)
        next_idx = 2
        for block, count in sorted(
            block_counts.items(), key=lambda x: x[1], reverse=True
        ):
            if block not in self.block_to_idx:
                self.block_to_idx[block] = next_idx
                self.idx_to_block[next_idx] = block
                next_idx += 1

        print(f"Found {len(self.file_info)} valid schematic files")
        print(f"Found {len(self.block_to_idx)} unique block types")

    def _preload_data(self):
        """Preload all data into memory."""
        print("Preloading data into memory...")
        self.preloaded_data = []

        for info in tqdm(self.file_info):
            try:
                blocks, _ = load_schematic_to_numpy(info["path"])
                self.preloaded_data.append(blocks)
            except Exception as e:
                print(f"Error preloading {info['path']}: {e}")
                self.preloaded_data.append(None)

    def _get_blocks(self, idx):
        """Get blocks array for a given index."""
        if self.preload and self.preloaded_data[idx] is not None:
            return self.preloaded_data[idx]
        else:
            blocks, _ = load_schematic_to_numpy(self.file_info[idx]["path"])
            return blocks

    def _extract_chunk(self, blocks, start_y, start_z, start_x):
        """
        Extract a chunk from the blocks array.

        Args:
            blocks: The blocks array
            start_y, start_z, start_x: Starting coordinates for the chunk

        Returns:
            chunk: The extracted chunk
            mask: Mask indicating valid positions (1) vs padding (0)
        """
        height, length, width = blocks.shape
        chunk_size = self.chunk_size

        # Initialize chunk and mask with zeros (padding)
        chunk = np.full(
            (chunk_size, chunk_size, chunk_size),
            self.block_to_idx["<pad>"],
            dtype=np.int64,
        )
        mask = np.zeros((chunk_size, chunk_size, chunk_size), dtype=np.int64)

        # Calculate end coordinates (clamped to array dimensions)
        end_y = min(start_y + chunk_size, height)
        end_z = min(start_z + chunk_size, length)
        end_x = min(start_x + chunk_size, width)

        # Calculate actual chunk dimensions
        chunk_height = end_y - start_y
        chunk_length = end_z - start_z
        chunk_width = end_x - start_x

        # Extract the chunk from the blocks array
        for y in range(chunk_height):
            for z in range(chunk_length):
                for x in range(chunk_width):
                    block_name = blocks[start_y + y, start_z + z, start_x + x]
                    chunk[y, z, x] = self.block_to_idx.get(
                        block_name, self.block_to_idx["<unk>"]
                    )
                    mask[y, z, x] = 1  # Mark as valid

        return chunk, mask

    def __len__(self):
        """Return the number of chunks in the dataset."""
        return len(self.file_info)

    def __getitem__(self, idx):
        """
        Get a chunk from the dataset.

        Args:
            idx (int): Index

        Returns:
            dict: A dictionary containing:
                - 'blocks': Tensor of shape (chunk_size, chunk_size, chunk_size) with block indices
                - 'mask': Tensor of shape (chunk_size, chunk_size, chunk_size) with 1 for valid positions, 0 for padding
                - 'file_path': Path to the source schematic file
        """
        # Get the blocks array
        blocks = self._get_blocks(idx)

        # Get dimensions
        height, length, width = blocks.shape

        # Choose a random starting position
        start_y = random.randint(0, max(0, height - self.chunk_size))
        start_z = random.randint(0, max(0, length - self.chunk_size))
        start_x = random.randint(0, max(0, width - self.chunk_size))

        # Extract the chunk
        chunk, mask = self._extract_chunk(blocks, start_y, start_z, start_x)

        # Convert to tensors
        chunk_tensor = torch.tensor(chunk, dtype=torch.long)
        mask_tensor = torch.tensor(mask, dtype=torch.float)

        # Apply transform if specified
        if self.transform:
            chunk_tensor = self.transform(chunk_tensor)

        return {
            "blocks": chunk_tensor,
            "mask": mask_tensor,
            "file_path": self.file_info[idx]["path"],
        }


# Example usage
if __name__ == "__main__":
    # Create the dataset
    dataset = MinecraftSchematicDataset(
        schematics_dir="minecraft-schematics-raw",
        chunk_size=16,
        cache_file="cache/block_mappings.pkl",
        max_files=100,  # Limit to 100 files for testing
    )

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    # Get a batch
    batch = next(iter(dataloader))

    # Print batch info
    print(f"Batch size: {batch['blocks'].shape}")
    print(f"Mask size: {batch['mask'].shape}")

    # Print some statistics
    print(f"\nDataset size: {len(dataset)}")
    print(f"Number of unique blocks: {len(dataset.block_to_idx)}")

    # Print the most common blocks
    print("\nBlock mapping:")
    for i in range(min(10, len(dataset.idx_to_block))):
        print(f"{i}: {dataset.idx_to_block.get(i, '<unknown>')}")

    # Calculate the percentage of valid (non-padded) positions in the batch
    valid_percentage = batch["mask"].float().mean().item() * 100
    print(f"\nPercentage of valid positions in batch: {valid_percentage:.2f}%")

