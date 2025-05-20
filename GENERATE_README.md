# Minecraft Structure Generation

This README explains how to use the `generate.py` script to generate Minecraft structures from a trained flow matching model.

## Prerequisites

Make sure you have the following dependencies installed:

```bash
pip install torch torchvision torchdyn nbtlib scikit-learn tqdm
```

## Usage

To generate Minecraft structures, run the `generate.py` script with the path to your trained model:

```bash
python generate.py --model_path /path/to/your/model_checkpoint.pt
```

### Command-line Arguments

- `--model_path`: Path to the model checkpoint (required)
- `--cache_file`: Path to the block mappings cache file (default: "cache/block_mappings.pkl")
- `--embedding_cache`: Path to the block embeddings cache file (default: "cache/block_embeddings.pt")
- `--output_dir`: Directory to save generated samples (default: "generated_samples")
- `--num_samples`: Number of samples to generate (default: 8)
- `--chunk_size`: Size of the generated chunks (default: 16)
- `--embedding_dim`: Dimension of block embeddings (default: 32)
- `--save_npy`: Save raw numpy arrays of block indices
- `--save_schematic`: Save as Minecraft schematic files (requires nbtlib)

### Example

```bash
# Generate 16 samples and save as both images and schematic files
python generate.py --model_path output/unet/unet_minecraft_epoch_160_checkpoint.pt --num_samples 16 --save_schematic
```

## Output

The script generates:

1. Image visualizations of the generated structures (saved as PNG files)
2. Optionally, numpy arrays of block indices (if `--save_npy` is specified)
3. Optionally, Minecraft schematic files that can be imported into Minecraft (if `--save_schematic` is specified)

## How It Works

1. The script loads the trained flow matching model
2. It generates continuous embeddings using Neural ODE with the correct dimensions (embedding_dim × 16 × 16 × 16)
3. It maps these continuous embeddings back to discrete Minecraft blocks using cosine similarity
4. It saves the results in the specified formats

## Importing into Minecraft

If you've generated schematic files (with the `--save_schematic` option), you can import them into Minecraft using tools like:

- [WorldEdit](https://www.curseforge.com/minecraft/mc-mods/worldedit)
- [Litematica](https://www.curseforge.com/minecraft/mc-mods/litematica)
- [MCEdit](https://www.mcedit.net/)

Simply load the .schem file in your preferred tool and place it in your Minecraft world.