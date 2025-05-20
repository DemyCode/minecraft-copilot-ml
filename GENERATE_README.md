# Minecraft Structure Generation

This README explains how to use the `generate.py` script to generate Minecraft structures from a trained flow matching model.

## Prerequisites

Make sure you have the following dependencies installed:

```bash
pip install torch torchvision torchdyn nbtlib scikit-learn tqdm
```

## Usage

The script supports two modes of operation:
1. **Generation Mode**: Generate new Minecraft structures from scratch
2. **Reconstruction Mode**: Take existing structures, partially destroy them, and reconstruct them

### Generation Mode

To generate Minecraft structures from scratch, run:

```bash
python generate.py --model_path /path/to/your/model_checkpoint.pt
```

### Reconstruction Mode

To reconstruct partially destroyed structures, run:

```bash
python generate.py --model_path /path/to/your/model_checkpoint.pt --reconstruction --destruction_percentage 0.3
```

### Command-line Arguments

#### Common Arguments
- `--model_path`: Path to the model checkpoint (required)
- `--cache_file`: Path to the block mappings cache file (default: "cache/block_mappings.pkl")
- `--embedding_cache`: Path to the block embeddings cache file (default: "cache/block_embeddings.pt")
- `--output_dir`: Directory to save generated samples (default: "generated_samples")
- `--embedding_dim`: Dimension of block embeddings (default: 32)
- `--save_npy`: Save raw numpy arrays of block indices
- `--save_schematic`: Save as Minecraft schematic files (requires nbtlib)

#### Generation Mode Arguments
- `--num_samples`: Number of samples to generate (default: 8)
- `--chunk_size`: Size of the generated chunks (default: 16)

#### Reconstruction Mode Arguments
- `--reconstruction`: Enable reconstruction mode
- `--destruction_percentage`: Percentage of blocks to destroy (0-1, default: 0.3)
- `--schematics_dir`: Directory containing schematic files for reconstruction (default: "minecraft-schematics-raw")
- `--num_structures`: Number of structures to reconstruct (default: 4)

### Examples

```bash
# Generate 16 samples from scratch and save as schematic files
python generate.py --model_path output/unet/unet_minecraft_epoch_160_checkpoint.pt --num_samples 16 --save_schematic

# Reconstruct 4 structures with 50% destruction
python generate.py --model_path output/unet/unet_minecraft_epoch_160_checkpoint.pt --reconstruction --destruction_percentage 0.5 --num_structures 4 --save_schematic
```

## Output

### Generation Mode Output

The script generates:

1. Optionally, numpy arrays of block indices (if `--save_npy` is specified)
2. Optionally, Minecraft schematic files that can be imported into Minecraft (if `--save_schematic` is specified)

### Reconstruction Mode Output

The script generates three sets of files in separate directories:

1. **original/** - The original structures before destruction
2. **destroyed/** - The partially destroyed structures
3. **reconstructed/** - The structures after reconstruction by the model

Each directory contains:
- Optionally, numpy arrays of block indices (if `--save_npy` is specified)
- Optionally, Minecraft schematic files (if `--save_schematic` is specified)

This allows you to compare the original, destroyed, and reconstructed versions side by side.

## How It Works

### Generation Mode

1. The script loads the trained flow matching model
2. It generates continuous embeddings using Neural ODE with the correct dimensions (embedding_dim × 16 × 16 × 16)
3. It maps these continuous embeddings back to discrete Minecraft blocks using cosine similarity
4. It saves the results in the specified formats

### Reconstruction Mode

1. The script loads the trained flow matching model
2. It loads real Minecraft structures from the dataset
3. It partially destroys the structures by replacing a percentage of blocks with random noise
4. It uses the model to reconstruct the original structures
5. It saves the original, destroyed, and reconstructed versions for comparison

The reconstruction process demonstrates the model's ability to understand and repair Minecraft structures, which is useful for:
- Repairing damaged structures in-game
- Completing partially built structures
- Understanding the structural patterns the model has learned

## Importing into Minecraft

If you've generated schematic files (with the `--save_schematic` option), you can import them into Minecraft using tools like:

- [WorldEdit](https://www.curseforge.com/minecraft/mc-mods/worldedit)
- [Litematica](https://www.curseforge.com/minecraft/mc-mods/litematica)
- [MCEdit](https://www.mcedit.net/)

Simply load the .schem file in your preferred tool and place it in your Minecraft world.