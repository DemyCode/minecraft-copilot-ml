# Minecraft Map Generation with VAE and Flow Matching

This repository contains code for generating Minecraft maps using a Variational Autoencoder (VAE) and Conditional Flow Matching (CFM).

## Overview

The project consists of the following components:

1. **Minecraft Dataset**: A PyTorch dataset for loading Minecraft schematic files.
2. **Variational Autoencoder (VAE)**: A neural network that encodes Minecraft maps into a latent space and decodes them back.
3. **Conditional Flow Matching (CFM)**: A generative model that learns to generate new Minecraft maps by modeling the flow in the latent space.

## Requirements

- Python 3.6+
- PyTorch
- NumPy
- Matplotlib
- tqdm
- nbtlib (for loading schematic files)
- scikit-learn (for t-SNE visualization)

## Usage

### 1. Prepare the Dataset

Place your Minecraft schematic files in a directory (e.g., `minecraft-schematics-raw`). The dataset will automatically load and process these files.

### 2. Train the VAE

```bash
python vae_demo.py --data_dir minecraft-schematics-raw --train --epochs 20 --save_model models/minecraft_vae.pth
```

### 3. Evaluate and Visualize the VAE

```bash
python vae_demo.py --data_dir minecraft-schematics-raw --load_model models/minecraft_vae.pth --evaluate --visualize_latent --generate --interpolate
```

### 4. Train the Flow Matching Network

```bash
python flow_matching.py --data_dir minecraft-schematics-raw --vae_model models/minecraft_vae.pth --train --epochs 20 --save_flow_model models/minecraft_flow.pth
```

### 5. Generate Samples with the Flow Matching Network

```bash
python flow_matching.py --data_dir minecraft-schematics-raw --vae_model models/minecraft_vae.pth --load_flow_model models/minecraft_flow.pth --generate --n_samples 10
```

## Model Architecture

### Variational Autoencoder (VAE)

The VAE consists of:
- An encoder that maps Minecraft maps to a latent space
- A decoder that reconstructs maps from the latent space
- A reparameterization trick for sampling from the latent space

The VAE takes into account the mask to handle variable-sized inputs.

### Conditional Flow Matching (CFM)

The CFM learns to model the vector field of a continuous normalizing flow in the latent space. It can be conditioned on additional context vectors for controlled generation.

## Files

- `minecraft_dataset.py`: PyTorch dataset for Minecraft schematic files
- `schematic_loader.py`: Utility for loading schematic files
- `minecraft_vae.py`: Implementation of the Variational Autoencoder
- `vae_demo.py`: Demo script for the VAE
- `flow_matching.py`: Implementation of the Conditional Flow Matching network
- `test_dataset.py`: Test script for the dataset

## Examples

### VAE Samples

The VAE can generate samples by sampling from the latent space and decoding:

```python
# Sample from the VAE
samples = vae.sample(num_samples=5, device="cuda")
```

### Flow Matching Samples

The Flow Matching network can generate samples by solving the ODE:

```python
# Generate samples with the Flow Matching network
samples = generate_samples_with_flow(vae, flow_model, device="cuda", num_samples=5)
```

## Customization

You can customize the models by adjusting the following parameters:

- `latent_dim`: Dimension of the latent space
- `embedding_dim`: Dimension of the block embeddings
- `hidden_dims`: Dimensions of the hidden layers
- `context_dim`: Dimension of the context vector for conditional generation

## License

This project is licensed under the MIT License - see the LICENSE file for details.