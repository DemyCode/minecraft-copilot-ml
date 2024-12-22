import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from minecraft_copilot_ml.data_loader import (
    MinecraftSchematicsDataset,
    MinecraftSchematicsDatasetItemType,
    get_working_files_and_unique_blocks,
    list_schematic_files_in_folder,
)


# Define the Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(input_dim, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(64 * 4 * 4 * 4, latent_dim)  # Adjust dimensions
        self.fc_logvar = nn.Linear(64 * 4 * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 64 * 4 * 4 * 4)  # Adjust dimensions
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(32, output_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.fc(z).view(-1, 64, 4, 4, 4)  # Adjust dimensions
        x = self.decoder(z)
        return x


# Define the VAE
class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)  # Reparameterization trick
        recon_x = self.decoder(z)
        return recon_x, mu, logvar


# Define the masked loss function
def masked_vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    # Ensure mask matches recon_x dimensions
    mask = mask.float()
    recon_loss = ((recon_x - x) ** 2) * mask
    recon_loss = recon_loss.sum() / mask.sum()  # Normalize by the number of valid voxels
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


# Training loop
def train_vae(
    model: VAE,
    dataloader: DataLoader,
    optimizer: optim.Adam,
    epochs: int = 10,
    device: str = "cuda",
) -> None:
    model.to(device)
    model.train()
    for epoch in range(epochs):
        total_loss: float = 0
        for batch in dataloader:
            x, mask = batch  # Ensure your DataLoader returns (data, mask)
            x, mask = torch.from_numpy(x), torch.from_numpy(mask)
            x, mask = x.to(device), mask.to(device)
            x, mask = x.float(), mask.float()
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss = masked_vae_loss(recon_x, x, mu, logvar, mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")


# Main
if __name__ == "__main__":
    latent_dim = 16
    batch_size = 32
    epochs = 10
    learning_rate = 1e-3

    # Replace with your dataset
    limit = 2
    path_to_schematics = "/home/mehdi/minecraft-copilot-ml/schematics_data_2_3_10"
    schematics_list_files = list_schematic_files_in_folder(path_to_schematics)
    schematics_list_files = sorted(schematics_list_files)
    unique_blocks_dict, loaded_schematic_files = get_working_files_and_unique_blocks(schematics_list_files)
    input_dim = len(unique_blocks_dict)

    dataset = MinecraftSchematicsDataset(schematics_list_files[:limit], unique_blocks_dict)
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

    def collate_fn(
        batch: List[MinecraftSchematicsDatasetItemType],
    ) -> MinecraftSchematicsDatasetItemType:
        block_map, block_map_mask = zip(*batch)
        return np.stack(block_map), np.stack(block_map_mask)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    vae = VAE(input_dim=input_dim, latent_dim=latent_dim)
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

    train_vae(
        vae,
        train_dataloader,
        optimizer,
        epochs=epochs,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
