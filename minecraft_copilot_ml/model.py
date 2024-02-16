# flake8: noqa: E203
from typing import Any, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(pl.LightningModule):
    def __init__(self, unique_blocks_dict, unique_counts_coefficients=None, latent_dim=64):
        super(VAE, self).__init__()
        self.unique_blocks_dict = unique_blocks_dict
        self.reverse_unique_blocks_dict = {v: k for k, v in unique_blocks_dict.items()}
        self.latent_dim = latent_dim
        if unique_counts_coefficients is None:
            unique_counts_coefficients = np.ones(len(unique_blocks_dict))
        self.unique_counts_coefficients = (
            torch.from_numpy(unique_counts_coefficients).float().to("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(len(unique_blocks_dict), 32, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(128 * 2 * 2 * 2, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 2 * latent_dim),  # 2 * latent_dim for mean and variance
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128 * 2 * 2 * 2),
            nn.LeakyReLU(),
            nn.Unflatten(1, (128, 2, 2, 2)),
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, output_padding=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, output_padding=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(32, len(unique_blocks_dict), kernel_size=3, stride=2, output_padding=1, padding=1),
        )

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def ml_core(self, x: torch.Tensor) -> torch.Tensor:
        # Encode input
        mean_variance = self.encoder(x)
        mean = mean_variance[:, : self.latent_dim]
        log_variance = mean_variance[:, self.latent_dim :]

        # Reparameterization trick
        z = self.reparameterize(mean, log_variance)

        # Decode
        reconstruction = self.decoder(z)
        return reconstruction, mean, log_variance

    def forward(self, x: torch.Tensor):
        reconstruction, mean, log_variance = self.ml_core(x)
        reconstruction = F.softmax(reconstruction, dim=1)
        return reconstruction, mean, log_variance

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int, mode: str) -> torch.Tensor:
        block_maps, noisy_block_maps, masks = batch
        pre_processed_block_maps = self.pre_process(block_maps)
        pre_processed_noisy_block_maps = self.pre_process_noisy(noisy_block_maps)
        masks = torch.from_numpy(masks).float().to("cuda" if torch.cuda.is_available() else "cpu").long()
        reconstruction, mean, log_var = self.ml_core(pre_processed_noisy_block_maps)

        # Compute reconstruction loss using categorical cross-entropy
        reconstruction_loss = F.cross_entropy(reconstruction, pre_processed_block_maps, reduction="none")
        reconstruction_loss = reconstruction_loss * masks
        reconstruction_loss = reconstruction_loss * self.unique_counts_coefficients[pre_processed_block_maps]
        reconstruction_loss = reconstruction_loss.mean()

        # Compute KL divergence
        kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        # Total loss
        loss = reconstruction_loss + kl_divergence
        loss_dict = {
            "reconstruction_loss": reconstruction_loss,
            "kl_divergence": kl_divergence,
            "loss": loss,
        }
        for name, value in loss_dict.items():
            self.log(
                f"{mode}_{name}",
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=block_maps.shape[0],
            )
        return loss

    def pre_process(self, x: np.ndarray) -> torch.Tensor:
        vectorized_x = np.vectorize(lambda x: self.unique_blocks_dict.get(x, self.unique_blocks_dict["minecraft:air"]))(
            x
        )
        vectorized_x = vectorized_x.astype(np.int64)
        x_tensor = torch.from_numpy(vectorized_x)
        x_tensor = x_tensor.to("cuda" if torch.cuda.is_available() else "cpu")
        return x_tensor

    def pre_process_noisy(self, x: np.ndarray) -> torch.Tensor:
        x_tensor = self.pre_process(x)
        x_tensor_one_hot_encoded: torch.Tensor = torch.functional.F.one_hot(
            x_tensor, num_classes=len(self.unique_blocks_dict)
        ).permute(0, 4, 1, 2, 3)
        x_tensor_one_hot_encoded = x_tensor_one_hot_encoded.float()
        return x_tensor_one_hot_encoded

    def post_process(self, x: torch.Tensor) -> np.ndarray:
        predicted_block_maps: np.ndarray = np.vectorize(self.reverse_unique_blocks_dict.get)(x.argmax(dim=1).numpy())
        return predicted_block_maps

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self.step(batch, batch_idx, "val")

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
                "monitor": "val_loss",
            },
        }
