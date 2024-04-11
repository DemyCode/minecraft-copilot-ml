# flake8: noqa: E203
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from minecraft_copilot_ml.data_loader import MinecraftSchematicsDatasetItemType


class ConvBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super(ConvBlock3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.relu(self.bn(self.conv(x)))
        return result


class UNet3d(pl.LightningModule):
    def __init__(
        self,
        unique_blocks_dict: Dict[str, int],
        unique_counts_coefficients: Optional[np.ndarray] = None,
        latent_dim: int = 64,
    ):
        super(UNet3d, self).__init__()
        self.unique_blocks_dict = unique_blocks_dict
        self.reverse_unique_blocks_dict = {v: k for k, v in unique_blocks_dict.items()}
        self.latent_dim = latent_dim
        if unique_counts_coefficients is None:
            unique_counts_coefficients = np.ones(len(unique_blocks_dict))
        self.unique_counts_coefficients = (
            torch.from_numpy(unique_counts_coefficients).float().to("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.conv_input = ConvBlock3d(1, 32)
        self.conv1 = ConvBlock3d(32, 64)
        self.conv2 = ConvBlock3d(64, 128)
        self.conv3 = ConvBlock3d(128, 256)
        self.conv4 = ConvBlock3d(256, 512)

        self.conv6 = ConvBlock3d(512, 256)
        self.conv7 = ConvBlock3d(256, 128)
        self.conv8 = ConvBlock3d(128, 64)
        self.conv9 = ConvBlock3d(64, 32)
        self.conv_output = nn.Conv3d(32, len(unique_blocks_dict), kernel_size=3, padding=1)

    def ml_core(self, x: torch.Tensor) -> torch.Tensor:
        # Encode input
        out_conv_input = self.conv_input(x)
        out_conv_1 = self.conv1(out_conv_input)
        out_conv_2 = self.conv2(out_conv_1)
        out_conv_3 = self.conv3(out_conv_2)
        out_conv_4 = self.conv4(out_conv_3)

        # Decode input
        out_conv_6 = self.conv6(out_conv_4) + out_conv_3
        out_conv_7 = self.conv7(out_conv_6) + out_conv_2
        out_conv_8 = self.conv8(out_conv_7) + out_conv_1
        out_conv_9 = self.conv9(out_conv_8) + out_conv_input
        out_conv_output: torch.Tensor = self.conv_output(out_conv_9)
        return out_conv_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reconstruction = self.ml_core(x)
        reconstruction = F.softmax(reconstruction, dim=1)
        return reconstruction

    def step(self, batch: MinecraftSchematicsDatasetItemType, batch_idx: int, mode: str) -> torch.Tensor:
        block_maps, noisy_block_maps, block_map_masks, loss_masks = batch
        pre_processed_block_maps = self.pre_process(block_maps)
        pre_processed_noisy_block_maps = self.pre_process(noisy_block_maps).float().unsqueeze(1)
        tensor_block_map_masks = (
            torch.from_numpy(block_map_masks).float().to("cuda" if torch.cuda.is_available() else "cpu").long()
        )
        tensor_loss_masks = (
            torch.from_numpy(loss_masks).float().to("cuda" if torch.cuda.is_available() else "cpu").long()
        )
        reconstruction = self.ml_core(pre_processed_noisy_block_maps)

        # Compute accuracy
        accuracy_truth_map = (reconstruction.argmax(dim=1) == pre_processed_block_maps).float()
        accuracy_on_block_map = accuracy_truth_map[tensor_block_map_masks.bool()].mean()
        accuracy_on_loss_map = accuracy_truth_map[tensor_loss_masks.bool()].mean()

        # Compute reconstruction loss using categorical cross-entropy
        reconstruction_loss = F.cross_entropy(reconstruction, pre_processed_block_maps, reduction="none")
        reconstruction_loss = reconstruction_loss * tensor_block_map_masks
        reconstruction_loss = reconstruction_loss * torch.where(tensor_loss_masks == 1, reconstruction_loss, 1)
        reconstruction_loss = reconstruction_loss * self.unique_counts_coefficients[pre_processed_block_maps]
        loss = reconstruction_loss.mean()

        # Total loss
        loss_dict = {
            "loss": loss,
            "accuracy_on_block_map": accuracy_on_block_map,
            "accuracy_on_loss_map": accuracy_on_loss_map,
            "learning_rate": self.trainer.optimizers[0].param_groups[0]["lr"],
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

    def post_process(self, x: torch.Tensor) -> np.ndarray:
        predicted_block_maps: np.ndarray = np.vectorize(self.reverse_unique_blocks_dict.get)(x.argmax(dim=1).numpy())
        return predicted_block_maps

    def training_step(self, batch: MinecraftSchematicsDatasetItemType, batch_idx: int) -> torch.Tensor:
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch: MinecraftSchematicsDatasetItemType, batch_idx: int) -> torch.Tensor:
        return self.step(batch, batch_idx, "val")

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, mode="max", factor=0.1),
                "monitor": "val_accuracy_on_loss_map",
            },
        }

    def on_train_start(self) -> None:
        print(self)
