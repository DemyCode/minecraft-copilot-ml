from typing import Any, Dict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet3D(pl.LightningModule):
    def __init__(self, unique_blocks_dict: Dict[str, int], unique_counts_coefficients: np.ndarray) -> None:
        super(UNet3D, self).__init__()
        self.unique_blocks_dict = unique_blocks_dict
        self.unique_counts_coefficients = torch.from_numpy(unique_counts_coefficients).float().to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.reverse_unique_blocks_dict = {v: k for k, v in unique_blocks_dict.items()}
        self.conv1 = nn.Sequential(
            nn.Conv3d(len(self.unique_blocks_dict), 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(num_features=64),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(num_features=128),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(num_features=256),
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(num_features=512),
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(512, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(num_features=256),
        )
        self.conv6 = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(num_features=128),
        )
        self.conv7 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(num_features=64),
        )
        self.conv8 = nn.Sequential(
            nn.Conv3d(64, len(self.unique_blocks_dict), kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(num_features=len(self.unique_blocks_dict)),
        )

    def ml_core(self, x: torch.Tensor) -> torch.Tensor:
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4) + out_conv3
        out_conv6 = self.conv6(out_conv5) + out_conv2
        out_conv7 = self.conv7(out_conv6) + out_conv1
        out_conv8 = self.conv8(out_conv7) + x
        return F.softmax(out_conv8, dim=1)

    def pre_process(self, x: np.ndarray) -> torch.Tensor:
        vectorized_x = np.vectorize(lambda x: self.unique_blocks_dict.get(x, self.unique_blocks_dict["minecraft:air"]))(
            x
        )
        vectorized_x = vectorized_x.astype(np.int64)
        x_tensor = torch.from_numpy(vectorized_x)
        x_tensor = x_tensor.to("cuda" if torch.cuda.is_available() else "cpu")
        x_tensor_one_hot_encoded: torch.Tensor = torch.functional.F.one_hot(
            x_tensor, num_classes=len(self.unique_blocks_dict)
        ).permute(0, 4, 1, 2, 3)
        x_tensor_one_hot_encoded = x_tensor_one_hot_encoded.float()
        return x_tensor_one_hot_encoded
    
    def post_process(self, x: torch.Tensor) -> np.ndarray:
        predicted_block_maps: np.ndarray = np.vectorize(self.reverse_unique_blocks_dict.get)(x.argmax(dim=1).numpy())
        return predicted_block_maps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ml_core(x)

    def step(self, batch: torch.Tensor, batch_idx: int, mode: str) -> torch.Tensor:
        block_maps, noisy_block_maps, masks = batch
        pre_processed_block_maps = self.pre_process(block_maps)
        pre_processed_noisy_block_maps = self.pre_process(noisy_block_maps)
        masks = torch.from_numpy(masks).float().to("cuda" if torch.cuda.is_available() else "cpu")
        predicted_one_hot_block_maps: torch.Tensor = self.ml_core(pre_processed_noisy_block_maps)
        loss = F.cross_entropy(predicted_one_hot_block_maps, pre_processed_block_maps, reduction="none")
        loss = loss * self.unique_counts_coefficients[pre_processed_block_maps.argmax(dim=1)]
        loss = loss * masks
        loss = loss.mean()
        self.log(
            f"{mode}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=block_maps.shape[0],
        )
        return loss

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
