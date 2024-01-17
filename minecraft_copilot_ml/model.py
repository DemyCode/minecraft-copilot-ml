from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import mlflow  # type: ignore


class UNet3D(pl.LightningModule):
    def __init__(self, n_unique_minecraft_blocks: int) -> None:
        super(UNet3D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(n_unique_minecraft_blocks, 64, kernel_size=3, padding=1),
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
            nn.Conv3d(64, n_unique_minecraft_blocks, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(num_features=n_unique_minecraft_blocks),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4) + out_conv3
        out_conv6 = self.conv6(out_conv5) + out_conv2
        out_conv7 = self.conv7(out_conv6) + out_conv1
        out_conv8 = self.conv8(out_conv7) + x
        return torch.softmax(out_conv8, dim=1)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat: torch.Tensor = self(x)
        loss = F.cross_entropy(y_hat, y)
        accuracy = (y_hat.argmax(dim=1) == y.argmax(dim=1)).float().mean()
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_accuracy", accuracy, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat: torch.Tensor = self(x)
        loss = F.cross_entropy(y_hat, y)
        accuracy = (y_hat.argmax(dim=1) == y.argmax(dim=1)).float().mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", accuracy, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def on_validation_end(self) -> None:
        mlflow.pytorch.log_model(self, f"model-{self.current_epoch}")
