from typing import Tuple, Dict
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lightning as pl
from improved_diffusion.unet import UNetModel, ClassicResBlock, Downsample, Upsample, conv_nd  # type: ignore[import-untyped]

from minecraft_copilot_ml.data_loader import MinecraftSchematicsDatasetItemType


class Encoder(nn.Module):
    def __init__(self, channels: int, hidden_channels: int):
        super(Encoder, self).__init__()
        self.input_conv = conv_nd(
            dims=3, in_channels=channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1
        )
        self.res_1 = ClassicResBlock(channels=hidden_channels, dropout=0.1, out_channels=hidden_channels * 2, dims=3)
        self.down_1 = Downsample(channels=hidden_channels * 2, use_conv=True, dims=3)
        self.res_2 = ClassicResBlock(
            channels=hidden_channels * 2, dropout=0.1, out_channels=hidden_channels * 3, dims=3
        )
        self.down_2 = Downsample(hidden_channels * 3, True, dims=3)
        self.res_3 = ClassicResBlock(
            channels=hidden_channels * 3, dropout=0.1, out_channels=hidden_channels * 4, dims=3
        )
        self.down_3 = Downsample(hidden_channels * 4, True, dims=3)
        self.mu_logvar = conv_nd(
            dims=3,
            in_channels=hidden_channels * 4,
            out_channels=hidden_channels * 4 * 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_conv(x)
        x = self.res_1(x)
        x = self.down_1(x)
        x = self.res_2(x)
        x = self.down_2(x)
        x = self.res_3(x)
        x = self.down_3(x)
        mu_logvar = self.mu_logvar(x)
        mu, logvar = torch.chunk(mu_logvar, 2, dim=1)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, activation: nn.Module = nn.Sigmoid()):
        super(Decoder, self).__init__()
        self.res_1 = ClassicResBlock(
            channels=hidden_channels * 4, dropout=0.1, out_channels=hidden_channels * 3, dims=3
        )
        self.up_1 = Upsample(hidden_channels * 3, True, dims=3)
        self.res_2 = ClassicResBlock(
            channels=hidden_channels * 3, dropout=0.1, out_channels=hidden_channels * 2, dims=3
        )
        self.up_2 = Upsample(hidden_channels * 2, True, dims=3)
        self.res_3 = ClassicResBlock(channels=hidden_channels * 2, dropout=0.1, out_channels=hidden_channels, dims=3)
        self.up_3 = Upsample(hidden_channels, True, dims=3)
        self.output_conv = conv_nd(
            dims=3,
            in_channels=hidden_channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res_1(x)
        x = self.up_1(x)
        x = self.res_2(x)
        x = self.up_2(x)
        x = self.res_3(x)
        x = self.up_3(x)
        x = self.output_conv(x)
        x = self.activation(x)
        return x


class VAE(nn.Module):
    def __init__(self, channels: int, hidden_channels: int):
        super(VAE, self).__init__()
        self.encoder = Encoder(channels, hidden_channels)
        self.decoder = Decoder(channels, hidden_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class VAETrainer(pl.LightningModule):
    def __init__(self, model: VAE, unique_blocks_dict: Dict[str, int]):  # type: ignore[no-any-unimported]
        super(VAETrainer, self).__init__()
        self.model = model
        self.recon_loss = nn.CrossEntropyLoss(reduction="none")
        self.unique_blocks_dict = unique_blocks_dict

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model(x)  # type: ignore[no-any-return]

    def pre_process(self, x: np.ndarray) -> torch.Tensor:
        vectorized_x = np.vectorize(lambda x: self.unique_blocks_dict.get(x, self.unique_blocks_dict["minecraft:air"]))(
            x
        )
        vectorized_x = vectorized_x.astype(np.int64)
        x_tensor = torch.from_numpy(vectorized_x)
        x_tensor = x_tensor.to("cuda" if torch.cuda.is_available() else "cpu")
        x_tensor = F.one_hot(x_tensor, num_classes=len(self.unique_blocks_dict)).permute(0, 4, 1, 2, 3).float()
        return x_tensor

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch: MinecraftSchematicsDatasetItemType, batch_idx: int) -> torch.Tensor:
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch: MinecraftSchematicsDatasetItemType, batch_idx: int) -> torch.Tensor:
        return self.step(batch, batch_idx, "val")

    def step(self, batch: MinecraftSchematicsDatasetItemType, batch_idx: int, mode: str) -> torch.Tensor:
        block_maps, block_map_masks = batch
        pre_processed_block_maps = self.pre_process(block_maps)
        recon_x, mu, logvar = self(pre_processed_block_maps)
        BCE = self.recon_loss(recon_x, pre_processed_block_maps)
        tensor_block_map_masks = torch.from_numpy(block_map_masks).to(self.device)
        BCE = BCE * tensor_block_map_masks
        BCE = BCE.mean()
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = BCE + KLD
        loss_dict = {
            "loss": loss,
            "loss_bce": BCE,
            "loss_kld": KLD,
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
        return loss  # type: ignore[no-any-return]

    def on_train_start(self) -> None:
        print(self)

import argparse
import json
import os
import subprocess
from typing import List, Optional, Set, Tuple

import boto3
import lightning as pl
import numpy as np
import torch
from improved_diffusion.unet import UNetModel  # type: ignore[import-untyped]
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger
from loguru import logger
from torch.utils.data import DataLoader

from minecraft_copilot_ml.data_loader import (
    MinecraftSchematicsDataset,
    MinecraftSchematicsDatasetItemType,
    get_working_files_and_unique_blocks,
    list_schematic_files_in_folder,
)
from minecraft_copilot_ml.model import MinecraftCopilotTrainer

if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name()
    if device_name is not None and device_name == "GeForce RTX 3090":
        torch.set_float32_matmul_precision("medium")
else:
    logger.warning("No CUDA device found.")


def main(argparser: argparse.ArgumentParser) -> None:
    path_to_schematics: str = argparser.parse_args().path_to_schematics
    path_to_output: str = argparser.parse_args().path_to_output
    epochs: int = argparser.parse_args().epochs
    batch_size: int = argparser.parse_args().batch_size
    dataset_limit: Optional[int] = argparser.parse_args().dataset_limit
    dataset_start: Optional[int] = argparser.parse_args().dataset_start

    if not os.path.exists(path_to_output):
        os.makedirs(path_to_output)

    sync_command = f"aws s3 sync s3://minecraft-schematics-raw {path_to_schematics} --acl public-read --no-sign-request"
    subprocess.run(sync_command, shell=True, check=True)

    schematics_list_files = list_schematic_files_in_folder(path_to_schematics)
    schematics_list_files = sorted(schematics_list_files)
    start = 0
    end = len(schematics_list_files)
    if dataset_start is not None:
        start = dataset_start
    if dataset_limit is not None:
        end = dataset_limit
    schematics_list_files = schematics_list_files[start:end]
    # Set the dictionary size to the number of unique blocks in the dataset.
    # And also select the right files to load.
    unique_blocks_dict, loaded_schematic_files = get_working_files_and_unique_blocks(schematics_list_files)

    logger.info(f"Unique blocks: {unique_blocks_dict}")
    logger.info(f"Number of unique blocks: {len(unique_blocks_dict)}")
    logger.info(f"Number of loaded schematics files: {len(loaded_schematic_files)}")

    train_loaded_schematic_files, test_loaded_schematic_files = train_test_split(
        loaded_schematic_files, test_size=0.2, random_state=42
    )

    train_schematics_dataset = MinecraftSchematicsDataset(train_loaded_schematic_files)
    test_schematics_dataset = MinecraftSchematicsDataset(test_loaded_schematic_files)

    def collate_fn(batch: List[MinecraftSchematicsDatasetItemType]) -> MinecraftSchematicsDatasetItemType:
        block_map, block_map_mask = zip(*batch)
        return np.stack(block_map), np.stack(block_map_mask)

    num_workers = os.cpu_count()
    if num_workers is None:
        num_workers = 0

    train_schematics_dataloader = DataLoader(
        train_schematics_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    test_schematics_dataloader = DataLoader(
        test_schematics_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    unet_model = VAE(
        channels=len(unique_blocks_dict),
        hidden_channels=32,
    )
    model = VAETrainer(unet_model, unique_blocks_dict)
    csv_logger = CSVLogger(save_dir=path_to_output)
    model_checkpoint = ModelCheckpoint(path_to_output, save_last=True, mode="min")
    trainer = pl.Trainer(
        logger=csv_logger,
        callbacks=[model_checkpoint],
        max_epochs=epochs,
        log_every_n_steps=1,
        accelerator="gpu" if torch.cuda.is_available() else "auto",
    )
    trainer.fit(model, train_schematics_dataloader, test_schematics_dataloader)

    # Save the best and last model locally
    last_model = VAETrainer.load_from_checkpoint(
        model_checkpoint.last_model_path,
        unet_model=unet_model,
        unique_blocks_dict=unique_blocks_dict,
        save_dir=path_to_output,
    )
    torch.save(last_model, os.path.join(path_to_output, "last_model.pth"))
    with open(os.path.join(path_to_output, "unique_blocks_dict.json"), "w") as f:
        json.dump(unique_blocks_dict, f)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--path-to-schematics", type=str, required=True)
    argparser.add_argument("--path-to-output", type=str, required=True)
    argparser.add_argument("--epochs", type=int, required=True)
    argparser.add_argument("--batch-size", type=int, required=True)
    argparser.add_argument("--dataset-limit", type=int)
    argparser.add_argument("--dataset-start", type=int)

    main(argparser)
