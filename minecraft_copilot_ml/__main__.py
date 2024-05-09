# flake8: noqa: E203
import argparse
import json
import os
from typing import List, Optional, Set, Tuple

import boto3
import lightning as pl
import numpy as np
import torch
from improved_diffusion.unet import UNetModel  # type: ignore[import-untyped]
from lightning.pytorch.callbacks import ModelCheckpoint
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

    schematics_dataset = MinecraftSchematicsDataset(loaded_schematic_files)

    def collate_fn(batch: List[MinecraftSchematicsDatasetItemType]) -> MinecraftSchematicsDatasetItemType:
        block_map, block_map_mask = zip(*batch)
        return np.stack(block_map), np.stack(block_map_mask)

    num_workers = os.cpu_count()
    if num_workers is None:
        num_workers = 0

    schematics_dataloader = DataLoader(
        schematics_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    unet_model = UNetModel(
        in_channels=len(unique_blocks_dict),
        model_channels=32,
        out_channels=len(unique_blocks_dict),
        num_res_blocks=2,
        num_heads=2,
        attention_resolutions=[1],
        dropout=0.1,
        channel_mult=(1, 2, 4, 8),
        dims=3,
    )
    model = MinecraftCopilotTrainer(unet_model, unique_blocks_dict, save_dir=path_to_output)
    csv_logger = CSVLogger(save_dir=path_to_output)
    model_checkpoint = ModelCheckpoint(path_to_output, save_last=True, mode="min")
    trainer = pl.Trainer(
        logger=csv_logger,
        callbacks=model_checkpoint,
        max_epochs=epochs,
        log_every_n_steps=1,
        accelerator="gpu" if torch.cuda.is_available() else "auto",
    )
    trainer.fit(model, schematics_dataloader)

    # Save the best and last model locally
    last_model = MinecraftCopilotTrainer.load_from_checkpoint(
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
