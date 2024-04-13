# flake8: noqa: E203
import argparse
import json
import os
from typing import List, Optional, Set, Tuple

import boto3
import numpy as np
import pytorch_lightning as pl
import torch
from loguru import logger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import train_test_split  # type: ignore
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchcfm.models.unet import UNetModel  # type: ignore[import-untyped]

from minecraft_copilot_ml.data_loader import (
    MinecraftSchematicsDataset,
    MinecraftSchematicsDatasetItemType,
    get_working_files_and_unique_blocks_and_counts,
    list_schematic_files_in_folder,
)
from minecraft_copilot_ml.model import LightningUNetModel, UnetModelWithDims


def export_to_onnx(model: LightningUNetModel, path_to_output: str) -> None:
    torch.onnx.export(
        model,
        (
            torch.randn(1).to("cuda" if torch.cuda.is_available() else "cpu"),
            torch.randn(1, 1, 16, 16, 16).to("cuda" if torch.cuda.is_available() else "cpu"),
        ),
        path_to_output,
        input_names=["timestep", "block_map"],
        # https://onnxruntime.ai/docs/reference/compatibility.html
        opset_version=17,
        output_names=["output"],
    )


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
    unique_blocks_dict, _, loaded_schematic_files = get_working_files_and_unique_blocks_and_counts(
        schematics_list_files
    )

    logger.info(f"Unique blocks: {unique_blocks_dict}")
    logger.info(f"Number of unique blocks: {len(unique_blocks_dict)}")
    logger.info(f"Number of loaded schematics files: {len(loaded_schematic_files)}")

    train_schematics_list_files, test_schematics_list_files = train_test_split(
        loaded_schematic_files, test_size=0.2, random_state=42
    )
    train_schematics_dataset = MinecraftSchematicsDataset(train_schematics_list_files)
    val_schematics_dataset = MinecraftSchematicsDataset(test_schematics_list_files)

    def collate_fn(batch: List[MinecraftSchematicsDatasetItemType]) -> MinecraftSchematicsDatasetItemType:
        block_map, noisy_block_map, mask, loss_mask = zip(*batch)
        return np.stack(block_map), np.stack(noisy_block_map), np.stack(mask), np.stack(loss_mask)

    train_schematics_dataloader = DataLoader(
        train_schematics_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_schematics_dataloader = DataLoader(val_schematics_dataset, batch_size=batch_size, collate_fn=collate_fn)

    unet_model = UnetModelWithDims(
        dims=3,
        dim=[len(unique_blocks_dict), 16, 16, 16],
        num_channels=32,
        num_res_blocks=4,
        channel_mult=(1, 2, 3, 4),
    )
    model = LightningUNetModel(unet_model, unique_blocks_dict)
    csv_logger = CSVLogger(save_dir=path_to_output)
    model_checkpoint = ModelCheckpoint(path_to_output, monitor="val_loss", save_top_k=1, save_last=True, mode="min")
    trainer = pl.Trainer(logger=csv_logger, callbacks=model_checkpoint, max_epochs=epochs, log_every_n_steps=1)
    trainer.fit(model, train_schematics_dataloader, val_schematics_dataloader)

    # Save the best and last model locally
    logger.info(f"Best val_loss is: {model_checkpoint.best_model_score}")
    best_model = LightningUNetModel.load_from_checkpoint(
        model_checkpoint.best_model_path, model=unet_model, unique_blocks_dict=unique_blocks_dict
    )
    torch.save(best_model, os.path.join(path_to_output, "best_model.pth"))
    last_model = LightningUNetModel.load_from_checkpoint(
        model_checkpoint.last_model_path, model=unet_model, unique_blocks_dict=unique_blocks_dict
    )
    torch.save(last_model, os.path.join(path_to_output, "last_model.pth"))
    export_to_onnx(best_model, os.path.join(path_to_output, "best_model.onnx"))
    export_to_onnx(last_model, os.path.join(path_to_output, "last_model.onnx"))
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
