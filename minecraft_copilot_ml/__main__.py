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
import json

from minecraft_copilot_ml.data_loader import (
    MinecraftSchematicsDataset,
    MinecraftBlockMapDataset,
    MinecraftSchematicsDatasetItemType,
    get_working_files_and_unique_blocks_and_counts,
    get_unique_blocks_from_block_maps,
    list_files_in_folder,
)
from minecraft_copilot_ml.model import LightningUNetModel


def export_to_onnx(model: LightningUNetModel, channel_n: int, path_to_output: str) -> None:
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    torch.onnx.export(
        model,
        (
            torch.randn(1).to("cuda" if torch.cuda.is_available() else "cpu"),
            torch.randn(1, channel_n, 16, 16, 16).to("cuda" if torch.cuda.is_available() else "cpu"),
        ),
        path_to_output,
        input_names=["timestep", "block_map"],
        # https://onnxruntime.ai/docs/reference/compatibility.html
        opset_version=17,
        output_names=["output"],
    )


def main(argparser: argparse.ArgumentParser) -> None:
    path_to_block_maps: str = "/home/mehdi/minecraft-copilot-ml/cut_datasets/minecraft-schematics/block_maps"
    path_to_block_map_masks: str = "/home/mehdi/minecraft-copilot-ml/cut_datasets/minecraft-schematics/block_map_masks"
    path_to_output: str = argparser.parse_args().path_to_output
    epochs: int = argparser.parse_args().epochs
    batch_size: int = argparser.parse_args().batch_size
    dataset_limit: Optional[int] = argparser.parse_args().dataset_limit
    dataset_start: Optional[int] = argparser.parse_args().dataset_start

    if not os.path.exists(path_to_output):
        os.makedirs(path_to_output)

    block_map_list_files = list_files_in_folder(path_to_block_maps)
    block_map_list_files = sorted(block_map_list_files)
    block_map_mask_list_files = list_files_in_folder(path_to_block_map_masks)
    block_map_mask_list_files = sorted(block_map_mask_list_files)

    start = 0
    end = len(block_map_list_files)
    if dataset_start is not None:
        start = dataset_start
    if dataset_limit is not None:
        end = dataset_limit
    block_map_list_files = block_map_list_files[start:end]
    block_map_mask_list_files = block_map_mask_list_files[start:end]

    unique_blocks_dict = json.load(
        open("/home/mehdi/minecraft-copilot-ml/cut_datasets/minecraft-schematics/unique_blocks.json", "r")
    )
    # (
    #     train_block_map_list_files,
    #     # test_block_map_list_files,
    #     train_block_map_mask_list_files,
    #     # test_block_map_mask_list_files,
    # ) = train_test_split(block_map_list_files, block_map_mask_list_files, test_size=0.2, random_state=42)
    train_block_map_dataset = MinecraftBlockMapDataset(block_map_list_files, block_map_mask_list_files)
    # val_block_map_dataset = MinecraftBlockMapDataset(test_block_map_list_files, test_block_map_mask_list_files)

    def collate_fn(batch: List[MinecraftSchematicsDatasetItemType]) -> MinecraftSchematicsDatasetItemType:
        block_map, noisy_block_map, mask, loss_mask = zip(*batch)
        return np.stack(block_map), np.stack(noisy_block_map), np.stack(mask), np.stack(loss_mask)

    train_schematics_dataloader = DataLoader(
        train_block_map_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        # num_workers=os.cpu_count() - 1,
        pin_memory=True,
        sampler=None
    )
    # val_schematics_dataloader = DataLoader(val_block_map_dataset, batch_size=1, collate_fn=collate_fn)

    unet_model = UNetModel(
        dims=3,
        dim=[len(unique_blocks_dict), 16, 16, 16],
        num_res_blocks=2,
        num_channels=32,
        channel_mult=(1, 2, 2, 2),
        dropout=0.1,
        num_heads=4,
        resblock_updown=True,
        updown=False,
    )
    model = LightningUNetModel(unet_model, unique_blocks_dict)
    csv_logger = CSVLogger(save_dir=path_to_output)
    model_checkpoint = ModelCheckpoint(path_to_output, monitor="train_loss", save_top_k=1, save_last=True, mode="min")
    trainer = pl.Trainer(
        logger=csv_logger,
        callbacks=model_checkpoint,
        max_epochs=epochs,
        log_every_n_steps=1,
        accelerator="gpu",
        devices=1,
    )
    trainer.fit(
        model,
        train_schematics_dataloader,
        # val_schematics_dataloader,
    )

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
    export_to_onnx(best_model, len(unique_blocks_dict), os.path.join(path_to_output, "best_model.onnx"))
    export_to_onnx(last_model, len(unique_blocks_dict), os.path.join(path_to_output, "last_model.onnx"))
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
