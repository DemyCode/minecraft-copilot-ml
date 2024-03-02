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
from tqdm import tqdm

from minecraft_copilot_ml.data_loader import (
    MinecraftSchematicsDataset,
    get_working_files_and_unique_blocks_and_counts,
    list_schematic_files_in_folder,
    nbt_to_numpy_minecraft_map,
)
from minecraft_copilot_ml.model import UNet3d


def export_to_onnx(model: UNet3d, path_to_output: str) -> None:
    torch.onnx.export(
        model,
        torch.randn(1, 1, 16, 16, 16).to("cuda" if torch.cuda.is_available() else "cpu"),
        path_to_output,
        input_names=["input"],
        output_names=["output"],
    )


def main(argparser: argparse.ArgumentParser) -> None:
    path_to_schematics: str = argparser.parse_args().path_to_schematics
    path_to_output: str = argparser.parse_args().path_to_output
    epochs: int = argparser.parse_args().epochs
    batch_size: int = argparser.parse_args().batch_size
    dataset_limit: Optional[int] = argparser.parse_args().dataset_limit

    if not os.path.exists(path_to_output):
        os.makedirs(path_to_output)

    schematics_list_files = list_schematic_files_in_folder(path_to_schematics)
    schematics_list_files = sorted(schematics_list_files)
    if dataset_limit is not None:
        schematics_list_files = schematics_list_files[:dataset_limit]
    # Set the dictionary size to the number of unique blocks in the dataset.
    # And also select the right files to load.
    unique_blocks_dict, unique_counts_coefficients, loaded_schematic_files = (
        get_working_files_and_unique_blocks_and_counts(schematics_list_files)
    )

    logger.info(f"Unique blocks: {unique_blocks_dict}")
    logger.info(f"Number of unique blocks: {len(unique_blocks_dict)}")
    logger.info(f"Number of loaded schematics files: {len(loaded_schematic_files)}")
    logger.info(f"Unique counts coefficients: {unique_counts_coefficients}")

    train_schematics_list_files, test_schematics_list_files = train_test_split(
        loaded_schematic_files, test_size=0.2, random_state=42
    )
    train_schematics_dataset = MinecraftSchematicsDataset(train_schematics_list_files)
    val_schematics_dataset = MinecraftSchematicsDataset(test_schematics_list_files)

    def collate_fn(batch: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        block_map, noisy_block_map, mask = zip(*batch)
        return np.stack(block_map), np.stack(noisy_block_map), np.stack(mask)

    train_schematics_dataloader = torch.utils.data.DataLoader(
        train_schematics_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_schematics_dataloader = torch.utils.data.DataLoader(
        val_schematics_dataset, batch_size=batch_size, collate_fn=collate_fn
    )

    model = UNet3d(
        unique_blocks_dict,
        unique_counts_coefficients=unique_counts_coefficients,
        train_len_dataloader=len(train_schematics_dataloader),
    )
    csv_logger = CSVLogger(save_dir=path_to_output)
    model_checkpoint = ModelCheckpoint(path_to_output, monitor="val_loss", save_top_k=1, save_last=True, mode="min")
    trainer = pl.Trainer(logger=csv_logger, callbacks=model_checkpoint, max_epochs=epochs, log_every_n_steps=1)
    trainer.fit(model, train_schematics_dataloader, val_schematics_dataloader)

    # Save the best and last model locally
    logger.info(f"Best val_loss is: {model_checkpoint.best_model_score}")
    best_model = UNet3d.load_from_checkpoint(
        model_checkpoint.best_model_path,
        unique_blocks_dict=unique_blocks_dict,
        train_len_dataloader=len(train_schematics_dataloader),
        unique_counts_coefficients=unique_counts_coefficients,
    )
    torch.save(best_model, os.path.join(path_to_output, "best_model.pth"))
    last_model = UNet3d.load_from_checkpoint(
        model_checkpoint.last_model_path,
        unique_blocks_dict=unique_blocks_dict,
        train_len_dataloader=len(train_schematics_dataloader),
        unique_counts_coefficients=unique_counts_coefficients,
    )
    torch.save(last_model, os.path.join(path_to_output, "last_model.pth"))
    export_to_onnx(best_model, os.path.join(path_to_output, "best_model.onnx"))
    export_to_onnx(last_model, os.path.join(path_to_output, "last_model.onnx"))
    with open(os.path.join(path_to_output, "unique_blocks_dict.json"), "w") as f:
        json.dump(unique_blocks_dict, f)

    # Save the best and last model to S3
    s3_client = boto3.client(
        "s3",
        region_name="eu-west-3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )
    s3_client.upload_file(os.path.join(path_to_output, "best_model.pth"), "minecraft-copilot-models", "best_model.pth")
    s3_client.upload_file(os.path.join(path_to_output, "last_model.pth"), "minecraft-copilot-models", "last_model.pth")
    s3_client.upload_file(
        os.path.join(path_to_output, "best_model.onnx"), "minecraft-copilot-models", "best_model.onnx"
    )
    s3_client.upload_file(
        os.path.join(path_to_output, "last_model.onnx"), "minecraft-copilot-models", "last_model.onnx"
    )
    s3_client.upload_file(
        os.path.join(path_to_output, "unique_blocks_dict.json"), "minecraft-copilot-models", "unique_blocks_dict.json"
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--path-to-schematics", type=str, required=True)
    argparser.add_argument("--path-to-output", type=str, required=True)
    argparser.add_argument("--epochs", type=int, required=True)
    argparser.add_argument("--batch-size", type=int, required=True)
    argparser.add_argument("--dataset-limit", type=int)

    main(argparser)
