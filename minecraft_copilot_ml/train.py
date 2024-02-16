# flake8: noqa: E203
import argparse
import json
import os
from typing import Dict, List, Set, Tuple

import boto3
import numpy as np
import pytorch_lightning as pl
import torch
from loguru import logger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import train_test_split  # type: ignore
from torch.utils.data import Dataset

from minecraft_copilot_ml.data_loader import (
    create_noisy_block_map,
    get_random_block_map_and_mask_coordinates,
    get_working_files_and_unique_blocks_and_counts,
    list_schematic_files_in_folder,
    nbt_to_numpy_minecraft_map,
)
from minecraft_copilot_ml.model import VAE


class MinecraftSchematicsDataset(Dataset):
    def __init__(
        self,
        schematics_list_files: List[str],
    ) -> None:
        self.schematics_list_files = schematics_list_files

    def __len__(self) -> int:
        return len(self.schematics_list_files)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        nbt_file = self.schematics_list_files[idx]
        numpy_minecraft_map = nbt_to_numpy_minecraft_map(nbt_file)
        block_map, (
            random_roll_x_value,
            random_y_height_value,
            random_roll_z_value,
            minimum_width,
            minimum_height,
            minimum_depth,
        ) = get_random_block_map_and_mask_coordinates(numpy_minecraft_map, 16, 16, 16)
        focused_block_map = block_map[
            random_roll_x_value : random_roll_x_value + minimum_width,
            random_y_height_value : random_y_height_value + minimum_height,
            random_roll_z_value : random_roll_z_value + minimum_depth,
        ]
        noisy_focused_block_map = create_noisy_block_map(focused_block_map)
        noisy_block_map = block_map.copy()
        noisy_block_map[
            random_roll_x_value : random_roll_x_value + minimum_width,
            random_y_height_value : random_y_height_value + minimum_height,
            random_roll_z_value : random_roll_z_value + minimum_depth,
        ] = noisy_focused_block_map
        mask = np.zeros((16, 16, 16), dtype=bool)
        mask[
            random_roll_x_value : random_roll_x_value + minimum_width,
            random_y_height_value : random_y_height_value + minimum_height,
            random_roll_z_value : random_roll_z_value + minimum_depth,
        ] = True
        return block_map, noisy_block_map, mask


def export_to_onnx(model: VAE, path_to_output: str) -> None:
    torch.onnx.export(
        model,
        torch.randn(1, len(model.unique_blocks_dict), 16, 16, 16).to("cuda" if torch.cuda.is_available() else "cpu"),
        path_to_output,
        input_names=["input"],
        output_names=["output"],
    )


def main(argparser: argparse.ArgumentParser) -> None:
    path_to_schematics: str = argparser.parse_args().path_to_schematics
    path_to_output: str = argparser.parse_args().path_to_output
    epochs: int = argparser.parse_args().epochs
    batch_size: int = argparser.parse_args().batch_size

    if not os.path.exists(path_to_output):
        os.makedirs(path_to_output)

    schematics_list_files = list_schematic_files_in_folder(path_to_schematics)
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

    num_workers = 1
    cpu_count = os.cpu_count()
    if cpu_count is not None:
        num_workers = cpu_count - 1

    def collate_fn(batch: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        block_map, noisy_block_map, mask = zip(*batch)
        return np.stack(block_map), np.stack(noisy_block_map), np.stack(mask)

    train_schematics_dataloader = torch.utils.data.DataLoader(
        train_schematics_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn
    )
    val_schematics_dataloader = torch.utils.data.DataLoader(
        val_schematics_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn
    )

    model = VAE(unique_blocks_dict, unique_counts_coefficients=unique_counts_coefficients)
    csv_logger = CSVLogger(save_dir=path_to_output)
    model_checkpoint = ModelCheckpoint(path_to_output, monitor="val_loss", save_top_k=1, save_last=True, mode="min")
    trainer = pl.Trainer(logger=csv_logger, callbacks=model_checkpoint, max_epochs=epochs, log_every_n_steps=1)
    trainer.fit(model, train_schematics_dataloader, val_schematics_dataloader)

    # Save the best and last model locally
    logger.info(f"Best val_loss is: {model_checkpoint.best_model_score}")
    best_model = VAE.load_from_checkpoint(
        model_checkpoint.best_model_path,
        unique_blocks_dict=unique_blocks_dict,
        unique_counts_coefficients=unique_counts_coefficients,
    )
    torch.save(best_model, os.path.join(path_to_output, "best_model.pth"))
    last_model = VAE.load_from_checkpoint(
        model_checkpoint.last_model_path,
        unique_blocks_dict=unique_blocks_dict,
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

    main(argparser)
