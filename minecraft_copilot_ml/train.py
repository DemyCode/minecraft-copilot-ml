# flake8: noqa: E203

import argparse
import os
from typing import Dict, List, Set, Tuple

import torch
from loguru import logger
from torch.utils.data import Dataset
from tqdm import tqdm

from minecraft_copilot_ml.data_loader import (
    create_noisy_block_map,
    get_random_block_map_and_mask_coordinates,
    nbt_to_numpy_minecraft_map,
)

# from minecraft_copilot_ml.minecraft_pre_flattening_id import default_palette
# from minecraft_copilot_ml.model import UNet3D


class MinecraftSchematicsDataset(Dataset):
    def __init__(
        self,
        schematics_list_files: List[str],
        unique_blocks_dict: Dict[str, int],
    ) -> None:
        self.schematics_list_files = schematics_list_files
        self.unique_blocks_dict = unique_blocks_dict

    def __len__(self) -> int:
        return len(self.schematics_list_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
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
        return torch.from_numpy(block_map).long(), torch.from_numpy(noisy_block_map).long()


def main(argparser: argparse.ArgumentParser) -> None:
    argparser.parse_args()
    path_to_schematics = argparser.parse_args().path_to_schematics
    path_to_output = argparser.parse_args().path_to_output
    # epochs = argparser.parse_args().epochs
    # batch_size = argparser.parse_args().batch_size
    # learning_rate = argparser.parse_args().learning_rate

    if not os.path.exists(path_to_output):
        os.makedirs(path_to_output)

    schematics_list_files = []
    for dirpath, _, filenames in os.walk(path_to_schematics):
        schematics_list_files.extend([os.path.join(dirpath, filename) for filename in filenames])
    logger.info(f"Found {len(schematics_list_files)} schematics files.")

    # Set the dictionary size to the number of unique blocks in the dataset.
    unique_blocks: Set[str] = set()
    for nbt_file in tqdm(schematics_list_files):
        numpy_minecraft_map = nbt_to_numpy_minecraft_map(nbt_file)
        uinque_blocks_in_map = set(numpy_minecraft_map.flatten())
        unique_blocks = unique_blocks.union(uinque_blocks_in_map)
    unique_blocks_dict = {block: idx for idx, block in enumerate(unique_blocks)}
    logger.info(f"Unique blocks: {unique_blocks_dict}")

    schematics_dataset = MinecraftSchematicsDataset(schematics_list_files, unique_blocks_dict)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--path-to-schematics", type=str, default="schematics_data")
    argparser.add_argument("--path-to-output", type=str, default="output")
    argparser.add_argument("--epochs", type=int, default=100)
    argparser.add_argument("--batch-size", type=int, default=32)
    argparser.add_argument("--learning-rate", type=float, default=0.001)

    main(argparser)
