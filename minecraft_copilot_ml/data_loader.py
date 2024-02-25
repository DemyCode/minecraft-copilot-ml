# flake8: noqa: E203
import gc
import re
from pathlib import Path
from typing import Dict, List, Tuple

import nbtlib  # type: ignore
import numpy as np
from loguru import logger
from torch.utils.data import Dataset

from minecraft_copilot_ml.minecraft_pre_flattening_id import default_palette

list_of_forbidden_files = [
    "243.schematic",
    "9171.schematic",
    "7519.schematic",
    "5381.schematic",
    "5984.schematic",
    "10189.schematic",
    "592.schematic",
    "14281.schematic",
    "12188.schematic",
    "8197.schematic",
    "576.schematic",
    "3322.schematic",
    "13197.schematic",
    "15716.schem",
    "11351.schematic",
    "11314.schematic",
    "14846.schem",
    "13441.schematic",
    "15111.schem",
    "452.schematic",
    "1924.schematic",
    "785.schematic",
    "4178.schematic",
]


def create_noisy_block_map(
    block_map: np.ndarray,
) -> np.ndarray:
    random_percentage = np.random.random()
    random_indices_from_focused_block_map = np.random.choice(
        np.arange(block_map.size), replace=False, size=int(block_map.size * random_percentage)
    )
    unraveled_indices = np.unravel_index(random_indices_from_focused_block_map, block_map.shape)
    returned_block_map = block_map.copy()
    returned_block_map[unraveled_indices] = "minecraft:air"
    return returned_block_map


def nbt_to_numpy_minecraft_map(
    nbt_file: str,
) -> np.ndarray:
    gc.collect()
    if any([Path(nbt_file).parts[-1] == x for x in list_of_forbidden_files]):
        raise Exception(
            f"File {nbt_file} is forbidden. Skipping. If this file is here it is because it generates a SIGKILL."
        )
    res = None
    try:
        res = nbtlib.load(nbt_file, gzipped=True, byteorder="big")
    except Exception as e:
        logger.warning(f"Could not load {nbt_file}: {e}")
        logger.info("Trying to load it as uncompressed")
        res = nbtlib.load(nbt_file, gzipped=False, byteorder="big")
    if "Palette" in res:
        palette = {int(value): key for key, value in res["Palette"].unpack().items()}
        palette = {key: re.sub(r"\[.*\]", "", value) for key, value in palette.items()}
    else:
        palette = default_palette
    if "BlockData" in res:
        block_data = res["BlockData"]
    else:
        block_data = res["Blocks"]
    block_map = np.asarray(block_data).reshape(res["Height"], res["Length"], res["Width"])
    block_map = np.vectorize(palette.get)(block_map)
    return block_map


def get_random_block_map_and_mask_coordinates(
    minecraft_map: np.ndarray,
    sliding_window_width: int,
    sliding_window_height: int,
    sliding_window_depth: int,
) -> Tuple[np.ndarray, Tuple[int, int, int, int, int, int]]:
    block_map = np.zeros((sliding_window_width, sliding_window_height, sliding_window_depth), dtype=object)
    minimum_width = min(sliding_window_width, minecraft_map.shape[0])
    minimum_height = min(sliding_window_height, minecraft_map.shape[1])
    minimum_depth = min(sliding_window_depth, minecraft_map.shape[2])
    x_start = np.random.randint(0, minecraft_map.shape[0] - minimum_width + 1)
    y_start = np.random.randint(0, minecraft_map.shape[1] - minimum_height + 1)
    z_start = np.random.randint(0, minecraft_map.shape[2] - minimum_depth + 1)
    x_end = x_start + minimum_width
    y_end = y_start + minimum_height
    z_end = z_start + minimum_depth
    random_roll_x_value = np.random.randint(0, sliding_window_width - minimum_width + 1)
    random_y_height_value = np.random.randint(0, sliding_window_height - minimum_height + 1)
    random_roll_z_value = np.random.randint(0, sliding_window_depth - minimum_depth + 1)
    block_map[
        random_roll_x_value : random_roll_x_value + minimum_width,
        random_y_height_value : random_y_height_value + minimum_height,
        random_roll_z_value : random_roll_z_value + minimum_depth,
    ] = minecraft_map[x_start:x_end, y_start:y_end, z_start:z_end]
    return block_map, (
        random_roll_x_value,
        random_y_height_value,
        random_roll_z_value,
        minimum_width,
        minimum_height,
        minimum_depth,
    )


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
