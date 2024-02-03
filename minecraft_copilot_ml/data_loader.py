# flake8: noqa: E203
import gc
import re
from pathlib import Path
from typing import Tuple

import nbtlib  # type: ignore
import numpy as np
from loguru import logger

from minecraft_copilot_ml.minecraft_pre_flattening_id import default_palette

list_of_forbidden_files = [
    "243.schematic",
    "9171.schematic",
    "7519.schematic",
    "5381.schematic",
    "5984.schematic",
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
