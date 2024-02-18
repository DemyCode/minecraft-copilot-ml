# flake8: noqa: E203
import gc
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import litemapy  # type: ignore
import nbtlib  # type: ignore
import numpy as np
from loguru import logger
from tqdm import tqdm

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
    "19231.schematic",
    "13942.schematic",
    "4766.schematic",
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


def litematic_to_numpy_minecraft_map(
    litematic_file: str,
) -> np.ndarray:
    nbt_loaded = litemapy.Schematic.load(litematic_file)
    regions = nbt_loaded.regions
    first_region = regions[list(regions.keys())[0]]
    reg = first_region
    # Print out the basic shape
    numpy_map = np.zeros((len(reg.xrange()), len(reg.yrange()), len(reg.zrange())), dtype=object)
    for x, i in zip(reg.xrange(), range(len(reg.xrange()))):
        for y, j in zip(reg.yrange(), range(len(reg.yrange()))):
            for z, k in zip(reg.zrange(), range(len(reg.zrange()))):
                b = reg.getblock(x, y, z)
                numpy_map[i, j, k] = b.blockid
    return numpy_map


def schematic_to_numpy_minecraft_map(
    nbt_file: str,
    gzipped: bool = True,
) -> np.ndarray:
    res = nbtlib.load(nbt_file, gzipped=gzipped, byteorder="big")
    if "Palette" in res:
        palette = {int(value): key for key, value in res["Palette"].unpack().items()}
        palette = {key: re.sub(r"\[.*\]", "", value) for key, value in palette.items()}
    else:
        palette = default_palette
    if "BlockData" in res:
        block_data = res["BlockData"]
    elif "Blocks" in res:
        block_data = res["Blocks"]
    else:
        raise Exception(f"Could not find Blocks or BlockData in {nbt_file}. Known keys: {res.keys()}")
    block_map = np.asarray(block_data).reshape(res["Height"], res["Length"], res["Width"])
    block_map = np.vectorize(palette.get)(block_map)
    return block_map


def nbt_to_numpy_minecraft_map(
    nbt_file: str,
) -> np.ndarray:
    gc.collect()
    if any([Path(nbt_file).parts[-1] == x for x in list_of_forbidden_files]):
        raise Exception(
            f"File {nbt_file} is forbidden. Skipping. If this file is here it is because it generates a SIGKILL."
        )
    from functools import partial

    function_to_file = [
        litematic_to_numpy_minecraft_map,
        partial(schematic_to_numpy_minecraft_map, gzipped=True),
        partial(schematic_to_numpy_minecraft_map, gzipped=False),
    ]
    if nbt_file.endswith(".litematic"):
        function_to_file = [
            litematic_to_numpy_minecraft_map,
            partial(schematic_to_numpy_minecraft_map, gzipped=True),
            partial(schematic_to_numpy_minecraft_map, gzipped=False),
        ]
    if nbt_file.endswith(".schematic") or nbt_file.endswith(".schem"):
        function_to_file = [
            partial(schematic_to_numpy_minecraft_map, gzipped=True),
            partial(schematic_to_numpy_minecraft_map, gzipped=False),
            litematic_to_numpy_minecraft_map,
        ]
    final_exception = None
    for function in function_to_file:
        try:
            return function(nbt_file)
        except Exception as e:
            logger.warning(f"Could not load {nbt_file} with {function}")
            final_exception = e
    logger.exception(final_exception)


def get_random_block_map_and_mask_coordinates(
    minecraft_map: np.ndarray,
    sliding_window_width: int,
    sliding_window_height: int,
    sliding_window_depth: int,
) -> Tuple[np.ndarray, Tuple[int, int, int, int, int, int]]:
    block_map = np.full(
        (sliding_window_width, sliding_window_height, sliding_window_depth), "minecraft:air", dtype=object
    )
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


def list_schematic_files_in_folder(path_to_schematics: str) -> list[str]:
    schematics_list_files = []
    tqdm_os_walk = tqdm(os.walk(path_to_schematics), smoothing=0)
    for dirpath, _, filenames in tqdm_os_walk:
        for filename in filenames:
            tqdm_os_walk.set_description(desc=f"Found {filename}")
            schematics_list_files.append(os.path.join(dirpath, filename))
    logger.info(f"Found {len(schematics_list_files)} schematics files.")
    return schematics_list_files


def get_working_files_and_unique_blocks_and_counts(schematics_list_files: list[str]) -> None:
    unique_blocks: Set[str] = set()
    unique_counts: Dict[str, int] = {}
    loaded_schematic_files: List[str] = []
    tqdm_list_files = tqdm(schematics_list_files, smoothing=0)
    for nbt_file in tqdm_list_files:
        tqdm_list_files.set_description(f"Processing {nbt_file}")
        try:
            numpy_minecraft_map = nbt_to_numpy_minecraft_map(nbt_file)
            unique_blocks_in_map, unique_counts_in_map = np.unique(numpy_minecraft_map, return_counts=True)
            for block, count in zip(unique_blocks_in_map, unique_counts_in_map):
                if block not in unique_counts:
                    unique_counts[block] = 0
                unique_counts[block] += count
            for block in unique_blocks_in_map:
                if block not in unique_blocks:
                    logger.info(f"Found new block: {block}")
            unique_blocks = unique_blocks.union(unique_blocks_in_map)
            loaded_schematic_files.append(nbt_file)
        except Exception as e:
            logger.error(f"Could not load {nbt_file}")
            logger.exception(e)
            continue
    unique_blocks_dict = {block: idx for idx, block in enumerate(unique_blocks)}
    unique_counts_coefficients = np.array([unique_counts[block] for block in unique_blocks_dict])
    unique_counts_coefficients = unique_counts_coefficients - unique_counts_coefficients.min()
    unique_counts_coefficients = unique_counts_coefficients / unique_counts_coefficients.max()
    unique_counts_coefficients = unique_counts_coefficients + 1
    return unique_blocks_dict, unique_counts_coefficients, loaded_schematic_files
