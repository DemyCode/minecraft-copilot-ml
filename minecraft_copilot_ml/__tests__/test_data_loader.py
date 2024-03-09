# flake8: noqa: E203

import numpy as np

from minecraft_copilot_ml.data_loader import create_noisy_block_map
from minecraft_copilot_ml.minecraft_pre_flattening_id import default_palette


def test_create_noisy_block_map() -> None:
    block_map = np.array(
        [
            [
                ["minecraft:stone", "minecraft:stone", "minecraft:stone"],
                ["minecraft:stone", "minecraft:stone", "minecraft:stone"],
                ["minecraft:stone", "minecraft:stone", "minecraft:stone"],
            ],
            [
                ["minecraft:stone", "minecraft:stone", "minecraft:stone"],
                ["minecraft:stone", "minecraft:stone", "minecraft:stone"],
                ["minecraft:stone", "minecraft:stone", "minecraft:stone"],
            ],
            [
                ["minecraft:stone", "minecraft:stone", "minecraft:stone"],
                ["minecraft:stone", "minecraft:stone", "minecraft:stone"],
                ["minecraft:stone", "minecraft:stone", "minecraft:stone"],
            ],
        ],
        dtype=object,
    )
    noisy_block_map, (
        x_coordinates,
        y_coordinates,
        z_coordinates,
        x_width,
        y_width,
        z_width,
    ) = create_noisy_block_map(block_map)
    assert noisy_block_map.shape == block_map.shape
    assert noisy_block_map.dtype == block_map.dtype
    assert (noisy_block_map != block_map).any()


def test_litematic_to_numpy_minecraft_map() -> None:
    pass


def test_schematic_to_numpy_minecraft_map() -> None:
    pass


def nbt_to_numpy_minecraft_map() -> None:
    pass


def get_random_block_map_and_mask_coordinates() -> None:
    pass


def test_list_schematic_files_in_folder() -> None:
    pass


def test_get_working_files_and_unique_blocks_and_counts() -> None:
    pass
