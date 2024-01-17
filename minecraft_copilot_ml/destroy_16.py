import argparse
import os
from pathlib import Path
import numpy as np
from loguru import logger
from tqdm import tqdm


def destroy_percentage_16_block(npy_file: str, percentage: int) -> np.ndarray | None:
    if os.path.exists(f"X/destroyed_{percentage}_{Path(npy_file).parts[-1]}") and os.path.exists(
        f"Y/destroyed_{percentage}_{Path(npy_file).parts[-1]}"
    ):
        logger.info(f"{npy_file} already exists, skipping")
        return None
    block_map_file_16: np.ndarray = np.load(npy_file, allow_pickle=True)
    flat_indices = np.random.choice(
        np.arange(block_map_file_16.size), replace=False, size=int(block_map_file_16.size * (percentage / 100))
    )
    unraveled_indices = np.unravel_index(flat_indices, block_map_file_16.shape)
    returned_block_map = block_map_file_16.copy()
    returned_block_map[unraveled_indices] = None
    return returned_block_map


def destroy_all_percentage_16_block(percentage: int) -> None:
    number_of_files = 0
    for root, _, files in os.walk("/home/mehdi/minecraft-copilot-ml/minecraft-schematics-16"):
        number_of_files += len(files)
    pbar = tqdm(
        os.walk("/home/mehdi/minecraft-copilot-ml/minecraft-schematics-16"),
        total=number_of_files,
        smoothing=0,
    )
    for root, _, files in pbar:
        for file in files:
            pbar.set_description(f"Processing {file}")
            pbar.refresh()
            pbar.update(1)
            block_map_destroyed = destroy_percentage_16_block(os.path.join(root, file), percentage)
            if block_map_destroyed is not None:
                np.save(f"X/destroyed_{percentage}_{Path(file).parts[-1]}", block_map_destroyed)
                np.save(
                    f"y/destroyed_{percentage}_{Path(file).parts[-1]}",
                    np.load(os.path.join(root, file), allow_pickle=True),
                )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--percentage", type=int, default=25)
    args = argparser.parse_args()
    if args.percentage < 0 or args.percentage > 100:
        raise ValueError("percentage must be between 0 and 100")
    destroy_all_percentage_16_block(args.percentage)
