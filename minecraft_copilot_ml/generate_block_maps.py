import os
from typing import Optional
import numpy as np
from tqdm import tqdm

from minecraft_copilot_ml.data_loader import (
    get_working_files_and_unique_blocks_and_counts,
    list_files_in_folder,
    MinecraftSchematicsDataset,
)

if __name__ == "__main__":
    dataset_start: Optional[int] = 0
    dataset_limit: Optional[int] = 64
    if not os.path.exists("/home/mehdi/minecraft-copilot-ml/cut_datasets/minecraft-schematics/block_maps"):
        os.makedirs("/home/mehdi/minecraft-copilot-ml/cut_datasets/minecraft-schematics/block_maps")
    if not os.path.exists("/home/mehdi/minecraft-copilot-ml/cut_datasets/minecraft-schematics/block_map_masks"):
        os.makedirs("/home/mehdi/minecraft-copilot-ml/cut_datasets/minecraft-schematics/block_map_masks")
    schematics_list_files = list_files_in_folder(
        "/home/mehdi/minecraft-copilot-ml/datasets/minecraft-schematics"
    )
    schematics_list_files = sorted(schematics_list_files)

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
    minecraft_schematic_dataset = MinecraftSchematicsDataset(loaded_schematic_files)
    tqdm_minecraft_schematic_dataset = tqdm(minecraft_schematic_dataset, smoothing=0)
    for idx, (block_map, _, block_map_mask, _) in enumerate(tqdm_minecraft_schematic_dataset):
        tqdm_minecraft_schematic_dataset.set_description(f"Processing block map {tqdm_minecraft_schematic_dataset.n}")
        np.save(f"/home/mehdi/minecraft-copilot-ml/cut_datasets/minecraft-schematics/block_maps/{idx}.npy", block_map)
        np.save(
            f"/home/mehdi/minecraft-copilot-ml/cut_datasets/minecraft-schematics/block_map_masks/{idx}.npy",
            block_map_mask,
        )
