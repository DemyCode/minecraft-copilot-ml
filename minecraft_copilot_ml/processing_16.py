import numpy as np
import os
from tqdm import tqdm
import nbtlib  # type: ignore
from loguru import logger
from pathlib import Path

from minecraft_copilot_ml.minecraft_pre_flattening_id import default_palette
from minecraft_copilot_ml.data_loader import get_random_block_map

import gc

list_of_forbidden_files = [
    "14281.schematic",
    "12188.schematic",
    "8197.schematic",
    "576.schematic",
    "3322.schematic",
    "243.schematic",
    "13197.schematic",
    "15716.schem",
    "11351.schematic",
    "11314.schematic",
    "14846.schem",
    "9171.schematic",
    "13441.schematic",
    "15111.schem",
    "452.schematic",
    "1924.schematic",
]


def convert_nbt_to_numpy_array_with_minecraft_ids(nbt_file: str) -> None:
    gc.collect()
    res = None
    if os.path.exists(f"minecraft-schematics-16/{Path(nbt_file).parts[-1]}.npy"):
        logger.info(f"File {nbt_file} already exists. Skipping.")
        return
    # if os.path.getsize(nbt_file) > 1500000:
    #     logger.error(f"File {nbt_file} is too big. {os.path.getsize(nbt_file)} bytes. Skipping.")
    #     return
    if any([Path(nbt_file).parts[-1] == x for x in list_of_forbidden_files]):
        logger.error(
            f"File {nbt_file} is forbidden. Skipping. If this file is here it is because it generates a SIGKILL."
        )
        return
    try:
        res = nbtlib.load(nbt_file, gzipped=True, byteorder="big")
    except Exception as e:
        logger.error(f"Could not load {nbt_file}: {e}")
        return
    block_data = None
    if "Palette" in res:
        palette = {int(value): key for key, value in res["Palette"].unpack().items()}
    else:
        palette = default_palette
    if "BlockData" in res:
        block_data = res["BlockData"]
    elif "Blocks" in res:
        block_data = res["Blocks"]
    else:
        logger.error(f"Could not find BlockData or Blocks in {nbt_file}")
        return
    if res["Width"] * res["Height"] * res["Length"] != len(block_data):
        logger.error(f"Length of BlockData is not equal to Width * Height * Length in {nbt_file}")
        return
    block_map = np.asarray(block_data).reshape(res["Width"], res["Height"], res["Length"])
    block_map = np.vectorize(palette.get)(block_map)
    block_map = get_random_block_map(
        block_map,
    )
    np.save(
        f"minecraft-schematics-16/{Path(nbt_file).parts[-1]}",
        block_map,
    )


def convert_all_nbts_to_numpy_array_with_minecraft_ids() -> None:
    number_of_files = 0
    for root, _, files in os.walk("/home/mehdi/minecraft-copilot-ml/minecraft-schematics-raw"):
        number_of_files += len(files)
    pbar = tqdm(
        os.walk("/home/mehdi/minecraft-copilot-ml/minecraft-schematics-raw"),
        total=number_of_files,
        smoothing=0,
    )
    for root, _, files in pbar:
        for file in files:
            pbar.set_description(f"Processing {file}")
            pbar.refresh()
            pbar.update(1)
            convert_nbt_to_numpy_array_with_minecraft_ids(os.path.join(root, file))


if __name__ == "__main__":
    if not os.path.exists("minecraft-schematics-16"):
        os.mkdir("minecraft-schematics-16")
    convert_all_nbts_to_numpy_array_with_minecraft_ids()
