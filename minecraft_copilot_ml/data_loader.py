import os

import numpy as np
import requests
import xmltodict
from tqdm import tqdm


def get_list_of_files(path: str) -> list[str]:
    list_files = os.listdir(path)
    concat_files = [os.path.join(path, f) for f in list_files]
    abs_path = [os.path.abspath(f) for f in concat_files]
    return abs_path


def random_block_destroyer(X: np.ndarray) -> np.ndarray:
    return X.copy()


def list_files_of_s3(
    s3_public_link: str = "https://minecraft-schematics-raw.s3.amazonaws.com",
    params: dict[str, str] = {"list-type": "2"},
) -> list[str]:
    r = requests.get("https://minecraft-schematics-raw.s3.amazonaws.com", params=params)
    xml = xmltodict.parse(r.text)
    list_of_files = [x["Key"] for x in xml["ListBucketResult"]["Contents"]]
    if (
        "ListBucketResult" in xml
        and "NextContinuationToken" in xml["ListBucketResult"]
        and xml["ListBucketResult"]["NextContinuationToken"] is not None
    ):
        list_of_files += list_files_of_s3(
            s3_public_link,
            params={"list-type": "2", "continuation-token": xml["ListBucketResult"]["NextContinuationToken"]},
        )
    return list_of_files


lf_s3 = list_files_of_s3()


def get_random_block_map(
    block_map: np.ndarray,
    sliding_window_width: int = 16,
    sliding_window_height: int = 16,
    sliding_window_depth: int = 16,
    rand_x: int | None = None,
    rand_y: int | None = None,
    rand_z: int | None = None,
) -> np.ndarray:
    if len(block_map.shape) != 3:
        raise ValueError("block_map must be a 3d array")
    PADDED_VALUE = 1
    x = np.random.randint(-sliding_window_width + 1, block_map.shape[0]) if rand_x is None else rand_x
    y = np.random.randint(-sliding_window_height + 1, block_map.shape[1]) if rand_y is None else rand_y
    z = np.random.randint(-sliding_window_depth + 1, block_map.shape[2]) if rand_z is None else rand_z
    return (
        block_map.take(range(x + PADDED_VALUE, x + PADDED_VALUE + sliding_window_width), mode="clip", axis=0)
        .take(range(y + PADDED_VALUE, y + PADDED_VALUE + sliding_window_height), mode="clip", axis=1)
        .take(range(z + PADDED_VALUE, z + PADDED_VALUE + sliding_window_depth), mode="clip", axis=2)
    )


def traverse_3d_array(
    block_map: np.ndarray,
    sliding_window_width: int = 16,
    sliding_window_height: int = 16,
    sliding_window_depth: int = 16,
) -> list[np.ndarray]:
    if len(block_map.shape) != 3:
        raise ValueError("block_map must be a 3d array")
    PADDED_VALUE = 1
    block_list = []
    padded = np.pad(block_map, (1, 1), mode="constant", constant_values=-1)
    for x in tqdm(range(-sliding_window_width + 1, block_map.shape[0]), leave=False):
        for y in tqdm(range(-sliding_window_height + 1, block_map.shape[1]), leave=False):
            for z in tqdm(range(-sliding_window_depth + 1, block_map.shape[2]), leave=False):
                new_block = (
                    padded.take(range(x + PADDED_VALUE, x + PADDED_VALUE + sliding_window_width), mode="clip", axis=0)
                    .take(range(y + PADDED_VALUE, y + PADDED_VALUE + sliding_window_height), mode="clip", axis=1)
                    .take(range(z + PADDED_VALUE, z + PADDED_VALUE + sliding_window_depth), mode="clip", axis=2)
                )
                block_list.append(new_block)
    return block_list
