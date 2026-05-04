import os
import pickle
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from schematic_loader import load_schematic


def _find_schematic_files(dirs):
    files = []
    for d in dirs:
        for root, _, names in os.walk(d):
            for n in names:
                if n.endswith(".schematic") or n.endswith(".schem"):
                    files.append(os.path.join(root, n))
    return sorted(files)


def _convert_to_indexed(blocks: np.ndarray, block_to_idx: dict) -> np.ndarray:
    unique_names, inverse = np.unique(blocks, return_inverse=True)
    global_indices = np.array(
        [block_to_idx.get(str(n), block_to_idx.get("minecraft:air", 0)) for n in unique_names],
        dtype=np.int16,
    )
    return global_indices[inverse].reshape(blocks.shape)


def _build_cache(files: list, chunk_size: int):
    block_counts: dict = defaultdict(int)
    raw_schematics = []

    for f in tqdm(files, desc="Scanning schematics"):
        try:
            blocks = load_schematic(f)
            for name in np.unique(blocks):
                block_counts[str(name)] += 1
            raw_schematics.append(blocks)
        except Exception as e:
            tqdm.write(f"Skip {f}: {e}")

    sorted_blocks = sorted(block_counts.items(), key=lambda x: -x[1])

    block_to_idx: dict = {}
    idx_to_block: dict = {}

    block_to_idx["minecraft:air"] = 0
    idx_to_block[0] = "minecraft:air"

    next_idx = 1
    for name, _ in sorted_blocks:
        if name not in block_to_idx:
            block_to_idx[name] = next_idx
            idx_to_block[next_idx] = name
            next_idx += 1

    indexed_schematics = []
    chunk_index = []

    cs = chunk_size
    stride = cs // 4

    for s_idx, blocks in enumerate(tqdm(raw_schematics, desc="Indexing schematics")):
        indexed = _convert_to_indexed(blocks, block_to_idx)
        indexed_schematics.append(indexed)

        h, l, w = indexed.shape
        ny = max(1, (h - cs) // stride + 1) if h >= cs else 1
        nz = max(1, (l - cs) // stride + 1) if l >= cs else 1
        nx = max(1, (w - cs) // stride + 1) if w >= cs else 1
        count = ny * nz * nx
        chunk_index.extend([s_idx] * count)

    vocab = {"block_to_idx": block_to_idx, "idx_to_block": idx_to_block}
    return vocab, indexed_schematics, chunk_index


def _sample_condition_mask(cs: int) -> np.ndarray:
    strategy = random.choices(
        ["random", "bottom_half", "top_half", "shell", "octant", "half_axis", "thin_strip"],
        weights=[0.35, 0.20, 0.08, 0.12, 0.12, 0.08, 0.05],
    )[0]

    mask = np.zeros((cs, cs, cs), dtype=bool)

    if strategy == "random":
        frac = random.uniform(0.15, 0.55)
        mask = np.random.rand(cs, cs, cs) < frac

    elif strategy == "bottom_half":
        cut = random.randint(cs // 4, cs * 3 // 4)
        mask[:cut] = True

    elif strategy == "top_half":
        cut = random.randint(cs // 4, cs * 3 // 4)
        mask[cut:] = True

    elif strategy == "shell":
        t = random.randint(1, max(1, cs // 8))
        mask[:t] = True
        mask[-t:] = True
        mask[:, :t] = True
        mask[:, -t:] = True
        mask[:, :, :t] = True
        mask[:, :, -t:] = True

    elif strategy == "octant":
        h = cs // 2
        y0 = random.choice([0, h])
        z0 = random.choice([0, h])
        x0 = random.choice([0, h])
        mask[y0 : y0 + h, z0 : z0 + h, x0 : x0 + h] = True

    elif strategy == "half_axis":
        axis = random.randint(0, 2)
        h = random.randint(cs // 4, cs * 3 // 4)
        slc = [slice(None)] * 3
        slc[axis] = slice(0, h) if random.random() < 0.5 else slice(cs - h, cs)
        mask[tuple(slc)] = True

    elif strategy == "thin_strip":
        axis = random.randint(0, 2)
        t = random.randint(2, max(2, cs // 6))
        start = random.randint(0, cs - t)
        slc = [slice(None)] * 3
        slc[axis] = slice(start, start + t)
        mask[tuple(slc)] = True

    return mask


class MinecraftDataset(Dataset):
    def __init__(
        self,
        data_dirs: list,
        chunk_size: int = 32,
        min_fill: float = 0.10,
        cache_dir: str = "cache",
    ):
        self.chunk_size = chunk_size
        self.min_fill = min_fill

        os.makedirs(cache_dir, exist_ok=True)
        vocab_path = os.path.join(cache_dir, "vocab.pkl")
        data_path = os.path.join(cache_dir, "schematics.pkl")
        index_path = os.path.join(cache_dir, "chunk_index.pkl")

        if all(os.path.exists(p) for p in [vocab_path, data_path, index_path]):
            print("Loading dataset cache...")
            with open(vocab_path, "rb") as f:
                vocab = pickle.load(f)
            with open(data_path, "rb") as f:
                self._schematics = pickle.load(f)
            with open(index_path, "rb") as f:
                self._chunk_index = pickle.load(f)
        else:
            print("Building dataset cache (first run only)...")
            files = _find_schematic_files(data_dirs)
            if not files:
                raise RuntimeError(f"No schematic files found in {data_dirs}")
            print(f"Found {len(files)} schematic files")

            vocab, self._schematics, self._chunk_index = _build_cache(files, chunk_size)

            with open(vocab_path, "wb") as f:
                pickle.dump(vocab, f)
            with open(data_path, "wb") as f:
                pickle.dump(self._schematics, f)
            with open(index_path, "wb") as f:
                pickle.dump(self._chunk_index, f)

        self.block_to_idx: dict = vocab["block_to_idx"]
        self.idx_to_block: dict = vocab["idx_to_block"]
        self.vocab_size: int = len(self.block_to_idx)
        self.mask_idx: int = self.vocab_size

        print(f"Dataset: {len(self._chunk_index)} chunks, vocab size {self.vocab_size}")

    def __len__(self) -> int:
        return len(self._chunk_index)

    def __getitem__(self, idx: int) -> dict:
        s_idx = self._chunk_index[idx]
        blocks = self._schematics[s_idx]
        cs = self.chunk_size

        for _ in range(10):
            chunk = self._random_crop(blocks, cs)
            if (chunk != 0).mean() >= self.min_fill:
                break

        k = random.randint(0, 3)
        if k > 0:
            chunk = np.rot90(chunk, k, axes=(1, 2)).copy()

        if random.random() < 0.5:
            chunk = chunk[:, :, ::-1].copy()

        condition_mask = _sample_condition_mask(cs)

        return {
            "blocks": torch.from_numpy(chunk.astype(np.int64)),
            "condition_mask": torch.from_numpy(condition_mask),
        }

    def _random_crop(self, blocks: np.ndarray, cs: int) -> np.ndarray:
        h, l, w = blocks.shape
        chunk = np.zeros((cs, cs, cs), dtype=blocks.dtype)

        def slices(dim_size):
            if dim_size >= cs:
                start = random.randint(0, dim_size - cs)
                return slice(start, start + cs), slice(0, cs)
            else:
                off = random.randint(0, cs - dim_size)
                return slice(0, dim_size), slice(off, off + dim_size)

        y_src, y_dst = slices(h)
        z_src, z_dst = slices(l)
        x_src, x_dst = slices(w)
        chunk[y_dst, z_dst, x_dst] = blocks[y_src, z_src, x_src]
        return chunk
