import json
import os
import random
import zlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from writcache import writcache, CacheConfig

from schematic_loader import load_any

_EXTS = (".schematic", ".schem", ".litematic")


def _find_schematic_files(dirs):
    files = []
    for d in dirs:
        for root, _, names in os.walk(d):
            for n in names:
                if n.lower().endswith(_EXTS):
                    files.append(Path(os.path.join(root, n)))
    return sorted(files)


def _file_key(f: Path) -> str:
    crc = zlib.crc32(str(f).encode()) & 0xFFFFFFFF
    return f"{f.stem}_{crc:08x}"


def _convert_to_indexed(blocks: np.ndarray, block_to_idx: dict) -> np.ndarray:
    unique_names, inverse = np.unique(blocks, return_inverse=True)
    global_indices = np.array(
        [block_to_idx.get(str(n), 0) for n in unique_names],
        dtype=np.int16,
    )
    return global_indices[inverse].reshape(blocks.shape)


def _build_dataset(files: list[Path], min_fill: float, npy_dir_str: str):
    """
    Per-file cache: each schematic is saved as an indexed .npy file keyed by path.
    A sidecar .json stores the block names so vocab can be reconstructed on any run
    without re-reading original files.
    npy_dir is passed as a string so writcache hashes it as a stable key (not by contents).
    """
    npy_dir = Path(npy_dir_str)
    npy_dir.mkdir(parents=True, exist_ok=True)

    block_to_idx: dict = {"minecraft:air": 0}
    idx_to_block: dict = {0: "minecraft:air"}
    result_paths: list[str] = []

    for f in tqdm(files, desc="Building dataset", smoothing=0):
        key = _file_key(f)
        npy_path = npy_dir / f"{key}.npy"
        meta_path = npy_dir / f"{key}.json"

        if npy_path.exists() and meta_path.exists():
            with open(meta_path) as mf:
                block_names: list[str] = json.load(mf)
            for name in block_names:
                if name not in block_to_idx:
                    idx = len(block_to_idx)
                    block_to_idx[name] = idx
                    idx_to_block[idx] = name
            result_paths.append(str(npy_path))
            continue

        try:
            blocks = load_any(str(f))
            if (blocks != "minecraft:air").mean() < min_fill:
                continue
            block_names = [str(n) for n in np.unique(blocks)]
            for name in block_names:
                if name not in block_to_idx:
                    idx = len(block_to_idx)
                    block_to_idx[name] = idx
                    idx_to_block[idx] = name
            indexed = _convert_to_indexed(blocks, block_to_idx)
            np.save(npy_path, indexed)
            with open(meta_path, "w") as mf:
                json.dump(block_names, mf)
            result_paths.append(str(npy_path))
        except Exception as e:
            tqdm.write(f"Skip {f}: {e}")

    return {"block_to_idx": block_to_idx, "idx_to_block": idx_to_block}, result_paths


def _sample_condition_mask(cs: int) -> np.ndarray:
    strategy = random.choices(
        [
            "random",
            "bottom_half",
            "top_half",
            "shell",
            "octant",
            "half_axis",
            "thin_strip",
        ],
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
        min_fill: float = 0.02,
        writcache_dir: str = "writcache",
        npy_dir: str = "npy",
        max_files: int | None = None,
    ):
        self.chunk_size = chunk_size

        files = _find_schematic_files(data_dirs)
        if max_files:
            files = files[:max_files]
        if not files:
            raise RuntimeError(f"No schematic files found in {data_dirs}")
        print(f"Found {len(files)} schematic files")

        cfg = CacheConfig(cache_dir=Path(writcache_dir))
        vocab, self._schematic_paths = writcache(_build_dataset, config=cfg)(
            files, min_fill, str(Path(npy_dir).resolve())
        )

        self.block_to_idx: dict = vocab["block_to_idx"]
        self.idx_to_block: dict = vocab["idx_to_block"]
        self.vocab_size: int = len(self.block_to_idx)
        self.mask_idx: int = self.vocab_size

        print(
            f"Dataset: {len(self._schematic_paths)} schematics, vocab size {self.vocab_size}"
        )

    def __len__(self) -> int:
        return len(self._schematic_paths)

    def __getitem__(self, idx: int) -> dict:
        blocks = np.load(self._schematic_paths[idx], mmap_mode="r")
        cs = self.chunk_size
        h, l, w = blocks.shape

        if h < cs or l < cs or w < cs:
            chunk, valid_mask = self._pad_chunk(blocks, cs)
        else:
            for _ in range(8):
                y = random.randint(0, h - cs)
                z = random.randint(0, l - cs)
                x = random.randint(0, w - cs)
                chunk = blocks[y : y + cs, z : z + cs, x : x + cs]
                if (chunk != 0).mean() >= 0.02:
                    break
            chunk = np.array(chunk)
            valid_mask = np.ones((cs, cs, cs), dtype=bool)

        k = random.randint(0, 3)
        if k > 0:
            chunk = np.rot90(chunk, k, axes=(1, 2)).copy()
            valid_mask = np.rot90(valid_mask, k, axes=(1, 2)).copy()

        if random.random() < 0.5:
            chunk = chunk[:, :, ::-1].copy()
            valid_mask = valid_mask[:, :, ::-1].copy()

        condition_mask = _sample_condition_mask(cs)

        return {
            "blocks": torch.from_numpy(chunk.astype(np.int64)),
            "condition_mask": torch.from_numpy(condition_mask),
            "valid_mask": torch.from_numpy(valid_mask),
        }

    def _pad_chunk(self, blocks: np.ndarray, cs: int) -> tuple[np.ndarray, np.ndarray]:
        h, l, w = blocks.shape
        chunk = np.zeros((cs, cs, cs), dtype=np.int16)
        valid_mask = np.zeros((cs, cs, cs), dtype=bool)

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
        valid_mask[y_dst, z_dst, x_dst] = True
        return chunk, valid_mask
