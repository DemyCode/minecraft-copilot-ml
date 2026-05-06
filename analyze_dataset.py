"""
Scan ~/Downloads for all Minecraft structure files, assess usability,
detect duplicates, and print summary statistics.

Usage:  uv run python analyze_dataset.py [--dir ~/Downloads] [--workers 8]
"""

import argparse
import hashlib
import os
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Must be importable from project root
sys.path.insert(0, str(Path(__file__).parent))
from schematic_loader import load_any

_EXTS = (".schematic", ".schem", ".litematic")
MIN_FILL = 0.02    # at least 2% non-air (dataset.py uses 0.10 by default)


def _file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _process_file(path: Path) -> dict:
    result = {
        "path": str(path),
        "ext": path.suffix.lower(),
        "file_bytes": path.stat().st_size,
        "hash": None,
        "ok": False,
        "reason": None,
        "h": 0, "l": 0, "w": 0,
        "total_voxels": 0,
        "non_air": 0,
        "fill": 0.0,
        "unique_blocks": 0,
        "blocks": None,  # top-N block counter, filled later
    }
    try:
        result["hash"] = _file_hash(path)
        arr = load_any(str(path))
        h, l, w = arr.shape
        result["h"], result["l"], result["w"] = h, l, w
        total = h * l * w
        result["total_voxels"] = total
        non_air = int((arr != "minecraft:air").sum())
        result["non_air"] = non_air
        result["fill"] = non_air / total if total > 0 else 0.0
        result["unique_blocks"] = len(np.unique(arr))

        if result["fill"] < MIN_FILL:
            result["reason"] = f"too_sparse ({result['fill']:.1%} fill)"
        else:
            result["ok"] = True
            # Store block counts for vocab analysis (cheap: just unique + counts)
            unique, counts = np.unique(arr, return_counts=True)
            result["blocks"] = {str(u): int(c) for u, c in zip(unique, counts)}

    except Exception as e:
        result["reason"] = f"load_error: {type(e).__name__}: {e}"

    return result


def _process_file_safe(path: Path) -> dict:
    try:
        return _process_file(path)
    except Exception as e:
        return {
            "path": str(path), "ext": path.suffix.lower(),
            "file_bytes": 0, "hash": None, "ok": False,
            "reason": f"unexpected: {e}",
            "h": 0, "l": 0, "w": 0, "total_voxels": 0,
            "non_air": 0, "fill": 0.0, "unique_blocks": 0, "blocks": None,
        }


def _percentile_str(values: list, label: str) -> str:
    if not values:
        return f"  {label}: (none)"
    a = sorted(values)
    n = len(a)
    p = lambda q: a[int(q * (n - 1))]
    return (f"  {label}: min={p(0)}, p25={p(0.25)}, median={p(0.5)}, "
            f"p75={p(0.75)}, p90={p(0.9)}, max={p(1.0)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=str(Path.home() / "Downloads"))
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 4, 8))
    args = parser.parse_args()

    src = Path(args.dir)
    all_paths = [
        p for p in src.iterdir()
        if p.is_file() and p.suffix.lower() in _EXTS
    ]
    all_paths.sort()

    print(f"Found {len(all_paths)} files in {src}")
    by_ext: Counter = Counter(p.suffix.lower() for p in all_paths)
    for ext, cnt in sorted(by_ext.items()):
        print(f"  {ext}: {cnt}")

    print(f"\nProcessing with {args.workers} workers...")
    results = []
    with Pool(args.workers) as pool:
        for r in tqdm(pool.imap_unordered(_process_file_safe, all_paths, chunksize=32),
                      total=len(all_paths), unit="file"):
            results.append(r)

    # ── Duplicate detection ──────────────────────────────────────────────────
    hash_to_paths: dict[str, list[str]] = defaultdict(list)
    for r in results:
        if r["hash"]:
            hash_to_paths[r["hash"]].append(r["path"])
    dup_groups = {h: ps for h, ps in hash_to_paths.items() if len(ps) > 1}
    dup_paths = {p for ps in dup_groups.values() for p in ps[1:]}  # keep first

    # ── Partition ─────────────────────────────────────────────────────────────
    ok = [r for r in results if r["ok"] and r["path"] not in dup_paths]
    ok_dups = [r for r in results if r["ok"] and r["path"] in dup_paths]
    failed = [r for r in results if not r["ok"]]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total files scanned : {len(results)}")
    print(f"  Usable (unique)   : {len(ok)}")
    print(f"  Duplicate copies  : {len(ok_dups)}  ({len(dup_groups)} groups)")
    print(f"  Unusable          : {len(failed)}")

    # Failure breakdown
    if failed:
        reason_counts: Counter = Counter()
        for r in failed:
            key = r["reason"].split(":")[0] if r["reason"] else "unknown"
            reason_counts[key] += 1
        print("\nUnusable reasons:")
        for reason, cnt in reason_counts.most_common():
            print(f"  {reason}: {cnt}")

    # ── Dimension analysis ────────────────────────────────────────────────────
    print("\nDimension analysis (usable unique files):")
    heights  = [r["h"] for r in ok]
    lengths  = [r["l"] for r in ok]
    widths   = [r["w"] for r in ok]
    volumes  = [r["total_voxels"] for r in ok]
    fills    = [r["fill"] for r in ok]

    print(_percentile_str(heights,  "height (Y)"))
    print(_percentile_str(lengths,  "length (Z)"))
    print(_percentile_str(widths,   "width  (X)"))
    print(_percentile_str(volumes,  "voxels    "))
    print(_percentile_str([round(f * 100, 1) for f in fills], "fill %    "))

    # Schematics that fit within a 32³ chunk vs those that are larger
    fits_in_32 = sum(1 for r in ok if r["h"] <= 32 and r["l"] <= 32 and r["w"] <= 32)
    print(f"\nFits in 32³ chunk: {fits_in_32} / {len(ok)} ({fits_in_32/len(ok)*100:.1f}%)")

    # Estimate chunk count (stride=16, chunk=32, min_fill=10%)
    def estimate_chunks(r: dict) -> int:
        h, l, w = r["h"], r["l"], r["w"]
        cs, stride = 32, 16
        if h < cs or l < cs or w < cs:
            return 1 if r["fill"] >= 0.10 else 0
        nh = (h - cs) // stride + 1
        nl = (l - cs) // stride + 1
        nw = (w - cs) // stride + 1
        return nh * nl * nw  # rough upper bound (doesn't filter by chunk fill)

    total_chunks = sum(estimate_chunks(r) for r in ok)
    print(f"Estimated chunks (32³, stride=16): ~{total_chunks:,}")

    # ── Vocabulary analysis ───────────────────────────────────────────────────
    global_counts: Counter = Counter()
    schematic_freq: Counter = Counter()  # how many schematics contain each block
    for r in ok:
        if r["blocks"]:
            global_counts.update(r["blocks"])
            schematic_freq.update(r["blocks"].keys())

    print(f"\nVocabulary size: {len(global_counts)} unique block types")
    print("\nTop 30 blocks by occurrence:")
    for block, cnt in global_counts.most_common(30):
        freq = schematic_freq[block]
        pct = freq / len(ok) * 100 if ok else 0
        print(f"  {block:<50s}  {cnt:>12,}  in {freq:>6} files ({pct:.1f}%)")

    # By format
    print("\nUsable files by format:")
    by_fmt: Counter = Counter(r["ext"] for r in ok)
    for ext, cnt in sorted(by_fmt.items()):
        print(f"  {ext}: {cnt}")

    # Duplicate group examples
    if dup_groups:
        print(f"\nDuplicate groups ({min(5, len(dup_groups))} examples):")
        for h, ps in list(dup_groups.items())[:5]:
            print(f"  hash={h[:8]}... → {[Path(p).name for p in ps]}")

    print("\nDone.")


if __name__ == "__main__":
    main()
