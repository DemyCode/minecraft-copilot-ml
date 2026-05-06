"""
Try loading every file in ~/schematics_scraped as a numpy array.
Results are cached via writcache — second run is instant.
Outputs dataset_report.csv (per-file) and dataset_report.json (summary).

Usage:  uv run python count_usable.py [--dir ~/schematics_scraped]
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm
from writcache import cache, CacheConfig

sys.path.insert(0, str(Path(__file__).parent))
from schematic_loader import load_any

_EXTS = (".schematic", ".schem", ".litematic")


def _scan_all(paths: list[Path]) -> list[tuple[str, str | None]]:
    """
    For each path, return (path_str, error_msg).
    error_msg is None on success, otherwise the exception string.
    """
    results = []
    for p in tqdm(paths, desc="Loading", unit="file"):
        try:
            arr = load_any(str(p))
            if arr.size == 0:
                results.append((str(p), "empty array"))
            else:
                results.append((str(p), None))
        except Exception as e:
            results.append((str(p), f"{type(e).__name__}: {e}"))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=str(Path.home() / "schematics_scraped"))
    parser.add_argument("--out", default="dataset_report")
    args = parser.parse_args()

    src = Path(args.dir)
    all_paths = sorted(p for p in src.iterdir() if p.is_file() and p.suffix.lower() in _EXTS)

    by_ext: Counter = Counter(p.suffix.lower() for p in all_paths)
    print(f"Found {len(all_paths)} files in {src}")
    for ext, cnt in sorted(by_ext.items()):
        print(f"  {ext}: {cnt}")

    cfg = CacheConfig(cache_dir=Path(".writcache_usable"))
    results = cache(_scan_all, config=cfg)(all_paths)

    ok     = [(p, e) for p, e in results if e is None]
    failed = [(p, e) for p, e in results if e is not None]

    print(f"\nUsable : {len(ok):,} / {len(results):,}  ({len(ok)/len(results)*100:.1f}%)")
    print(f"Failed : {len(failed):,}")

    err_types: Counter = Counter(e.split(":")[0] for _, e in failed)
    print("\nFailure types:")
    for err, cnt in err_types.most_common(10):
        print(f"  {err}: {cnt:,}")

    fmt_ok:   Counter = Counter(Path(p).suffix.lower() for p, e in results if e is None)
    fmt_fail: Counter = Counter(Path(p).suffix.lower() for p, e in results if e is not None)
    print("\nBy format:")
    for ext in sorted(_EXTS):
        print(f"  {ext}: {fmt_ok[ext]:,} ok, {fmt_fail[ext]:,} failed")

    # ── CSV: one row per file ────────────────────────────────────────────────
    csv_path = Path(f"{args.out}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "format", "usable", "error"])
        for p, e in results:
            w.writerow([Path(p).name, Path(p).suffix.lower(), e is None, e or ""])
    print(f"\nPer-file report : {csv_path}")

    # ── JSON: summary ────────────────────────────────────────────────────────
    json_path = Path(f"{args.out}.json")
    summary = {
        "source_dir": str(src),
        "total_files": len(results),
        "usable": len(ok),
        "failed": len(failed),
        "usable_pct": round(len(ok) / len(results) * 100, 1),
        "by_format": {
            ext: {"ok": fmt_ok[ext], "failed": fmt_fail[ext]}
            for ext in sorted(_EXTS)
        },
        "failure_types": dict(err_types.most_common()),
    }
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary report  : {json_path}")


if __name__ == "__main__":
    main()
