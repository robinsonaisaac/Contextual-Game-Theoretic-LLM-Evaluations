#!/usr/bin/env python3
"""Rebuild index.parquet from .pt bundle files on disk.

Usage:
    python3 scripts/rebuild_index.py --run-dir local_data/mech-interp-v1 --split train
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from game_theory_llm.steering.storage import load_activation_bundle, write_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--bundle-dir", default=None,
                        help="Override bundle directory (default: run-dir/activations/split)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    bundle_dir = Path(args.bundle_dir) if args.bundle_dir else run_dir / "activations" / args.split
    index_path = run_dir / "index.parquet"

    pts = sorted(bundle_dir.glob("*.pt"))
    if not pts:
        print(f"No .pt files found in {bundle_dir}")
        sys.exit(1)

    print(f"Found {len(pts)} bundles, loading...")
    bundles, paths = [], []
    for i, p in enumerate(pts):
        if i % 100 == 0:
            print(f"  {i}/{len(pts)}...", flush=True)
        b = load_activation_bundle(p)
        # Remap path to volume-style path for compatibility with load_index_and_bundles
        bundles.append(b)
        paths.append(p)

    write_index(bundles, paths, index_path, split=args.split)
    print(f"Wrote index with {len(bundles)} rows → {index_path}")


if __name__ == "__main__":
    main()
