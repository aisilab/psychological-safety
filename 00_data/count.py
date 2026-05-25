#!/usr/bin/env python3
"""Output split statistics for train/val/test datasets."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
SPLITS = ("train", "val", "test")
CLUSTER_KEY = "risk cluster"


def load_split(path: Path) -> list[dict]:
    with path.open() as f:
        return json.load(f)


def summarize_split(split_name: str, data: list[dict]) -> None:
    cluster_counts = Counter(item.get(CLUSTER_KEY, "UNKNOWN") for item in data)
    print(f"{split_name}: {len(data)} examples")
    for cluster, count in sorted(cluster_counts.items()):
        print(f"  {cluster}: {count}")


def main() -> None:
    for split in SPLITS:
        split_path = DATA_DIR / f"{split}.json"
        split_data = load_split(split_path)
        summarize_split(split, split_data)
        print()


if __name__ == "__main__":
    main()
