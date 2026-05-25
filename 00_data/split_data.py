#!/usr/bin/env python3
"""Split the dataset into train/val/test, stratified by risk cluster.

Usage:
    python split_data.py [--n N] [--seed SEED]

where N is the number of validation (and test) examples per risk cluster (default: 100).
"""

import argparse
import json
import re
import random
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent
PROJECT_DIR = DATA_DIR.parent
INPUT_FILE = DATA_DIR / "final_dataset.json"
SYSTEM_PROMPT_FILE = PROJECT_DIR / "01_prompting" / "systemprompt-v1.txt"


def extract_system_prompt_examples(path: Path) -> set[str]:
    """Extract user prompts used as examples in the system prompt."""
    text = path.read_text()
    # Matches lines like: User: "some prompt here"
    return {m.group(1) for m in re.finditer(r'^User: "(.+)"', text, re.MULTILINE)}


def main():
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test.")
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of val (and test) examples per risk cluster (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()
    n = args.n

    # Load system prompt examples to keep them in train split
    reserved_prompts = set()
    if SYSTEM_PROMPT_FILE.exists():
        reserved_prompts = extract_system_prompt_examples(SYSTEM_PROMPT_FILE)
        print(f"Reserved {len(reserved_prompts)} example(s) from system prompt for train split.")

    with open(INPUT_FILE) as f:
        data = json.load(f)

    # Group by risk cluster
    by_cluster = defaultdict(list)
    for item in data:
        by_cluster[item["risk cluster"]].append(item)

    train, val, test = [], [], []
    rng = random.Random(args.seed)

    for cluster, items in sorted(by_cluster.items()):
        # Separate reserved items (must go to train) from the rest
        reserved = [it for it in items if it["prompt"] in reserved_prompts]
        available = [it for it in items if it["prompt"] not in reserved_prompts]

        if len(available) < 2 * n:
            raise ValueError(
                f"Cluster '{cluster}' has only {len(available)} available examples "
                f"({len(reserved)} reserved for train), "
                f"need at least {2 * n} for {n} val + {n} test."
            )
        rng.shuffle(available)
        val.extend(available[:n])
        test.extend(available[n : 2 * n])
        train.extend(reserved)
        train.extend(available[2 * n :])

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        out_path = DATA_DIR / f"{split_name}.json"
        with open(out_path, "w") as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        # Count per cluster for summary
        counts = defaultdict(int)
        for item in split_data:
            counts[item["risk cluster"]] += 1
        print(f"{split_name}: {len(split_data)} examples ({dict(sorted(counts.items()))})")


if __name__ == "__main__":
    main()
