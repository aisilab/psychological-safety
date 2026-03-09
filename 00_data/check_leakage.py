#!/usr/bin/env python3
"""Check if any val/test examples leak into system prompt files.

Compares the 'prompt' and 'answer' fields of val.json and test.json against
all system prompt files in 01_prompting/. Reports any matches found.

Usage:
    python check_leakage.py [--prompts-dir DIR]
"""

import argparse
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent
PROJECT_DIR = DATA_DIR.parent
DEFAULT_PROMPTS_DIR = PROJECT_DIR / "01_prompting"


def load_split(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def load_prompt_texts(prompts_dir: Path) -> dict[str, str]:
    """Load all .txt files from the prompts directory."""
    texts = {}
    for p in sorted(prompts_dir.glob("*.txt")):
        texts[p.name] = p.read_text()
    return texts


def check_leakage(split_name: str, split_data: list[dict], prompt_texts: dict[str, str]):
    """Check if any example's prompt or answer appears in any system prompt file."""
    leaks = []
    for i, item in enumerate(split_data):
        for field in ("prompt", "answer"):
            text = item[field].strip()
            # Check substrings: use the first 80 chars of prompt as a reasonable match
            # For prompts, check the full text; for answers, check a representative chunk
            check_text = text if field == "prompt" else text[:200]
            for fname, prompt_content in prompt_texts.items():
                if check_text in prompt_content:
                    leaks.append({
                        "split": split_name,
                        "index": i,
                        "field": field,
                        "file": fname,
                        "risk_cluster": item.get("risk cluster", "N/A"),
                        "text_preview": text[:100],
                    })
    return leaks


def main():
    parser = argparse.ArgumentParser(description="Check for val/test leakage into system prompts.")
    parser.add_argument(
        "--prompts-dir",
        type=Path,
        default=DEFAULT_PROMPTS_DIR,
        help=f"Directory containing system prompt .txt files (default: {DEFAULT_PROMPTS_DIR})",
    )
    args = parser.parse_args()

    prompt_texts = load_prompt_texts(args.prompts_dir)
    if not prompt_texts:
        print(f"No .txt files found in {args.prompts_dir}")
        return

    print(f"Checking against {len(prompt_texts)} system prompt file(s): {list(prompt_texts.keys())}")
    print()

    total_leaks = []
    for split_name in ("val", "test"):
        split_path = DATA_DIR / f"{split_name}.json"
        if not split_path.exists():
            print(f"WARNING: {split_path} not found, skipping.")
            continue
        split_data = load_split(split_path)
        leaks = check_leakage(split_name, split_data, prompt_texts)
        total_leaks.extend(leaks)

    if total_leaks:
        print(f"LEAKAGE DETECTED: {len(total_leaks)} match(es) found!\n")
        for leak in total_leaks:
            print(f"  Split: {leak['split']}, Index: {leak['index']}, Field: {leak['field']}")
            print(f"  System prompt file: {leak['file']}")
            print(f"  Risk cluster: {leak['risk_cluster']}")
            print(f"  Preview: {leak['text_preview']}...")
            print()
    else:
        print("No leakage detected. Val/test examples are clean.")


if __name__ == "__main__":
    main()
