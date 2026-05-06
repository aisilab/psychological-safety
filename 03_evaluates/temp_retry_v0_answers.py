import argparse
import json
from pathlib import Path


DEFAULT_RETRIES_PATH = Path("03_evaluates/output/qwen3.5-397b-a17b_judgements_v1_retries.json")
DEFAULT_V0_ANSWERS_PATH = Path("01_prompting/results/svenharms_val_v0_latest_splitted_with_category.json")
DEFAULT_OUTPUT_PATH = Path("03_evaluates/output/temp_v0_retries.json")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_extract_failed_requests_like_rows(retries_payload, v0_payload):
    criteria_file = retries_payload.get("criteria_file")
    v0_by_prompt = {
        item["prompt"]: item
        for item in v0_payload.get("results", [])
        if item.get("stucked") != "true" and "prompt" in item
    }

    rows = []
    missing_prompts = []
    for item in retries_payload.get("judgments", []):
        prompt = item.get("prompt")
        v0_item = v0_by_prompt.get(prompt)
        if v0_item is None:
            missing_prompts.append(prompt)
            continue

        rows.append(
            {
                "id": item.get("id"),
                "prompt": prompt,
                "model_answer": v0_item.get("answer"),
                "risk cluster": v0_item.get("risk cluster"),
                "risk category": v0_item.get("risk category"),
                "criteria_file": criteria_file
            }
        )

    if missing_prompts:
        raise ValueError(
            f"Could not find {len(missing_prompts)} prompts in v0 results. "
            f"First missing prompt: {missing_prompts[0]!r}"
        )

    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create extract_failed_requests-like rows for v0 using ids/prompts from v1 retries."
    )
    parser.add_argument("--retries", type=Path, default=DEFAULT_RETRIES_PATH)
    parser.add_argument("--v0", type=Path, default=DEFAULT_V0_ANSWERS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    retries_payload = load_json(args.retries)
    v0_payload = load_json(args.v0)
    rows = build_extract_failed_requests_like_rows(retries_payload, v0_payload)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
