"""Aggregate SORRY-Bench judge outputs for the PsychoSafe paper.

Reads `data/sorry_bench/model_judgment{_<mut>}/ft-mistral-7b-instruct-v0.2.jsonl`
across every mutation directory inside a sorry-bench checkout, then emits:

  * a summary table to the terminal           — base + mut-avg per model/prompt
  * `results/sorrybench_overall.csv`      — that summary table
  * `results/sorrybench_per_mutation.csv` — long-form data for the appendix
  * `results/sorrybench_per_category.csv` — per-category compliance rates

Compliance rate = fraction of (question, response) pairs the judge labels `1`
(complies with the unsafe instruction). Lower is better.

A subset of SORRY-Bench base prompts appears verbatim in the PsychoSafe SFT
training split (`00_data/train.json`), so reporting raw rates would be
train/test contamination. Pass `--exclude-ids-file` (defaults to the sibling
`sorrybench_leaked_ids.json`) to drop those question_ids from every rate; the
emitted artifacts then describe the decontaminated subset.

Usage:
    python sorrybench_aggregate.py \\
        --sorry-bench-dir /path/to/sorry-bench \\
        --out-dir results/sorrybench \\
        --exclude-ids-file sorrybench_leaked_ids.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

JUDGE_FILE = "ft-mistral-7b-instruct-v0.2.jsonl"

# Suffixes that decode.py adds; the judge writes these into the `model` field
# of its output rows. Stripping them recovers the original model_id.
JUDGE_SUFFIXES = ("_decoded", "_translated_to_en")


def strip_judge_suffix(model_id: str) -> str:
    for suffix in JUDGE_SUFFIXES:
        if model_id.endswith(suffix):
            return model_id[: -len(suffix)]
    return model_id


def discover_judgment_files(sorry_dir: Path, bench_name: str) -> dict[str, Path]:
    """Map mutation_key → judgment jsonl path. Base set uses key 'question'."""
    out: dict[str, Path] = {}
    base = sorry_dir / "data" / bench_name / "model_judgment" / JUDGE_FILE
    if base.exists():
        out["question"] = base
    pattern = re.compile(r"model_judgment_(.+)$")
    for d in (sorry_dir / "data" / bench_name).glob("model_judgment_*"):
        m = pattern.search(d.name)
        if not m:
            continue
        candidate = d / JUDGE_FILE
        if candidate.exists():
            out[m.group(1)] = candidate
    return out


def load_categories(sorry_dir: Path, bench_name: str) -> dict[int, str]:
    """question_id → category, from the base question.jsonl."""
    qfile = sorry_dir / "data" / bench_name / "question.jsonl"
    cats: dict[int, str] = {}
    if not qfile.exists():
        return cats
    with qfile.open() as f:
        for line in f:
            row = json.loads(line)
            cats[row["question_id"]] = row.get("category", "unknown")
    return cats


def load_exclude_ids(path: Path | None) -> set[int]:
    """Load question_ids to drop (train/test contamination). Empty set if none."""
    if path is None or not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open() as f:
        payload = json.load(f)
    ids = payload["question_ids"] if isinstance(payload, dict) else payload
    return {int(i) for i in ids}


def load_judgments(path: Path) -> list[dict]:
    """Take the LAST judgment per (question_id, model_id) in case of duplicates."""
    by_key: dict[tuple[int, str], dict] = {}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            key = (row["question_id"], row["model"])
            by_key[key] = row
    return list(by_key.values())


def parse_model_id(model_id: str) -> tuple[str, str]:
    """`qwen35-a3b_v1` → ('qwen35-a3b', 'v1'). Falls back to (full_id, 'default')."""
    base = strip_judge_suffix(model_id)
    if "_" in base:
        head, tail = base.rsplit("_", 1)
        return head, tail
    return base, "default"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sorry-bench-dir", required=True, type=Path)
    p.add_argument("--bench-name", default="sorry_bench")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument(
        "--prompt-order",
        default="default,v0,v1",
        help="Column order in the main table.",
    )
    p.add_argument(
        "--exclude-ids-file",
        type=Path,
        default=Path(__file__).with_name("sorrybench_leaked_ids.json"),
        help="JSON with a `question_ids` list to drop (SFT-train contamination). "
        "Pass an empty/nonexistent path to disable decontamination.",
    )
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    exclude_ids = load_exclude_ids(args.exclude_ids_file)
    if exclude_ids:
        print(
            f"Excluding {len(exclude_ids)} contaminated question_id(s) "
            f"from {args.exclude_ids_file}"
        )
    categories = load_categories(args.sorry_bench_dir, args.bench_name)
    files = discover_judgment_files(args.sorry_bench_dir, args.bench_name)
    if not files:
        raise SystemExit(
            f"No judgment files found under {args.sorry_bench_dir}/data/{args.bench_name}/"
        )
    print(f"Found judgments for {len(files)} mutation(s): {sorted(files)}")

    # rate[mutation][model_root][prompt] = {"complied": int, "total": int}
    rate: dict[str, dict[str, dict[str, dict[str, int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: {"complied": 0, "total": 0}))
    )
    # cat[mutation][model_root][prompt][category] = {"complied": int, "total": int}
    cat_rate: dict = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(lambda: {"complied": 0, "total": 0})
            )
        )
    )

    for mutation, path in sorted(files.items()):
        for row in load_judgments(path):
            if row["question_id"] in exclude_ids:
                continue
            model_root, prompt_key = parse_model_id(row["model"])
            score = row.get("score")
            if score is None:
                continue
            bucket = rate[mutation][model_root][prompt_key]
            bucket["total"] += 1
            if int(score) == 1:
                bucket["complied"] += 1
            category = categories.get(row["question_id"], "unknown")
            cb = cat_rate[mutation][model_root][prompt_key][category]
            cb["total"] += 1
            if int(score) == 1:
                cb["complied"] += 1

    def pct(b: dict[str, int]) -> float:
        return 100.0 * b["complied"] / b["total"] if b["total"] else float("nan")

    # ---- per-mutation CSV ----
    per_mut = args.out_dir / "sorrybench_per_mutation.csv"
    with per_mut.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "mutation",
                "model",
                "system_prompt",
                "n",
                "complied",
                "compliance_rate_pct",
            ]
        )
        for mutation in sorted(rate):
            for model in sorted(rate[mutation]):
                for prompt in sorted(rate[mutation][model]):
                    b = rate[mutation][model][prompt]
                    w.writerow(
                        [
                            mutation,
                            model,
                            prompt,
                            b["total"],
                            b["complied"],
                            f"{pct(b):.2f}",
                        ]
                    )

    # ---- per-category CSV ----
    per_cat = args.out_dir / "sorrybench_per_category.csv"
    with per_cat.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "mutation",
                "model",
                "system_prompt",
                "category",
                "n",
                "complied",
                "compliance_rate_pct",
            ]
        )
        for mutation in sorted(cat_rate):
            for model in sorted(cat_rate[mutation]):
                for prompt in sorted(cat_rate[mutation][model]):
                    for category in sorted(cat_rate[mutation][model][prompt]):
                        b = cat_rate[mutation][model][prompt][category]
                        w.writerow(
                            [
                                mutation,
                                model,
                                prompt,
                                category,
                                b["total"],
                                b["complied"],
                                f"{pct(b):.2f}",
                            ]
                        )

    # ---- main paper table ----
    prompt_order = [p.strip() for p in args.prompt_order.split(",")]
    base_rate = rate.get("question", {})
    models = sorted(base_rate.keys())

    # Effective base set size after decontamination (prompts the judge scored).
    base_n = max(
        (b["total"] for pp in base_rate.values() for b in pp.values()),
        default=0,
    )

    # Average over non-base mutations.
    n_mutations = sum(1 for m in rate if m != "question")
    mut_rate: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for mutation, mm in rate.items():
        if mutation == "question":
            continue
        for model, pp in mm.items():
            for prompt, b in pp.items():
                if b["total"]:
                    mut_rate[model][prompt].append(pct(b))

    def base_cell(model: str, prompt: str) -> str:
        b = base_rate.get(model, {}).get(prompt)
        return f"{pct(b):.1f}" if b and b["total"] else "--"

    def mut_cell(model: str, prompt: str) -> str:
        vals = mut_rate.get(model, {}).get(prompt, [])
        return f"{sum(vals) / len(vals):.1f}" if vals else "--"

    # ---- overall CSV (the former main table) ----
    overall = args.out_dir / "sorrybench_overall.csv"
    with overall.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["model"]
            + [f"{p}_base" for p in prompt_order]
            + [f"{p}_mutavg" for p in prompt_order]
        )
        for model in models:
            w.writerow(
                [model]
                + [base_cell(model, p) for p in prompt_order]
                + [mut_cell(model, p) for p in prompt_order]
            )

    # ---- terminal summary ----
    headers = (
        ["model"]
        + [f"{p} (base)" for p in prompt_order]
        + [f"{p} (mut)" for p in prompt_order]
    )
    rows_out = [
        [model]
        + [base_cell(model, p) for p in prompt_order]
        + [mut_cell(model, p) for p in prompt_order]
        for model in models
    ]
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows_out))
        if rows_out
        else len(headers[i])
        for i in range(len(headers))
    ]

    def fmt(cells: list[str]) -> str:
        return "  ".join(
            c.ljust(widths[i]) if i == 0 else c.rjust(widths[i])
            for i, c in enumerate(cells)
        )

    print(
        f"\nSORRY-Bench compliance rate (%, lower is safer) | "
        f"base={base_n} prompts, {n_mutations} mutations"
    )
    if exclude_ids:
        print(
            f"decontaminated: {len(exclude_ids)} base prompts removed "
            f"(verbatim in SFT train split)"
        )
    print(fmt(headers))
    print("-" * len(fmt(headers)))
    for r in rows_out:
        print(fmt(r))

    print(f"\nWrote {per_mut}")
    print(f"Wrote {per_cat}")
    print(f"Wrote {overall}")


if __name__ == "__main__":
    main()
