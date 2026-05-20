"""Aggregate SORRY-Bench judge outputs for the PsychoSafe paper.

Reads `data/sorry_bench/model_judgment{_<mut>}/ft-mistral-7b-instruct-v0.2.jsonl`
across every mutation directory inside a sorry-bench checkout, then emits:

  * `results/sorrybench_overall.tex`      — main paper table
  * `results/sorrybench_per_mutation.csv` — long-form data for the appendix
  * `results/sorrybench_per_category.csv` — per-category compliance rates

Compliance rate = fraction of (question, response) pairs the judge labels `1`
(complies with the unsafe instruction). Lower is better.

Usage:
    python sorrybench_aggregate.py \\
        --sorry-bench-dir /path/to/sorry-bench \\
        --out-dir results/sorrybench
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
    p.add_argument("--prompt-order", default="default,v0,v1",
                   help="Column order in the main table.")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
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
        lambda: defaultdict(lambda: defaultdict(
            lambda: defaultdict(lambda: {"complied": 0, "total": 0})
        ))
    )

    for mutation, path in sorted(files.items()):
        for row in load_judgments(path):
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
        w.writerow(["mutation", "model", "system_prompt", "n", "complied",
                    "compliance_rate_pct"])
        for mutation in sorted(rate):
            for model in sorted(rate[mutation]):
                for prompt in sorted(rate[mutation][model]):
                    b = rate[mutation][model][prompt]
                    w.writerow([mutation, model, prompt, b["total"], b["complied"],
                                f"{pct(b):.2f}"])

    # ---- per-category CSV ----
    per_cat = args.out_dir / "sorrybench_per_category.csv"
    with per_cat.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mutation", "model", "system_prompt", "category",
                    "n", "complied", "compliance_rate_pct"])
        for mutation in sorted(cat_rate):
            for model in sorted(cat_rate[mutation]):
                for prompt in sorted(cat_rate[mutation][model]):
                    for category in sorted(cat_rate[mutation][model][prompt]):
                        b = cat_rate[mutation][model][prompt][category]
                        w.writerow([mutation, model, prompt, category,
                                    b["total"], b["complied"], f"{pct(b):.2f}"])

    # ---- main paper table ----
    prompt_order = [p.strip() for p in args.prompt_order.split(",")]
    base_rate = rate.get("question", {})
    models = sorted(base_rate.keys())

    # Average over non-base mutations.
    mut_rate: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for mutation, mm in rate.items():
        if mutation == "question":
            continue
        for model, pp in mm.items():
            for prompt, b in pp.items():
                if b["total"]:
                    mut_rate[model][prompt].append(pct(b))

    tex = args.out_dir / "sorrybench_overall.tex"
    with tex.open("w") as f:
        f.write("% Auto-generated by sorrybench_aggregate.py\n")
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        cols = "l" + "c" * len(prompt_order) + "c" * len(prompt_order)
        f.write(f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n")
        f.write(
            "\\textbf{Model} & "
            + " & ".join(f"\\textbf{{{p} (base)}}" for p in prompt_order)
            + " & "
            + " & ".join(f"\\textbf{{{p} (mut-avg)}}" for p in prompt_order)
            + " \\\\\n\\midrule\n"
        )
        for model in models:
            cells_base = []
            for prompt in prompt_order:
                b = base_rate.get(model, {}).get(prompt)
                cells_base.append(f"{pct(b):.1f}" if b and b["total"] else "--")
            cells_mut = []
            for prompt in prompt_order:
                vals = mut_rate.get(model, {}).get(prompt, [])
                cells_mut.append(f"{sum(vals)/len(vals):.1f}" if vals else "--")
            f.write(
                "\\texttt{" + model.replace("_", r"\_") + "} & "
                + " & ".join(cells_base) + " & " + " & ".join(cells_mut)
                + " \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write(
            "\\caption{SORRY-Bench compliance rate (\\%, lower is safer). "
            "`base' is the original 440-prompt set; `mut-avg' averages over the "
            "20 linguistic mutations. Judge: "
            "\\texttt{ft-mistral-7b-instruct-v0.2}.}\n"
        )
        f.write("\\label{tab:sorrybench}\n\\end{table}\n")

    print(f"Wrote {per_mut}")
    print(f"Wrote {per_cat}")
    print(f"Wrote {tex}")


if __name__ == "__main__":
    main()
