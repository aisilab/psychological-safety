"""Convert the XSTest prompt set into a SORRY-Bench-compatible question.jsonl.

XSTest~\\citep{rottger2024xstest} ships 250 safe prompts that look unsafe (and
should therefore be answered) plus 200 contrast prompts that are genuinely
unsafe (and should be refused) — 450 prompts in total. The aggregator
separates the two by the `category` field, which we tag as `safe_<subcat>` or
`unsafe_<subcat>` based on the CSV row's `label` column.

CSV schema (paul-rottger/xstest, xstest_prompts.csv):
    id, prompt, type, label, focus, note
where `label` ∈ {"safe", "unsafe"} and `type` is the subcategory (e.g.
`homonyms`, `figurative_language`, `contrast_homonyms`, ...).

Output schema, one record per line:
    {"question_id": int,
     "category":    "safe_<subcat>" | "unsafe_<subcat>",
     "turns":       [prompt]}
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from urllib.request import urlopen

XSTEST_CSV_URL = (
    "https://raw.githubusercontent.com/paul-rottger/xstest/main/"
    "xstest_prompts.csv"
)


def fetch_csv(src: str) -> str:
    if src.startswith(("http://", "https://")):
        with urlopen(src, timeout=30) as r:
            return r.read().decode("utf-8")
    return Path(src).read_text(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output question.jsonl path (e.g. data/xstest/question.jsonl).",
    )
    p.add_argument(
        "--csv-source",
        default=XSTEST_CSV_URL,
        help="HTTPS URL or local path for the XSTest prompts CSV.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    text = fetch_csv(args.csv_source)
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        raise SystemExit(f"!! no rows parsed from {args.csv_source}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_safe = 0
    n_unsafe = 0
    n_written = 0
    with args.out.open("w", encoding="utf-8") as fout:
        for row in rows:
            prompt = (row.get("prompt") or "").strip()
            label = (row.get("label") or "").strip().lower()
            subcat = (row.get("type") or "").strip()
            try:
                qid = int((row.get("id") or "").strip())
            except ValueError:
                continue
            if not prompt or label not in {"safe", "unsafe"}:
                continue
            # Strip the redundant `contrast_` prefix on unsafe rows so the
            # subcategory matches its safe counterpart for paired analyses.
            if label == "unsafe":
                subcat = subcat.removeprefix("contrast_")
                category = f"unsafe_{subcat}"
                n_unsafe += 1
            else:
                category = f"safe_{subcat}"
                n_safe += 1
            fout.write(
                json.dumps(
                    {
                        "question_id": qid,
                        "category": category,
                        "turns": [prompt],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n_written += 1

    print(
        f"[xstest] wrote {n_written} prompts to {args.out} "
        f"({n_safe} safe, {n_unsafe} unsafe)"
    )


if __name__ == "__main__":
    main()
