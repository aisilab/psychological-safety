from typing import Tuple
import re
import json
import statistics
import os
import random

def split_reasoning_traces(text: str, reasoning_token: str = "</think>") -> Tuple[str|None, str]:
    if reasoning_token in text:
        reasoning, answer = text.split(reasoning_token, 1)
        return reasoning.strip(), answer.strip()
    else:
        return None, text.strip()
    

def extract_criteria_from_judge_answer(judge_answer):
    """
    Parse criterion outcomes from a judge answer text.

    Returns a dict with keys criterion_1..criterion_4 where values are either:
    - int/float for rating-based criteria
    - "Yes"/"No" for presence-based criteria
    - None if not found
    """
    extracted = {f"criterion_{i}": None for i in range(1, 5)}
    if not judge_answer:
        return extracted

    criterion_block_pattern = re.compile(
        r"Criterion\s*(\d+)\b(.*?)(?=Criterion\s*\d+\b|$)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    rating_pattern = re.compile(r"Rating\s*:\s*([0-9]+(?:\.[0-9]+)?)", flags=re.IGNORECASE)
    yes_no_pattern = re.compile(
        r"(?:Refusal\s+presence|External\s+references?\s+presence|Presence)\s*:\s*(Yes|No)",
        flags=re.IGNORECASE,
    )

    for match in criterion_block_pattern.finditer(judge_answer):
        criterion_id = int(match.group(1))
        if criterion_id < 1 or criterion_id > 4:
            continue

        block = match.group(2)
        rating_match = rating_pattern.search(block)
        if rating_match:
            raw_rating = rating_match.group(1)
            if raw_rating.isdigit():
                extracted[f"criterion_{criterion_id}"] = int(raw_rating)
            else:
                extracted[f"criterion_{criterion_id}"] = float(raw_rating)
            continue

        yes_no_match = yes_no_pattern.search(block)
        if yes_no_match:
            extracted[f"criterion_{criterion_id}"] = yes_no_match.group(1).capitalize()

    return extracted


def append_criteria_to_judgements_json(json_filename, output_filename=None):
    """
    Append criterion_1..criterion_4 fields to each judgment object in a JSON file.
    If output_filename is None, the input file is updated in-place.
    """
    if output_filename is None:
        output_filename = json_filename

    with open(json_filename, "r") as f:
        payload = json.load(f)

    judgments = payload.get("judgments")
    if judgments is None:
        judgments = payload.get("judgements", [])

    for item in judgments:
        judge_answer = item.get("judge_answer") or item.get("judgment") or item.get("full_judge_output", "")
        item.update(extract_criteria_from_judge_answer(judge_answer))

    with open(output_filename, "w") as f:
        json.dump(payload, f, indent=4)

    return payload


def aggregate_criteria_by_model(json_file_paths, model_names, output_filename=None, include_model_answers=False):
    """
    Aggregate criterion values from multiple judgment JSON files.

    Parameters:
    - json_file_paths: list of paths to JSON files produced by append_criteria_to_judgements_json.
    - model_names: list of model names in the same order as json_file_paths.
    - output_filename: optional path to save the aggregated payload.
        - include_model_answers: if True, include model_answer_v0/model_answer_v1 columns when
            v0/v1 files are present in json_file_paths.

    Output schema:
    {
        "models": [...],
        "num_items": N,
        "results": [
            {
                "id": 0,
                "<model_1>_criterion_1": ...,
                ...,
                "<model_n>_criterion_4": ...
            },
            ...
        ]
    }
    """
    if len(json_file_paths) != len(model_names):
        raise ValueError(
            f"json_file_paths and model_names must have same length, got {len(json_file_paths)} and {len(model_names)}"
        )

    if not json_file_paths:
        raise ValueError("json_file_paths cannot be empty")

    normalized_rows_per_model = []
    split_tags_per_file = []

    def _detect_split_tag(file_path):
        path_lower = file_path.lower()
        if "v0" in path_lower:
            return "v0"
        if "v1" in path_lower:
            return "v1"
        return None

    for file_path in json_file_paths:
        with open(file_path, "r") as f:
            payload = json.load(f)

        split_tags_per_file.append(_detect_split_tag(file_path))

        judgments = payload.get("judgments")
        if judgments is None:
            judgments = payload.get("judgements", [])

        rows = []
        for idx, item in enumerate(judgments):
            row_id = item.get("id", idx)
            rows.append(
                {
                    "row_id": row_id,
                    "model_answer": item.get("model_answer"),
                    "criterion_1": item.get("criterion_1"),
                    "criterion_2": item.get("criterion_2"),
                    "criterion_3": item.get("criterion_3"),
                    "criterion_4": item.get("criterion_4"),
                }
            )

        normalized_rows_per_model.append(rows)

    base_rows = normalized_rows_per_model[0]
    base_ids = [row["row_id"] for row in base_rows]

    # Validate that all files can be aligned by id/index and have the same size.
    for rows in normalized_rows_per_model[1:]:
        current_ids = [row["row_id"] for row in rows]
        if current_ids != base_ids:
            raise ValueError("Input judgment files do not align by id/index in the same order")

    v0_answers = [None] * len(base_rows)
    v1_answers = [None] * len(base_rows)

    if include_model_answers:
        for rows, split_tag in zip(normalized_rows_per_model, split_tags_per_file):
            if split_tag not in {"v0", "v1"}:
                continue

            for i, row in enumerate(rows):
                model_answer = row.get("model_answer")
                if model_answer is None:
                    continue

                if split_tag == "v0" and v0_answers[i] is None:
                    v0_answers[i] = model_answer
                if split_tag == "v1" and v1_answers[i] is None:
                    v1_answers[i] = model_answer

    has_v0 = include_model_answers and any(tag == "v0" for tag in split_tags_per_file)
    has_v1 = include_model_answers and any(tag == "v1" for tag in split_tags_per_file)

    aggregated_results = []
    for i, base_row in enumerate(base_rows):
        aggregated_row = {
            "id": base_row.get("row_id", i),
        }

        if has_v0:
            aggregated_row["model_answer_v0"] = v0_answers[i]
        if has_v1:
            aggregated_row["model_answer_v1"] = v1_answers[i]

        for model_name, model_rows in zip(model_names, normalized_rows_per_model):
            model_row = model_rows[i]
            aggregated_row[f"{model_name}_criterion_1"] = model_row.get("criterion_1")
            aggregated_row[f"{model_name}_criterion_2"] = model_row.get("criterion_2")
            aggregated_row[f"{model_name}_criterion_3"] = model_row.get("criterion_3")
            aggregated_row[f"{model_name}_criterion_4"] = model_row.get("criterion_4")

        aggregated_results.append(aggregated_row)

    output_payload = {
        "models": model_names,
        "num_items": len(aggregated_results),
        "results": aggregated_results,
    }

    if output_filename:
        with open(output_filename, "w") as f:
            json.dump(output_payload, f, indent=4)

    return output_payload


def compute_and_print_judge_metrics(
    json_filename,
    scalar_criteria=(2, 4),
    boolean_criteria=(1, 3),
):
    """
    Compute and print per-judge, per-version metrics from an aggregated compare file.

    Expected key format inside each result row:
        <judge_name>_v0_criterion_<n>
        <judge_name>_v1_criterion_<n>

    Metrics:
    - Scalar criteria: mean and population std dev
    - Boolean criteria: Yes-rate (mean over 1/0) and population std dev
    """
    with open(json_filename, "r") as f:
        payload = json.load(f)

    rows = payload.get("results", [])
    criteria_to_track = set(scalar_criteria).union(boolean_criteria)
    key_pattern = re.compile(r"^(?P<judge>.+)_(?P<version>v[01])_criterion_(?P<criterion>\d+)$")

    def _to_float(value):
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                return float(stripped)
            except ValueError:
                return None
        return None

    def _to_binary(value):
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            if value in (0, 1):
                return float(value)
            return None
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"yes", "true", "1"}:
                return 1.0
            if normalized in {"no", "false", "0"}:
                return 0.0
        return None

    values_by_judge = {}

    def _ensure_judge(judge_name):
        if judge_name not in values_by_judge:
            values_by_judge[judge_name] = {
                "v0": {f"criterion_{c}": [] for c in sorted(criteria_to_track)},
                "v1": {f"criterion_{c}": [] for c in sorted(criteria_to_track)},
            }

    for row in rows:
        for key, value in row.items():
            match = key_pattern.match(key)
            if not match:
                continue

            judge = match.group("judge")
            version = match.group("version")
            criterion = int(match.group("criterion"))

            if criterion not in criteria_to_track:
                continue

            _ensure_judge(judge)
            criterion_key = f"criterion_{criterion}"

            if criterion in scalar_criteria:
                numeric_value = _to_float(value)
                if numeric_value is not None:
                    values_by_judge[judge][version][criterion_key].append(numeric_value)

            if criterion in boolean_criteria:
                binary_value = _to_binary(value)
                if binary_value is not None:
                    values_by_judge[judge][version][criterion_key].append(binary_value)

    def _stats(values):
        if not values:
            return {"mean": None, "std_dev": None, "n": 0}
        return {
            "mean": statistics.mean(values),
            "std_dev": statistics.pstdev(values),
            "n": len(values),
        }

    def _fmt(number):
        return "n/a" if number is None else f"{number:.4f}"

    def _judge_sort_key(name):
        priority = payload.get("models", [])
        for i, model_name in enumerate(priority):
            if model_name.startswith(f"{name}_"):
                return i
        return len(priority)

    judge_names = sorted(values_by_judge.keys(), key=_judge_sort_key)

    metrics = {}
    print(f"Metric summary from: {json_filename}")
    for judge in judge_names:
        metrics[judge] = {}
        print(f"\nJudge: {judge}")

        for version in ("v0", "v1"):
            metrics[judge][version] = {"scalar": {}, "boolean": {}}
            print(f"  Version: {version}")
            print("    Scalar criteria")

            for criterion in sorted(scalar_criteria):
                criterion_key = f"criterion_{criterion}"
                scalar_stats = _stats(values_by_judge[judge][version][criterion_key])
                metrics[judge][version]["scalar"][criterion_key] = scalar_stats
                print(
                    f"      {criterion_key}: mean={_fmt(scalar_stats['mean'])}, "
                    f"std_dev={_fmt(scalar_stats['std_dev'])}, n={scalar_stats['n']}"
                )

            print("    Boolean criteria")
            for criterion in sorted(boolean_criteria):
                criterion_key = f"criterion_{criterion}"
                bool_stats = _stats(values_by_judge[judge][version][criterion_key])
                yes_rate = bool_stats["mean"]
                metrics[judge][version]["boolean"][criterion_key] = {
                    "yes_rate": yes_rate,
                    "std_dev": bool_stats["std_dev"],
                    "n": bool_stats["n"],
                }

                yes_rate_pct = "n/a" if yes_rate is None else f"{yes_rate * 100:.2f}%"
                print(
                    f"      {criterion_key}: yes_rate={_fmt(yes_rate)} ({yes_rate_pct}), "
                    f"std_dev={_fmt(bool_stats['std_dev'])}, n={bool_stats['n']}"
                )

    return metrics

def create_annotator_set(model_answer_path):
    with open(model_answer_path, "r") as f:
        payload = json.load(f)

    selected_v0 = payload.get("selected_responses_v0", [])
    selected_v1 = payload.get("selected_responses_v1", [])

    if len(selected_v0) != len(selected_v1):
        raise ValueError(
            "selected_responses_v0 and selected_responses_v1 must have the same length "
            f"(got {len(selected_v0)} and {len(selected_v1)})"
        )

    base_path, ext = os.path.splitext(model_answer_path)
    if not ext:
        ext = ".json"

    annotator_output_path = f"{base_path}_annotator{ext}"
    trace_output_path = f"{base_path}_annotator_trace{ext}"

    rng = random.Random(42)

    def _extract_answer(item):
        answer = item.get("answer")
        if isinstance(answer, str) and answer:
            return answer

        response = item.get("response")
        if isinstance(response, str) and response:
            _, extracted_answer = split_reasoning_traces(response, reasoning_token="</think>")
            return extracted_answer

        return ""

    annotator_pairs = []
    trace_rows = []

    for idx, (item_v0, item_v1) in enumerate(zip(selected_v0, selected_v1)):
        pair_id = f"pair_{idx:04d}"

        pair_items = [("v0", item_v0), ("v1", item_v1)]
        if rng.random() < 0.5:
            pair_items.reverse()

        pair_payload = {
            "pair_id": pair_id,
        }

        for slot, (source_split, source_item) in zip(("a", "b"), pair_items):
            item_id = f"{pair_id}_{slot}"
            anonymized_item = {
                "id": item_id,
                "prompt": source_item.get("prompt", ""),
                "answer": _extract_answer(source_item),
                "criterion_1": None,
                "justification_1": None,
                "criterion_2": None,
                "justification_2": None,
                "criterion_3": None,
                "justification_3": None,
                "criterion_4": None,
                "justification_4": None,
            }

            pair_payload[slot] = anonymized_item

            trace_rows.append(
                {
                    "id": item_id,
                    "pair_id": pair_id,
                    "slot": slot,
                    "source_split": source_split,
                    "source_index": idx,
                    "full_example": source_item,
                }
            )

        annotator_pairs.append(pair_payload)

    annotator_payload = {
        "paired_responses": annotator_pairs,
        "num_pairs": len(annotator_pairs),
    }

    trace_payload = {
        "source_file": model_answer_path,
        "random_seed": 42,
        "mapping": trace_rows,
    }

    with open(annotator_output_path, "w") as f:
        json.dump(annotator_payload, f, indent=4)

    with open(trace_output_path, "w") as f:
        json.dump(trace_payload, f, indent=4)

    return {
        "annotator_output_path": annotator_output_path,
        "trace_output_path": trace_output_path,
        "num_pairs": len(annotator_pairs),
    }


def merge_annotator_with_trace(annotator_path, trace_path, output_path=None):
    """
    Merge annotator labels back to source metadata using the trace file.
    """
    with open(annotator_path, "r") as f:
        annotator_payload = json.load(f)

    with open(trace_path, "r") as f:
        trace_payload = json.load(f)

    pairs = annotator_payload.get("paired_responses", [])
    trace_mapping = trace_payload.get("mapping", [])
    trace_by_id = {row.get("id"): row for row in trace_mapping}

    merged_rows = []

    for pair in pairs:
        for slot in ("a", "b"):
            item = pair.get(slot, {})
            item_id = item.get("id")

            if item_id not in trace_by_id:
                raise ValueError(f"Missing trace mapping for annotator item id: {item_id}")

            trace_row = trace_by_id[item_id]

            merged_rows.append(
                {
                    "source_index": trace_row.get("source_index"),
                    "source_split": trace_row.get("source_split"),
                    "full_example": trace_row.get("full_example"),
                    "criteria_1": item.get("criteria_1", item.get("criterion_1")),
                    "justification_1": item.get("justification_1"),
                    "criteria_2": item.get("criteria_2", item.get("criterion_2")),
                    "justification_2": item.get("justification_2"),
                    "criteria_3": item.get("criteria_3", item.get("criterion_3")),
                    "justification_3": item.get("justification_3"),
                    "criteria_4": item.get("criteria_4", item.get("criterion_4")),
                    "justification_4": item.get("justification_4"),
                }
            )

    output_payload = {
        "source_file": trace_payload.get("source_file"),
        "num_rows": len(merged_rows),
        "rows": merged_rows,
    }

    if output_path is None:
        base_path, ext = os.path.splitext(annotator_path)
        if not ext:
            ext = ".json"
        output_path = f"{base_path}_merged{ext}"

    with open(output_path, "w") as f:
        json.dump(output_payload, f, indent=4)

    return {
        "output_path": output_path,
        "num_rows": len(merged_rows),
    }


if __name__ == "__main__":
    judgment_files = [
        "03_evaluates/output/glm-4.7_judgements_v0.json",
        "03_evaluates/output/glm-4.7_judgements_v1.json",
        "03_evaluates/output/mistral-large-3-675b-instruct-2512_judgements_v0.json",
        "03_evaluates/output/mistral-large-3-675b-instruct-2512_judgements_v1.json",
        "03_evaluates/output/qwen3.5-397b-a17b_judgements_v0.json",
        "03_evaluates/output/qwen3.5-397b-a17b_judgements_v1.json",
    ]

    compare_all_names = [
        "glm-4.7_v0",
        "glm-4.7_v1",
        "mistral-large-3-675b-instruct_v0",
        "mistral-large-3-675b-instruct_v1",
        "qwen3.5-397b-a17b_v0",
        "qwen3.5-397b-a17b_v1",
    ]
    compare_all_files = judgment_files
    compare_all_output = "03_evaluates/output/compare_all_judges.json"
    # aggregate_criteria_by_model(compare_all_files, compare_all_names, compare_all_output, include_model_answers=True)
    
    # compute_and_print_judge_metrics(compare_all_output)

    selected_answers_path = "03_evaluates/output/selected_responses.json"

    create_annotator_set(selected_answers_path)

    merge_annotator_with_trace(
        annotator_path="03_evaluates/output/selected_responses_annotator.json",
        trace_path="03_evaluates/output/selected_responses_annotator_trace.json",
        output_path="03_evaluates/output/selected_responses_annotator_merged.json",
    )

    # for file in judgment_files:
    #     append_criteria_to_judgements_json(file)

    # compare_v_glm = [
    #     "03_evaluates/output/glm-4.7_judgements_v0.json",
    #     "03_evaluates/output/glm-4.7_judgements_v1.json",
    # ]
    # compare_v_glm_names = ["glm-4.7_v0", "glm-4.7_v1"]
    # aggregate_criteria_by_model(compare_v_glm, compare_v_glm_names, "03_evaluates/output/compare_v_glm-4.7.json")


    # compare_v_mistral = [
    #     "03_evaluates/output/mistral-large-3-675b-instruct-2512_judgements_v0.json",
    #     "03_evaluates/output/mistral-large-3-675b-instruct-2512_judgements_v1.json",
    # ] 
    # compare_v_mistral_names = ["mistral-large-3-675b-instruct_v0", "mistral-large-3-675b-instruct_v1"]
    # aggregate_criteria_by_model(compare_v_mistral, compare_v_mistral_names, "03_evaluates/output/compare_v_mistral-large-3-675b-instruct.json")


    # compare_v_qwen = [
    #     "03_evaluates/output/qwen3.5-397b-a17b_judgements_v0.json",
    #     "03_evaluates/output/qwen3.5-397b-a17b_judgements_v1.json",
    # ]
    # compare_v_qwen_names = ["qwen3.5-397b-a17b_v0", "qwen3.5-397b-a17b_v1"]
    # aggregate_criteria_by_model(compare_v_qwen, compare_v_qwen_names, "03_evaluates/output/compare_v_qwen3.5-397b-a17b.json")

