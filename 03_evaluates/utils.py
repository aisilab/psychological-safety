from typing import Tuple
import re
import json

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
    aggregate_criteria_by_model(compare_all_files, compare_all_names, "03_evaluates/output/compare_all_judges.json", include_model_answers=True)

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


