import requests
import os
from dotenv import load_dotenv
import json
import random
import time
from openai import OpenAI
from utils import append_criteria_to_judgements_json

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from utils import split_reasoning_traces

load_dotenv(os.getcwd() + '/03_evaluates/envs.env')
CHAT_AI_TOKEN = os.getenv("CHAT_AI_TOKEN")
CHAT_AI_TIMEOUT_SECONDS = float(os.getenv("CHAT_AI_TIMEOUT_SECONDS", "1000"))
CHAT_AI_MAX_RETRIES = int(os.getenv("CHAT_AI_MAX_RETRIES", "50"))
CHAT_AI_RETRY_BASE_SECONDS = float(os.getenv("CHAT_AI_RETRY_BASE_SECONDS", "1.0"))

def save_json_response(response, filename = "03_evaluates/output/response.json"):
    with open(filename, 'w') as f:
        if isinstance(response, dict):
            json.dump(response, f, indent=4)
        else:
            json.dump(response.json(), f, indent=4)

def extract_markdown_judgements_from_json(
    json_filename,
    markdown_filename=None,
):
    if markdown_filename is None:
        markdown_filename = json_filename.rsplit(".", 1)[0] + ".md"

    with open(json_filename, 'r') as f:
        payload = json.load(f)

    judge_model_id = payload.get("judge_model_id", "")
    judgements = payload.get("judgments", [])

    lines = [f"## Model: {judge_model_id}", ""]

    for item in judgements:
        prompt = item.get("prompt", "")
        reasoning = item.get("reasoning", "")
        judgement = item.get("judge_answer") or item.get("judgment", "")
        risk_cluster = item.get("risk cluster", "")
        risk_category = item.get("risk category", "")
        model_answer = item.get("model_answer", "")

        if item.get("error"):
            judgement = f"ERROR: {item['error']}"

        lines.extend([
            f"### Prompt: {prompt}",
            "",
            "### Model Answer",
            f"{model_answer}",
            "",
            "### Reasoning",
            f"{reasoning}",
            "",
            "### Judgement",
            f"{judgement}",
            "",
            f"### Risk Cluster: {risk_cluster}",
            f"### Risk Category: {risk_category}",
            "",
            "---",
            "",
        ])

    with open(markdown_filename, "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")

def get_models():
    url = "https://chat-ai.academiccloud.de/v1/models"

    headers = {
        "Authorization": f"Bearer {CHAT_AI_TOKEN}"
    }

    response = requests.get(url, headers=headers)

    print(response.status_code)
    print(response.json())

    save_json_response(response)

def create_chat_completion(
    model_id,
    messages,
    temperature=0.1,
    filename="03_evaluates/output/chat_completion_response.json",
    save_response=True,
    print_response=True,
    timeout_seconds=None,
):
    if not model_id:
        raise ValueError("model_id must be set for create_chat_completion")

    url = "https://chat-ai.academiccloud.de/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {CHAT_AI_TOKEN}",
        "Content-Type": "application/json"
    }

    data = {
        "model": f"{model_id}",
        "messages": messages,
        "temperature": temperature,
    }

    request_timeout = CHAT_AI_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    max_attempts = max(1, CHAT_AI_MAX_RETRIES)
    retriable_status_codes = {429, 500, 502, 503, 504}
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=request_timeout)

            if print_response:
                print(response.status_code)

            if response.status_code >= 400:
                body_preview = (response.text or "").strip()[:500]
                is_retriable = response.status_code in retriable_status_codes
                error = requests.HTTPError(
                    f"{response.status_code} error for url: {url}. "
                    f"Response body (first 500 chars): {body_preview}"
                )

                if is_retriable and attempt < max_attempts:
                    sleep_seconds = CHAT_AI_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                    sleep_seconds += random.uniform(0, 0.3)
                    if print_response:
                        print(f"Transient HTTP {response.status_code}, retrying in {sleep_seconds:.2f}s (attempt {attempt}/{max_attempts})")
                    time.sleep(sleep_seconds)
                    continue

                raise error

            try:
                response_payload = response.json()
            except ValueError as exc:
                body_preview = (response.text or "").strip()[:500]
                is_retriable = response.status_code in retriable_status_codes or not body_preview
                error = ValueError(
                    f"Chat completion returned a non-JSON or empty response body. "
                    f"Status={response.status_code}, body (first 500 chars): {body_preview}"
                )
                if is_retriable and attempt < max_attempts:
                    sleep_seconds = CHAT_AI_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                    sleep_seconds += random.uniform(0, 0.3)
                    if print_response:
                        print(f"Transient parse failure, retrying in {sleep_seconds:.2f}s (attempt {attempt}/{max_attempts})")
                    time.sleep(sleep_seconds)
                    continue
                raise error from exc

            if print_response:
                print(response_payload)

            if filename and save_response:
                save_json_response(response_payload, filename=filename)

            return response_payload

        except (requests.Timeout, requests.ConnectionError, requests.RequestException) as exc:
            last_error = exc
            if attempt >= max_attempts:
                break
            sleep_seconds = CHAT_AI_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
            sleep_seconds += random.uniform(0, 0.3)
            if print_response:
                print(f"Request failed ({type(exc).__name__}), retrying in {sleep_seconds:.2f}s (attempt {attempt}/{max_attempts})")
            time.sleep(sleep_seconds)

    if last_error is not None:
        raise RuntimeError(
            f"Chat completion failed after {max_attempts} attempts."
        ) from last_error
    raise RuntimeError(f"Chat completion failed after {max_attempts} attempts.")

def get_judge_reasoning_and_judgement(judge_response):
    judge_text = judge_response["choices"][0]["message"]["content"]

    if judge_response["choices"][0]["message"]["reasoning"] is not None:
        judge_answer = judge_text
        reasoning = judge_response["choices"][0]["message"]["reasoning"]
    elif "<think>" in judge_text and "</think>" in judge_text:
        reasoning, judge_answer = split_reasoning_traces(judge_text, reasoning_token="</think>")
    else:
        judge_answer = judge_text
        reasoning = None

    return reasoning, judge_answer

def select_data_to_test_judges(sample_per_cluster=5, random_seed=42):
    """
    Take 5 random model answers for each risk cluster (for both prompts v0 and v1) and ask all the judges to evaluate them.
    """

    model_answers_v0_path = "01_prompting/results/svenharms_val_v0_latest_splitted_with_category.json"
    model_answers_v1_path = "01_prompting/results/svenharms_val_v1_latest_splitted_with_category.json"

    with open(model_answers_v0_path, 'r') as f:
        model_answers_v0 = json.load(f)
    with open(model_answers_v1_path, 'r') as f:
        model_answers_v1 = json.load(f)
    
    copy_model_answers_v0 = model_answers_v0.copy()
    copy_model_answers_v1 = model_answers_v1.copy()
    
    # Filter out model answers where the model got stuck in reasoning ("stucked": "true")
    copy_model_answers_v0["results"] = [item for item in model_answers_v0["results"] if item.get("stucked") != "true"]
    copy_model_answers_v1["results"] = [item for item in model_answers_v1["results"] if item.get("stucked") != "true"]

    model_answers_v0 = copy_model_answers_v0
    model_answers_v1 = copy_model_answers_v1

    clusters = ["crimes (sexual)", "suicide and selfharm", "substance", "weapon", "violence"]

    # select randomly 5 answers (both v0 and v1) for each category
    selected_responses_v0 = []
    selected_responses_v1 = []
    rng = random.Random(random_seed)

    for cluster in clusters:
        v0_responses = [item for item in model_answers_v0["results"] if item["risk cluster"] == cluster]
        v1_responses = [item for item in model_answers_v1["results"] if item["risk cluster"] == cluster]

        # Pair by prompt so both splits select the exact same elements.
        v0_by_prompt = {item["prompt"]: item for item in v0_responses if "prompt" in item}
        v1_by_prompt = {item["prompt"]: item for item in v1_responses if "prompt" in item}
        shared_prompts = sorted(set(v0_by_prompt).intersection(v1_by_prompt))

        sample_size = min(sample_per_cluster, len(shared_prompts))
        if sample_size == 0:
            print(f"Skipping cluster '{cluster}' because one split has no samples.")
            continue

        sampled_prompts = rng.sample(shared_prompts, sample_size)
        
        if len(sampled_prompts) != sample_per_cluster:
            raise ValueError(f"Expected to sample {sample_per_cluster} prompts for cluster '{cluster}', but only found {len(sampled_prompts)} shared prompts between v0 and v1.")

        for prompt in sampled_prompts:
            selected_responses_v0.append(v0_by_prompt[prompt])
            selected_responses_v1.append(v1_by_prompt[prompt])
    
    if len(selected_responses_v0) != len(selected_responses_v1):
        raise ValueError("Mismatched number of selected responses between v0 and v1")

    for res0, res1 in zip(selected_responses_v0, selected_responses_v1):
        if res0.get("prompt") != res1.get("prompt"):
            raise ValueError("Mismatched prompt pairing between v0 and v1 samples")
        print(res0["prompt"])
        print()

    output_path = "03_evaluates/output/selected_responses.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "selected_responses_v0": selected_responses_v0,
                "selected_responses_v1": selected_responses_v1,
            },
            f,
            indent=4,
        )

    print(f"Selected {sample_per_cluster} responses for each cluster: {clusters}")
    print(f"Total selected responses: {len(selected_responses_v0)}\n")

    print(f"Saved paired selected responses to {output_path}")

def validate_input_data(v0_path, v1_path):
    """
    Checks that model answers in v0 and v1 are properly paired by prompt and that there are no "stucked" responses.
    If one stuck response is found remove it from both v0 and v1 to keep the pairing consistent.
    """
    with open(v0_path, "r") as f:
        data_v0 = json.load(f)
    with open(v1_path, "r") as f:
        data_v1 = json.load(f)

    results_v0 = data_v0.get("results", [])
    results_v1 = data_v1.get("results", [])
    if len(results_v0) != len(results_v1):
        raise ValueError(
            f"Mismatched number of raw responses between v0 and v1: "
            f"{len(results_v0)} vs {len(results_v1)}"
        )

    cleaned_results_v0 = []
    cleaned_results_v1 = []
    stucked_v1 = 0
    stucked_v0 = 0
    stucked_both = 0

    for idx, (item_v0, item_v1) in enumerate(zip(results_v0, results_v1)):
        prompt_v0 = item_v0.get("prompt")
        prompt_v1 = item_v1.get("prompt")
        if prompt_v0 != prompt_v1:
            raise ValueError(
                f"Mismatched prompt at index {idx}: v0='{prompt_v0}' vs v1='{prompt_v1}'"
            )
        is_stucked_v1 = item_v1.get("stucked") == "true"
        is_stucked_v0 = item_v0.get("stucked") == "true"

        if is_stucked_v0 and is_stucked_v1:
            stucked_both += 1
        if is_stucked_v1 and not is_stucked_v0:
            stucked_v1 += 1
        if is_stucked_v0 and not is_stucked_v1:
            stucked_v0 += 1
        if is_stucked_v0 or is_stucked_v1:
            continue

        cleaned_results_v0.append(item_v0)
        cleaned_results_v1.append(item_v1)

    cleaned_v0 = dict(data_v0)
    cleaned_v1 = dict(data_v1)
    cleaned_v0["results"] = cleaned_results_v0
    cleaned_v1["results"] = cleaned_results_v1

    print(
        f"Validated input data: {len(cleaned_results_v0)} paired responses "
        f"(removed {stucked_v0 + stucked_v1 + stucked_both} pairs because at least one side was 'stucked').\n"
        f"Stucked responses only in v0: {stucked_v0}, stucked responses only in v1: {stucked_v1}, stucked responses in both: {stucked_both}."
    )

    return cleaned_v0, cleaned_v1



def judge_baseline(model_id, temperature, mode = "selected", limit=None):
    if mode == "test":
        selected_response_path = "03_evaluates/output/test_requests/test_selected_responses.json"
    elif mode == "selected":
        selected_response_path = "03_evaluates/output/selected_responses.json"
    elif mode == "full":
        full_response_path_v0 = "01_prompting/results/svenharms_val_v0_latest_splitted_with_category.json"
        full_response_path_v1 = "01_prompting/results/svenharms_val_v1_latest_splitted_with_category.json"
        data_v0, data_v1 = validate_input_data(full_response_path_v0, full_response_path_v1)

        v0_by_prompt = {
            item["prompt"]: item
            for item in data_v0.get("results", [])
            if item.get("stucked") != "true" and "prompt" in item
        }
        v1_by_prompt = {
            item["prompt"]: item
            for item in data_v1.get("results", [])
            if item.get("stucked") != "true" and "prompt" in item
        }

        shared_prompts = sorted(set(v0_by_prompt).intersection(v1_by_prompt))
        selected_responses_v0 = [v0_by_prompt[prompt] for prompt in shared_prompts]
        selected_responses_v1 = [v1_by_prompt[prompt] for prompt in shared_prompts]
    else:
        raise ValueError(f"Unsupported mode '{mode}'. Expected one of: test, selected, full.")
    
    print(f"Loaded {len(selected_responses_v0)} responses for v0 and {len(selected_responses_v1)} responses for v1 in mode '{mode}'")
    if limit is not None:
        print(f"Limit is set to {limit}.")

    if mode != "full":
        with open(selected_response_path, 'r') as f:
            data = json.load(f)
            selected_responses_v1 = data["selected_responses_v1"]
            selected_responses_v0 = data["selected_responses_v0"]

    with open("03_evaluates/CRITERIA_llm.md", 'r') as f:
        criteria = f.read()
    
    # Extract only answers
    model_answers_v1 = []
    model_answers_v0 = []
    for response_v1, response_v0 in zip(selected_responses_v1, selected_responses_v0):
        model_answers_v1.append(response_v1['answer'])
        model_answers_v0.append(response_v0['answer'])

    judgments_v1 = []
    judgments_v0 = []

    judge_model_id = model_id

    paired_answers = list(zip(model_answers_v1, model_answers_v0))
    for idx, (model_answer_v1, model_answer_v0) in enumerate(
        tqdm(paired_answers, total=len(paired_answers), desc="Judging responses")
    ):
        if limit is not None and idx >= limit:
            break
        prompt = selected_responses_v1[idx].get("prompt", "")

        messages_v1 = [
            {
                "role": "system",
                "content": (
                    f"{criteria}"
                )
            },
            {
                "role": "user",
                "content": (
                    f"User Request:\n{prompt}\n\n"
                    f"Assistant Response:\n{model_answer_v1}\n\n"
                    "Evaluate this assistant response now following the specified criteria and output format."
                )
            }
        ]

        messages_v0 = [
            {
                "role": "system",
                "content": (
                    f"{criteria}"
                )
            },
            {
                "role": "user",
                "content": (
                    f"User Request:\n{prompt}\n\n"
                    f"Assistant Response:\n{model_answer_v0}\n\n"
                    "Evaluate this assistant response now following the specified criteria and output format."
                )
            }
        ]

        try:
            judge_response_v1 = create_chat_completion(
                judge_model_id,
                messages_v1,
                temperature=temperature,
                filename=f"03_evaluates/output/completions_v1/{model_id}_{idx}_v1_judge_completion_{mode}.json"
            )

            judge_response_v0 = create_chat_completion(
                judge_model_id,
                messages_v0,
                temperature=temperature,
                filename=f"03_evaluates/output/completions_v0/{model_id}_{idx}_v0_judge_completion_{mode}.json"
            )
            
            reasoning_v1, judge_answer_v1 = get_judge_reasoning_and_judgement(judge_response_v1)
            reasoning_v0, judge_answer_v0 = get_judge_reasoning_and_judgement(judge_response_v0)

            full_judge_output_v1 = judge_response_v1["choices"][0]["message"]["content"]
            full_judge_output_v0 = judge_response_v0["choices"][0]["message"]["content"]

            if judge_response_v1["choices"][0]["message"]["reasoning"] is not None:
                judge_answer_v1 = full_judge_output_v1
                reasoning_v1 = judge_response_v1["choices"][0]["message"]["reasoning"]
            elif "<think>" in full_judge_output_v1 and "</think>" in full_judge_output_v1:
                reasoning_v1, judge_answer_v1 = split_reasoning_traces(full_judge_output_v1, reasoning_token="</think>")
            else:
                judge_answer_v1 = full_judge_output_v1
                reasoning_v1 = None

            judgments_v1.append({
                "id": idx,
                "prompt": prompt,
                "model_answer": model_answer_v1,
                "full_judge_output": full_judge_output_v1,
                "reasoning": reasoning_v1,
                "judge_answer": judge_answer_v1,
                "risk cluster": selected_responses_v1[idx].get("risk cluster", ""),
                "risk category": selected_responses_v1[idx].get("risk category", ""),
            })

            judgments_v0.append({
                "id": idx,
                "prompt": prompt,
                "model_answer": model_answer_v0,
                "full_judge_output": full_judge_output_v0,
                "reasoning": reasoning_v0,
                "judge_answer": judge_answer_v0,
                "risk cluster": selected_responses_v0[idx].get("risk cluster", ""),
                "risk category": selected_responses_v0[idx].get("risk category", ""),
            })

        except Exception as exc:
            judgments_v1.append({
                "id": idx,
                "prompt": prompt,
                "model_answer": model_answer_v1,
                "error": str(exc),
            })

    judgements_v1_path = f"03_evaluates/output/{model_id}_judgements_v1_{mode}.json"
    judgements_v0_path = f"03_evaluates/output/{model_id}_judgements_v0_{mode}.json"

    save_json_response(
        {
            "judge_model_id": judge_model_id,
            "criteria_file": "03_evaluates/CRITERIA_llm.md",
            "num_answers": len(model_answers_v1),
            "judgments": judgments_v1,
        },
        filename=judgements_v1_path,
    )

    save_json_response(
        {
            "judge_model_id": judge_model_id,
            "criteria_file": "03_evaluates/CRITERIA_llm.md",
            "num_answers": len(model_answers_v0),
            "judgments": judgments_v0,
        },
        filename=judgements_v0_path,
    )

    extract_markdown_judgements_from_json(json_filename=judgements_v1_path)
    extract_markdown_judgements_from_json(json_filename=judgements_v0_path)

    return judgments_v0, judgments_v1

def judge_single_response(model_id, prompt, model_answer, risk_category, risk_cluster, temperature=0.1):
    # Run a single judge completion and persist results in the same schema as judge_baseline.
    with open("03_evaluates/CRITERIA_llm.md", "r") as f:
        criteria = f.read()

    os.makedirs("03_evaluates/output/completions_single", exist_ok=True)

    messages = [
        {
            "role": "system",
            "content": f"{criteria}",
        },
        {
            "role": "user",
            "content": (
                f"User Request:\n{prompt}\n\n"
                f"Assistant Response:\n{model_answer}\n\n"
                "Evaluate this assistant response now following the specified criteria and output format."
            ),
        },
    ]

    judgements_path = f"03_evaluates/output/{model_id}_judgements_single.json"
    completion_path = f"03_evaluates/output/completions_single/{model_id}_single_judge_completion.json"

    judge_response = create_chat_completion(
        model_id,
        messages,
        temperature=temperature,
        filename=completion_path,
    )
    reasoning, judge_answer = get_judge_reasoning_and_judgement(judge_response)
    full_judge_output = judge_response["choices"][0]["message"]["content"]

    judgments = [
        {
            "id": 0,
            "prompt": prompt,
            "model_answer": model_answer,
            "full_judge_output": full_judge_output,
            "reasoning": reasoning,
            "judge_answer": judge_answer,
            "risk cluster": risk_cluster,
            "risk category": risk_category,
        }
    ]

    save_json_response(
        {
            "judge_model_id": model_id,
            "criteria_file": "03_evaluates/CRITERIA_llm.md",
            "num_answers": 1,
            "judgments": judgments,
        },
        filename=judgements_path,
    )

    append_criteria_to_judgements_json(json_filename=judgements_path, output_filename=judgements_path)
    extract_markdown_judgements_from_json(json_filename=judgements_path)

    return judgments[0]
    

def test_completion(model_id):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Who are you?"
        }
    ]

    # messages = [
    #     {
    #         "role": "system",
    #         "content": "You are a helpful assistant. Always reason before answering."
    #     },
    #     {
    #         "role": "user",
    #         "content": "What's the square root of 1725?"
    #     }
    # ]
    
    output_file = f"03_evaluates/output/test_requests/test_completion_{model_id}.json"

    create_chat_completion(model_id, messages, temperature=0.1, filename=output_file)
    # create_chat_completion_2(model_id, messages, temperature=1.0, filename=output_file)

def extract_failed_requests(json_filename):
    with open(json_filename, 'r') as f:
        payload = json.load(f)

    judgements = payload.get("judgments", [])
    criteria_file = payload.get("criteria_file")

    full_metadata_by_id = {}
    if "_full" in json_filename:
        full_response_path_v0 = "01_prompting/results/svenharms_val_v0_latest_splitted_with_category.json"
        full_response_path_v1 = "01_prompting/results/svenharms_val_v1_latest_splitted_with_category.json"
        data_v0, data_v1 = validate_input_data(full_response_path_v0, full_response_path_v1)

        v0_by_prompt = {
            item["prompt"]: item
            for item in data_v0.get("results", [])
            if item.get("stucked") != "true" and "prompt" in item
        }
        v1_by_prompt = {
            item["prompt"]: item
            for item in data_v1.get("results", [])
            if item.get("stucked") != "true" and "prompt" in item
        }
        shared_prompts = sorted(set(v0_by_prompt).intersection(v1_by_prompt))
        selected_responses_v0 = [v0_by_prompt[prompt] for prompt in shared_prompts]
        selected_responses_v1 = [v1_by_prompt[prompt] for prompt in shared_prompts]

        is_v1 = "_v1_" in json_filename
        selected_responses = selected_responses_v1 if is_v1 else selected_responses_v0

        full_metadata_by_id = {
            idx: {
                "risk cluster": item.get("risk cluster"),
                "risk category": item.get("risk category"),
            }
            for idx, item in enumerate(selected_responses)
        }

    failed_requests = []
    for item in judgements:
        if item.get("error"):
            req_id = item.get("id")
            fallback_meta = full_metadata_by_id.get(req_id, {})
            failed_requests.append({
                "id": req_id,
                "prompt": item.get("prompt"),
                "model_answer": item.get("model_answer"),
                "error": item.get("error"),
                "risk cluster": item.get("risk cluster") or fallback_meta.get("risk cluster"),
                "risk category": item.get("risk category") or fallback_meta.get("risk category"),
                "criterion_1": item.get("criterion_1"),
                "criterion_2": item.get("criterion_2"),
                "criterion_3": item.get("criterion_3"),
                "criterion_4": item.get("criterion_4"),
                "criteria_file": criteria_file,
            })

    return failed_requests


def validate_judges():
    select_data_to_test_judges()

    selected_judges = [
        "glm-4.7",
        "mistral-large-3-675b-instruct-2512",
        "qwen3.5-397b-a17b",
    ]

    for model_id in tqdm(selected_judges, desc="Judging models"):
        judge_baseline(model_id=model_id, temperature=0.1, mode="selected")
        judgements_v1 = f"03_evaluates/output/{model_id}_judgements_v1.json"
        judgements_v0 = f"03_evaluates/output/{model_id}_judgements_v0.json"
        append_criteria_to_judgements_json(json_filename=judgements_v1, output_filename=judgements_v1)
        append_criteria_to_judgements_json(json_filename=judgements_v0, output_filename=judgements_v0)
        extract_markdown_judgements_from_json(json_filename=judgements_v1)
        extract_markdown_judgements_from_json(json_filename=judgements_v0)

def run_judge_on_full_data(judge_model_id, temperature, limit=None):
    judge_baseline(model_id=judge_model_id, temperature=temperature, mode="full", limit=limit)
    judgements_v1 = f"03_evaluates/output/{judge_model_id}_judgements_v1_full.json"
    judgements_v0 = f"03_evaluates/output/{judge_model_id}_judgements_v0_full.json"
    append_criteria_to_judgements_json(json_filename=judgements_v1, output_filename=judgements_v1)
    append_criteria_to_judgements_json(json_filename=judgements_v0, output_filename=judgements_v0)
    extract_markdown_judgements_from_json(json_filename=judgements_v1)
    extract_markdown_judgements_from_json(json_filename=judgements_v0)

def rerun_judges_for_failed_http_requests(judge_model_id, temperature):
    failed_requests_v1 = extract_failed_requests(f"03_evaluates/output/{judge_model_id}_judgements_v1_full.json")
    failed_requests_v0 = extract_failed_requests(f"03_evaluates/output/{judge_model_id}_judgements_v0_full.json")

    # print failed requests for manual inspection
    print(f"Failed requests for v1 ({len(failed_requests_v1)}):")
    for req in failed_requests_v1:
        print(f"ID: {req['id']}, Prompt: {req['prompt']}, Error: {req['error']}")
        print()
    print(f"Failed requests for v0 ({len(failed_requests_v0)}):")
    for req in failed_requests_v0:
        print(f"ID: {req['id']}, Prompt: {req['prompt']}, Error: {req['error']}")
        print()
    
    print("\n=== Full preview of v1 requests ===")
    for req in failed_requests_v1:
        print(
            f"\n[Preview v1] ID: {req.get('id')}\n"
            f"Prompt:\n{req.get('prompt', '')}\n\n"
            f"Model answer:\n{req.get('model_answer', '')}\n"
        )

    print("\n=== Full preview of v0 requests ===")
    for req in failed_requests_v0:
        print(
            f"\n[Preview v0] ID: {req.get('id')}\n"
            f"Prompt:\n{req.get('prompt', '')}\n\n"
            f"Model answer:\n{req.get('model_answer', '')}\n"
        )

    input("Press Enter to send all requests...")

    with open("03_evaluates/CRITERIA_llm.md", 'r') as f:
        criteria = f.read()
    
    os.makedirs("03_evaluates/output/completions_v1", exist_ok=True)
    os.makedirs("03_evaluates/output/completions_v0", exist_ok=True)

    judgments_v1 = []
    judgments_v0 = []

    for req in tqdm(failed_requests_v1, desc="Retrying failed requests v1"):
        prompt = req.get("prompt", "")
        model_answer = req.get("model_answer", "")
        req_id = req.get("id")
        messages_v1 = [
            {
                "role": "system",
                "content": f"{criteria}",
            },
            {
                "role": "user",
                "content": (
                    f"User Request:\n{prompt}\n\n"
                    f"Assistant Response:\n{model_answer}\n\n"
                    "Evaluate this assistant response now following the specified criteria and output format."
                ),
            },
        ]
        try:
            judge_response_v1 = create_chat_completion(
                judge_model_id,
                messages_v1,
                temperature=temperature,
                filename=f"03_evaluates/output/completions_v1/{judge_model_id}_{req_id}_v1_judge_completion_retries.json",
            )
            reasoning_v1, judge_answer_v1 = get_judge_reasoning_and_judgement(judge_response_v1)
            full_judge_output_v1 = judge_response_v1["choices"][0]["message"]["content"]
            judgments_v1.append({
                "id": req_id,
                "prompt": prompt,
                "model_answer": model_answer,
                "full_judge_output": full_judge_output_v1,
                "reasoning": reasoning_v1,
                "judge_answer": judge_answer_v1,
                "risk cluster": req.get("risk cluster", ""),
                "risk category": req.get("risk category", ""),
            })
            print(f"Successfully retried request ID {req_id} for v1.")
        except Exception as exc:
            print(f"Failed again to process request ID {req_id} for v1. Error: {exc}")
            judgments_v1.append({
                "id": req_id,
                "prompt": prompt,
                "model_answer": model_answer,
                "error": str(exc),
                "risk cluster": req.get("risk cluster", ""),
                "risk category": req.get("risk category", ""),
                "criterion_1": req.get("criterion_1"),
                "criterion_2": req.get("criterion_2"),
                "criterion_3": req.get("criterion_3"),
                "criterion_4": req.get("criterion_4"),
            })

    for req in tqdm(failed_requests_v0, desc="Retrying failed requests v0"):
        prompt = req.get("prompt", "")
        model_answer = req.get("model_answer", "")
        req_id = req.get("id")
        messages_v0 = [
            {
                "role": "system",
                "content": f"{criteria}",
            },
            {
                "role": "user",
                "content": (
                    f"User Request:\n{prompt}\n\n"
                    f"Assistant Response:\n{model_answer}\n\n"
                    "Evaluate this assistant response now following the specified criteria and output format."
                ),
            },
        ]
        try:
            judge_response_v0 = create_chat_completion(
                judge_model_id,
                messages_v0,
                temperature=temperature,
                filename=f"03_evaluates/output/completions_v0/{judge_model_id}_{req_id}_v0_judge_completion_retries.json",
            )
            reasoning_v0, judge_answer_v0 = get_judge_reasoning_and_judgement(judge_response_v0)
            full_judge_output_v0 = judge_response_v0["choices"][0]["message"]["content"]
            judgments_v0.append({
                "id": req_id,
                "prompt": prompt,
                "model_answer": model_answer,
                "full_judge_output": full_judge_output_v0,
                "reasoning": reasoning_v0,
                "judge_answer": judge_answer_v0,
                "risk cluster": req.get("risk cluster", ""),
                "risk category": req.get("risk category", ""),
            })
            print(f"Successfully retried request ID {req_id} for v0.")
        except Exception as exc:
            print(f"Failed again to process request ID {req_id} for v0. Error: {exc}")
            judgments_v0.append({
                "id": req_id,
                "prompt": prompt,
                "model_answer": model_answer,
                "error": str(exc),
                "risk cluster": req.get("risk cluster", ""),
                "risk category": req.get("risk category", ""),
                "criterion_1": req.get("criterion_1"),
                "criterion_2": req.get("criterion_2"),
                "criterion_3": req.get("criterion_3"),
                "criterion_4": req.get("criterion_4"),
            })

    judgements_v1_path = f"03_evaluates/output/{judge_model_id}_judgements_v1_retries.json"
    judgements_v0_path = f"03_evaluates/output/{judge_model_id}_judgements_v0_retries.json"

    save_json_response(
        {
            "judge_model_id": judge_model_id,
            "criteria_file": "03_evaluates/CRITERIA_llm.md",
            "num_answers": len(failed_requests_v1),
            "judgments": judgments_v1,
        },
        filename=judgements_v1_path,
    )

    save_json_response(
        {
            "judge_model_id": judge_model_id,
            "criteria_file": "03_evaluates/CRITERIA_llm.md",
            "num_answers": len(failed_requests_v0),
            "judgments": judgments_v0,
        },
        filename=judgements_v0_path,
    )

    append_criteria_to_judgements_json(json_filename=judgements_v1_path, output_filename=judgements_v1_path)
    append_criteria_to_judgements_json(json_filename=judgements_v0_path, output_filename=judgements_v0_path)

    extract_markdown_judgements_from_json(json_filename=judgements_v1_path)
    extract_markdown_judgements_from_json(json_filename=judgements_v0_path)

    return judgments_v0, judgments_v1

# TODO remove this temp function
def rerun_judges_for_v0_requests_from_file(
    judge_model_id,
    temperature,
    requests_path="03_evaluates/output/temp_v0_retries.json",
):
    with open(requests_path, "r") as f:
        loaded = json.load(f)

    if isinstance(loaded, dict):
        v0_requests = loaded.get("judgments", [])
    else:
        v0_requests = loaded

    print(f"Loaded {len(v0_requests)} v0 requests from {requests_path}.")
    print("\n=== Full preview of selected v0 requests ===")
    for req in v0_requests:
        print(
            f"\n[Preview v0] ID: {req.get('id')}\n"
            f"Prompt:\n{req.get('prompt', '')}\n\n"
            f"Model answer:\n{req.get('model_answer', '')}\n"
        )
    input("Press Enter to send all selected v0 requests...")

    with open("03_evaluates/CRITERIA_llm.md", "r") as f:
        criteria = f.read()

    os.makedirs("03_evaluates/output/completions_v0", exist_ok=True)
    judgments_v0 = []

    for req in tqdm(v0_requests, desc="Retrying selected v0 requests"):
        prompt = req.get("prompt", "")
        model_answer = req.get("model_answer", "")
        req_id = req.get("id")
        messages_v0 = [
            {
                "role": "system",
                "content": f"{criteria}",
            },
            {
                "role": "user",
                "content": (
                    f"User Request:\n{prompt}\n\n"
                    f"Assistant Response:\n{model_answer}\n\n"
                    "Evaluate this assistant response now following the specified criteria and output format."
                ),
            },
        ]

        try:
            judge_response_v0 = create_chat_completion(
                judge_model_id,
                messages_v0,
                temperature=temperature,
                filename=f"03_evaluates/output/completions_v0/{judge_model_id}_{req_id}_v0_judge_completion_selected_retries.json",
            )
            reasoning_v0, judge_answer_v0 = get_judge_reasoning_and_judgement(judge_response_v0)
            full_judge_output_v0 = judge_response_v0["choices"][0]["message"]["content"]
            judgments_v0.append({
                "id": req_id,
                "prompt": prompt,
                "model_answer": model_answer,
                "full_judge_output": full_judge_output_v0,
                "reasoning": reasoning_v0,
                "judge_answer": judge_answer_v0,
                "risk cluster": req.get("risk cluster", ""),
                "risk category": req.get("risk category", ""),
            })
            print(f"Successfully retried request ID {req_id} for v0.")
        except Exception as exc:
            print(f"Failed again to process request ID {req_id} for v0. Error: {exc}")
            judgments_v0.append({
                "id": req_id,
                "prompt": prompt,
                "model_answer": model_answer,
                "error": str(exc),
                "risk cluster": req.get("risk cluster", ""),
                "risk category": req.get("risk category", ""),
                "criterion_1": req.get("criterion_1"),
                "criterion_2": req.get("criterion_2"),
                "criterion_3": req.get("criterion_3"),
                "criterion_4": req.get("criterion_4"),
            })

    output_path = f"03_evaluates/output/{judge_model_id}_judgements_v0_selected_retries.json"
    save_json_response(
        {
            "judge_model_id": judge_model_id,
            "criteria_file": "03_evaluates/CRITERIA_llm.md",
            "num_answers": len(v0_requests),
            "judgments": judgments_v0,
        },
        filename=output_path,
    )

    append_criteria_to_judgements_json(json_filename=output_path, output_filename=output_path)
    extract_markdown_judgements_from_json(json_filename=output_path)

    return judgments_v0


def run_single_judge_request():
    model_id = "qwen3.5-397b-a17b"
    prompt = ""
    model_answer = ""
    risk_category = ""
    risk_cluster = ""

    judge_single_response(model_id, prompt, model_answer, risk_category, risk_cluster, temperature=0.1)


    
def main():
    judge_model_id = "qwen3.5-397b-a17b"
    # run_judge_on_full_data(judge_model_id="qwen3.5-397b-a17b", temperature=0.1, limit=None)
    # rerun_judges_for_failed_http_requests(judge_model_id=judge_model_id, temperature=0.1)

    rerun_judges_for_v0_requests_from_file(judge_model_id=judge_model_id, temperature=0.1)

    


if __name__ == "__main__":
    main()


""" Unused code

def create_chat_completion_2(model_id, messages, temperature=0.0, filename="03_evaluates/output/chat_completion_response.json"):
    openai_api_key = CHAT_AI_TOKEN
    openai_api_base = "https://chat-ai.academiccloud.de/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    # Thinking ON (default if you omit chat_template_kwargs)
    resp_on = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=temperature,
        extra_body={
            "thinking": {
                "type": "enabled",
                "clear_thinking": True
            }
        }
    )
    print("thinking=on, think content:\n", resp_on.choices[0].message)

    # save_json_response(resp_on, filename=filename)


"""
