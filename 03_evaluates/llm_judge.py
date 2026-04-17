import requests
import os
from dotenv import load_dotenv
import json
import random
from openai import OpenAI

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from utils import split_reasoning_traces

load_dotenv(os.getcwd() + '/03_evaluates/envs.env')
CHAT_AI_TOKEN = os.getenv("CHAT_AI_TOKEN")

def save_json_response(response, filename = "03_evaluates/output/response.json"):
    with open(filename, 'w') as f:
        if isinstance(response, dict):
            json.dump(response, f, indent=4)
        else:
            json.dump(response.json(), f, indent=4)

def load_json_payload(json_filename):
    with open(json_filename, 'r') as f:
        raw_text = f.read().strip()

    if not raw_text:
        raise ValueError(f"JSON file is empty: {json_filename}")

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        # Fallback for files accidentally written as multiple concatenated JSON objects.
        decoder = json.JSONDecoder()
        idx = 0
        last_obj = None

        while idx < len(raw_text):
            while idx < len(raw_text) and raw_text[idx].isspace():
                idx += 1

            if idx >= len(raw_text):
                break

            obj, next_idx = decoder.raw_decode(raw_text, idx)
            last_obj = obj
            idx = next_idx

        if last_obj is None:
            raise ValueError(f"Could not parse JSON content from {json_filename}")

        return last_obj

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

def create_chat_completion(model_id, messages, temperature=0.0, filename="03_evaluates/output/chat_completion_response.json", save_response=True, print_response=False):
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

    response = requests.post(url, headers=headers, json=data, timeout=120)

    if print_response:
        print(response.status_code)
    response_payload = response.json()
    if print_response:
        print(response_payload)

    if filename and save_response:
        save_json_response(response_payload, filename=filename)

    response.raise_for_status()
    return response_payload

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


def judge_baseline(model_id, temperature, test = False):
    if test:
        selected_response_path = "03_evaluates/output/test_requests/test_selected_responses.json"
    else:
        selected_response_path = "03_evaluates/output/selected_responses.json"

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
        # print(response_v0['answer'])
        # exit()
        model_answers_v1.append(response_v1['answer'])
        model_answers_v0.append(response_v0['answer'])

    judgments_v1 = []
    judgments_v0 = []

    judge_model_id = model_id

    paired_answers = list(zip(model_answers_v1, model_answers_v0))
    for idx, (model_answer_v1, model_answer_v0) in enumerate(
        tqdm(paired_answers, total=len(paired_answers), desc="Judging responses")
    ):
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
                filename=f"03_evaluates/output/completions_v1/{idx}_v1_judge_completion.json"
            )

            judge_response_v0 = create_chat_completion(
                judge_model_id,
                messages_v0,
                temperature=temperature,
                filename=f"03_evaluates/output/completions_v0/{idx}_v0_judge_completion.json"
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

    save_json_response(
        {
            "judge_model_id": judge_model_id,
            "criteria_file": "03_evaluates/CRITERIA_llm.md",
            "num_answers": len(model_answers_v1),
            "judgments": judgments_v1,
        },
        filename="03_evaluates/output/judgements_v1.json",
    )

    save_json_response(
        {
            "judge_model_id": judge_model_id,
            "criteria_file": "03_evaluates/CRITERIA_llm.md",
            "num_answers": len(model_answers_v0),
            "judgments": judgments_v0,
        },
        filename="03_evaluates/output/judgements_v0.json",
    )

    extract_markdown_judgements_from_json(json_filename="03_evaluates/output/judgements_v1.json")
    extract_markdown_judgements_from_json(json_filename="03_evaluates/output/judgements_v0.json")

    return judgments_v0, judgments_v1
    

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

def main():
    # get_models()

    # model_id = "qwen3-32b"
    # model_id = "qwen3.5-27b"

    # model_id = "glm-4.7"

    model_id = "qwen3.5-397b-a17b"

    # model_id = "mistral-large-3-675b-instruct-2512"

    # test_completion(model_id)

    # select_data_to_test_judges()

    judge_baseline(model_id=model_id, temperature=0.6, test=False)
    
    # judgements_v1 = "03_evaluates/output/judgements_v1.json"
    # judgements_v0 = "03_evaluates/output/judgements_v0.json"
    # extract_markdown_judgements_from_json(json_filename=judgements_v1)
    # extract_markdown_judgements_from_json(json_filename=judgements_v0)




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