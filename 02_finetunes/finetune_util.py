import json
from tqdm import tqdm

def is_peft_model(model):
    is_peft = isinstance(model.active_adapters, list) and len(model.active_adapters) > 0
    try:
        is_peft = is_peft or len(model.active_adapters()) > 0
    except Exception as e:
        print(f"Error checking if model is PEFT: {e}, continuing...")
        pass
    return is_peft

def load_jsonl(file_id):
    with open(file_id, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]

def generate_conversation(rows, think_start="<think>", think_end="</think>"):
    """Convert rows with 'prompt'/'answer' fields to 'messages' format.

    If a row has a non-empty 'reasoning_language' field, it is wrapped with
    think_start/think_end tokens and prepended to the assistant answer.
    """
    converted = []

    for r in tqdm(rows, desc="Generating conversations"):
        if 'prompt' in r and 'answer' in r:
            answer = r["answer"]
            reasoning = r.get("reasoning_language") or r.get("reasoning")
            if reasoning:
                answer = f"{think_start}\n{reasoning}\n{think_end}\n{answer}"
            else:
                answer = f"{think_start}\n{think_end}\n{answer}"
            messages = [
                {"role": "user", "content": r["prompt"]},
                {"role": "assistant", "content": answer},
            ]
            converted.append(dict(messages=messages))
        elif 'messages' in r:
            converted.append(dict(messages=r['messages']))
        else:
            raise ValueError(f"Row must have either 'prompt'+'answer' or 'messages' keys, got: {list(r.keys())}")
    return {"conversations": converted}

def formatting_prompts_func(samples, tokenizer, verbose=False):
    convos = samples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }
    # [Keeping this for know if we need to debug]
    # texts = []
    # for convo in tqdm(samples, desc="Formatting prompts"):
    #     try:
    #         text = tokenizer.apply_chat_template(
    #             convo,
    #             tokenize = False,
    #             add_generation_prompt = False,
    #         )
    #         if verbose:
    #             print(text)
    #             print("=="*42)
    #         texts.append(text)
    #     except Exception as e:
    #         print("Error processing convo:", convo)
    #         print("Exception:", e)