"""
Run a HuggingFace model with a system prompt against safety benchmark datasets.

Usage:
    python run_benchmark.py \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --dataset Anthropic/hh-rlhf \
        --dataset-split test \
        --prompt-field chosen \
        --output results.json \
        --max-samples 100

Common safety benchmarks on HuggingFace:
    - Anthropic/hh-rlhf          (field: chosen / rejected)
    - truthful_qa                 (field: question, config: generation)
    - allenai/wildjailbreak       (field: vanilla_prompt)
    - walledai/AdvBench           (field: prompt)
    - LibrAI/do-not-answer        (field: question)
    - LLM-LAT/harmful-dataset     (field: prompt)
    - PKU-Alignment/BeaverTails   (field: prompt, config: 30k_test)
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def load_system_prompt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def extract_prompts(dataset, field: str, max_samples: int | None) -> list[str]:
    """Extract text prompts from a dataset, handling nested fields with dot notation."""
    prompts = []
    for i, row in enumerate(dataset):
        if max_samples and i >= max_samples:
            break
        value = row
        for key in field.split("."):
            value = value[key]
        # Some fields contain full conversation strings; extract the first human turn
        if isinstance(value, str):
            prompts.append(value.strip())
        elif isinstance(value, list):
            # e.g. truthful_qa best_answer is a list
            prompts.append(str(value[0]).strip())
    return prompts


def run_model(
    model_name: str,
    system_prompt: str,
    prompts: list[str],
    max_new_tokens: int,
    device: str,
    batch_size: int,
) -> list[dict]:
    print(f"Loading tokenizer and model: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map="auto" if device == "auto" else device,
    )
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    results = []
    total = len(prompts)

    for i, prompt in enumerate(prompts):
        print(f"  [{i+1}/{total}] Generating...", end="\r", flush=True)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        # Use apply_chat_template if available, else fall back to plain formatting
        try:
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(model.device)
        except Exception:
            text = f"<|system|>{system_prompt}\n<|user|>{prompt}\n<|assistant|>"
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_tokens = output_ids[0][input_ids.shape[-1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        results.append({"prompt": prompt, "response": response})

    print(f"\nDone. Generated {total} responses.")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run HF model on safety benchmarks.")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--system-prompt", default="systemprompt-v0.txt",
                        help="Path to system prompt file (default: systemprompt-v0.txt)")
    parser.add_argument("--dataset", required=True,
                        help="HuggingFace dataset ID (e.g. walledai/AdvBench)")
    parser.add_argument("--dataset-config", default=None,
                        help="Dataset config/subset name if needed")
    parser.add_argument("--dataset-split", default="train",
                        help="Dataset split to use (default: train)")
    parser.add_argument("--prompt-field", default="prompt",
                        help="Field name in the dataset containing the prompt text (default: prompt). "
                             "Supports dot notation for nested fields (e.g. question.stem).")
    parser.add_argument("--output", required=True,
                        help="Output JSON file path")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to process (default: all)")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Maximum new tokens to generate per response (default: 512)")
    parser.add_argument("--device", default="auto",
                        help="Device: auto, cpu, cuda, mps (default: auto)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size (default: 1, batching not yet implemented)")
    args = parser.parse_args()

    # Load system prompt
    system_prompt = load_system_prompt(args.system_prompt)
    print(f"System prompt loaded ({len(system_prompt)} chars): {args.system_prompt}")

    # Load dataset
    print(f"Loading dataset: {args.dataset} / config={args.dataset_config} / split={args.dataset_split}")
    ds = load_dataset(args.dataset, args.dataset_config, split=args.dataset_split)
    print(f"Dataset size: {len(ds)} rows. Field: '{args.prompt_field}'")

    prompts = extract_prompts(ds, args.prompt_field, args.max_samples)
    print(f"Extracted {len(prompts)} prompts.")

    # Run model
    results = run_model(
        model_name=args.model,
        system_prompt=system_prompt,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        batch_size=args.batch_size,
    )

    # Save output
    output = {
        "model": args.model,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "prompt_field": args.prompt_field,
        "system_prompt_file": args.system_prompt,
        "n_samples": len(results),
        "results": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
