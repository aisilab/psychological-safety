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
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def load_system_prompt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def extract_first_human_turn(text: str) -> str:
    """Extract the first human turn from hh-rlhf style conversation strings.

    These have the format: '\\n\\nHuman: ...\\n\\nAssistant: ...'
    Returns just the first human message, or the original text if no match.
    """
    match = re.search(r"Human:\s*(.+?)(?:\n\nAssistant:|$)", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def extract_prompts(dataset, field: str, max_samples: int | None) -> list[str]:
    """Extract text prompts from a dataset, handling nested fields with dot notation."""
    prompts = []
    for i, row in enumerate(dataset):
        if max_samples and i >= max_samples:
            break
        value = row
        for key in field.split("."):
            value = value[key]
        if isinstance(value, str):
            # hh-rlhf chosen/rejected fields contain full conversation strings
            if "\n\nHuman:" in value:
                prompts.append(extract_first_human_turn(value))
            else:
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
    existing_results: list[dict] | None = None,
    hf_token: str | None = None,
) -> list[dict]:
    print(f"Loading tokenizer and model: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map="auto" if device == "auto" else device,
        token=hf_token,
    )
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Resume from existing results if provided
    results = list(existing_results) if existing_results else []
    start_idx = len(results)
    total = len(prompts)

    if start_idx > 0:
        print(f"Resuming from sample {start_idx + 1}/{total}")

    t0 = time.time()
    for i, prompt in enumerate(prompts[start_idx:], start=start_idx):
        elapsed = time.time() - t0
        rate = (i - start_idx + 1) / max(elapsed, 1e-6)
        remaining = (total - i - 1) / max(rate, 1e-6)
        print(
            f"  [{i+1}/{total}] Generating... "
            f"({rate:.1f} samples/s, ~{remaining:.0f}s remaining)",
            end="\r", flush=True,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            # Use apply_chat_template if available, else fall back to plain formatting
            try:
                result = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                # Newer transformers may return a BatchEncoding instead of a raw tensor
                if isinstance(result, torch.Tensor):
                    input_ids = result.to(model.device)
                else:
                    input_ids = result["input_ids"].to(model.device)
            except Exception:
                print("  [INFO] apply_chat_template failed, using fallback formatting.")
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

        except Exception as e:
            print(f"\n  [WARNING] Sample {i+1} failed: {e}")
            results.append({"prompt": prompt, "response": None, "error": str(e)})

    elapsed_total = time.time() - t0
    n_generated = total - start_idx
    print(f"\nDone. Generated {n_generated} responses in {elapsed_total:.1f}s.")
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
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file if it exists")
    parser.add_argument("--hf-token", default=None,
                        help="HuggingFace API token for accessing gated models/datasets")
    args = parser.parse_args()

    # Load system prompt
    system_prompt = load_system_prompt(args.system_prompt)
    print(f"System prompt loaded ({len(system_prompt)} chars): {args.system_prompt}")

    # Load dataset
    print(f"Loading dataset: {args.dataset} / config={args.dataset_config} / split={args.dataset_split}")
    ds = load_dataset(args.dataset, args.dataset_config, split=args.dataset_split, token=args.hf_token)
    print(f"Dataset size: {len(ds)} rows. Field: '{args.prompt_field}'")

    prompts = extract_prompts(ds, args.prompt_field, args.max_samples)
    print(f"Extracted {len(prompts)} prompts.")

    # Check for resume
    out_path = Path(args.output)
    existing_results = None
    if args.resume and out_path.exists():
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        existing_results = prev.get("results", [])
        print(f"Resuming: found {len(existing_results)} existing results in {out_path}")

    # Run model
    results = run_model(
        model_name=args.model,
        system_prompt=system_prompt,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        existing_results=existing_results,
        hf_token=args.hf_token,
    )

    # Save output
    output = {
        "model": args.model,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "prompt_field": args.prompt_field,
        "system_prompt_file": args.system_prompt,
        "system_prompt": system_prompt,
        "max_new_tokens": args.max_new_tokens,
        "n_samples": len(results),
        "n_errors": sum(1 for r in results if r.get("error")),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
