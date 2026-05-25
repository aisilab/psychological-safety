import argparse
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

# Reduce noisy non-fatal warnings from optional extension paths.
logging.getLogger("torchao").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_rope_utils").setLevel(logging.ERROR)

from datasets import load_dataset
from vllm import LLM, SamplingParams

try:
    from .utils import split_reasoning_traces
except ImportError:
    # Support direct execution, e.g. `uv run run_benchmark_vllm.py` from 01_prompting/
    from utils import split_reasoning_traces

logger = logging.getLogger("vllm")

def load_system_prompt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def extract_first_human_turn(text: str) -> str:
    match = re.search(r"Human:\s*(.+?)(?:\n\nAssistant:|$)", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def extract_prompts(dataset, field: str, max_samples: int | None) -> list[str]:
    prompts = []
    for i, row in enumerate(dataset):
        if max_samples and i >= max_samples:
            break
        value = row
        for key in field.split("."):
            value = value[key]
        if isinstance(value, str):
            if "\n\nHuman:" in value:
                prompts.append(extract_first_human_turn(value))
            else:
                prompts.append(value.strip())
        elif isinstance(value, list):
            prompts.append(str(value[0]).strip())
    return prompts


def run_model(
    model_name: str,
    tokenizer_name: str | None,
    system_prompt: str,
    prompts: list[str],
    max_new_tokens: int,
    hf_token: str | None = None,
    gpu_memory_utilization: float = 0.95,
    language_model_only: bool = False,
    reasoning_parser: str | None = None,
    enable_prefix_caching: bool = False,
    attention_backend: str = "FLASH_ATTN",
    enforce_eager: bool = False,
    trust_remote_code: bool = False,
) -> list[dict]:
    import os
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    attention_config = {
        "backend": attention_backend.upper() if attention_backend else None,
        # Avoid TRTLLM FlashInfer kernels that require local CUDA dev headers.
        "use_trtllm_attention": False,
        "disable_flashinfer_prefill": True,
    }

    print(f"Loading model: {model_name}", flush=True)
    llm = LLM(
        model=model_name,
        tokenizer=tokenizer_name or model_name,
        tokenizer_mode="auto",
        trust_remote_code=trust_remote_code,
        # max_model_len=4096,
        dtype="auto",
        attention_config=attention_config,
        enforce_eager=enforce_eager,
        gpu_memory_utilization=gpu_memory_utilization,
        language_model_only=language_model_only,
        reasoning_parser=reasoning_parser,
        enable_prefix_caching=enable_prefix_caching,
    )

    sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=0)

    # Build chat messages for each prompt
    tokenizer = llm.get_tokenizer()
    formatted = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = f"<|system|>{system_prompt}\n<|user|>{prompt}\n<|assistant|>"
        formatted.append(text)

    print(f"Running batch inference on {len(formatted)} prompts...", flush=True)
    t0 = time.time()
    outputs = llm.generate(formatted, sampling_params)
    elapsed = time.time() - t0
    print(f"Done. {len(outputs)} responses in {elapsed:.1f}s ({len(outputs)/elapsed:.1f} samples/s).")

    results = []
    for prompt, out in zip(prompts, outputs):
        results.append({"prompt": prompt, "response": out.outputs[0].text.strip()})
    return results


def main() -> Path:
    parser = argparse.ArgumentParser(description="Run HF model on safety benchmarks via vLLM.")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--tokenizer", default=None,
                        help="Optional tokenizer model ID override. Use when the model repo has broken custom tokenizer metadata.")
    parser.add_argument("--system-prompt", default="systemprompt-v0.txt",
                        help="Path to system prompt file (default: systemprompt-v0.txt)")
    parser.add_argument("--dataset", required=True,
                        help="HuggingFace dataset ID or local JSON file path")
    parser.add_argument("--dataset-config", default=None,
                        help="Dataset config/subset name if needed")
    parser.add_argument("--dataset-split", default="train",
                        help="Dataset split to use (default: train)")
    parser.add_argument("--prompt-field", default="prompt",
                        help="Field name containing the prompt text (default: prompt). "
                             "Supports dot notation for nested fields.")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to process (default: all)")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Maximum new tokens to generate per response (default: 512)")
    parser.add_argument("--hf-token", default=None,
                        help="HuggingFace API token for accessing gated models/datasets")
    parser.add_argument("--attention-backend", default="FLASH_ATTN",
                        help="vLLM attention backend, e.g. FLASH_ATTN, FLASHINFER, XFORMERS")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95,
                        help="Fraction of GPU memory vLLM may use for KV cache (default: 0.95).")
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Force eager mode (disables cudagraph/compile optimizations).")
    parser.add_argument("--language-model-only", action="store_true",
                        help="Pass through to vLLM engine arg language_model_only.")
    parser.add_argument("--reasoning-parser", default=None,
                        help="Pass through to vLLM engine arg reasoning_parser (e.g. qwen3).")
    parser.add_argument("--enable-prefix-caching", action="store_true",
                        help="Pass through to vLLM engine arg enable_prefix_caching.")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Allow execution of custom code from model/tokenizer repos.")
    args = parser.parse_args()

    # Set vLLM defaults for Qwen 3.5 reasoning models
    auto_language_model_only = args.language_model_only
    auto_reasoning_parser = args.reasoning_parser
    if "qwen3.5" in args.model.lower():
        if not auto_language_model_only:
            auto_language_model_only = True
            logger.info("Auto-enabled --language-model-only for Qwen 3.5 model.")
        if auto_reasoning_parser is None:
            auto_reasoning_parser = "qwen3"
            logger.info("Auto-set --reasoning-parser qwen3 for Qwen 3.5 model.")

    system_prompt = load_system_prompt(args.system_prompt)
    logger.info(f"System prompt loaded ({len(system_prompt)} chars): {args.system_prompt}")

    dataset_path = Path(args.dataset)
    if dataset_path.exists() and dataset_path.suffix == ".json":
        logger.info(f"Loading local JSON file: {args.dataset}")
        ds = json.loads(dataset_path.read_text(encoding="utf-8"))
    else:
        logger.info(f"Loading dataset: {args.dataset} / config={args.dataset_config} / split={args.dataset_split}")
        ds = load_dataset(args.dataset, args.dataset_config, split=args.dataset_split,
                          token=args.hf_token)
    logger.info(f"Dataset size: {len(ds)} rows. Field: '{args.prompt_field}'")

    prompts = extract_prompts(ds, args.prompt_field, args.max_samples)
    logger.info(f"Extracted {len(prompts)} prompts.")
    
    start = time.time()
    results = run_model(
        model_name=args.model,
        tokenizer_name=args.tokenizer,
        system_prompt=system_prompt,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        hf_token=args.hf_token,
        attention_backend=args.attention_backend,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        language_model_only=auto_language_model_only,
        reasoning_parser=auto_reasoning_parser,
        enable_prefix_caching=args.enable_prefix_caching,
        trust_remote_code=args.trust_remote_code,
    )
    elapsed = time.time() - start

    output = {
        "model": args.model,
        "tokenizer": args.tokenizer or args.model,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "prompt_field": args.prompt_field,
        "system_prompt_file": args.system_prompt,
        "system_prompt": system_prompt,
        "max_new_tokens": args.max_new_tokens,
        "attention_backend": args.attention_backend,
        "enforce_eager": args.enforce_eager,
        "language_model_only": auto_language_model_only,
        "reasoning_parser": auto_reasoning_parser,
        "enable_prefix_caching": args.enable_prefix_caching,
        "trust_remote_code": args.trust_remote_code,
        "n_samples": len(results),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Generation completed for model {args.model} on dataset {args.dataset}.")
    logger.info(f"Total samples: {len(results)}. Time: {int(elapsed/3600):02d}:{int(elapsed/60)%60:02d}:{elapsed%60:.1f}. Speed: {len(results)/elapsed:.1f} samples/s.")
    logger.info(f"Full outputs saved to: {out_path}")
    return out_path


if __name__ == "__main__":
    generation_path = main()

    splitted_out_path = generation_path.with_name(generation_path.stem + "_splitted.json")
    split_reasoning_traces(generation_path, output_json_file=splitted_out_path, logger=logger)
    logger.info(f"Reasoning traces split and saved to: {splitted_out_path}")
