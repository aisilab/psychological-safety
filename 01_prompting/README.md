# Safety Benchmark Runner

Run a HuggingFace model with a system prompt against safety benchmark datasets and store responses in JSON. Supports both HuggingFace dataset IDs and local JSON files.

## Dependencies

```bash
pip install transformers datasets torch accelerate
```

## Usage

### HuggingFace Transformers Version

```bash
python run_benchmark.py \
  --model <model-id> \
  --dataset <dataset-id-or-path.json> \
  --prompt-field <field> \
  --output <output.json> \
  --hf-token <hf-token> \
  [--system-prompt systemprompt-v0.txt] \
  [--dataset-config <config>] \
  [--dataset-split <split>] \
  [--max-samples 100] \
  [--max-new-tokens 512] \
  [--device auto] \
  [--hf-token <hf-token>] \
```

The `--dataset` argument accepts either a HuggingFace dataset ID (e.g., `walledai/AdvBench`) or a path to a local `.json` file. Local JSON files should contain an array of objects, each with at least the field specified by `--prompt-field`.

### vLLM Version for Fast Inference

The same run_benchmark pipeline is also implemented using vLLM for faster inference. The command-line arguments are mostly the same, with additional options for vLLM-specific settings.

```bash
python run_benchmark_vllm.py \
  --model <model-id> \
  --dataset <dataset-id-or-path.json> \
  --prompt-field <field> \
  --output <output.json> \
  --hf-token <hf-token> \
  [--system-prompt systemprompt-v0.txt] \
  [--dataset-config <config>] \
  [--dataset-split <split>] \
  [--max-samples 100] \
  [--max-new-tokens 512] \
  [--device auto] \
  [--hf-token <hf-token>] \
  [--attention-backend <attention-backend>] \
  [--gpu-memory-utilization <value>] \
  [--language-model-only] \
  [--reasoning-parser <parser-name>]
  [--enable-prefix-caching]
```

Some of the additional vLLM-specific arguments are model specific. In particular `--language-model-only` is used for the text-only mode of Qwen3.5 models (which are multimodal). The `--reasoning-parser` is also used only for reasoning models.


## Common Safety Benchmarks

| Dataset | `--dataset` | `--prompt-field` | Notes |
|---|---|---|---|
| AdvBench | `walledai/AdvBench` | `prompt` | |
| TruthfulQA | `truthful_qa` | `question` | add `--dataset-config generation --dataset-split validation` |
| BeaverTails | `PKU-Alignment/BeaverTails` | `prompt` | add `--dataset-config 30k_test` |
| Do-Not-Answer | `LibrAI/do-not-answer` | `question` | |
| WildJailbreak | `allenai/wildjailbreak` | `vanilla_prompt` | |
| HH-RLHF | `Anthropic/hh-rlhf` | `chosen` | |
| Harmful Dataset | `LLM-LAT/harmful-dataset` | `prompt` | |

## Examples

```bash
# AdvBench
python run_benchmark.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --dataset walledai/AdvBench \
  --prompt-field prompt \
  --output results/advbench.json \
  --max-samples 100 \
  --hf-token your-hf-token

# TruthfulQA
python run_benchmark.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --dataset truthful_qa \
  --dataset-config generation \
  --dataset-split validation \
  --prompt-field question \
  --output results/truthfulqa.json \
  --hf-token your-hf-token

# BeaverTails
python run_benchmark.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --dataset PKU-Alignment/BeaverTails \
  --dataset-config 30k_test \
  --prompt-field prompt \
  --output results/beavertails.json \
  --hf-token your-hf-token

# Do-Not-Answer
python run_benchmark.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --dataset LibrAI/do-not-answer \
  --prompt-field question \
  --output results/do-not-answer.json \
  --hf-token your-hf-token

# Local JSON file (e.g., our own val set)
python run_benchmark.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --dataset ../00_data/val.json \
  --prompt-field prompt \
  --output results/val.json

# Local JSON file (e.g. our own val set) with Qwen 3.5 27B on vLLM
python run_benchmark_vllm.py \
    --model Qwen/Qwen3.5-27B \
    --dataset ../00_data/val.json \
    --prompt-field prompt \
    --output results/output.json \
    --max-new-tokens 4096 \
    --max-samples 10 \
    --language-model-only \
    --reasoning-parser qwen3 \
    --enable-prefix-caching
```

## Output Format

```json
{
  "model": "meta-llama/Llama-3.2-1B-Instruct",
  "dataset": "walledai/AdvBench",
  "dataset_config": null,
  "dataset_split": "train",
  "prompt_field": "prompt",
  "system_prompt_file": "systemprompt-v0.txt",
  "n_samples": 100,
  "results": [
    { "prompt": "...", "response": "..." },
    ...
  ]
}
```
