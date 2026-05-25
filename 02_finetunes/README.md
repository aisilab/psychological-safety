# Fine-Tuning

This folder contains the training pipeline for supervised fine-tuning of the project's safety models with Unsloth, TRL, and LoRA adapters.

## Contents

- `run_finetune.py`: main training entry point
- `finetune_util.py`: dataset loading and chat-format conversion helpers
- `upload_to_hub.py`: uploads the merged model and checkpoint folders to the Hugging Face Hub
- `configs/`: example training configurations

## Dependencies

Project dependencies are defined in `pyproject.toml`. The fine-tuning scripts rely in particular on:

- `torch`
- `datasets`
- `transformers`
- `trl`
- `unsloth`
- `wandb`
- `huggingface-hub`

## Training Data

`run_finetune.py` supports three input styles:

### JSON / JSONL in prompt-answer format

Each row can contain:

```json
{
  "prompt": "User prompt",
  "answer": "Assistant answer",
  "reasoning_language": "Optional chain-of-thought style reasoning"
}
```

If `reasoning_language` or `reasoning` is present, it is wrapped in the configured `think_start` / `think_end` tokens before the answer.

### JSON / JSONL in chat format

Each row can also already be in message format:

```json
{
  "messages": [
    { "role": "user", "content": "User prompt" },
    { "role": "assistant", "content": "Assistant answer" }
  ]
}
```

### Parquet

Parquet files are supported for plain-text training. The loader combines `title` and `content` into a single `text` field.

## Usage

Run training from the repository root:

```bash
python 02_finetunes/run_finetune.py 02_finetunes/configs/c3.yaml
```

The script will:

1. Load the base model with Unsloth.
2. Apply the configured LoRA adapters.
3. Load and format the train and validation datasets.
4. Train with `SFTTrainer`.
5. Save a merged model into `output_dir`.
6. Copy the config file into the output directory for reproducibility.

If `val_file` is empty, the script creates a 90/10 train/validation split from the training set.

## Configuration

The YAML files in `configs/` control:

- model name and sequence length
- 4-bit loading
- dataset paths
- reasoning tokens such as `<think>` and `</think>`
- batch size, epochs, learning rate, scheduler, and checkpointing
- LoRA rank, alpha, dropout, bias, and target modules
- optional Weights & Biases project name

Current example configs:

- `c1.yaml`: base SFT setup using `00_data/train.json` and `00_data/val.json`
- `c2.yaml`: reasoning-data SFT on `Qwen3.5-35B-A3B`
- `c3.yaml`: reasoning-data SFT on `Qwen3.5-27B`

## Path Note

Dataset paths in the YAML files are resolved from the current working directory, not from the config file location. In practice, that means you should double-check `training_file`, `val_file`, and `output_dir` before launching a run. The configs currently mix path styles, so running from the repository root is the safest default.

## Uploading To Hugging Face Hub

After training, upload the merged model and checkpoint folders with:

```bash
python 02_finetunes/upload_to_hub.py <model_dir> <repo_id> [--private]
```

Example:

```bash
python 02_finetunes/upload_to_hub.py checkpoints/ org-or-team/psych-safety-qwen --private
```

This script uploads:

- the merged model on the `main` branch
- each `checkpoint-*` directory as its own branch

## Outputs

A successful run writes artifacts to the configured `output_dir`, typically including:

- merged model weights
- tokenizer files
- checkpoint directories
- a copy of the YAML config used for the run
