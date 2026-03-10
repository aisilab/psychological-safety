import os
import sys
import yaml
import json
import argparse

import torch
from trainer import sft_train
from tqdm import tqdm
import pandas as pd
from datasets import Dataset, concatenate_datasets
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from finetune_util import load_jsonl, convert_prompt_answer_to_messages, formatting_prompts_func

def get_parser():
    parser = argparse.ArgumentParser("LGP", add_help=False)
    parser.add_argument("--config", default="runs/", type=str, required=False)
    return parser

def load_parquet_as_text(file_path):
    """Load a parquet file and combine title + content into a text field."""
    df = pd.read_parquet(file_path)
    rows = []
    for _, row in df.iterrows():
        title = str(row.get("title", "")).strip()
        content = str(row.get("content", "")).strip()
        if title and content:
            text = f"{title}\n\n{content}"
        elif title:
            text = title
        elif content:
            text = content
        else:
            continue
        rows.append(dict(text=text))
    return rows


def load_training_data(file_path, loss_type):
    """Load training data from JSON, JSONL, or parquet, returning a Dataset."""
    if file_path.endswith(".parquet"):
        rows = load_parquet_as_text(file_path)
        return Dataset.from_list(rows)

    if file_path.endswith(".json"):
        with open(file_path, 'r') as f:
            rows = json.load(f)
    else:
        rows = load_jsonl(file_path)

    if loss_type == "sft":
        return Dataset.from_list(convert_prompt_answer_to_messages(rows))
    else:
        return Dataset.from_list(rows)

def train(config):
    """Prepare lora model, call training function, and push to hub"""
    print(f"Loading model {config['model']} with load_in_4bit={config['load_in_4bit']}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_id=config["model"],
        dtype=torch.bfloat16,
        device_map="auto",
        load_in_4bit=config["load_in_4bit"],
        max_seq_length=2048,
    )
    target_modules = config["target_modules"]

    print(f"Using PEFT target modules: {target_modules}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["r"],
        target_modules=target_modules,
        layers_to_transform=config["layers_to_transform"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias=config["lora_bias"],
        use_gradient_checkpointing=True,
        random_state=config["seed"],
        use_rslora=config["use_rslora"],
        loftq_config=None,
        use_dora=False,
    )
    # load datasets 
    train_dataset = load_training_data(config["training_file"], config["loss"])
    eval_dataset = None
    if config["val_file"]:
        eval_dataset = load_training_data(config["val_file"], config["loss"])
    else:
        print("No test file provided, splitting 10% of training data for validation")
        # Split 10% of train data for testing when no test set provided
        # Use seed from training config to make the split deterministic
        split = train_dataset.train_test_split(test_size=0.1, seed=config["seed"])
        train_dataset = split["train"]
        eval_dataset = split["test"]

    if config["concatenate_datasets"]:
        print("Concatenating train and eval datasets for training...")
        train_dataset = concatenate_datasets([train_dataset, eval_dataset])
   
    train_dataset = train_dataset.map(
        formatting_prompts_func, 
        batched = True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "verbose": config["verbose_formatting"]
        },
    )
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(
            formatting_prompts_func, 
            batched = True,
            fn_kwargs={
                "tokenizer": tokenizer,
                "verbose": config["verbose_formatting"]
            },
        )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset if not config["concatenate_datasets"] else None,
        args = SFTConfig(
            per_device_train_batch_size = config["batch_size"],
            gradient_accumulation_steps = config["gradient_accumulation_steps"], 
            per_device_eval_batch_size=config["per_device_eval_batch_size"],  
            warmup_steps = config["warmup_steps"],
            num_train_epochs = config["num_train_epochs"],
            max_steps = config["max_steps"],
            save_strategy=config["save_strategy"],
            learning_rate = float(config["learning_rate"]),
            logging_steps = int(config["logging_steps"]),
            optim = config["optim"],
            weight_decay = float(config["weight_decay"]),
            lr_scheduler_type = config["lr_scheduler_type"],
            seed = int(config["random_seed"]),
            output_dir = config["output_dir"],
            report_to = config["report_to"], # Use this for WandB etc
            do_eval=config["do_eval"],
            eval_strategy=config["eval_strategy"],
        ),
    )
    trainer.train(resume_from_checkpoint=config["resume_from_checkpoint"])
    trainer.save_model(config["output_dir"])

def main(config_path: str):
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
    
    if "wandb_project" in config_data:
        os.environ["WANDB_PROJECT"] = config_data["wandb_project"]
    os.environ["TOKENIZERS_PARALLELISM"] = "false" 

    train(config_data)

    print(f"Saving config to {config_data.output_dir}/{config_data['name']}.yaml")
    with open(os.path.join(config_data.output_dir, f"{config_data['name']}.yaml"), "w") as f:
        yaml.dump(config_data, f)

if __name__ == "__main__":
    main(sys.argv[1])
