import yaml
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import torch
import shutil
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
import datasets
from datetime import datetime
import copy
from peft import LoraConfig, get_peft_model, TaskType
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# Define the arguments required for the main program.
@dataclass
class ModelArguments:
    """Arguments for model"""
    model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The path to the LLM to fine-tune or its name on the Hugging Face Hub."}
    )
    torch_dtype: Optional[str] = field(
        default=None, metadata={"help": "Override the default `torch.dtype`.", "choices": ["bfloat16", "float16", "float32"]}
    )


@dataclass
class DataArguments:
    """Arguments for data"""
    dataset_path: Optional[str] = field(
        default=None, metadata={"help": "The path to the fine-tuning dataset or its name on the Hugging Face Hub."}
    )


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def finetune(config_path: str, loss: str, use_lora: bool):
    # Load config from the provided YAML file
    config = load_config(config_path)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Parse model, data, and training arguments
    model_args = ModelArguments(**config.get("model_args", {}))
    data_args = DataArguments(**config.get("data_args", {}))
    training_args = TrainingArguments(**config.get("training_args", {}))
    log_dir = current_time + "_" + loss
    training_args.output_dir = os.path.join(training_args.output_dir, log_dir)
    

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=getattr(torch, model_args.torch_dtype) if model_args.torch_dtype else None,
    )
    if use_lora:
        lora_config = LoraConfig(
            r=8,                      # LoRA rank
            lora_alpha=32,           # LoRA alpha
            lora_dropout=0.05,       # LoRA dropout
            inference_mode=False,
            task_type=TaskType.CAUSAL_LM,
            target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj"
            ]
        )
        model = get_peft_model(model, lora_config)
        model.config.use_cache = False
        model.print_trainable_parameters()

    # Load dataset
    dataset = datasets.load_dataset("json", data_files={"train": data_args.dataset_path})
    
    PROMPT_DICT = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        ),
    }
    def tokenize_function(example):
        IGNORE_INDEX = -100
        if example.get("input", "") == "":
            prompt = PROMPT_DICT["prompt_no_input"].format_map(example)
        else:
            prompt = PROMPT_DICT["prompt_input"].format_map(example)
        whole = prompt + example["output"]
        whole = tokenizer(
            whole,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=2048,
        )
        labels = copy.deepcopy(whole["input_ids"].squeeze(0))
        
        if loss == "whole":
            labels[labels==tokenizer.pad_token_id] = IGNORE_INDEX
        elif loss == "output":
            labels[labels==tokenizer.pad_token_id] = IGNORE_INDEX
            prompt = tokenizer.encode(prompt)
            output_pos = len(prompt)
            if output_pos > 2048:
                output_pos = 2048
            labels[:output_pos] = IGNORE_INDEX
        else:
            print("Please input correct loss type")
            
        return {
            "input_ids": whole["input_ids"].squeeze(0),
            "attention_mask": whole["attention_mask"].squeeze(0),
            "labels": labels,
        }

    dataset = dataset.map(tokenize_function)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()
    
    config_save_path = os.path.join(training_args.output_dir, "config.yaml")
    shutil.copy(config_path, config_save_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune LLMs with Huggingface.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the YAML configuration file.")
    parser.add_argument("--loss", type=str, default="output", help="Using output or whole as labels to compute loss")
    parser.add_argument("--use_lora", type=bool, default=True, help="Using lora to finetune or not")
    args = parser.parse_args()

    finetune(args.config, args.loss, args.use_lora)
