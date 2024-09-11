import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
# Add the project root to Python path

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

print("Python path in pipeline:", sys.path)  # This will help us debug

from kron.analyzer import Analyzer
from tasks import TinyStoriesTask

MODEL_NAME = "roneneldan/TinyStories-1M"
MAX_LENGTH = 512

def construct_tinystories_model():
    return AutoModelForCausalLM.from_pretrained(MODEL_NAME)

def get_tinystories_dataset(indices=None):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Check for existing tokenized dataset
    dataset_path = "/data/scratch/kaivuh/data/tokenized_dataset.pt"
    try:
        tokenized_datasets = torch.load(dataset_path)
        print("Loaded tokenized dataset from disk.")
    except FileNotFoundError:
        raw_datasets = load_dataset("roneneldan/TinyStories")

        def tokenize_function(examples):
            results = tokenizer(examples['text'], truncation=True, padding=True, max_length=MAX_LENGTH)
            results["labels"] = results["input_ids"].copy()
            results["labels"] = [
                [-100 if token == tokenizer.pad_token_id else token for token in label]
                for label in results["labels"]
            ]
            return results

        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

        # Save the tokenized dataset for future use
        torch.save(tokenized_datasets, dataset_path)
        print("Saved tokenized dataset to disk.")

    if indices is not None:
        tokenized_datasets = tokenized_datasets.select(indices)

    return tokenized_datasets["train"]


if __name__ == "__main__":
    model = construct_tinystories_model()
    print(Analyzer.get_module_summary(model))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = get_tinystories_dataset()
    task = TinyStoriesTask(tokenizer)

    # Example of how to use the pipeline
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample input: {dataset[0]['input_ids'][:10]}...")
    print(f"Tracked modules: {task.get_influence_tracked_modules()[:5]}...")