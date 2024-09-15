import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
# Add the project root to Python path

import os
import sys
from tqdm import tqdm


# Add the parent directory (kronfluencer) to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# print("Current file:", __file__)
# print("Project root:", project_root)
# print("Python path:", sys.path)
# print("Contents of project root:", os.listdir(project_root))



from kron.analyzer import Analyzer
from tasks import TinyStoriesTask

MODEL_NAME = "roneneldan/TinyStories-1M"
MAX_LENGTH = 512
DATASET_PATH = "/workspace/data/tokenized_dataset.pt"


def construct_tinystories_model():
    return AutoModelForCausalLM.from_pretrained(MODEL_NAME)

def get_tinystories_dataset(indices=None):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Check for existing tokenized dataset
    try:
        tokenized_datasets = torch.load(DATASET_PATH)
        print("Loaded tokenized dataset from disk.")
    except FileNotFoundError:
        raw_datasets = load_dataset("roneneldan/TinyStories")

        error_count = 0
        total_count = 0

        def tokenize_function(examples):
            nonlocal error_count, total_count
            results = tokenizer(examples['text'], truncation=True, padding="max_length", max_length=MAX_LENGTH)
            results["labels"] = results["input_ids"].copy()
            results["labels"] = [
                [-100 if token == tokenizer.pad_token_id else token for token in label]
                for label in results["labels"]
            ]
            
            # Check for errors
            for input_ids in results["input_ids"]:
                total_count += 1
                if len(input_ids) != MAX_LENGTH:
                    error_count += 1
            
            return results

        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

        # Print error statistics
        print(f"Total sequences processed: {total_count}")
        print(f"Sequences with incorrect length: {error_count}")
        print(f"Percentage of errors: {error_count/total_count*100:.2f}%")

        # Save the tokenized dataset for future use
        torch.save(tokenized_datasets, DATASET_PATH)
        print(f"Saved tokenized dataset to {DATASET_PATH}")

    if indices is not None:
        tokenized_datasets = tokenized_datasets.select(indices)

    return tokenized_datasets["train"]

if __name__ == "__main__":
    model = construct_tinystories_model()
    print(Analyzer.get_module_summary(model))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = get_tinystories_dataset()
    task = TinyStoriesTask()

    # Example of how to use the pipeline
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample input: {dataset[0]['input_ids'][:10]}...")
    print(f"Tracked modules: {task.get_influence_tracked_modules()[:5]}...")