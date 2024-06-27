import copy
from typing import List

from datasets import load_dataset
from torch import nn
from torch.utils import data
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def construct_llama3() -> nn.Module:
    config = AutoConfig.from_pretrained(
        "meta-llama/Meta-Llama-3-8B",
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B",
        from_tf=False,
        config=config,
        ignore_mismatched_sizes=False,
        trust_remote_code=True,
    )
    return model


def get_openwebtext_dataset(
    indices: List[int] = None,
) -> data.Dataset:
    raw_datasets = load_dataset("stas/openwebtext-10k")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_fast=True, trust_remote_code=True)

    tokenizer.pad_token = tokenizer.eos_token

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        results = tokenizer(examples[text_column_name], truncation=True, padding=True, max_length=64)
        results["labels"] = results["input_ids"].copy()
        return results

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=None,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    train_dataset = tokenized_datasets["train"]
    ds = train_dataset

    if indices is not None:
        ds = ds.select(indices)

    return ds


def get_custom_dataset(
    indices: List[int] = None,
) -> data.Dataset:
    data_kwargs = {
        "path": "json",
        "data_files": "./data/data.json",
        "num_proc": 4,
    }
    raw_datasets = load_dataset(**data_kwargs)["train"]
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_fast=True, trust_remote_code=True)

    def tokenize_function(examples):
        data_dict = {}
        prompt_results = tokenizer(text=examples["prompt"])
        completion_results = tokenizer(text=examples["completion"])
        input_ids = prompt_results["input_ids"] + completion_results["input_ids"][1:]
        attention_mask = prompt_results["attention_mask"] + completion_results["attention_mask"][1:]
        data_dict["input_ids"] = input_ids
        data_dict["labels"] = copy.deepcopy(input_ids)
        data_dict["labels"][: len(prompt_results["input_ids"])] = [
            -100 for _ in range(len(prompt_results["input_ids"]))
        ]
        data_dict["attention_mask"] = attention_mask
        return data_dict

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=False,
        num_proc=None,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    if indices is not None:
        tokenized_datasets = tokenized_datasets.select(indices)

    return tokenized_datasets


if __name__ == "__main__":
    from kronfluence import Analyzer

    model = construct_llama3()
    print(Analyzer.get_module_summary(model))
