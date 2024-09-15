import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

MODEL_NAME = "roneneldan/TinyStories-1M"
MAX_LENGTH = 512
SAMPLE_SIZE = 10000  # Number of examples to process for investigation

def investigate_tokenizer():
    print(f"Investigating tokenizer for model: {MODEL_NAME}")
    print(f"Max length: {MAX_LENGTH}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Pad token: '{tokenizer.pad_token}', ID: {tokenizer.pad_token_id}")

    raw_datasets = load_dataset("roneneldan/TinyStories")
    sample_dataset = raw_datasets["train"].select(range(SAMPLE_SIZE))

    error_count = 0
    total_count = 0

    def tokenize_function(examples):
        nonlocal error_count, total_count
        results = tokenizer(examples['text'], truncation=True, padding="max_length", max_length=MAX_LENGTH)
        
        for i, (input_ids, attention_mask) in enumerate(zip(results["input_ids"], results["attention_mask"])):
            total_count += 1
            if len(input_ids) != MAX_LENGTH:
                error_count += 1
                print(f"\nError in sequence {i}:")
                print(f"  Original text: {examples['text'][i][:100]}...")
                print(f"  Tokenized length: {len(input_ids)}")
                print(f"  First few tokens: {input_ids[:10]}")
                print(f"  Last few tokens: {input_ids[-10:]}")
                print(f"  Attention mask sum: {sum(attention_mask)}")
                print(f"  Number of non-pad tokens: {sum(1 for t in input_ids if t != tokenizer.pad_token_id)}")
        
        return results

    print(f"Processing {SAMPLE_SIZE} examples...")
    tokenized_datasets = sample_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=sample_dataset.column_names,
        desc="Investigating tokenization",
    )

    print("\nTokenization Investigation Results:")
    print(f"Total sequences processed: {total_count}")
    print(f"Sequences with incorrect length: {error_count}")
    print(f"Percentage of errors: {error_count/total_count*100:.2f}%")

    if error_count == 0:
        print("\nNo errors found in the sample. Investigating a few correct examples:")
        for i in range(min(5, len(tokenized_datasets))):
            input_ids = tokenized_datasets[i]['input_ids']
            attention_mask = tokenized_datasets[i]['attention_mask']
            print(f"\nExample {i}:")
            print(f"  Tokenized length: {len(input_ids)}")
            print(f"  First few tokens: {input_ids[:10]}")
            print(f"  Last few tokens: {input_ids[-10:]}")
            print(f"  Attention mask sum: {sum(attention_mask)}")
            print(f"  Number of non-pad tokens: {sum(1 for t in input_ids if t != tokenizer.pad_token_id)}")

if __name__ == "__main__":
    investigate_tokenizer()