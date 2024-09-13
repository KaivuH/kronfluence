import argparse
import logging
from datetime import timedelta
import os
import sys

import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from transformers import default_data_collator

from pipeline import construct_tinystories_model, get_tinystories_dataset
from tasks import TinyStoriesTask

# Add the parent directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from kron.analyzer import Analyzer, prepare_model
from kron.utils.common.factor_arguments import extreme_reduce_memory_factor_arguments
from kron.utils.common.score_arguments import extreme_reduce_memory_score_arguments
from kron.utils.dataset import DataLoaderKwargs

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

INFLUENCE_RESULTS_DIR = os.path.join(project_root, "influence_results")
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def clear_cuda_cache():
    torch.cuda.empty_cache()
    import gc
    gc.collect()

def parse_args():
    parser = argparse.ArgumentParser(description="Influence score computation on TinyStories dataset.")

    parser.add_argument(
        "--factors_name",
        type=str,
        required=True,
        help="Name of the factor.",
    )
    parser.add_argument(
        "--scores_name",
        type=str,
        required=True,
        help="Name of the score.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of examples to sample from the dataset.",
    )
    parser.add_argument(
        "--query_gradient_rank",
        type=int,
        default=-1,
        help="Rank for the low-rank query gradient approximation.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=8,
        help="Batch size for computing query gradients.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Boolean flag to profile computations.",
    )
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    # Prepare the dataset.
    train_dataset = get_tinystories_dataset()

    # Sample the dataset if sample_size is specified
    if args.sample_size is not None:
        train_dataset = train_dataset.select(range(min(args.sample_size, len(train_dataset))))
        print(f"Sampled dataset size: {len(train_dataset)}")

    # Use a subset of the train dataset as the eval dataset
    eval_dataset = train_dataset.select(range(min(1000, len(train_dataset))))

    # Prepare the trained model.
    model = construct_tinystories_model()

    # Define task and prepare model.
    task = TinyStoriesTask()
    model = prepare_model(model, task)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))  # 1.5 hours.
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    model = accelerator.prepare_model(model)

    clear_cuda_cache()

    analyzer = Analyzer(
        analysis_name="tinystories",
        model=model,
        task=task,
        profile=args.profile,
        output_dir=INFLUENCE_RESULTS_DIR,
    )

    dataloader_kwargs = DataLoaderKwargs(num_workers=8, collate_fn=default_data_collator, pin_memory=True)
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    rank = args.query_gradient_rank if args.query_gradient_rank != -1 else None
    score_args = extreme_reduce_memory_score_arguments(
        damping_factor=None, module_partitions=1, query_gradient_low_rank=rank, dtype=torch.bfloat16
    )
    score_args.query_gradient_accumulation_steps = 10
    score_args.use_full_svd = True
    score_args.precondition_dtype = torch.float32
    score_args.per_sample_gradient_dtype = torch.float32

    analyzer.compute_pairwise_scores(
        scores_name=args.scores_name,
        score_args=score_args,
        factors_name=args.factors_name,
        query_dataset=eval_dataset,
        train_dataset=train_dataset,
        per_device_query_batch_size=25,
        per_device_train_batch_size=args.train_batch_size,
        overwrite_output_dir=True,
    )
    scores = analyzer.load_pairwise_scores(args.scores_name)["all_modules"]
    logging.info(f"Scores shape: {scores.shape}")

if __name__ == "__main__":
    main()