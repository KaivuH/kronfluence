#%%
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


import sys
import os

# Add the parent directory (kronfluencer) to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# print("Current file:", __file__)
# print("Project root:", project_root)
# print("Python path:", sys.path)
# print("Contents of project root:", os.listdir(project_root))


from kron.analyzer import prepare_model, Analyzer
from kron.utils.common.factor_arguments import extreme_reduce_memory_factor_arguments
from kron.utils.dataset import DataLoaderKwargs

#%%

def parse_args():
    parser = argparse.ArgumentParser(description="Influence factor computation on TinyStories dataset.")

    parser.add_argument(
        "--factors_name",
        type=str,
        default="tinystories",
        help="Name of the factor.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of examples to sample from the dataset.",
    )
    parser.add_argument(
        "--factor_strategy",
        type=str,
        default="ekfac",
        help="Strategy to compute influence factors.",
    )
    parser.add_argument(
        "--factor_batch_size",
        type=int,
        default=4,
        help="Batch size for computing influence factors.",
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


    # Prepare the trained model.
    model = construct_tinystories_model()

    # Define task and prepare model.
    task = TinyStoriesTask()
    model = prepare_model(model, task)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))  # 1.5 hours.
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    model = accelerator.prepare_model(model)

    analyzer = Analyzer(
        analysis_name="tinystories",
        model=model,
        task=task,
        profile=args.profile,
    )

    dataloader_kwargs = DataLoaderKwargs(num_workers=8, collate_fn=default_data_collator, pin_memory=True)
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    factors_name = args.factors_name
    factor_args = extreme_reduce_memory_factor_arguments(
        strategy=args.factor_strategy, 
        module_partitions=2,  # Increase partitions to reduce memory per operation
        dtype=torch.bfloat16,  # Use bfloat16 for better numerical stability
    )

    # Further customize the arguments
    # factor_args.covariance_module_partitions = 2
    # factor_args.lambda_module_partitions = 4
    # factor_args.covariance_data_partitions = 4
    # factor_args.lambda_data_partitions = 4
    factor_args.amp_dtype = torch.bfloat16
    factor_args.activation_covariance_dtype = torch.bfloat16
    factor_args.gradient_covariance_dtype = torch.bfloat16
    factor_args.per_sample_gradient_dtype = torch.bfloat16
    factor_args.lambda_dtype = torch.bfloat16
    factor_args.eigendecomposition_dtype = torch.bfloat16  # Add this line


    analyzer.fit_all_factors(
        factors_name=factors_name,
        dataset=train_dataset,
        per_device_batch_size=args.factor_batch_size,
        factor_args=factor_args,
        overwrite_output_dir=True,
    )




if __name__ == "__main__":
    main()