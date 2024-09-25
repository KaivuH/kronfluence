import argparse
import logging
import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import default_data_collator

from pipeline import get_grokking_model, get_dataset, GroupDataset
from kron.analyzer import Analyzer, prepare_model
from kron.arguments import FactorArguments, ScoreArguments
from kron.task import Task
from kron.utils.dataset import DataLoaderKwargs

from omegaconf import DictConfig, OmegaConf
import hydra

BATCH_TYPE = Dict[str, torch.Tensor]
INFLUENCE_RESULTS_DIR = '/data/scratch/kaivuh/kronfluencer/influence_results'



def parse_args():
    parser = argparse.ArgumentParser(description="Influence analysis on WikiText dataset.")

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="A path that is storing the final checkpoint of the model.",
    )

    parser.add_argument(
        "--query_gradient_rank",
        type=int,
        default=-1,
        help="Rank for the low-rank query gradient approximation.",
    )
    parser.add_argument(
        "--use_half_precision",
        type=bool,
        default=False,
        help="Whether to use half precision for computing factors and scores.",
    )
    parser.add_argument(
        "--query_batch_size",
        type=int,
        default=32,
        help="Batch size for computing query gradients.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=64,
        help="Batch size for computing query gradients.",
    )
    parser.add_argument(
        "--factor_strategy",
        type=str,
        default="ekfac",
        help="Strategy to compute influence factors.",
    )

    args = parser.parse_args()

    if args.checkpoint_dir is not None:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    return args


class GrokkingModelingTask(Task):
    def compute_train_loss(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:

        input = batch["input"]
        labels = batch["labels"]

        loss, __ = model.get_loss(input, labels)
        return loss

    def compute_measurement(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
    ) -> torch.Tensor:
        # We could also compute the log-likelihood or averaged margin.
        return self.compute_train_loss(batch, model)

    def tracked_modules(self) -> List[str]:
        total_modules = []

        for i in range(2):
            total_modules.append(f"transformer.transformer_blocks.{i}.attn.key_proj")
            total_modules.append(f"transformer.transformer_blocks.{i}.attn.val_proj")
            total_modules.append(f"transformer.transformer_blocks.{i}.attn.query_proj")
            total_modules.append(f"transformer.transformer_blocks.{i}.attn.output_proj")
            total_modules.append(f"transformer.transformer_blocks.{i}.ff1")
            total_modules.append(f"transformer.transformer_blocks.{i}.ff2")

        return total_modules

@hydra.main(config_path="grokking/config", config_name="train_grokk")
def main(cfg : DictConfig):
    args = parse_args()
    
    logging.basicConfig(level=logging.INFO)

    # Prepare the dataset.
    config = OmegaConf.to_container(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = get_dataset(config)
    model = get_grokking_model(config, dataset, device)
    # Prepare the trained model.
    train_dataset = GroupDataset(dataset, 'train')
    eval_dataset = GroupDataset(dataset, 'val')
    # checkpoint_path = "checkpoints/model_step_1000.pth"
    # if not os.path.isfile(checkpoint_path):
    #     raise ValueError(f"No checkpoint found at {os.path.abspath(checkpoint_path)}.")
    # model.load_state_dict(torch.load(checkpoint_path))

    # Define task and prepare model.
    task = GrokkingModelingTask()
    model = prepare_model(model, task)

    analyzer = Analyzer(
        analysis_name="wikitext",
        model=model,
        task=task,
        output_dir=INFLUENCE_RESULTS_DIR,
    )
    # Configure parameters for DataLoader.
    dataloader_kwargs = DataLoaderKwargs(collate_fn=default_data_collator)
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    # Compute influence factors.
    factors_name = args.factor_strategy
    factor_args = FactorArguments(strategy=args.factor_strategy)
    if args.use_half_precision:
        factor_args.activation_covariance_dtype = torch.bfloat16
        factor_args.gradient_covariance_dtype = torch.bfloat16
        factor_args.lambda_dtype = torch.bfloat16
        factors_name += "_half"

    analyzer.fit_all_factors(
        factors_name=factors_name,
        dataset=train_dataset,
        per_device_batch_size=None,
        factor_args=factor_args,
        overwrite_output_dir=True,
        initial_per_device_batch_size_attempt=128,
    )

    # Compute pairwise scores.
    rank = args.query_gradient_rank if args.query_gradient_rank != -1 else None
    score_args = ScoreArguments(query_gradient_svd_dtype=torch.float32)
    scores_name = f"{factor_args.strategy}_pairwise"
    if rank is not None:
        scores_name += f"_qlr{rank}"

    if args.use_half_precision:
        score_args.per_sample_gradient_dtype = torch.bfloat16
        score_args.score_dtype = torch.bfloat16
        score_args.cached_activation_cpu_offload = True
        scores_name += "_half"

    analyzer.compute_pairwise_scores(
        scores_name=scores_name,
        score_args=score_args,
        factors_name=args.factor_strategy,
        query_dataset=train_dataset,
        query_indices=list(range(len(train_dataset))),
        train_dataset=train_dataset,
        per_device_query_batch_size=args.query_batch_size,
        per_device_train_batch_size=args.train_batch_size,
        overwrite_output_dir=True,
    )
    scores = analyzer.load_pairwise_scores(scores_name)["all_modules"]
    logging.info(f"Scores shape: {scores.shape}")



if __name__ == "__main__":
    main()
