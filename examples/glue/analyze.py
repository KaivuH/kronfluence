import argparse
import logging
import os
from typing import Dict, Tuple, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from examples.glue.pipeline import construct_bert, get_glue_dataset
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments
from kronfluence.task import Task
from kronfluence.utils.dataset import DataLoaderKwargs

BATCH_TYPE = Dict[str, torch.Tensor]


def parse_args():
    parser = argparse.ArgumentParser(description="Influence analysis on CIFAR-10 dataset.")

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="sst2",
        help="A name of GLUE dataset.",
    )

    parser.add_argument(
        "--query_batch_size",
        type=int,
        default=8,
        help="Batch size for computing query gradients.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=8,
        help="Batch size for computing query gradients.",
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="A path to store the final checkpoint.",
    )

    parser.add_argument(
        "--factor_strategy",
        type=str,
        default="ekfac",
        help="Strategy to compute preconditioning factors.",
    )

    args = parser.parse_args()

    if args.checkpoint_dir is not None:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset_name)
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    return args


class TextClassificationTask(Task):
    def compute_train_loss(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
        ).logits

        if not sample:
            return F.cross_entropy(logits, batch["labels"], reduction="sum")
        with torch.no_grad():
            probs = torch.nn.functional.softmax(logits, dim=-1)
            sampled_labels = torch.multinomial(
                probs,
                num_samples=1,
            ).flatten()
        return F.cross_entropy(logits, sampled_labels.detach(), reduction="sum")

    def compute_measurement(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
    ) -> torch.Tensor:
        # Copied from: https://github.com/MadryLab/trak/blob/main/trak/modelout_functions.py.
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
        ).logits

        labels = batch["labels"]
        bindex = torch.arange(logits.shape[0]).to(device=logits.device, non_blocking=False)
        logits_correct = logits[bindex, labels]

        cloned_logits = logits.clone()
        cloned_logits[bindex, labels] = torch.tensor(-torch.inf, device=logits.device, dtype=logits.dtype)

        margins = logits_correct - cloned_logits.logsumexp(dim=-1)
        return -margins.sum()

    def get_attention_mask(self, batch: BATCH_TYPE) -> Optional[torch.Tensor]:
        print(type(batch["attention_mask"]))
        return batch["attention_mask"]


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    train_dataset = get_glue_dataset(
        data_name=args.dataset_name,
        split="eval_train",
    )
    eval_dataset = get_glue_dataset(
        data_name=args.dataset_name,
        split="valid",
    )

    model = construct_bert()
    checkpoint_path = os.path.join(args.checkpoint_dir, "model.pth")
    if not os.path.isfile(checkpoint_path):
        raise ValueError(f"No checkpoint found at {checkpoint_path}.")
    model.load_state_dict(torch.load(checkpoint_path))

    task = TextClassificationTask()
    model = prepare_model(model, task)

    analyzer = Analyzer(
        analysis_name=args.dataset_name,
        model=model,
        task=task,
        cpu=False,
    )

    factor_args = FactorArguments(strategy=args.factor_strategy)
    analyzer.fit_all_factors(
        factors_name=args.factor_strategy,
        dataset=train_dataset,
        per_device_batch_size=None,
        factor_args=factor_args,
        overwrite_output_dir=True,
        initial_per_device_batch_size_attempt=64,
    )
    analyzer.compute_pairwise_scores(
        scores_name="pairwise",
        factors_name=args.factor_strategy,
        query_dataset=eval_dataset,
        query_indices=list(range(2000)),
        train_dataset=train_dataset,
        per_device_query_batch_size=args.query_batch_size,
        per_device_train_batch_size=args.train_batch_size,
        overwrite_output_dir=True,
    )
    scores = analyzer.load_pairwise_scores("pairwise")
    print(scores)


if __name__ == "__main__":
    main()
