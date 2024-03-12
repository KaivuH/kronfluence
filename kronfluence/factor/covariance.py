from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from accelerate.utils import find_batch_size, send_to_device
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils import data
from tqdm import tqdm

from kronfluence.arguments import FactorArguments
from kronfluence.module.constants import (
    COVARIANCE_FACTOR_NAMES,
    FACTOR_TYPE,
    PARTITION_TYPE,
)
from kronfluence.module.tracked_module import ModuleMode
from kronfluence.module.utils import (
    load_factors,
    remove_attention_mask,
    set_attention_mask,
    set_mode,
    synchronize_covariance_matrices,
    update_factor_args,
)
from kronfluence.task import Task
from kronfluence.utils.logger import TQDM_BAR_FORMAT
from kronfluence.utils.state import State, no_sync


def covariance_matrices_save_path(
    output_dir: Path,
    covariance_factor_name: str,
    partition: Optional[PARTITION_TYPE] = None,
) -> Path:
    """Generates the path for saving/loading covariance matrices."""
    assert covariance_factor_name in COVARIANCE_FACTOR_NAMES
    if partition is not None:
        data_partition, module_partition = partition
        return output_dir / (
            f"{covariance_factor_name}_covariance_data_partition{data_partition}"
            f"_module_partition{module_partition}.safetensors"
        )
    return output_dir / f"{covariance_factor_name}_covariance.safetensors"


def save_covariance_matrices(
    output_dir: Path,
    covariance_factors: Dict[str, Dict[str, torch.Tensor]],
    partition: Optional[PARTITION_TYPE] = None,
) -> None:
    """Saves covariance matrices to disk."""
    assert set(covariance_factors.keys()) == set(COVARIANCE_FACTOR_NAMES)
    for name in covariance_factors:
        save_path = covariance_matrices_save_path(
            output_dir=output_dir,
            covariance_factor_name=name,
            partition=partition,
        )
        save_file(tensors=covariance_factors[name], filename=save_path)


def load_covariance_matrices(
    output_dir: Path,
    partition: Optional[PARTITION_TYPE] = None,
) -> FACTOR_TYPE:
    """Loads covariance matrices from disk."""
    covariance_factors = {}
    for name in COVARIANCE_FACTOR_NAMES:
        save_path = covariance_matrices_save_path(
            output_dir=output_dir,
            covariance_factor_name=name,
            partition=partition,
        )
        covariance_factors[name] = load_file(filename=save_path)
    return covariance_factors


def covariance_matrices_exist(
    output_dir: Path,
    partition: Optional[PARTITION_TYPE] = None,
) -> bool:
    """Checks if covariance matrices exist at specified directory."""
    for name in COVARIANCE_FACTOR_NAMES:
        save_path = covariance_matrices_save_path(
            output_dir=output_dir,
            covariance_factor_name=name,
            partition=partition,
        )
        if not save_path.exists():
            return False
    return True


def fit_covariance_matrices_with_loader(
    model: nn.Module,
    state: State,
    task: Task,
    loader: data.DataLoader,
    factor_args: FactorArguments,
    tracked_module_names: Optional[List[str]] = None,
) -> Tuple[torch.Tensor, FACTOR_TYPE]:
    """Computes activation and pseudo-gradient covariance matrices for a given model and task.

    Args:
        model (nn.Module):
            The model that covariance matrices will be computed.
        state (State):
            The current process's information (e.g., device being used).
        task (Task):
            The specific task associated with the model.
        loader (data.DataLoader):
            The data loader that will be used to compute covariance matrices.
        factor_args (FactorArguments):
            Arguments related to computing covariance matrices.
        tracked_module_names (List[str], optional):
            A list of module names that covariance matrices will be computed. If not specified, covariance
            matrices will be computed for all tracked modules.

    Returns:
        Tuple[torch.Tensor, FACTOR_TYPE]:
            A tuple containing the number of data points processed, and computed covariance matrices in CPU.
            The covariance matrices are organized in nested dictionaries, where the first key in the name of the
            covariance matrix (e.g., activation covariance and gradient covariance) and the second key is
            the module name.
    """
    with torch.no_grad():
        update_factor_args(model=model, factor_args=factor_args)
        set_mode(model=model, mode=ModuleMode.DEFAULT, keep_factors=False)
        set_mode(
            model=model,
            tracked_module_names=tracked_module_names,
            mode=ModuleMode.COVARIANCE,
        )
    num_data_processed = torch.zeros((1,), dtype=torch.int64, device=state.device, requires_grad=False)

    with tqdm(
        total=len(loader),
        desc="Fitting covariance matrices",
        bar_format=TQDM_BAR_FORMAT,
        disable=not state.is_main_process,
    ) as pbar:
        for batch in loader:
            batch = send_to_device(batch, device=state.device)
            with torch.no_grad():
                attention_mask = task.get_attention_mask(batch=batch)
                set_attention_mask(model=model, attention_mask=attention_mask)

            with no_sync(model=model, state=state):
                model.zero_grad(set_to_none=True)
                loss = task.compute_train_loss(
                    batch=batch,
                    model=model,
                    sample=not factor_args.use_empirical_fisher,
                )
                loss.backward()
            num_data_processed += find_batch_size(batch)
            pbar.update(1)

    with torch.no_grad():
        remove_attention_mask(model=model)

    if state.use_distributed:
        # Aggregate covariance matrices across multiple devices or nodes.
        synchronize_covariance_matrices(model=model)
        dist.all_reduce(tensor=num_data_processed, op=torch.distributed.ReduceOp.SUM)

    with torch.no_grad():
        saved_factors: FACTOR_TYPE = {}
        for covariance_factor_name in COVARIANCE_FACTOR_NAMES:
            saved_factors[covariance_factor_name] = load_factors(model=model, factor_name=covariance_factor_name)
        set_mode(model=model, mode=ModuleMode.DEFAULT, keep_factors=False)
    return num_data_processed, saved_factors
