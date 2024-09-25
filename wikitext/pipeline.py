import torch
from grokking.grokk_replica.load_objs import load_item
from torch.utils.data import IterableDataset
from grokking.grokk_replica.datasets import AbstractDataset
from omegaconf import DictConfig, OmegaConf
import hydra

import os
import sys

# Add the parent directory (kronfluencer) to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from kron.analyzer import Analyzer



class GroupDataset(IterableDataset):
    def __init__(self, dataset: AbstractDataset, split: str):
        super(GroupDataset, self).__init__()
        assert split in {'train', 'val'}
        self.dataset = dataset
        self.split = split
        self.fetch_f = None
        if self.split == 'train':
            self.fetch_f = self.dataset.fetch_train_example
        elif self.split == 'val':
            self.fetch_f = self.dataset.fetch_val_example
        else:
            raise NotImplementedError

    def __iter__(self):
        return self

    def __next__(self):
        x, y, _ = self.fetch_f()
        return torch.tensor(x), torch.tensor(y)
    
    def __len__(self):
        if self.split == 'train':
            return len(self.dataset.train_pairs)
        elif self.split == 'val':
            return len(self.dataset.val_pairs)
        else:
            raise NotImplementedError
        
    def __getitem__(self, index):
        if self.split == 'train':
            example = self.dataset.fetch_example(self.dataset.train_pairs[index])
        elif self.split == 'val':
            example = self.dataset.fetch_example(self.dataset.val_pairs[index])
        x, y, _ = example
        return {"input": torch.tensor(x), "label": torch.tensor(y)}
    
def get_dataset(config):
    dataset = load_item(config['dataset'])
    print(f'loaded dataset: {dataset}')
    train_data = GroupDataset(dataset, 'train')
    val_data = GroupDataset(dataset, 'val')
    return dataset

def get_grokking_model(config, dataset, device):
    return load_item(config['model'], dataset.n_vocab, dataset.n_out, device)

@hydra.main(config_path="grokking/config", config_name="train_grokk")
def main(cfg : DictConfig):
    config = OmegaConf.to_container(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = get_dataset(config)
    model = get_grokking_model(config, dataset, device)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    print(Analyzer.get_module_summary(model))


if __name__ == "__main__":
    main()
