import torch
from torch.optim import lr_scheduler
from torch.utils import data
from torch.utils.data import IterableDataset
from dataset_maker import AbstractDataset
from utils import combine_logs
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm.auto import tqdm
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from load_objs import load_item
from grokk_model import GrokkModel
from transformer import xavier_init
from tqdm import tqdm
import os

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

def train(config):
    print('using config:', config)
    train_cfg = config['train']
    wandb_cfg = config['wandb']
    if wandb_cfg['use_wandb']:
        wandb.init(project=wandb_cfg['wandb_project'], config=config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = load_item(config['dataset'])
    train_data = GroupDataset(dataset, 'train')
    val_data = GroupDataset(dataset, 'val')
    try:
        model = load_item(config['model'], dataset.n_vocab, dataset.n_out, device)
    except Exception as e:
        print('Failed to load model:', e)
        print('Initializing GrokkModel from config...')
        model_cfg = config['model']['transformer_config']
        model = GrokkModel(
            transformer_config=model_cfg,
            vocab_size=dataset.n_vocab,
            output_size=dataset.n_out,
            device=device
        )
        xavier_init(model)  # Initialize weights
        model = model.to(device)

    
    # Check if checkpoint path exists and load if it does
    checkpoint_path = config['model'].get('checkpoint_path', None)
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint, strict=config['model'].get('strict_load', True))
    
    model.train()
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    train_dataloader = DataLoader(train_data, num_workers=train_cfg['num_workers'], batch_size=train_cfg['bsize'])
    val_dataloader = DataLoader(val_data, num_workers=train_cfg['num_workers'], batch_size=train_cfg['bsize'])
    optim = torch.optim.AdamW(model.parameters(), lr=train_cfg['lr'], 
                              weight_decay=train_cfg['weight_decay'], 
                              betas=train_cfg['betas'])
    lr_schedule = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda s: min(s / train_cfg['warmup_steps'], 1))
    
    # Ensure checkpoint directory exists
    checkpoint_dir = train_cfg.get('checkpoint_dir', '../checkpoints')
    checkpoint_prefix = train_cfg.get('checkpoint_prefix', 'model_step')
    os.makedirs(checkpoint_dir, exist_ok=True)
    step = 0
    train_pbar = tqdm(total=train_cfg['max_steps'], desc="Training")
    for x, y in train_dataloader:
        loss, logs = model.get_loss(x.to(device), y.to(device))
        optim.zero_grad()
        loss.backward()
        optim.step()
        lr_schedule.step()

        # Update progress bar
        train_pbar.update(1)
        train_pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{logs['accuracy'][0]:.4f}",
            'lr': f"{lr_schedule.get_last_lr()[0]:.2e}"
        })

        if (step+1) % train_cfg['eval_every'] == 0:
            model.eval()
            with torch.no_grad():
                all_val_logs = []
                val_pbar = tqdm(total=train_cfg['eval_batches'], desc="Validating")
                for i, (val_x, val_y) in enumerate(val_dataloader):
                    if i >= train_cfg['eval_batches']:
                        break
                    _, val_logs = model.get_loss(val_x.to(device), val_y.to(device))
                    all_val_logs.append(val_logs)
                    val_pbar.update(1)
                val_pbar.close()

            val_metrics = combine_logs(all_val_logs)
            train_metrics = combine_logs([logs])
            
            print(f"\nStep {step+1}:")
            print(f"Train - Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
            
            out_log = {
                'val': val_metrics,
                'train': train_metrics,
                'step': (step+1),
                'lr': float(lr_schedule.get_last_lr()[0])
            }
            if wandb_cfg['use_wandb']:
                wandb.log(out_log)
            model.train()

        step += 1
        if step % 1000 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_prefix}_{step}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"\nCheckpoint saved: {checkpoint_path}")

        if train_cfg['max_steps'] is not None and step >= train_cfg['max_steps']:
            break

    train_pbar.close()


@hydra.main(config_path="../config", config_name="train_grokk")
def main(cfg : DictConfig):
    cfg = OmegaConf.to_container(cfg)
    train(cfg)

if __name__ == "__main__":
    main()