import logging

import numpy as np
import torch
import tqdm
from scipy.stats import spearmanr
from pipeline import get_dataset, GroupDataset
from kronfluence.analyzer import Analyzer

from omegaconf import DictConfig, OmegaConf
import hydra
import matplotlib.pyplot as plt
from scipy.stats import powerlaw
import pandas as pd

def create_dataframe_for_index(index, checkpoints, checkpoint_names, dataset):
    data = []

    # Iterate over each checkpoint (epoch)
    for i, checkpoint_scores in enumerate(checkpoints):
        influence_scores = checkpoint_scores[index].cpu().numpy()

        # Create rows for the DataFrame
        for idx, score in enumerate(influence_scores):
            data.append({
                "Epoch": checkpoint_names[i],
                "Influence Score": score,
                "Index": idx,
                "Input": str(dataset[idx]["input"][0].item()) + str('-') + str(dataset[idx]["input"][2].item()),
                "Label": dataset[idx]["label"].item()
            })

    df = pd.DataFrame(data)

    return df

def get_tokens(data_example):
    return {t.item() for t in data_example}

@hydra.main(config_path="grokking/config", config_name="train_grokk")
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO)
    config = OmegaConf.to_container(cfg)

    # Datasets, comment out one of them
    # Eval Dataset
    scores_0 = Analyzer.load_file("scores/eval_0.safetensors")["all_modules"].to(dtype=torch.float32)
    scores_1000 = Analyzer.load_file("scores/eval_1k.safetensors")["all_modules"].to(dtype=torch.float32)
    scores_5000 = Analyzer.load_file("scores/eval_5k.safetensors")["all_modules"].to(dtype=torch.float32)
    scores_20k = Analyzer.load_file("scores/eval_20k.safetensors")["all_modules"].to(dtype=torch.float32)
    eval_idx = 0

    # Train Dataset
    scores_0 = Analyzer.load_file("scores/train_0.safetensors")["all_modules"].to(dtype=torch.float32)
    scores_1000 = Analyzer.load_file("scores/train_1k.safetensors")["all_modules"].to(dtype=torch.float32)
    scores_5000 = Analyzer.load_file("scores/train_5k.safetensors")["all_modules"].to(dtype=torch.float32)
    scores_20k = Analyzer.load_file("scores/train_20k.safetensors")["all_modules"].to(dtype=torch.float32)
    eval_idx = 0

    dataset = get_dataset(config)
    train_dataset = GroupDataset(dataset, 'train')
    print("Train Sample Count:", len(train_dataset))
    eval_dataset = GroupDataset(dataset, 'val')
    print("Eval Sample Count:", len(eval_dataset))
    print("Query Data Example:")
    print(train_dataset[eval_idx])

    checkpoints = [scores_0, scores_1000, scores_5000, scores_20k]
    checkpoint_names = ["0", "1000", "5000", "20k"]

    # Create the DataFrame for the given index
    # df_for_index = create_dataframe_for_index(eval_idx, checkpoints, checkpoint_names, train_dataset)
    # print(df_for_index)
    # file_name = f"influence_scores_index_0.csv"
    # df_for_index.to_csv(file_name, index=False)

    for i, checkpoint_scores in enumerate(checkpoints):
        top_idx = torch.argsort(checkpoint_scores[eval_idx], descending=True)[:500]
        top_influential_scores = checkpoint_scores[eval_idx, top_idx]

        inf_scores = top_influential_scores.cpu().numpy()
        assert len(inf_scores.shape) == 1
        inf_scores = np.sort(inf_scores) / np.maximum(np.max(inf_scores), 1e-6)
        survival = np.arange(0, len(inf_scores), dtype=float)[::-1] / len(inf_scores)

        checkpoint_names = ["0", "1000", "5000", "20k"]
        name = checkpoint_names[i]

        print(f"Top Influential Example at Epoch {name}:")
        print(train_dataset[top_idx[0]])
        # query_tokens = get_tokens(set(train_dataset[eval_idx]["input"]))
        # top_influence_tokens = get_tokens(set(train_dataset[top_idx[0]]["input"]))
        # union_tokens = query_tokens.union(top_influence_tokens)
        # print(union_tokens)
        # num_duplicates =  len(query_tokens) + len(top_influence_tokens) - len(union_tokens)
        # print(f"Number of duplicate tokens: {num_duplicates}")

    #     fig, ax = plt.subplots()
    #     plt.title(f"The tail end of influence scores at Epoch {name}")
    #     plt.xlabel("Influence Score")
    #     plt.ylabel("Survival Rate")
    #     plt.plot(inf_scores, survival, label="Data")
    #     plt.legend()
        
    #     # Log-log axis
    #     ax.set_xscale('log')
    #     ax.set_yscale('log')
    #     plt.savefig(f"influence_scores_checkpoint_{i}.png")

if __name__ == "__main__":
    main()

    # # Load the CSV file
    # data = pd.read_csv('influence_scores_index_0_train.csv')

    # unique_epochs = data['Epoch'].unique()
    # unique_epochs.sort()
    # global_max = data['Influence Score'].max()
    # fig, ax = plt.subplots()
    # plt.title("The tail end of influence scores by Epoch")
    # plt.xlabel("Influence Score")
    # plt.ylabel("Survival Rate")

    # # Process data for each epoch and plot
    # for i, epoch in enumerate(unique_epochs):
    #     # Filter the data for the current epoch
    #     epoch_data = data[data['Epoch'] == epoch]['Influence Score'].dropna()

    #     top_influential_scores = np.sort(epoch_data)[-500:]

    #     inf_scores = np.sort(top_influential_scores) / np.maximum(np.max(top_influential_scores), 1e-6)
    #     survival = np.arange(1, len(inf_scores) + 1, dtype=float)[::-1] / len(inf_scores)
    #     ax.plot(inf_scores, survival, label=f"Epoch {epoch}")

    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # plt.legend()

    # Save the figure
    # plt.savefig("influence_scores_summary.png")
