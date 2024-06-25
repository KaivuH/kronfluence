import logging

import matplotlib.pyplot as plt
from tueplots import markers

from kronfluence.analyzer import Analyzer


def main():
    logging.basicConfig(level=logging.INFO)

    # Load the scores. You might need to modify the path.
    scores = Analyzer.load_file("influence_results/cifar10/scores_ekfac/pairwise_scores.safetensors")["all_modules"] / 50_000
    half_scores = Analyzer.load_file("influence_results/cifar10/scores_ekfac_half/pairwise_scores.safetensors")["all_modules"].float() / 50_000

    plt.rcParams.update({"figure.dpi": 150})
    plt.rcParams.update(markers.with_edge())
    plt.rcParams["axes.axisbelow"] = True

    # Only plot first 3000 points to avoid clutter.
    idx = 0
    plt.scatter(half_scores[idx][:3000], scores[idx][:3000], edgecolor="k")
    plt.grid()
    plt.xlabel("bfloat16")
    plt.ylabel("float32")
    plt.show()


if __name__ == "__main__":
    main()
