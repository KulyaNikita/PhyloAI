#!/usr/bin/env python3

"""
(c) Nikita Kulikov, Iker Irisarri, Christoph Mayer; LIB 2025

Perform PCA on the given dataset and plot the cumulative explained variance for the first 8 components.
"""
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-muted')
from sklearn.decomposition import PCA
import sys
#from sklearn.decomposition import IncrementalPCA
#from joblib import Parallel, delayed
import os
import pickle

os.environ["OMP_NUM_THREADS"] = "30"

pca_model_path  = sys.argv[1]
substitution_model = sys.argv[2]


def plot_pca_cumulative_variance_first_8(pca_model):
    with open(pca_model, "rb") as f:
        pca = pickle.load(f)

    # Compute explained variance ratio
    explained_variance = np.cumsum(pca.explained_variance_ratio_[:8])

    plt.figure(figsize=(8, 5))
    plt.bar(range(1, 9), explained_variance, color='royalblue', alpha=0.7)
    plt.xlabel('Principal Components', fontsize=12)
    plt.ylabel('Cumulative Explained Variance', fontsize=12)
    plt.title(f'Cumulative Explained Variance of First 8 Components\nSubstitution model: {substitution_model}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.grid(True, alpha=0.7)
    plt.savefig(f"pca_variance_plot_{substitution_model}.png", dpi=500, bbox_inches='tight')


if __name__ == "__main__":
    np.random.seed(42)
    plot_pca_cumulative_variance_first_8(pca_model_path)