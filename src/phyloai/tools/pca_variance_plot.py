"""Plot the cumulative explained variance of the first 8 components of a fitted PCA model
(`phyloai pca-variance-plot`). Needs matplotlib.

(c) Nikita Kulikov, Iker Irisarri, Christoph Mayer; LIB 2025

Writes pca_variance_plot_<substitution model>.png in the current folder.
"""
import argparse
import importlib.util
import os
import pickle


def plot_pca_cumulative_variance_first_8(pca_model, substitution_model):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.style.use('seaborn-v0_8-muted')

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


def main(argv=None, prog=None):
    parser = argparse.ArgumentParser(prog=prog, description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pca_model", help="Pickled scikit-learn PCA model")
    parser.add_argument("substitution_model", help="Substitution model name, used in the title and file name")
    args = parser.parse_args(argv)
    if importlib.util.find_spec("matplotlib") is None:
        parser.error("this command needs matplotlib: conda install -c conda-forge matplotlib, "
                     "or pip install 'phyloai[extras]'")
    os.environ["OMP_NUM_THREADS"] = "30"
    import numpy as np
    np.random.seed(42)
    plot_pca_cumulative_variance_first_8(args.pca_model, args.substitution_model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
