"""How much variance the first 8 PCA components keep (`phyloai pca-variance-top8`).

(c) Nikita Kulikov, Iker Irisarri, Christoph Mayer; LIB 2025

Fits PCA-8 to a frequency array and reports the retained variance. The array is loaded into RAM;
provide frequencies without an appended length column.
"""
import argparse


def main(argv=None, prog=None):
    parser = argparse.ArgumentParser(prog=prog, description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("frequencies", help="Frequency array (.npy)")
    args = parser.parse_args(argv)

    import numpy as np
    from sklearn.decomposition import PCA

    X = np.load(args.frequencies)

    pca = PCA(n_components=8)
    pca.fit_transform(X)

    # Compute the total variance explained by the first 8 components
    explained_variance = np.sum(pca.explained_variance_ratio_)

    print(f"Explained variance by the first 8 components: {explained_variance:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
