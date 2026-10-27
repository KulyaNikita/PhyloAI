#!/usr/bin/env python3

"""

(c) Nikita Kulikov, Iker Irisarri, Christoph Mayer; LIB 2025

A code to see how much information is preserved by first 8 components

"""
import sys
import numpy as np
from sklearn.decomposition import PCA

frequency_path = sys.argv[1]

X = np.load(frequency_path)

pca = PCA(n_components=8)
X_transformed = pca.fit_transform(X)

# Compute the total variance explained by the first 8 components
explained_variance = np.sum(pca.explained_variance_ratio_)

print(f"Explained variance by the first 8 components: {explained_variance:.4f}")