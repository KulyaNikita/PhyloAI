#!/usr/bin/env python3
"""Deprecated: use `phyloai pca-variance-plot`. Kept for compatibility with PhyloAI 1.x; will be removed later."""
import sys
from pathlib import Path

try:
    from phyloai.cli import main
except ImportError:  # Source checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from phyloai.cli import main

print("Note: pca_plot_variance_explanation.py is deprecated; use `phyloai pca-variance-plot`.", file=sys.stderr)
sys.exit(main(["pca-variance-plot", *sys.argv[1:]]))
