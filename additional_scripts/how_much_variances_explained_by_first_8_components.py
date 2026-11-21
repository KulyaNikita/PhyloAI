#!/usr/bin/env python3
"""Deprecated: use `phyloai pca-variance-top8`. Kept for compatibility with PhyloAI 1.x; will be removed later."""
import sys
from pathlib import Path

try:
    from phyloai.cli import main
except ImportError:  # Source checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from phyloai.cli import main

print("Note: how_much_variances_explained_by_first_8_components.py is deprecated; use `phyloai pca-variance-top8`.", file=sys.stderr)
sys.exit(main(["pca-variance-top8", *sys.argv[1:]]))
