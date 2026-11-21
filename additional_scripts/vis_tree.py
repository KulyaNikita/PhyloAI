#!/usr/bin/env python3
"""Deprecated: use `phyloai vis-tree`. Kept for compatibility with PhyloAI 1.x; will be removed later."""
import sys
from pathlib import Path

try:
    from phyloai.cli import main
except ImportError:  # Source checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from phyloai.cli import main

print("Note: vis_tree.py is deprecated; use `phyloai vis-tree`.", file=sys.stderr)
sys.exit(main(["vis-tree", *sys.argv[1:]]))
