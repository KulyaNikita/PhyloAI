#!/usr/bin/env python3
"""Deprecated: use `phyloai train`. Kept for compatibility with PhyloAI 1.x; will be removed later."""
import sys
from pathlib import Path

try:
    from phyloai.cli import main
except ImportError:  # Source checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from phyloai.cli import main

print("Note: train_phyloai.py is deprecated; use `phyloai train`.", file=sys.stderr)
sys.exit(main(["train", *sys.argv[1:]]))
