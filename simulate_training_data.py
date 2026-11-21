#!/usr/bin/env python3
"""Deprecated: use `phyloai simulate`. Kept for compatibility with PhyloAI 1.x; will be removed later."""
import sys
from pathlib import Path

try:
    from phyloai.cli import main
except ImportError:  # Source checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from phyloai.cli import main

print("Note: simulate_training_data.py is deprecated; use `phyloai simulate`.", file=sys.stderr)
sys.exit(main(["simulate", *sys.argv[1:]]))
