#!/usr/bin/env python3
"""Deprecated: use `phyloai recode-6state`. Kept for compatibility with PhyloAI 1.x; will be removed later."""
import sys
from pathlib import Path

try:
    from phyloai.cli import main
except ImportError:  # Source checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from phyloai.cli import main

print("Note: recode_aa_frequencies_6state.py is deprecated; use `phyloai recode-6state`.", file=sys.stderr)
sys.exit(main(["recode-6state", *sys.argv[1:]]))
