#!/usr/bin/env python3
"""Deprecated: use `phyloai download-models`. Kept for compatibility with PhyloAI 1.x; will be removed later."""
import sys
from pathlib import Path

try:
    from phyloai.cli import main
except ImportError:  # Source checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from phyloai.cli import main

print("Note: download_protein_models.py is deprecated; use `phyloai download-models`.", file=sys.stderr)
# The old --output option (the networks folder) is now --models.
arguments = ['--models' if a == '--output' else a for a in sys.argv[1:]]
sys.exit(main(["download-models", *arguments]))
