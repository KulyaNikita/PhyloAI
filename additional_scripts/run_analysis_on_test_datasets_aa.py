#!/usr/bin/env python3
"""Deprecated: use `phyloai test-datasets`. Kept for compatibility with PhyloAI 1.x; will be removed later."""
import sys
from pathlib import Path

try:
    from phyloai.cli import main
except ImportError:  # Source checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from phyloai.cli import main

print("Note: run_analysis_on_test_datasets_aa.py is deprecated; use `phyloai test-datasets`.", file=sys.stderr)
sys.exit(main(["test-datasets", *sys.argv[1:]]))
