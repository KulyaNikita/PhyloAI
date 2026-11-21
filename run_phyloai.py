#!/usr/bin/env python3
"""Deprecated: use `phyloai run`. Kept for compatibility with PhyloAI 1.x; will be removed later."""
import sys
from pathlib import Path

try:
    from phyloai.cli import main
except ImportError:  # Source checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from phyloai.cli import main

print("Note: run_phyloai.py is deprecated; use `phyloai run`.", file=sys.stderr)
# PhyloAI 1.x kept the models in the repository folder. Use them if they are there.
arguments = sys.argv[1:]
here = Path(__file__).resolve().parent
for option, old_location in (("--models", here / "PhyloAI-Fisher-Kaggle"), ("--detector", here / "ModelDetector.h5")):
    given = any(a == option or a.startswith(option + "=") for a in arguments)
    if not given and old_location.exists():
        arguments += [option, str(old_location)]
sys.exit(main(["run", *arguments]))
