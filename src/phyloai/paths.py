"""Where PhyloAI keeps its downloaded models.

The data folder is, in this order:
  1. the PHYLOAI_DATA environment variable
  2. $XDG_DATA_HOME/phyloai, or ~/.local/share/phyloai (Linux)
     ~/Library/Application Support/phyloai (macOS)

It contains the protein networks in `models/` and ModelDetector in `ModelDetector.h5`.
Command-line options (--models, --detector, --output) take precedence over the data folder.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

MODELS_SUBDIR = "models"
DETECTOR_FILE = "ModelDetector.h5"


def data_dir() -> Path:
    if os.environ.get("PHYLOAI_DATA"):
        return Path(os.environ["PHYLOAI_DATA"]).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "phyloai"
    base = os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share"
    return Path(base) / "phyloai"


def models_dir() -> Path:
    return data_dir() / MODELS_SUBDIR


def detector_path() -> Path:
    return data_dir() / DETECTOR_FILE
