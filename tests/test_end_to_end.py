"""`phyloai run` on the example alignment, with the downloaded networks.

Skipped unless the Fisher networks are in the data folder (`phyloai download-models --family fisher
--no-detector`, or set PHYLOAI_DATA) and ASTRAQ is installed. --substitution-model LG avoids ModelDetector.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from phyloai import paths

EXAMPLE = Path(__file__).resolve().parents[1] / "proteins.fasta"


@pytest.mark.skipif(not (paths.models_dir() / "PhyloAI-Fisher-LG" / "saved_model.pb").is_file(),
                    reason="Fisher networks not downloaded")
@pytest.mark.skipif(shutil.which("ASTRAQ") is None, reason="ASTRAQ not installed")
def test_run_lg_astraq(tmp_path):
    pytest.importorskip("tensorflow")
    output = tmp_path / "results"
    result = subprocess.run([sys.executable, "-m", "phyloai", "run", "--alignment", str(EXAMPLE),
                             "--substitution-model", "LG", "--method", "astraq", "--output", str(output)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    tree = (output / "astraq.tre").read_text()
    names = [line[1:].split()[0] for line in EXAMPLE.read_text().splitlines() if line.startswith(">")]
    assert all(name in tree for name in names)
    assert (output / "run_summary.json").is_file()
