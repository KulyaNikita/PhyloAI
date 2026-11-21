"""Check the PhyloAI installation (`phyloai check`).

Reports the Python packages, the external programs (ASTRAQ, ASTRAL, QFM-FI, PolyMoSim, Java)
and the downloaded models. Exits with status 1 if something required is missing.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import shutil
import subprocess
import sys
from importlib import metadata

from phyloai import __version__, paths

# (import name, distribution name for the version)
PACKAGES = [("numpy", "numpy"), ("tensorflow", "tensorflow"), ("sklearn", "scikit-learn"),
            ("Bio", "biopython"), ("QuartetPatternCounterModule", "QuartetPatternCounterModule")]
# (import name, distribution name, what it is needed for)
OPTIONAL_PACKAGES = [("tf_keras", "tf-keras", "phyloai train with TensorFlow >=2.16"),
                     ("matplotlib", "matplotlib", "phyloai pca-variance-plot"),
                     ("ete3", "ete3", "phyloai vis-tree")]
# (command, version arguments or None, what it is used for)
PROGRAMS = [("ASTRAQ", ["--version"], "phyloai run --method astraq/astraqw (default)"),
            ("astral", None, "phyloai run --method astral"),
            ("QFM-FI", None, "phyloai run --method qfm"),
            ("java", ["-version"], "ASTRAL and QFM-FI"),
            ("PolyMoSim", ["--version"], "phyloai simulate, phyloai test-datasets")]
FISHER_NETWORKS = 15
SIX_STATE_NETWORKS = 3


def package_version(module: str, distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        pass
    try:
        return getattr(importlib.import_module(module), "__version__", "installed")
    except ImportError:
        return None


def program_version(command: str, arguments: list[str] | None) -> str:
    if arguments is None:
        return ""
    try:
        result = subprocess.run([command, *arguments], capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    lines = [line.strip() for line in (result.stdout + result.stderr).splitlines() if line.strip()]
    return next((line for line in lines if "version" in line.lower()), lines[0] if lines else "")


def main(argv=None, prog=None):
    parser = argparse.ArgumentParser(prog=prog, description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-models", action="store_true", help="Do not check the downloaded models")
    args = parser.parse_args(argv)
    problems = []

    def report(ok: bool, name: str, detail: str, required: bool = True):
        mark = "ok" if ok else ("MISSING" if required else "missing (optional)")
        print(f"  {mark:<20s} {name:<28s} {detail}")
        if required and not ok:
            problems.append(name)

    print(f"phyloai {__version__}, Python {sys.version.split()[0]} ({sys.executable})")

    print("\nPython packages:")
    for module, distribution in PACKAGES:
        version = package_version(module, distribution)
        report(version is not None, distribution, version or "")
    try:
        import QuartetPatternCounterModule as qpc

        from phyloai.run import check_qpc
        check_qpc(qpc)
    except ImportError:
        pass  # Already reported above.
    except RuntimeError as exc:
        report(False, "QPC version and pattern order", str(exc))
    for module, distribution, needed_for in OPTIONAL_PACKAGES:
        version = package_version(module, distribution)
        report(version is not None, distribution, version or f"needed for {needed_for}", required=False)

    print("\nPrograms:")
    for command, version_arguments, used_for in PROGRAMS:
        path = shutil.which(command)
        detail = f"{path}  {program_version(command, version_arguments)}".strip() if path else f"needed for {used_for}"
        report(path is not None, command, detail)

    if not args.no_models:
        models, detector = paths.models_dir(), paths.detector_path()
        print(f"\nModels (data folder {paths.data_dir()}; set PHYLOAI_DATA to change it):")
        fisher = len(list(models.glob("PhyloAI-Fisher-*/saved_model.pb")))
        six_state = len(list(models.glob("PhyloAI-*-6-LA/saved_model.pb")))
        report(fisher == FISHER_NETWORKS, "Fisher networks", f"{fisher}/{FISHER_NETWORKS} in {models}")
        report(six_state == SIX_STATE_NETWORKS, "6-state-LA networks", f"{six_state}/{SIX_STATE_NETWORKS} in {models}")
        if detector.is_file():
            from phyloai.models import DETECTOR_SHA256
            verified = hashlib.sha256(detector.read_bytes()).hexdigest() == DETECTOR_SHA256
            report(verified, "ModelDetector", f"{detector}" + ("" if verified else " (checksum differs)"))
        else:
            report(False, "ModelDetector", f"{detector} not found")
        if fisher != FISHER_NETWORKS or six_state != SIX_STATE_NETWORKS or not detector.is_file():
            print("  Download the models with: phyloai download-models")

    if problems:
        print(f"\n{len(problems)} problem(s): {', '.join(problems)}")
        return 1
    print("\nEverything needed is installed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
