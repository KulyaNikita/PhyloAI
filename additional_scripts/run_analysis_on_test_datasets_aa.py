#!/usr/bin/env python3
"""Simulate and evaluate amino-acid quartet neural networks.

The runner supports raw 20-state DeepNNPhylogeny networks, PCA/GRP/SRP,
Fisher-selected Fisher-selected networks, length-aware PCA networks, and
Dayhoff/SR/KGB 6-state networks with or without alignment length.
Quartet frequencies and effective lengths are read in memory with
Quartet-Pattern-Counter 1.6. QPC 1.6 emits amino-acid site patterns in
the same alphabetical order used to train the PolyMoSim-based models.

For completely unambiguous model discovery, put ``model_manifest.json`` in
the network directory.  See ``--write-manifest-template``.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib
import json
import os
import pickle
import random
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from main_simulation_functions import make_tree_file_model
from recode_aa_frequencies_6state import SCHEMES, recode_batch


AA_MODELS = (
    "JTT", "LG", "WAG_OLD", "WAG", "WAG_STAR", "DAY",
    "Q.insect", "Q.mammal", "Q.pfam", "Q.plant", "Q.bird", "Q.yeast",
    "VT", "Blosum62", "PMB", "DCMut", "JTTDCMut",
)
RUNNER_VERSION = "20261022-protein-release"
IQTREE_MODELS = {
    "JTT": "JTT", "LG": "LG", "WAG_OLD": "WAG", "WAG": "WAG",
    "WAG_STAR": "WAG", "DAY": "DAYHOFF",
    "Q.insect": "Q.INSECT", "Q.mammal": "Q.MAMMAL", "Q.pfam": "Q.PFAM",
    "Q.plant": "Q.PLANT", "Q.bird": "Q.BIRD", "Q.yeast": "Q.YEAST",
    "VT": "VT", "Blosum62": "BLOSUM62", "PMB": "PMB",
    "DCMut": "DCMUT", "JTTDCMut": "JTTDCMUT",
}
POLYMOSIM_MODELS = {
    model: model for model in AA_MODELS
}
POLYMOSIM_MODELS.update({
    "Q.insect": "Qinsect", "Q.mammal": "Qmammal", "Q.pfam": "Qpfam",
    "Q.plant": "Qplant", "Q.bird": "Qbird", "Q.yeast": "Qyeast",
})
FEATURES_20 = 20**4
FEATURES_6 = 6**4
POLYMOSIM_AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
QPC_AA_ORDER = POLYMOSIM_AA_ORDER
TOPOLOGY_NAMES = ("AB|CD", "AC|BD", "AD|BC")
BRANCH_RANGES = {
    "farris": ((.1, .5), (.1, .5), (.01, .05), (.001, .02), (.001, .02), (.01, .05)),
    "felsenstein": ((.1, .5), (.001, .1), (.1, .5), (.01, .05), (.02, .05), (.01, .05)),
    "star": ((.001, .5), (.001, .5), (0., 0.), (.001, .5), (.001, .5), (0., 0.)),
    "longbr": ((.1, .5), (.1, .5), (.01, .25), (.1, .5), (.1, .5), (.01, .25)),
    "extralong": ((.5, 1.), (.5, 1.), (.01, .25), (.5, 1.), (.5, 1.), (.01, .25)),
    "shortbr": ((.001, .01), (.001, .01), (.01, .25), (.001, .01), (.001, .01), (.01, .25)),
    "singlong": ((.1, .5), (.001, .02), (.01, .25), (.001, .02), (.001, .02), (.01, .25)),
    "singshort": ((.001, .02), (.1, .5), (.01, .25), (.1, .5), (.1, .5), (.01, .25)),
    "shortint": ((.1, .5), (.1, .5), (.0005, .01), (.1, .5), (.1, .5), (.0005, .01)),
    "normbr": ((.1, .5), (.1, .5), (.05, .25), (.1, .5), (.1, .5), (.05, .25)),
    "intermedbr": ((.02, .5), (.02, .5), (.01, .05), (.02, .5), (.02, .5), (.01, .05)),
}


@dataclass
class ModelSpec:
    name: str
    model: str
    kind: str = "raw"              # raw, pca, grp, srp, length_aware, topology_aware, 6state
    reducer: str | None = None
    selector: str | None = None
    recoding: str | None = None     # dayhoff-6, sr-6, kgb-6
    length_input: str = "none"     # none, separate or appended
    length_transform: str = "raw"  # raw or log10


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--version", action="version", version=RUNNER_VERSION)
    p.add_argument("--models", "-ml_path", type=Path, required=True,
                   help="substitution-model directory containing networks and companion files, or collection root")
    p.add_argument("--substitution-model", "-subs_model", choices=AA_MODELS, default="WAG")
    p.add_argument("--latest-fisher-only", action="store_true",
                   help="evaluate only the current Fisher 10k/b1/log10 length-aware model; skip IQ-TREE and default to no bootstrap")
    p.add_argument("--model-name", action="append", default=[],
                   help="evaluate only this exact network directory name (repeatable)")
    p.add_argument("--branch-mode", "-brlen_mode", choices=tuple(BRANCH_RANGES), default="normbr")
    p.add_argument("--alignment-length", "-seqlen", type=int, default=1000)
    p.add_argument("--replicates", "-seqnum", type=int, default=1000,
                   help="simulated alignments per topology")
    p.add_argument("--bootstrap-replicates", type=int, default=None,
                   help="bootstrap replicates; default 100, or 0 with --latest-fisher-only")
    p.add_argument("--processes", type=int, default=1,
                   help="threads for Quartet-Pattern-Counter 1.6")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--output-root", type=Path, default=Path("aa_analysis_results"))
    p.add_argument("--run-name", help="stable output directory name; timestamp is used otherwise")
    p.add_argument("--force-rerun", action="store_true",
                   help="ignore matching completed runs and perform a new evaluation")
    p.add_argument("--poly-mosim", default="PolyMoSim",
                   help="PolyMoSim executable")
    p.add_argument("--iqtree", default="iqtree3", help="IQ-TREE executable")
    p.add_argument("--skip-iqtree", action="store_true",
                   help="evaluate networks only (useful if IQ-TREE is unavailable)")
    p.add_argument("--iqtree-only", action="store_true",
                   help="evaluate only IQ-TREE on simulated quartets; do not load or evaluate networks")
    p.add_argument("--counter-dir", type=Path,
                   default=PROJECT_ROOT / "quartet-pattern-counter-v.1.6")
    p.add_argument("--alignments", type=Path,
                   help="instead of simulation, use FASTA files below this directory; names must contain top0/top1/top2")
    p.add_argument("--save-features", action="store_true",
                   help="save raw 160000-feature arrays (large; off by default)")
    p.add_argument("--predict-batch-size", type=int, default=32)
    p.add_argument("--counter-verbose", type=int, default=0)
    p.add_argument("--write-manifest-template", action="store_true")
    p.add_argument("--inventory-only", action="store_true",
                   help="discover models and write metadata without simulating")
    a = p.parse_args(argv)
    if a.iqtree_only and (a.skip_iqtree or a.latest_fisher_only or a.model_name):
        p.error("--iqtree-only cannot be combined with --skip-iqtree, --latest-fisher-only or --model-name")
    if a.latest_fisher_only:
        a.skip_iqtree = True
    if a.bootstrap_replicates is None:
        a.bootstrap_replicates = 0 if a.latest_fisher_only else 100
    for key in ("alignment_length", "replicates", "processes", "predict_batch_size"):
        if getattr(a, key) <= 0:
            p.error(f"--{key.replace('_', '-')} must be positive")
    if a.bootstrap_replicates < 0:
        p.error("--bootstrap-replicates cannot be negative")
    if not a.skip_iqtree and 0 < a.bootstrap_replicates < 100:
        p.error("IQ-TREE standard bootstrap requires --bootstrap-replicates 0 or at least 100")
    return a


def _norm(text: str) -> str:
    return "".join(c for c in text.lower() if c.isalnum())

def _topology_selector_identity(text: str) -> str:
    """Normalize matching network/selector names to the same identity."""
    name = str(text)
    if name.lower().endswith(".npz"):
        name = name[:-4]
    identity = _norm(name)
    identity = identity.replace("fisher", "topologyawarelength")
    for prefix in ("phyloai", "toppred"):
        if identity.startswith(prefix):
            identity = identity[len(prefix):]
            break
    # Network directories describe their input convention, whereas selector
    # files commonly end in ``selector``. Neither suffix identifies the
    # feature set, so remove them for exact pairing.
    changed = True
    while changed:
        changed = False
        for suffix in ("selector", "nolength", "log10", "raw"):
            if identity.endswith(suffix):
                identity = identity[:-len(suffix)]
                changed = True
    return identity


def recoding_group_indices(scheme: str, aa_order: str) -> tuple[np.ndarray, ...]:
    """Build six-state groups without depending on helper API version.

    Older copies of ``recode_aa_frequencies_6state.py`` accepted only the
    scheme argument.  The runner needs indices in its already-reordered
    PolyMoSim feature order, so construct them explicitly here.
    """
    groups = SCHEMES[scheme]
    flattened = "".join(groups)
    if len(aa_order) != 20 or len(flattened) != 20 or set(flattened) != set(aa_order):
        raise ValueError(f"Invalid six-state scheme {scheme!r} or AA order {aa_order!r}")
    positions = {amino_acid: index for index, amino_acid in enumerate(aa_order)}
    return tuple(
        np.asarray([positions[amino_acid] for amino_acid in group], dtype=np.intp)
        for group in groups
    )


def write_manifest_template(folder: Path) -> None:
    template = {
        "models": [
            {"name": "DeepNNPhylogeny-WAG", "model": "DeepNNPhylogeny-WAG", "kind": "raw"},
            {"name": "PhyloAI-WAG-PCA", "model": "PhyloAI-WAG-PCA", "kind": "pca", "reducer": "WAG-pca.pkl"},
            {"name": "PhyloAI-WAG-length-aware-raw", "model": "PhyloAI-WAG-length-aware-raw", "kind": "length_aware", "reducer": "WAG-length-pca.pkl", "length_input": "separate", "length_transform": "raw"},
            {"name": "PhyloAI-WAG-length-aware-log10", "model": "PhyloAI-WAG-length-aware-log10", "kind": "length_aware", "reducer": "WAG-length-pca.pkl", "length_input": "separate", "length_transform": "log10"},
            {"name": "PhyloAI-KGB-6", "model": "PhyloAI-KGB-6", "kind": "6state", "recoding": "kgb-6"},
        ]
    }
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "model_manifest.template.json"
    path.write_text(json.dumps(template, indent=2) + "\n")
    print(path)


def is_current_fisher_name(name: str) -> bool:
    """Current empirical-test family, not the newest filesystem timestamp."""
    low = name.lower()
    # Public release names omit k10000/b1/log10; the release metadata
    # records those properties and is checked during discovery below.
    if low.startswith("phyloai-fisher-"):
        return True
    family = "fisher" in low or "topologyawarelength" in _norm(low)
    return bool(family and re.search(r"(?:^|[-_])k10000(?:[-_]|$)", low)
                and re.search(r"(?:^|[-_])b1(?:[-_]|$)", low) and low.endswith("log10"))


def discover_specs(folder: Path, latest_fisher_only: bool = False,
                   model_names: list[str] | None = None) -> list[ModelSpec]:
    manifest = folder / "model_manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text())
        items = [item for item in data.get("models", [])
                 if (not latest_fisher_only or is_current_fisher_name(item["name"]))
                 and (not model_names or item["name"] in model_names)]
        specs = [ModelSpec(**dict(item, length_input=item.get("length_input", "separate" if item.get("kind") == "length_aware" else "none"))) for item in items]
    else:
        network_dirs = sorted(p for p in folder.iterdir() if p.is_dir() and
                              ((p / "saved_model.pb").exists() or (p / "config.json").exists()))
        # Filter before companion-file discovery, so missing reducers for
        # excluded older networks cannot prevent a Fisher-only evaluation.
        network_dirs = [p for p in network_dirs
                        if (not latest_fisher_only or is_current_fisher_name(p.name))
                        and (not model_names or p.name in model_names)]
        reducers = sorted(folder.rglob("*.pkl"))
        selectors = sorted(folder.rglob("*.npz"))
        specs = []
        for model_path in network_dirs:
            low = model_path.name.lower()
            metadata_path = folder / (model_path.name + ".training.json")
            if not metadata_path.exists():
                metadata_path = folder / (model_path.name + ".json")
            metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
            if latest_fisher_only and low.startswith("phyloai-fisher-"):
                if (metadata.get("architecture") != "b1"
                        or metadata.get("selected_patterns") != 10000
                        or metadata.get("length_transform") != "log10"):
                    raise ValueError(f"{model_path.name} is not documented as a 10k/B1/log10 model")
            recoding = metadata.get("recoding") or next((s for s in SCHEMES if _norm(s) in _norm(low)), None)
            selector = None
            if recoding:
                kind = "6state"
            elif "fisher" in low or "topology-aware" in low or "topology_aware" in low or metadata.get("selector"):
                kind = "topology_aware"
            elif (
                "lenaw" in low
                or ("length" in low and "aware" in low)
                or "length-balanced" in low
                or "length_balanced" in low
            ):
                kind = "length_aware"
            else:
                kind = next((k for k in ("pca", "grp", "srp") if k in low), "pca" if low.endswith(".shallow") else "raw")
            reducer = None
            if kind == "topology_aware":
                count_match = re.search(r"(?:^|[-_])k(\d+)(?:[-_]|$)", low)
                candidates = [
                    path for path in selectors
                    if "topologyaware" in _norm(path.name) or "fisher" in _norm(path.name)
                ]
                shared = [
                    path for path in candidates
                    if any(_norm(model) in _norm(low) and
                           _norm(model) in _norm(path.name) for model in AA_MODELS)
                ]
                candidates = shared or candidates
                exact_identity = [
                    path for path in candidates
                    if _topology_selector_identity(path.name)
                    == _topology_selector_identity(model_path.name)
                ]
                # Exact identities prevent the basic selector from accidentally
                # matching a different training series or feature count.
                declared_selector = metadata.get("selector")
                declared_path = folder / declared_selector if declared_selector else None
                if exact_identity:
                    candidates = exact_identity
                    if declared_selector and declared_path not in exact_identity:
                        print(f"WARNING: {model_path.name}: stale selector metadata {declared_selector!r}; using exact filename match", file=sys.stderr)
                elif declared_path and declared_path in candidates:
                    candidates = [declared_path]
                if count_match:
                    token = "k" + count_match.group(1)
                    candidates = [
                        path for path in candidates if re.search(r"(?:^|[-_])" + token + r"(?:[-_.]|$)", path.name.lower())
                    ]
                architecture = next(
                    (name for name in ("b1", "b3")
                     if f"-{name}-" in low or f"_{name}_" in low), None
                )
                if architecture:
                    architecture_matches = [
                        path for path in candidates
                        if architecture in _norm(path.name)
                    ]
                    candidates = architecture_matches or candidates
                for qualifier in ("long1m",):
                    if qualifier in _norm(low):
                        qualifier_matches = [
                            path for path in candidates
                            if qualifier in _norm(path.name)
                        ]
                        candidates = qualifier_matches or candidates
                if len(candidates) != 1:
                    raise ValueError(
                        f"Expected one topology-aware selector for {model_path.name}; "
                        f"found {candidates}"
                    )
                selector = str(candidates[0].relative_to(folder))
            if kind in {"pca", "grp", "srp", "length_aware"}:
                token = "pca" if kind == "length_aware" else kind
                candidates = [r for r in reducers if token in r.name.lower()]
                # Keep ordinary PCA reducers and length-aware PCA reducers in
                # separate namespaces when both are stored in one directory.
                if kind == "length_aware":
                    length_candidates = [
                        r for r in candidates
                        if "length" in _norm(r.name) or "lenaw" in _norm(r.name)
                    ]
                    candidates = length_candidates or candidates
                elif kind == "pca":
                    ordinary_candidates = [
                        r for r in candidates
                        if "length" not in _norm(r.name) and "lenaw" not in _norm(r.name)
                    ]
                    candidates = ordinary_candidates or candidates
                # Balanced experimental networks must use their balanced PCA;
                # ordinary length-aware networks must not silently switch to
                # it when both reducer families share the same component count.
                model_is_balanced = "balanced" in _norm(low)
                candidates = [
                    r for r in candidates
                    if ("balanced" in _norm(r.name)) == model_is_balanced
                ]
                # Keep reducers from named experimental series separate.  In
                # particular, Iker-combined PCA32/64/128 reducers coexist with
                # older length-aware reducers having the same dimensions.
                for qualifier in ("iker", "combined"):
                    model_has_qualifier = qualifier in _norm(low)
                    candidates = [
                        r for r in candidates
                        if (qualifier in _norm(r.name)) == model_has_qualifier
                    ]
                # Prefer reducers sharing a substitution-model token.
                shared = [
                    r for r in candidates
                    if any(_norm(m) in _norm(low) and _norm(m) in _norm(r.name)
                           for m in AA_MODELS)
                ]
                candidates = shared or candidates
                # Length-aware model names and reducers include pca16, pca32,
                # etc.; use that component count to resolve multiple reducers.
                component_match = re.search(r"pca[_-]?(\d+)", low)
                if component_match:
                    dimension_matches = [
                        r for r in candidates if re.search(r"pca[_-]?" + component_match.group(1) + r"(?:\D|$)", r.name.lower())
                    ]
                    candidates = dimension_matches or candidates
                if len(candidates) == 1:
                    reducer = str(candidates[0].relative_to(folder))
                elif not candidates:
                    raise ValueError(f"No {token.upper()} reducer found for {model_path.name}")
                else:
                    raise ValueError(f"Ambiguous reducer for {model_path.name}: {candidates}; use model_manifest.json")
            inferred_length_transform = (
                "log10" if
                ("log10" in low or "loglength" in _norm(low)) else "raw"
            )
            # Selected-pattern networks were originally trained without an
            # alignment-length input.  Treat that as the safe default: a
            # topology-aware model is length-aware only when its directory
            # name explicitly says so (for example, "...-length-...-log10").
            topology_has_length = kind == "topology_aware" and (
                "fisher" in low
                or
                "lenaw" in low
                or ("length" in low and "aware" in low)
                or "log10" in low
                or "loglength" in _norm(low)
                or low.endswith("-raw")
                or low.endswith("_raw")
            ) and not ("no-length" in low or "nolength" in _norm(low))
            inferred_length_input = (
                "separate" if kind == "length_aware" or topology_has_length or (kind == "6state" and re.search(r"[-_]la(?:[-_]|$)", low))
                else "none"
            )
            if metadata.get("length_transform") == "none":
                inferred_length_input = "none"
                inferred_length_transform = "raw"
            elif metadata.get("length_transform") in {"raw", "log10"}:
                inferred_length_input = metadata.get("length_input", "separate")
                inferred_length_transform = metadata["length_transform"]
            specs.append(ModelSpec(
                name=model_path.name, model=str(model_path.relative_to(folder)),
                kind=kind, reducer=reducer, recoding=recoding,
                selector=selector,
                length_input=inferred_length_input,
                length_transform=inferred_length_transform,
            ))
    if model_names and set(model_names) != {s.name for s in specs}:
        raise ValueError(f"Requested networks not found or excluded: {set(model_names) - {s.name for s in specs}}")
    if latest_fisher_only and len(specs) != 1:
        raise ValueError(f"Expected exactly one current Fisher k10000/b1/log10 model in {folder}; found {[s.name for s in specs]}")
    valid_kinds = {"raw", "pca", "grp", "srp", "length_aware", "topology_aware", "6state"}
    for spec in specs:
        if ("fisher" in spec.name.lower()) and (
                spec.kind != "topology_aware" or spec.length_input == "none"):
            raise ValueError(f"Fisher model {spec.name} must use kind='topology_aware', its matching selector, and a separate length input. Check for a stale model_manifest.json overriding automatic discovery.")
        if spec.kind not in valid_kinds:
            raise ValueError(f"Unsupported kind {spec.kind!r} for {spec.name}")
        if spec.kind == "6state" and spec.recoding not in SCHEMES:
            raise ValueError(f"6-state model {spec.name} requires recoding in {tuple(SCHEMES)}")
        if spec.kind in {"pca", "grp", "srp", "length_aware"} and not spec.reducer:
            raise ValueError(f"Dimensionality-reduction model {spec.name} has no reducer")
        if spec.kind == "topology_aware" and not spec.selector:
            raise ValueError(f"Topology-aware model {spec.name} has no selector")
        if spec.length_transform not in {"raw", "log10"}:
            raise ValueError(
                f"Invalid length_transform {spec.length_transform!r} for {spec.name}; "
                "use 'raw' or 'log10'"
            )
        if spec.length_input not in {"none", "separate", "appended"}:
            raise ValueError(f"Invalid length_input for {spec.name}: {spec.length_input}")
        if not (folder / spec.model).exists():
            raise FileNotFoundError(folder / spec.model)
        if spec.reducer and not (folder / spec.reducer).exists():
            raise FileNotFoundError(folder / spec.reducer)
        if spec.selector and not (folder / spec.selector).exists():
            raise FileNotFoundError(folder / spec.selector)
    if not specs:
        raise ValueError(f"No TensorFlow/Keras networks found in {folder}")
    return specs


def import_counter(counter_dir: Path) -> Any:
    counter_dir = counter_dir.resolve()
    if re.search(r"v\.?1\.5(?:\D|$)", counter_dir.name, re.IGNORECASE):
        raise RuntimeError("QPC 1.5 emits the old amino-acid pattern order. This runner requires QPC 1.6; compile it in the cluster environment and update --counter-dir.")
    sys.path.insert(0, str(counter_dir))
    try:
        return importlib.import_module("QuartetPatternCounterModule")
    except ImportError as exc:
        raise RuntimeError(f"Cannot import Quartet-Pattern-Counter 1.6 from {counter_dir}. The extension must match this Python version.") from exc


def count_alignment(path: Path, counter_module: Any, threads: int, verbose: int = 0) -> tuple[np.ndarray, np.ndarray]:
    counter = counter_module.QuartetPatternCounter(str(path), "P", threads, verbose)
    frequencies = np.empty((1, FEATURES_20), dtype=np.float32)
    lengths = np.empty(1, dtype=np.uint32)
    n = counter.next_quartet_pattern_frequencies_with_lengths(frequencies, lengths)
    if n != 1:
        raise ValueError(f"Expected exactly four sequences (one quartet) in {path}; counter returned {n}")
    extra = counter.next_quartet_pattern_frequencies_with_lengths(frequencies, lengths)
    if extra != 0:
        raise ValueError(f"Expected one quartet in {path}, but more were returned")
    return frequencies[0].copy(), lengths.copy()[0:1]


def bootstrap_frequencies(path: Path, count: int, rng: np.random.Generator,
                          counter_module: Any, threads: int, work: Path) -> tuple[np.ndarray, np.ndarray]:
    if count == 0:
        return np.empty((0, FEATURES_20), np.float32), np.empty(0, np.uint32)
    names, seqs = read_fasta(path)
    widths = {len(s) for s in seqs}
    if len(seqs) != 4 or len(widths) != 1:
        raise ValueError(f"Bootstrap requires a four-sequence rectangular FASTA: {path}")
    site_count = widths.pop()
    output = np.empty((count, FEATURES_20), np.float32)
    lengths = np.empty(count, np.uint32)
    boot_path = work / "bootstrap.fasta"
    for index in range(count):
        sites = rng.integers(0, site_count, site_count)
        write_fasta(boot_path, names, ["".join(seq[j] for j in sites) for seq in seqs])
        counted, lengths[index:index + 1] = count_alignment(
            boot_path, counter_module, threads
        )
        output[index] = counted
    return output, lengths


def read_fasta(path: Path) -> tuple[list[str], list[str]]:
    names, seqs, current = [], [], []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if names:
                seqs.append("".join(current))
            names.append(line[1:].split()[0])
            current = []
        else:
            if not names:
                raise ValueError(f"Sequence before FASTA header in {path}")
            current.append(line)
    if names:
        seqs.append("".join(current))
    return names, seqs


def write_fasta(path: Path, names: Iterable[str], seqs: Iterable[str]) -> None:
    with path.open("w") as handle:
        for name, sequence in zip(names, seqs):
            handle.write(f">{name}\n{sequence}\n")


def felsenstein_branch(q: float) -> float:
    numerator = -18*q + 24*q*q + np.sqrt(243*q - 567*q*q + 648*q**3 - 288*q**4)
    return float(numerator / (9 - 24*q + 32*q*q))


def simulate_alignment(args: argparse.Namespace, topology: int, replicate: int,
                       work: Path) -> tuple[Path, dict[str, Any]]:
    seed = args.seed + topology * args.replicates + replicate
    rng = random.Random(seed)
    ranges = BRANCH_RANGES[args.branch_mode]
    branches = [rng.uniform(*limits) for limits in ranges]
    if args.branch_mode == "felsenstein":
        q = rng.uniform(*ranges[1])
        long = felsenstein_branch(q) * rng.uniform(1., 1.5)
        branches = [long, q, q/2, long, q, q/2]
    elif args.branch_mode != "star":
        branches[5] = branches[2]
    shape, pinv = rng.uniform(.01, 4.), rng.uniform(0., .5)
    prefix = f"top{topology}_rep{replicate:06d}"
    model_file, tree_file, fasta = work / f"{prefix}.model", work / f"{prefix}.tree", work / f"{prefix}.fasta"
    model_file.write_text(
        "begin aa-model\n"
        f"name: {args.substitution_model}_model\n"
        f"modeltype: {POLYMOSIM_MODELS[args.substitution_model]}\n"
        f"shape: {shape}\n"
        f"pinv: {pinv}\n"
        "end model\n"
    )
    make_tree_file_model(str(tree_file), topology, *branches, args.alignment_length,
                         f"{args.substitution_model}_model")
    with fasta.open("w") as output:
        result = subprocess.run([args.poly_mosim, "-s", str(seed), "-m", str(model_file),
                                 "-t", str(tree_file), "-f", "fasta", "-n", "1"],
                                stdout=output, stderr=subprocess.PIPE, text=True)
    if result.returncode:
        raise RuntimeError(f"PolyMoSim failed for {prefix}: {result.stderr.strip()}")
    model_file.unlink(); tree_file.unlink()
    return fasta, {"seed": seed, "shape": shape, "pinv": pinv,
                   **{f"branch_{i+1}": x for i, x in enumerate(branches)}}


def alignment_inputs(args: argparse.Namespace, work: Path) -> list[tuple[Path, int, int, dict[str, Any]]]:
    if args.alignments:
        paths = sorted(p for p in args.alignments.rglob("*") if p.is_file() and p.suffix.lower() in {".fa", ".fas", ".fasta", ".faa"})
        result = []
        counts = [0, 0, 0]
        for path in paths:
            match = re.search(r"(?:^|[_-])top(?:ology)?[_-]?([012])(?:[_-]|\.|$)", path.name, re.I)
            if not match:
                raise ValueError(f"Cannot determine true topology from filename: {path.name}")
            topology = int(match.group(1)); replicate = counts[topology]; counts[topology] += 1
            result.append((path, topology, replicate, {}))
        if not result:
            raise ValueError(f"No FASTA alignments found below {args.alignments}")
        return result
    result = []
    for topology in range(3):
        for replicate in range(args.replicates):
            path, parameters = simulate_alignment(args, topology, replicate, work)
            result.append((path, topology, replicate, parameters))
    return result


def newick_topology(newick: str, taxa: list[str] | None = None) -> int:
    """Return the unrooted A/B/C/D quartet class without BioNJ/ETE dependencies."""
    # Every binary quartet contains at least one two-leaf cherry. Either cherry
    # determines its unrooted split, including rooted IQ-TREE representations.
    taxa = taxa or ["A", "B", "C", "D"]
    if len(taxa) != 4:
        raise ValueError(f"Exactly four taxon names are required; got {taxa}")
    label = r"([^\s,:()]+)"
    cherries = re.findall(r"\(\s*" + label + r"(?:\s*:[^,()]+)?\s*,\s*" + label + r"(?:\s*:[^,()]+)?\s*\)", newick)
    a, b, c, d = taxa
    mapping = {frozenset((a, b)): 0, frozenset((c, d)): 0,
               frozenset((a, c)): 1, frozenset((b, d)): 1,
               frozenset((a, d)): 2, frozenset((b, c)): 2}
    for pair in cherries:
        if frozenset(pair) in mapping:
            return mapping[frozenset(pair)]
    raise ValueError(f"Cannot recognize A/B/C/D quartet topology: {newick.strip()}")


def run_iqtree(args: argparse.Namespace, alignment: Path, prefix: Path) -> tuple[int, list[int]]:
    taxa, _ = read_fasta(alignment)
    command = [args.iqtree, "-s", str(alignment), "-st", "AA", "-T", "1", "-m",
               f"{IQTREE_MODELS[args.substitution_model]}+I+G", "-pre", str(prefix), "-redo"]
    if args.bootstrap_replicates:
        # The original analysis used the standard nonparametric bootstrap.
        command.extend(("-b", str(args.bootstrap_replicates)))
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise RuntimeError(f"IQ-TREE failed for {alignment}:\n{result.stdout[-4000:]}")
    treefile = Path(f"{prefix}.treefile")
    if not treefile.exists():
        raise RuntimeError(f"IQ-TREE did not create {treefile}")
    inferred = newick_topology(treefile.read_text().splitlines()[0], taxa)
    boottrees = Path(f"{prefix}.boottrees")
    bootstrap = [newick_topology(line, taxa) for line in boottrees.read_text().splitlines() if line.strip()] if boottrees.exists() else []
    return inferred, bootstrap


def load_network(path: Path) -> Any:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required to load neural networks") from exc
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as first:
        # Keras 3 cannot load legacy SavedModel directories; TFSMLayer provides
        # inference while retaining the SavedModel endpoint.
        try:
            import keras
            return keras.layers.TFSMLayer(str(path), call_endpoint="serving_default")
        except Exception:
            raise RuntimeError(f"Could not load neural network {path}: {first}") from first


def prepare_features(spec: ModelSpec, raw: np.ndarray, lengths: np.ndarray,
                     reducer: Any | None) -> tuple[Any, tuple[int, ...]]:
    x = raw
    if spec.kind == "6state":
        # QPC 1.6 already emits the PolyMoSim training order; index the
        # six-state groups in that same order without rearranging features.
        x = recode_batch(
            np.asarray(raw, dtype=np.float32),
            recoding_group_indices(spec.recoding, POLYMOSIM_AA_ORDER),
        )
    elif spec.kind in {"pca", "grp", "srp", "length_aware"}:
        x = np.asarray(reducer.transform(raw))
    elif spec.kind == "topology_aware":
        indices = np.asarray(reducer["indices"], dtype=np.int64)
        x = np.asarray(raw[:, indices], dtype=np.float32)
        if spec.length_input == "none":
            return x, tuple(x.shape[1:])
    if spec.length_input != "none" and (spec.kind in {"length_aware", "topology_aware", "6state"}):
        if x.ndim != 2 or x.shape[1] < 1:
            raise ValueError(
                f"Length-aware model {spec.name} requires a non-empty 2D "
                f"reduced feature matrix; got {x.shape}"
            )
        length_column = lengths.astype(np.float32).reshape(-1, 1)
        if spec.length_transform == "log10":
            if np.any(length_column <= 0):
                raise ValueError(f"Cannot log-transform non-positive lengths for {spec.name}")
            length_column = np.log10(length_column)
        value = np.concatenate((x, length_column), axis=1) if spec.length_input == "appended" else [x, length_column]
        return value, (x.shape[1], 1)
    return x, tuple(x.shape[1:])


def validate_network_inputs(network: Any, features: Any, spec: ModelSpec) -> None:
    """Fail with model context instead of an opaque Keras input-count error."""
    expected = getattr(network, "inputs", None)
    if not expected:
        return
    supplied = features if isinstance(features, (list, tuple)) else [features]
    expected_shapes = [tuple(tensor.shape) for tensor in expected]
    supplied_shapes = [tuple(value.shape) for value in supplied]
    if len(expected) != len(supplied) or any(
            len(tensor.shape) != value.ndim or any(
                want is not None and int(want) != got
                for want, got in zip(tuple(tensor.shape)[1:], value.shape[1:]))
            for tensor, value in zip(expected, supplied)):
        raise ValueError(f"Model {spec.name}: expects {expected_shapes}, prepared {supplied_shapes}; kind={spec.kind}, selector={spec.selector}, reducer={spec.reducer}, length_input={spec.length_input}, length_transform={spec.length_transform}. Check the uploaded runner version and any stale model_manifest.json.")


def predict(network: Any, features: Any, batch_size: int) -> np.ndarray:
    try:
        output = network.predict(features, batch_size=batch_size, verbose=0)
    except AttributeError:
        output = network(features, training=False)
    if isinstance(output, dict):
        if len(output) != 1:
            raise ValueError(f"Network returned multiple outputs: {tuple(output)}")
        output = next(iter(output.values()))
    output = np.asarray(output)
    if output.ndim != 2 or output.shape[1] != 3:
        raise ValueError(f"Expected network output (samples, 3); got {output.shape}")
    return output


def predict_in_chunks(network: Any, spec: ModelSpec, raw: np.ndarray,
                      lengths: np.ndarray, reducer: Any | None,
                      batch_size: int) -> tuple[np.ndarray, tuple[int, ...]]:
    """Transform and predict without materialising the complete feature matrix.

    This is especially important for bootstrap data: even when ``raw`` is a
    disk-backed memmap, a reducer (or TensorFlow) may otherwise copy all rows
    into RAM at once.
    """
    probabilities = np.empty((len(raw), 3), dtype=np.float32)
    input_shape: tuple[int, ...] | None = None
    for start in range(0, len(raw), batch_size):
        stop = min(start + batch_size, len(raw))
        features, chunk_shape = prepare_features(
            spec, raw[start:stop], lengths[start:stop], reducer
        )
        if input_shape is None:
            validate_network_inputs(network, features, spec)
        chunk_probabilities = predict(network, features, batch_size)
        probabilities[start:stop] = chunk_probabilities
        if input_shape is None:
            input_shape = chunk_shape
    return probabilities, input_shape or ()


def write_csv_gz(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def write_human_summary(output: Path, summaries: list[dict[str, Any]]) -> None:
    """Write percentage-based rankings that are convenient to inspect or plot."""
    ranked = sorted(
        summaries,
        key=lambda row: (
            -float(row["accuracy"]),
            -(float(row["bootstrap_accuracy"]) if row["bootstrap_accuracy"] is not None else -1.0),
            str(row["model"]),
        ),
    )
    # Competition ranks: equal percentages share a rank (1, 1, 3), avoiding
    # arbitrary alphabetical ordering of tied models.
    accuracy_order = {
        row["model"]: 1 + sum(float(other["accuracy"]) > float(row["accuracy"]) for other in summaries)
        for row in summaries
    }
    bootstrap_order = {
        row["model"]: 1 + sum(
            other["bootstrap_accuracy"] is not None
            and float(other["bootstrap_accuracy"]) > float(row["bootstrap_accuracy"])
            for other in summaries
        )
        for row in summaries if row["bootstrap_accuracy"] is not None
    }
    rows = []
    for overall_rank, row in enumerate(ranked, 1):
        samples = int(row["samples"])
        bootstrap_total = int(row["bootstrap_predictions"])
        accuracy = float(row["accuracy"])
        bootstrap_accuracy = row["bootstrap_accuracy"]
        rows.append({
            "overall_rank": overall_rank,
            "model": row["model"],
            "model_type": row["kind"],
            "correct_quartets": int(round(accuracy * samples)),
            "total_quartets": samples,
            "quartets_correct_percent": round(100.0 * accuracy, 2),
            "quartet_accuracy_rank": accuracy_order[row["model"]],
            "correct_bootstraps": int(round(float(bootstrap_accuracy) * bootstrap_total)) if bootstrap_accuracy is not None else "",
            "total_bootstraps": bootstrap_total,
            "bootstraps_correct_percent": round(100.0 * float(bootstrap_accuracy), 2) if bootstrap_accuracy is not None else "",
            "bootstrap_accuracy_rank": bootstrap_order[row["model"]] if bootstrap_accuracy is not None else "",
        })
    fields = ["overall_rank", "model", "model_type", "correct_quartets", "total_quartets",
              "quartets_correct_percent", "quartet_accuracy_rank", "correct_bootstraps",
              "total_bootstraps", "bootstraps_correct_percent", "bootstrap_accuracy_rank"]
    with (output / "model_ranking_percent.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    lines = [
        "MODEL RANKING",
        "Ranking: quartet accuracy, with bootstrap accuracy used as the tie-breaker.",
        "",
    ]
    for row in rows:
        bootstrap = (f'{row["correct_bootstraps"]}/{row["total_bootstraps"]} '
                     f'({row["bootstraps_correct_percent"]:.2f}%)') if row["bootstraps_correct_percent"] != "" else "not calculated"
        lines.extend((
            f'{row["overall_rank"]}. {row["model"]} [{row["model_type"]}]',
            f'   Correct quartets:   {row["correct_quartets"]}/{row["total_quartets"]} ({row["quartets_correct_percent"]:.2f}%)',
            f'   Correct bootstraps: {bootstrap}',
            "",
        ))
    (output / "MODEL_RANKING.txt").write_text("\n".join(lines))


def find_completed_run(args: argparse.Namespace, specs: list[ModelSpec]) -> Path | None:
    """Reuse only finalized results with matching settings and model inventory."""
    if args.inventory_only or getattr(args, "force_rerun", False) or not args.output_root.exists():
        return None
    keys = ("substitution_model", "branch_mode", "alignment_length", "replicates",
            "bootstrap_replicates", "seed", "skip_iqtree",
            "save_features", "poly_mosim", "iqtree")
    expected = [asdict(spec) for spec in specs]
    for folder in sorted(args.output_root.iterdir(), reverse=True):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        try:
            config = json.loads((folder / "config.json").read_text())
            if not config.get("finished_utc") or config.get("inventory_only"):
                continue
            if any(config.get(key) != getattr(args, key) for key in keys):
                continue
            if any((Path(config[key]).resolve() if config.get(key) else None) !=
                   (Path(getattr(args, key)).resolve() if getattr(args, key) else None)
                   for key in ("models", "counter_dir", "alignments")):
                continue
            if config.get("feature_order_conversion") != "none_QPC_1.6_matches_PolyMoSim":
                continue
            if json.loads((folder / "model_inventory.json").read_text()) != expected:
                continue
            required = {"samples.csv", "summary.csv", "predictions.csv.gz", "bootstrap_predictions.csv.gz"}
            recorded = config.get("files", {})
            if not required.issubset(recorded):
                continue
            if any(not (folder / name).is_file() or (folder / name).stat().st_size != entry["bytes"]
                   for name, entry in recorded.items()):
                continue
            with (folder / "summary.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            names = {spec.name for spec in specs}
            if not args.skip_iqtree:
                names.add("IQ-TREE")
            if {row["model"] for row in rows} != names or len(rows) != len(names):
                continue
            return folder
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


def run(args: argparse.Namespace) -> Path:
    # Accept either a single substitution-model directory or the collection root.
    if not any((p / "saved_model.pb").exists() or (p / "config.json").exists()
               for p in args.models.iterdir() if p.is_dir()) and not (args.models / "model_manifest.json").exists():
        model_token = "dayhoff" if args.substitution_model == "DAY" else _norm(args.substitution_model)
        matches = [p for p in args.models.iterdir() if p.is_dir() and _norm(p.name) == model_token]
        if len(matches) != 1:
            raise ValueError(f"Cannot resolve {args.substitution_model} model directory inside {args.models}: {matches}")
        args.models = matches[0]
    specs = [] if args.iqtree_only else discover_specs(args.models, args.latest_fisher_only, args.model_name)
    completed = find_completed_run(args, specs)
    if completed is not None:
        print(f"SKIPPED: matching completed run already exists: {completed}")
        return completed
    started = datetime.now(timezone.utc)
    run_name = args.run_name or started.strftime("%Y%m%dT%H%M%SZ") + f"_{args.substitution_model}_{args.branch_mode}_L{args.alignment_length}_n{args.replicates}"
    final_output = args.output_root / run_name
    output = args.output_root / f".{run_name}.incomplete"
    if final_output.exists() or output.exists():
        raise FileExistsError(f"Run or incomplete run directory already exists for: {final_output}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True)
    (output / "model_inventory.json").write_text(json.dumps([asdict(s) for s in specs], indent=2) + "\n")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"run_name": run_name, "started_utc": started.isoformat(), "schema_version": 2,
                   "runner_version": RUNNER_VERSION,
                   "topology_labels": dict(enumerate(TOPOLOGY_NAMES)), "counter_version": "1.6",
                   "qpc_amino_acid_order": QPC_AA_ORDER,
                   "model_feature_amino_acid_order": POLYMOSIM_AA_ORDER,
                   "feature_order_conversion": "none_QPC_1.6_matches_PolyMoSim"})
    (output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    if args.inventory_only:
        output.rename(final_output)
        return final_output

    counter_module = import_counter(args.counter_dir)
    with tempfile.TemporaryDirectory(prefix="aa_quartets_") as tmp:
        work = Path(tmp)
        inputs = alignment_inputs(args, work)
        # The raw 20-state representation is 640 KB per row.  Bootstrap runs
        # can therefore be hundreds of GB, so keep both matrices on disk and
        # only page small prediction chunks into memory.
        # Put the cache beside the requested output, rather than in /tmp (which
        # is commonly a small RAM-backed filesystem on compute nodes).
        feature_dir = output
        raw_path = feature_dir / ("site_pattern_frequencies.npy" if args.save_features else ".site_pattern_frequencies.tmp.npy")
        boot_raw_path = feature_dir / ("bootstrap_site_pattern_frequencies.npy" if args.save_features else ".bootstrap_site_pattern_frequencies.tmp.npy")
        raw = np.lib.format.open_memmap(
            raw_path, mode="w+", dtype=np.float32,
            shape=(len(inputs), FEATURES_20),
        )
        bootstrap_count = len(inputs) * args.bootstrap_replicates
        boot_raw_array = (
            np.lib.format.open_memmap(
                boot_raw_path, mode="w+", dtype=np.float32,
                shape=(bootstrap_count, FEATURES_20),
            )
            if bootstrap_count else np.empty((0, FEATURES_20), np.float32)
        )
        lengths = np.empty(len(inputs), np.uint32)
        boot_length_array = np.empty(bootstrap_count, np.uint32)
        meta_rows, boot_parent = [], []
        rng = np.random.default_rng(args.seed)
        for row, (path, topology, replicate, parameters) in enumerate(inputs):
            counted, lengths[row:row+1] = count_alignment(
                path, counter_module, args.processes, args.counter_verbose
            )
            raw[row] = counted
            braw, blength = bootstrap_frequencies(path, args.bootstrap_replicates, rng, counter_module, args.processes, work)
            boot_start = row * args.bootstrap_replicates
            boot_stop = boot_start + len(blength)
            boot_raw_array[boot_start:boot_stop] = braw
            boot_length_array[boot_start:boot_stop] = blength
            boot_parent.extend([row] * len(blength))
            meta_rows.append({"sample_id": row, "source": str(path), "topology": topology,
                              "topology_name": TOPOLOGY_NAMES[topology], "replicate": replicate,
                              "effective_length": int(lengths[row]), **parameters})
        raw.flush()
        if isinstance(boot_raw_array, np.memmap):
            boot_raw_array.flush()
        prediction_rows, bootstrap_rows, summaries, inventory = [], [], [], []
        true = np.asarray([int(r["topology"]) for r in meta_rows])
        if not args.skip_iqtree:
            iq_classes = np.empty(len(inputs), dtype=int)
            iq_boot_correct: list[int] = []
            iq_boot_count = 0
            for i, ((path, _, _, _), meta) in enumerate(zip(inputs, meta_rows)):
                inferred, iq_boot = run_iqtree(args, path, work / f"iqtree_{i:06d}")
                iq_classes[i] = inferred
                prediction_rows.append({"run_name": run_name, "model": "IQ-TREE", **meta,
                                        "predicted_topology": inferred, "correct": int(inferred == true[i]),
                                        "confidence": "", "prob_top0": "", "prob_top1": "", "prob_top2": ""})
                for boot_id, boot_topology in enumerate(iq_boot):
                    correct = int(boot_topology == true[i]); iq_boot_correct.append(correct)
                    bootstrap_rows.append({"run_name": run_name, "model": "IQ-TREE", "sample_id": i,
                                           "bootstrap_id": boot_id, "topology": int(true[i]),
                                           "predicted_topology": boot_topology, "correct": correct,
                                           "confidence": "", "prob_top0": "", "prob_top1": "", "prob_top2": "",
                                           "effective_length": int(lengths[i])})
                    iq_boot_count += 1
            iq_by_topology = [float((iq_classes[true == t] == t).mean()) if np.any(true == t) else None for t in range(3)]
            summaries.append({"run_name": run_name, "model": "IQ-TREE", "kind": "maximum_likelihood",
                              "recoding": "", "input_shape": "alignment", "samples": len(true),
                              "accuracy": float((iq_classes == true).mean()),
                              "accuracy_top0": iq_by_topology[0], "accuracy_top1": iq_by_topology[1],
                              "accuracy_top2": iq_by_topology[2], "bootstrap_predictions": iq_boot_count,
                              "bootstrap_accuracy": float(np.mean(iq_boot_correct)) if iq_boot_correct else None,
                              "mean_confidence": None})
        for spec in specs:
            if spec.selector:
                with np.load(args.models / spec.selector, allow_pickle=False) as selector_data:
                    if "training_order" in selector_data and str(selector_data["training_order"].item()) != POLYMOSIM_AA_ORDER:
                        raise ValueError(f"Selector {spec.selector} uses a different amino-acid pattern order")
                    selector_key = next(
                        (key for key in ("indices", "selected_indices")
                         if key in selector_data), None
                    )
                    if selector_key is None:
                        raise ValueError(
                            f"Selector {spec.selector} must contain 'indices' "
                            "(or 'selected_indices')"
                        )
                    indices = np.asarray(selector_data[selector_key], dtype=np.int64).reshape(-1)
                    if not len(indices):
                        raise ValueError(f"Selector {spec.selector} contains no pattern indices")
                    if np.any(indices < 0) or np.any(indices >= FEATURES_20):
                        raise ValueError(
                            f"Selector {spec.selector} contains indices outside 0..{FEATURES_20 - 1}"
                        )
                    if len(np.unique(indices)) != len(indices):
                        raise ValueError(f"Selector {spec.selector} contains duplicate indices")
                    reducer = {"indices": indices.copy()}
            elif spec.reducer:
                with (args.models / spec.reducer).open("rb") as reducer_handle:
                    reducer = pickle.load(reducer_handle)
            else:
                reducer = None
            network = load_network(args.models / spec.model)
            print(f"Model {spec.name}: kind={spec.kind}, selector={spec.selector}, reducer={spec.reducer}, length_input={spec.length_input}, length_transform={spec.length_transform}", flush=True)
            probabilities, input_shape = predict_in_chunks(
                network, spec, raw, lengths, reducer, args.predict_batch_size
            )
            classes = probabilities.argmax(axis=1)
            if len(boot_raw_array):
                boot_prob, _ = predict_in_chunks(
                    network, spec, boot_raw_array, boot_length_array, reducer,
                    args.predict_batch_size,
                )
            else:
                boot_prob = np.empty((0, 3))
            boot_classes = boot_prob.argmax(axis=1) if len(boot_prob) else np.empty(0, int)
            for i, (meta, prob, predicted) in enumerate(zip(meta_rows, probabilities, classes)):
                prediction_rows.append({"run_name": run_name, "model": spec.name, **meta,
                                        "predicted_topology": int(predicted), "correct": int(predicted == true[i]),
                                        "confidence": float(prob.max()), "prob_top0": float(prob[0]),
                                        "prob_top1": float(prob[1]), "prob_top2": float(prob[2])})
            parent_boot_correct = []
            for j, (parent, prob, predicted) in enumerate(zip(boot_parent, boot_prob, boot_classes)):
                expected = true[parent]; correct = int(predicted == expected); parent_boot_correct.append((parent, correct))
                bootstrap_rows.append({"run_name": run_name, "model": spec.name, "sample_id": parent,
                                       "bootstrap_id": j % max(1, args.bootstrap_replicates), "topology": int(expected),
                                       "predicted_topology": int(predicted), "correct": correct,
                                       "confidence": float(prob.max()), "prob_top0": float(prob[0]),
                                       "prob_top1": float(prob[1]), "prob_top2": float(prob[2]),
                                       "effective_length": int(boot_length_array[j])})
            accuracy_by_topology = [float((classes[true == t] == t).mean()) if np.any(true == t) else None for t in range(3)]
            summaries.append({"run_name": run_name, "model": spec.name, "kind": spec.kind,
                              "recoding": spec.recoding or "", "input_shape": json.dumps(input_shape),
                              "samples": len(true), "accuracy": float((classes == true).mean()),
                              "accuracy_top0": accuracy_by_topology[0], "accuracy_top1": accuracy_by_topology[1],
                              "accuracy_top2": accuracy_by_topology[2],
                              "bootstrap_predictions": len(boot_classes),
                              "bootstrap_accuracy": float(np.mean(boot_classes == true[np.asarray(boot_parent)])) if len(boot_classes) else None,
                              "mean_confidence": float(probabilities.max(axis=1).mean())})
            inventory.append({**asdict(spec), "input_shape": input_shape})
            del network

        # Closing/unlinking the temporary memmaps releases the large disk
        # cache before result checksums are calculated.
        del raw, boot_raw_array
        if not args.save_features:
            raw_path.unlink()
            if boot_raw_path.exists():
                boot_raw_path.unlink()

    meta_fields = sorted({key for row in meta_rows for key in row})
    with (output / "samples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=meta_fields); writer.writeheader(); writer.writerows(meta_rows)
    pred_fields = ["run_name", "model", "sample_id", "source", "topology", "topology_name", "replicate",
                   "effective_length", "predicted_topology", "correct", "confidence", "prob_top0", "prob_top1", "prob_top2"]
    write_csv_gz(output / "predictions.csv.gz", prediction_rows, pred_fields)
    boot_fields = ["run_name", "model", "sample_id", "bootstrap_id", "topology", "effective_length",
                   "predicted_topology", "correct", "confidence", "prob_top0", "prob_top1", "prob_top2"]
    write_csv_gz(output / "bootstrap_predictions.csv.gz", bootstrap_rows, boot_fields)
    summary_fields = ["run_name", "model", "kind", "recoding", "input_shape", "samples", "accuracy",
                      "accuracy_top0", "accuracy_top1", "accuracy_top2", "bootstrap_predictions",
                      "bootstrap_accuracy", "mean_confidence"]
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields); writer.writeheader(); writer.writerows(summaries)
    write_human_summary(output, summaries)
    config["finished_utc"] = datetime.now(timezone.utc).isoformat()
    config["duration_seconds"] = (datetime.now(timezone.utc) - started).total_seconds()
    config["files"] = {p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                       for p in output.iterdir() if p.is_file() and p.name != "config.json"}
    (output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    output.rename(final_output)
    return final_output


class _nullcontext:
    def __enter__(self): return None
    def __exit__(self, *args): return False


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.write_manifest_template:
        write_manifest_template(args.models)
        return 0
    try:
        output = run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
