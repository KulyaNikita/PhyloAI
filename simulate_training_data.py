#!/usr/bin/env python3
"""Simulate protein quartet training data with PolyMoSim.

The scheme contains '<alignment length> <replicates per topology/branch regime>'.
All scheme entries are simulated in one run, without sharding. Alternatively,
use --alignment-length and --replicates for fixed-length data. Outputs include
full pattern frequencies, labels, and raw lengths for train_phyloai.py.
"""

import argparse
import os
import random
import subprocess
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import main_simulation_functions as msf


AA_MODELS = [
    "JTT", "LG", "WAG_OLD", "WAG", "WAG_STAR", "DAY", "Qplant",
    "Qpfam", "Qmammal", "Qinsect", "Qbird", "Qlg", "Qyeast", "VT",
    "PMB", "JTTDCMut", "DCMut", "Blosum62",
]


def parse_length_scheme(filename):
    scheme = []
    with open(filename, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) != 2:
                raise ValueError(
                    f"{filename}:{line_number}: expected '<length> <replicates>'"
                )
            length, replicates = map(int, fields)
            if length <= 0 or replicates <= 0:
                raise ValueError(
                    f"{filename}:{line_number}: values must be positive integers"
                )
            scheme.append((length, replicates))
    if not scheme:
        raise ValueError(f"{filename}: the length scheme is empty")
    return scheme


@lru_cache(maxsize=None)
def parse_custom_aa_matrix(filename):
    rows = []
    with open(filename, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                rows.append([float(value) for value in stripped.split()])
    expected_widths = list(range(1, 20)) + [20]
    if [len(row) for row in rows] != expected_widths:
        raise ValueError(
            f"{filename}: expected 19 lower-triangle rows of widths 1..19 "
            "followed by one row of 20 amino-acid frequencies"
        )
    rates = tuple(value for row in rows[:19] for value in row)
    frequencies = tuple(rows[19])
    if any(value <= 0 for value in rates):
        raise ValueError(f"{filename}: all exchangeabilities must be positive")
    if any(value <= 0 for value in frequencies):
        raise ValueError(f"{filename}: all amino-acid frequencies must be positive")
    if not np.isclose(sum(frequencies), 1.0, atol=1e-5):
        raise ValueError(f"{filename}: amino-acid frequencies must sum to 1")
    return rates, frequencies


def make_custom_aa_model_file(filename, matrix_filename, shape, pinv, model_name):
    rates, frequencies = parse_custom_aa_matrix(matrix_filename)
    with open(filename, "w", encoding="utf-8") as model_file:
        model_file.write("begin aa-model\n")
        model_file.write(f"name: {model_name}\n")
        model_file.write("modeltype: User\n")
        model_file.write("rrates_lower: " + " ".join(map(str, rates)) + "\n")
        model_file.write("basefreq: " + " ".join(map(str, frequencies)) + "\n")
        model_file.write(f"shape: {shape}\n")
        model_file.write(f"pinv: {pinv}\n")
        model_file.write("end model\n")


def simulate_one(task):
    (
        topology,
        seed,
        internal_min,
        internal_max,
        sequence_length,
        substitution_model,
        run_id,
        workdir,
        custom_matrix,
        poly_mosim,
    ) = task

    rng = random.Random(seed)
    model_filename = workdir / f"Modelfile_free{run_id}_{seed}.txt"
    sequence_filename = workdir / f"sim-sequences{run_id}_{seed}.txt"
    tree_filename = workdir / f"sim_tree{run_id}_{seed}.txt"
    model_name = substitution_model + "_model"

    shape = rng.uniform(0.01, 4.0)
    pinv = rng.uniform(0.0, 0.5)
    r1 = rng.uniform(0.02, 0.5)
    r2 = rng.uniform(0.02, 0.5)
    r3 = rng.uniform(internal_min, internal_max) / 2
    r4 = rng.uniform(0.02, 0.5)
    r5 = rng.uniform(0.02, 0.5)
    r6 = r3
    tstv = rng.uniform(1.0, 3.0)
    basefreq_1 = rng.gauss(0.5, 0.03) / 2
    while not 0.2 <= basefreq_1 <= 0.3:
        basefreq_1 = rng.gauss(0.5, 0.03) / 2
    basefreq_2 = (1 - 2 * basefreq_1) / 2
    rates = [rng.uniform(0.1, 1.0) for _ in range(6)]

    try:
        if custom_matrix is None:
            msf.make_model_file(
                str(model_filename), substitution_model, shape, pinv, tstv,
                float(basefreq_1), float(basefreq_2), *rates, model_name,
            )
        else:
            make_custom_aa_model_file(
                model_filename, custom_matrix, shape, pinv, model_name
            )
        msf.make_tree_file_model(
            str(tree_filename), topology, r1, r2, r3, r4, r5, r6,
            sequence_length, model_name,
        )
        with open(sequence_filename, "w", encoding="utf-8") as output:
            subprocess.run(
                [
                    poly_mosim, "-s", str(seed),
                    "-m", str(model_filename), "-t", str(tree_filename),
                    "-f", "site_pattern_freq_relative_fill", "-n", "1",
                ],
                stdout=output,
                check=True,
            )
        return msf.find_frequency(str(sequence_filename))
    finally:
        for filename in (sequence_filename, tree_filename, model_filename):
            try:
                filename.unlink()
            except FileNotFoundError:
                pass


def output_filenames(model, scheme_file, alignment_length=None, replicates=None):
    tag = f"scheme_{Path(scheme_file).stem}" if scheme_file else f"L{alignment_length}_R{replicates}"
    return (f"site_pattern_frequencies_{model}_{tag}.npy",
            f"trainTopoClass_{model}_{tag}.npy",
            f"trainLength_{model}_{tag}.npy")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate length-aware amino-acid simulations for quartet topology training."
    )
    parser.add_argument("-sequence_type", choices=["AA"], default="AA")
    parser.add_argument("-substitution_model", choices=AA_MODELS, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--training-scheme", "-length_dependence", dest="length_dependence",
                        help="Simulate every '<length> <replicates>' row; counts are per topology and branch regime")
    source.add_argument("--alignment-length", type=int,
                        help="Fixed alignment length in amino-acid sites (e.g. 1000000)")
    parser.add_argument("--replicates", type=int,
                        help="Fixed-length replicates per topology and branch regime; total = 6 times this count")
    parser.add_argument("-processes", type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1000,
                        help='Maximum simulations buffered before writing to disk (default: 1000)')
    parser.add_argument("-output_dir", default=".")
    parser.add_argument("--poly-mosim", default="PolyMoSim",
                        help="PolyMoSim executable installed by install.sh, or its path")
    return parser


def build_custom_parser():
    parser = argparse.ArgumentParser(
        description="Generate length-aware amino-acid simulations using a custom Q matrix."
    )
    parser.add_argument("-sequence_type", choices=["AA"], default="AA")
    parser.add_argument("-custom_model", required=True)
    parser.add_argument(
        "-length_dependence", required=True,
        help="Text file containing '<alignment length> <replicate count>' per line.",
    )
    parser.add_argument("-processes", type=int, default=1)
    parser.add_argument("-output_dir", default=".")
    parser.add_argument("--poly-mosim", default="PolyMoSim",
                        help="PolyMoSim executable installed by install.sh, or its path")
    return parser


def run(args):
    if args.processes <= 0:
        raise ValueError("-processes must be a positive integer")

    if args.length_dependence:
        if getattr(args, 'replicates', None) is not None:
            raise ValueError('--replicates is only used with --alignment-length')
        scheme = parse_length_scheme(args.length_dependence)
    else:
        if args.alignment_length <= 0 or args.replicates is None or args.replicates <= 0:
            raise ValueError('Fixed-length simulation requires positive --alignment-length and --replicates')
        scheme = [(args.alignment_length, args.replicates)]
    custom_matrix = getattr(args, "custom_model", None)
    if custom_matrix is not None:
        custom_matrix = str(Path(custom_matrix).resolve())
        parse_custom_aa_matrix(custom_matrix)
    total_scheme_replicates = sum(replicates for _, replicates in scheme)
    replicates_per_regime = total_scheme_replicates
    batch_size = getattr(args, 'batch_size', 1000)
    if batch_size < 1 or replicates_per_regime == 0:
        raise ValueError('Batch size must be positive and the scheme must contain simulations')
    run_id = f"{os.getpid()}_{random.randint(1, 100000)}"
    workdir = Path.cwd()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frequency_name, topology_name, length_name = output_filenames(
        args.substitution_model, args.length_dependence,
        getattr(args, 'alignment_length', None), getattr(args, 'replicates', None)
    )
    paths = [output_dir / name for name in (frequency_name, topology_name, length_name)]
    partials = [path.with_name(path.name + '.partial') for path in paths]
    for path in paths + partials:
        if path.exists():
            raise FileExistsError(f'Output or incomplete output already exists: {path}')
    total_rows = 6 * replicates_per_regime
    shapes = [(total_rows, 160000), (total_rows, 1), (total_rows, 1)]
    dtypes = [np.float32, np.int64, np.float32]
    for path, shape, dtype in zip(partials, shapes, dtypes):
        array = np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=shape)
        array.flush()
        del array

    scheme_prefix = regime_rows_written = completed = 0
    # Bounded batches prevent Pool from retaining a whole regime's results.
    with Pool(args.processes) as pool:
        for sequence_length, replicates in scheme:
            short_min = 0.001 * max(1, 1000 / sequence_length)
            for group in range(6):
                topology = group % 3
                internal_min, internal_max = (0.02, 0.5) if group < 3 else (short_min, 0.02)
                for start in range(0, replicates, batch_size):
                    stop = min(start + batch_size, replicates)
                    tasks = [
                        (topology, group * total_scheme_replicates + scheme_prefix
                         + index, internal_min, internal_max,
                         sequence_length, args.substitution_model, run_id, workdir,
                         custom_matrix, args.poly_mosim)
                        for index in range(start, stop)
                    ]
                    block = np.asarray(pool.map(simulate_one, tasks), dtype=np.float32)
                    if block.shape != (stop - start, 160000) or not np.isfinite(block).all():
                        raise ValueError('PolyMoSim returned invalid site-pattern frequencies')
                    offset = group * replicates_per_regime + regime_rows_written + start
                    rows = slice(offset, offset + len(block))
                    # Close each mapping after flushing: mapped pages do not accumulate in RSS.
                    for number, path in enumerate(partials):
                        array = np.lib.format.open_memmap(path, mode='r+')
                        if number == 0:
                            array[rows] = block
                        else:
                            array[rows, 0] = topology if number == 1 else sequence_length
                        array.flush()
                        del array
                    completed += len(block)
                    del block
                    print(f'Written {completed:,}/{total_rows:,} simulations', flush=True)
            regime_rows_written += replicates
            scheme_prefix += replicates
    if completed != total_rows:
        raise RuntimeError('Incomplete simulation output')
    for partial, path in zip(partials, paths):
        partial.rename(path)
    print("Frequency array shape:", shapes[0])
    print("Topology array shape:", shapes[1])
    print("Simulations:", completed)
    print("Seeds are unique in global seed space: 0 -", 6 * total_scheme_replicates - 1)
    print("Saved:", output_dir / frequency_name)
    print("Saved:", output_dir / topology_name)
    print("Saved:", output_dir / length_name)


def main():
    run(build_parser().parse_args())


def custom_main():
    args = build_custom_parser().parse_args()
    args.substitution_model = Path(args.custom_model).stem.replace(".", "_")
    run(args)


if __name__ == "__main__":
    main()
