#!/usr/bin/env python3
"""Protein MSA -> ModelDetector -> PhyloAI quartets -> amalgamated tree.

Examples (after installation, with downloaded models available locally):
  python run_phyloai.py --alignment proteins.fasta --models /path/to/models \
      --family fisher --method astraq --output results
  python run_phyloai.py --alignment genes/ --models /path/to/models \
      --family 6-state-la --method astral --output results_per_gene
  python run_phyloai.py --alignment proteins.fasta --models /path/to/models \
      --substitution-model DCMut --family fisher --method astraq --output results_dcmut

A directory is a per-gene analysis: select a model independently for each MSA,
pool the predicted quartets, and reconstruct one final tree. Input taxon IDs
must be consistent across genes. Supported families are Fisher, 6-state-LA,
and original full-pattern DeepNNPhylogeny. No PCA or random projection is used.
Use --substitution-model to skip ModelDetector and apply the specified model
to every input alignment. ModelDetector and its weights are then not required.
Use --method astraqw for confidence-weighted ASTRAQ, with per-alignment
normalization and integer scaling as in the empirical analyses.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
DETECTOR_LABELS = (
    "Q.plant", "Q.bird", "Q.yeast", "Q.mammal", "Q.insect", "Q.pfam",
    "LG", "WAG", "JTT", "VT", "PMB", "Blosum62", "Dayhoff",
)
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
DETECTOR_ORDER = "ARNDCQEGHILKMFPSTWYV"


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--alignment', type=Path, required=True, help='One MSA or directory of per-gene MSAs')
    p.add_argument('--format', default='fasta', choices=['fasta', 'phylip', 'phylip-relaxed', 'clustal', 'nexus'])
    p.add_argument('--models', type=Path, default=ROOT / 'PhyloAI-Fisher-Kaggle')
    p.add_argument('--detector', type=Path, default=ROOT / 'ModelDetector.h5')
    p.add_argument('--substitution-model', choices=(*DETECTOR_LABELS, 'DCMut', 'JTTDCMut'),
                   help='Skip ModelDetector and use this substitution model for every alignment')
    p.add_argument('--family', choices=['fisher', '6-state-la', 'deepnnphylogeny'], default='fisher')
    p.add_argument('--method', choices=['astral', 'qfm', 'astraq', 'astraqw', 'all'], default='astraq',
                   help='all runs ASTRAL, QFM-FI, ASTRAQ and ASTRAQw')
    p.add_argument('--output', type=Path, required=True, help='New output directory; existing directories are not overwritten')
    p.add_argument('--batch-size', type=int, default=64, help='QPC/prediction batch; raw frequency buffer uses 640000 bytes per row')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--seed', type=int, default=13)
    p.add_argument('--astral', default='astral', help='Installed command or JAR')
    p.add_argument('--qfm', default='QFM-FI', help='Installed command or JAR')
    p.add_argument('--astraq', default='ASTRAQ', help='Installed executable')
    a = p.parse_args()
    if a.batch_size < 1 or a.threads < 1:
        p.error('--batch-size and --threads must be positive')
    return a


def alignment_paths(path):
    if path.is_file():
        return [path.resolve()]
    extensions = {'.fa', '.fas', '.fasta', '.faa', '.phy', '.phylip', '.aln', '.nex', '.nexus'}
    paths = sorted(p.resolve() for p in path.iterdir() if p.is_file() and p.suffix.lower() in extensions)
    if not paths:
        raise ValueError(f'No supported alignment files in {path}')
    return paths


def read_alignment(path, format_name):
    from Bio import AlignIO
    alignment = AlignIO.read(str(path), format_name)
    names = [record.id for record in alignment]
    sequences = [str(record.seq).upper() for record in alignment]
    if len(names) < 4 or len(set(names)) != len(names) or not alignment.get_alignment_length():
        raise ValueError(f'{path}: require >=4 distinct taxon IDs and a nonempty alignment')
    invalid = set(''.join(sequences)) - set(AA_ORDER + '-?.XBZJUO*')
    if invalid:
        raise ValueError(f'{path}: unsupported sequence characters: {sorted(invalid)}')
    return names, sequences


def detector_features(sequences, seed):
    """Match the original ModelDetector pair-frequency extraction, including its pseudocount."""
    import numpy as np
    rng = random.Random(seed)
    mapping = {aa: i for i, aa in enumerate(DETECTOR_ORDER)}
    counts = np.zeros((20, 20), np.float32)
    total = 1
    for _ in range(max(int(400000 / len(sequences[0])), len(sequences))):
        first = rng.randrange(len(sequences))
        second = rng.randrange(len(sequences))
        while first == second:
            second = rng.randrange(len(sequences))
        for a, b in zip(sequences[first], sequences[second]):
            if a in mapping and b in mapping:
                counts[mapping[a], mapping[b]] += 1
                total += 1
    if total == 1:
        raise ValueError('ModelDetector found no valid amino-acid pairs')
    return (counts / total).reshape(1, 400)


def load_detector(path, tf):
    # The author distributes weights; rebuild the original Dense/BN architecture.
    model = tf.keras.Sequential([tf.keras.Input((400,))])
    for units in (128, 64, 32, 16):
        model.add(tf.keras.layers.Dense(units))
        model.add(tf.keras.layers.Dropout(0))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dense(13, activation='softmax'))
    model.load_weights(str(path))
    return model


def model_files(root, substitution, family):
    from additional_scripts.run_analysis_on_test_datasets_aa import discover_specs, _norm
    def identity(value):
        normalized = _norm(value)
        return 'dayhoff' if normalized == 'day' else normalized
    if not root.is_dir():
        raise FileNotFoundError(f'Model directory not found: {root}')
    folders = [root]
    folders += [p for p in root.iterdir() if p.is_dir() and identity(p.name) == identity(substitution)]
    found = []
    for folder in folders:
        if (folder / 'model_manifest.json').is_file():
            specs = discover_specs(folder)
        else:
            candidates = []
            for path in folder.iterdir():
                if not path.is_dir() or not (path / 'saved_model.pb').is_file():
                    continue
                low = path.name.lower()
                meta_path = folder / (path.name + '.training.json')
                if not meta_path.exists():
                    meta_path = folder / (path.name + '.json')
                metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
                declared = metadata.get('substitution_model')
                if declared and identity(declared) != identity(substitution):
                    continue
                if not declared and _norm(substitution) not in _norm(low) and folder == root:
                    recoding_model = {'dayhoff6': 'Dayhoff', 'sr6': 'JTT', 'kgb6': 'WAG'}
                    inferred = next((model for scheme, model in recoding_model.items() if scheme in _norm(low)), None)
                    if inferred != substitution:
                        continue
                is_ta = any(s in low for s in ('fisher', 'topology-aware', 'topology_aware'))
                is_six = any(s in _norm(low) for s in ('dayhoff6', 'sr6', 'kgb6'))
                allowed = (is_ta if family == 'fisher' else is_six if family == '6-state-la'
                           else not is_ta and not is_six and not any(s in low for s in ('pca', 'grp', 'srp', 'length', 'lenaw')))
                if allowed:
                    candidates.append(path.name)
            specs = discover_specs(folder, model_names=candidates) if candidates else []
        for spec in specs:
            matches = ((spec.kind == 'topology_aware' and spec.length_input != 'none') if family == 'fisher'
                       else (spec.kind == '6state' and spec.length_input != 'none') if family == '6-state-la'
                       else spec.kind == 'raw' and spec.length_input == 'none')
            if matches:
                found.append((folder, spec))
    if len(found) != 1:
        if family == 'fisher':
            from additional_scripts.run_analysis_on_test_datasets_aa import is_current_fisher_name
            current = [(folder, spec) for folder, spec in found if is_current_fisher_name(spec.name)]
            if len(current) == 1:
                return current[0]
        raise ValueError(f'{substitution}/{family}: expected one matching network; found {[s.name for _, s in found]}. '
                         'Provide a model directory/manifest containing one desired network per substitution model.')
    return found[0]


def write_weighted_quartets(output, taxa):
    """Empirical weighting: chance-corrected confidence, unit total per gene.

    Use the historical 1e6 scale unless that would exceed uint16 weights.
    A common scale across genes preserves their relative contributions.
    """
    def rows(path):
        with gzip.open(path, 'rt') as handle:
            for row in csv.DictReader(handle, delimiter='\t'):
                confidence = max(0.0, (max(float(row[f'p{i}']) for i in range(3)) - 1 / 3) / (2 / 3))
                yield row, confidence

    genes = []
    maximum = 0.0
    for path in sorted(output.glob('*/predictions.tsv.gz')):
        total = largest = 0.0
        for _, confidence in rows(path):
            total += confidence
            largest = max(largest, confidence)
        if total <= 0:
            raise ValueError(f'{path}: all quartet confidence weights are zero')
        genes.append((path, total))
        maximum = max(maximum, largest / total)
    if not genes:
        raise ValueError('No quartet predictions available for ASTRAQw')
    scale = min(1_000_000.0, 65535.0 / maximum)
    zero = count = 0
    with (output / 'quartets.astraqw.tre').open('w') as handle:
        for path, total in genes:
            for row, confidence in rows(path):
                weight = int(confidence / total * scale + 0.5)
                if not 0 <= weight <= 65535:
                    raise ValueError('Scaled ASTRAQw weight exceeds uint16 range')
                zero += weight == 0
                count += 1
                ids = [taxa[row[f'taxon{i}']] for i in range(1, 5)]
                topology = int(row['topology'])
                a, b, c, d = [ids[i] for i in ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2))[topology]]
                handle.write(f'(({a},{b}),({c},{d}));\t{weight}\n')
    if zero == count:
        raise ValueError('All ASTRAQw weights became zero after scaling')
    return dict(method='chance-corrected confidence', normalization='unit total per alignment before rounding',
                scale=scale, zero_weights=zero, quartets=count)


def reconstruct(method, args, taxa, environment):
    from Bio import Phylo
    executable = {'astral': args.astral, 'qfm': args.qfm, 'astraq': args.astraq, 'astraqw': args.astraq}[method]
    command = ['java', '-jar', str(Path(executable).resolve())] if executable.endswith('.jar') else [executable]
    tree_path = args.output / f'{method}.numeric.tre'
    if method == 'qfm':
        command += [str(args.output / 'quartets.qmc'), str(tree_path)]
    else:
        quartet_file = 'quartets.astraqw.tre' if method == 'astraqw' else 'quartets.tre'
        command += ['-i' if method == 'astral' else '-q', str(args.output / quartet_file), '-o', str(tree_path)]
    if method in ('astraq', 'astraqw'):
        command += ['-t', str(args.threads), '--seed', str(args.seed)]
    start = time.perf_counter()
    with (args.output / f'{method}.log').open('w') as log:
        subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    elapsed = time.perf_counter() - start
    tree = Phylo.read(str(tree_path), 'newick')
    leaves = [leaf.name for leaf in tree.get_terminals()]
    if len(leaves) != len(set(leaves)) or set(leaves) != set(taxa.values()):
        raise ValueError(f'{method}: reconstructed tree has missing, duplicate, or unexpected taxa')
    inverse = {value: key for key, value in taxa.items()}
    for leaf in tree.get_terminals():
        leaf.name = inverse[leaf.name]
    Phylo.write(tree, str(args.output / f'{method}.tre'), 'newick')
    return elapsed


def run(args):
    start = time.perf_counter()
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                     'NUMEXPR_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS'):
        os.environ[variable] = str(args.threads)
    os.environ['TF_NUM_INTEROP_THREADS'] = '1'
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import numpy as np
    import tensorflow as tf
    import QuartetPatternCounterModule as qpc
    from additional_scripts.run_analysis_on_test_datasets_aa import load_network, prepare_features, validate_network_inputs, predict
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    if getattr(qpc, 'AA_PATTERN_ORDER', None) != AA_ORDER or getattr(qpc, '__version__', None) != '1.6.1':
        raise RuntimeError(f'Requires project QPC 1.6.1; found {qpc.__file__}. Run install.sh and remove old PYTHONPATH entries.')
    if args.substitution_model is None and not args.detector.is_file():
        raise FileNotFoundError(f'ModelDetector not found: {args.detector}; run install.sh or use --detector')
    methods = ['astral', 'qfm', 'astraq', 'astraqw'] if args.method == 'all' else [args.method]
    for method in methods:
        executable = {'astral': args.astral, 'qfm': args.qfm, 'astraq': args.astraq, 'astraqw': args.astraq}[method]
        if not (Path(executable).is_file() if executable.endswith('.jar') else shutil.which(executable)):
            raise FileNotFoundError(f'{method} command/JAR not found: {executable}')
    paths = alignment_paths(args.alignment)
    taxa = {}
    for path in paths:
        names, _ = read_alignment(path, args.format)
        for name in names:
            if name not in taxa:
                taxa[name] = str(len(taxa) + 1)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'taxa.json').write_text(json.dumps(taxa, indent=2) + '\n')
    detector = load_detector(args.detector, tf) if args.substitution_model is None else None
    summaries = []
    with (args.output / 'quartets.tre').open('w') as newick, (args.output / 'quartets.qmc').open('w') as qmc:
        for index, path in enumerate(paths):
            record_start = time.perf_counter()
            names, sequences = read_alignment(path, args.format)
            if detector is None:
                substitution = args.substitution_model
                scores = None
                detector_seconds = 0.0
                selection_source = 'user'
            else:
                selected_start = time.perf_counter()
                scores = np.asarray(detector(detector_features(sequences, args.seed + index), training=False))[0]
                if scores.shape != (13,) or not np.isfinite(scores).all():
                    raise ValueError('Invalid ModelDetector output')
                substitution = DETECTOR_LABELS[int(np.argmax(scores))]
                detector_seconds = time.perf_counter() - selected_start
                selection_source = 'ModelDetector'
            folder, spec = model_files(args.models, substitution, args.family)
            print(f'{path.name}: {selection_source}={substitution}; network={spec.name}', flush=True)
            network_start = time.perf_counter()
            network = load_network(folder / spec.model)
            load_seconds = time.perf_counter() - network_start
            selector = None
            if spec.selector:
                with np.load(folder / spec.selector, allow_pickle=False) as archive:
                    selector = {'indices': archive['indices'].copy()}
                    if 'training_order' in archive and str(archive['training_order']) != AA_ORDER:
                        raise ValueError(f'Incompatible selector pattern order: {folder / spec.selector}')
                indices = selector['indices']
                if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
                        or not len(indices) or len(np.unique(indices)) != len(indices)
                        or indices.min() < 0 or indices.max() >= 160000):
                    raise ValueError('Invalid pattern selector indices')
            gene = args.output / f'{index + 1:04d}'
            gene.mkdir()
            normalized = gene / 'alignment.fasta'
            normalized.write_text(''.join(f'>{taxa[name]}\n{sequence}\n' for name, sequence in zip(names, sequences)))
            counter_start = time.perf_counter()
            counter = qpc.QuartetPatternCounter(str(normalized), 'P', args.threads, 0)
            counter_seconds = time.perf_counter() - counter_start
            frequency = np.empty((args.batch_size, 160000), np.float32)
            lengths = np.empty(args.batch_size, np.uint32)
            combinations = itertools.combinations(range(len(names)), 4)
            counted = written = skipped = 0
            preprocessing_seconds = prediction_seconds = 0.0
            with gzip.open(gene / 'predictions.tsv.gz', 'wt') as handle:
                writer = csv.writer(handle, delimiter='\t')
                writer.writerow(['taxon1', 'taxon2', 'taxon3', 'taxon4', 'effective_length', 'topology', 'p0', 'p1', 'p2'])
                while True:
                    tick = time.perf_counter()
                    n = counter.next_quartet_pattern_frequencies_with_lengths(frequency, lengths)
                    counter_seconds += time.perf_counter() - tick
                    if not n:
                        break
                    positions = [next(combinations) for _ in range(n)]
                    counted += n
                    valid = (lengths[:n] > 0) & np.isfinite(frequency[:n]).all(axis=1)
                    skipped += int((~valid).sum())
                    if not valid.any():
                        continue
                    tick = time.perf_counter()
                    features, _ = prepare_features(spec, frequency[:n][valid], lengths[:n][valid], selector)
                    validate_network_inputs(network, features, spec)
                    preprocessing_seconds += time.perf_counter() - tick
                    tick = time.perf_counter()
                    probabilities = predict(network, features, args.batch_size)
                    prediction_seconds += time.perf_counter() - tick
                    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-4):
                        raise ValueError(f'{spec.name}: invalid topology probabilities')
                    for row, probability in zip(np.flatnonzero(valid), probabilities):
                        local = positions[row]
                        ids = [taxa[names[i]] for i in local]
                        topology = int(probability.argmax())
                        a, b, c, d = [ids[i] for i in ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2))[topology]]
                        newick.write(f'(({a},{b}),({c},{d}));\n')
                        qmc.write(f'{a},{b}|{c},{d}\n')
                        writer.writerow([*(names[i] for i in local), int(lengths[row]), topology, *map(float, probability)])
                        written += 1
            if counted != math.comb(len(names), 4):
                raise ValueError(f'QPC returned {counted} quartets; expected {math.comb(len(names), 4)}')
            if not written:
                raise ValueError(f'{path}: no quartets with usable overlapping sites')
            summaries.append(dict(alignment=str(path), taxa=len(names), sites=len(sequences[0]),
                                  substitution_model=substitution, network=spec.name, model_directory=str(folder.resolve()),
                                  selector=spec.selector, recoding=spec.recoding, length_transform=spec.length_transform,
                                  model_selection=selection_source,
                                  detector_probabilities=dict(zip(DETECTOR_LABELS, map(float, scores))) if scores is not None else None,
                                  quartets=written, skipped_quartets=skipped, detector_seconds=detector_seconds,
                                  network_load_seconds=load_seconds, counter_seconds=counter_seconds,
                                  preprocessing_seconds=preprocessing_seconds, prediction_seconds=prediction_seconds,
                                  alignment_seconds=time.perf_counter() - record_start))
            (gene / 'summary.json').write_text(json.dumps(summaries[-1], indent=2) + '\n')
            del network, counter
    environment = os.environ.copy()
    environment['JAVA_TOOL_OPTIONS'] = (environment.get('JAVA_TOOL_OPTIONS', '') + f' -XX:ActiveProcessorCount={args.threads}').strip()
    weighting_start = time.perf_counter()
    weighting = write_weighted_quartets(args.output, taxa) if 'astraqw' in methods else None
    weighting_seconds = time.perf_counter() - weighting_start if weighting is not None else 0.0
    reconstruction = {method: reconstruct(method, args, taxa, environment) for method in methods}
    summary = dict(family=args.family, seed=args.seed, threads=args.threads, qpc_version=qpc.__version__,
                   requested_substitution_model=args.substitution_model,
                   alignments=summaries, amalgamation_seconds=reconstruction,
                   total_seconds=time.perf_counter() - start, weighted=weighting is not None,
                   weighting=weighting, weighting_seconds=weighting_seconds)
    (args.output / 'run_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (args.output / 'running_times.tsv').open('w') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(['stage', 'seconds'])
        for key in ('detector_seconds', 'network_load_seconds', 'counter_seconds', 'preprocessing_seconds', 'prediction_seconds'):
            writer.writerow([key, sum(row[key] for row in summaries)])
        writer.writerows((method, elapsed) for method, elapsed in reconstruction.items())
        writer.writerow(['weighting_seconds', weighting_seconds])
        writer.writerow(['total', summary['total_seconds']])
    print(f'Completed. Trees and timings: {args.output}', flush=True)


if __name__ == '__main__':
    args = arguments()
    try:
        run(args)
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(1)
