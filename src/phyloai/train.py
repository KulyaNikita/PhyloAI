"""Train fixed-length full-pattern, Fisher, or six-state-LA networks (`phyloai train`).

Frequency .npy shards contain 160,000 alphabetical-order site-pattern columns.
Fisher and six-state-LA require matching --length-data .npy files containing raw
alignment lengths. No PCA, random projection, or shallow architecture is used.
Feature selection and normalization use training rows only.
Raw Fisher/six-state runs save transformed .features.npy arrays. Reuse them
with --input-format selected (plus --selector-model) or recoded respectively.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

PATTERN_COLUMNS = 20 ** 4
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
SCHEMES = {
    "dayhoff-6": ("AGPST", "DENQ", "HKR", "MIVL", "WFY", "C"),
    "sr-6": ("APST", "DENG", "QKR", "MIVL", "WC", "FYH"),
    "kgb-6": ("AGPS", "DENQHKRT", "MIL", "W", "FY", "CV"),
}


def arguments(argv=None, prog=None):
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--training-data", type=Path, nargs="+", required=True,
                        help="One or more frequency-array shards in row order")
    parser.add_argument("--topology-data", type=Path, nargs="+", required=True,
                        help="Matching topology-label shards in the same order")
    parser.add_argument("--substitution-model", required=True,
                        help="Training-data substitution model, e.g. LG, Dayhoff, Q.plant")
    parser.add_argument("--method", choices=("fisher", "six-state", "raw"),
                        default="fisher")
    parser.add_argument("--input-format", choices=("raw", "recoded", "selected"), default="raw")
    parser.add_argument("--recoding", choices=tuple(SCHEMES),
                        help="Required for six-state training")
    parser.add_argument("--selected-patterns", type=int, help='Fisher feature count (default: 10000; inferred from selector for selected input)')
    parser.add_argument('--selector-model', type=Path,
                        help='Matching .npz selector with original train/validation rows; required for --input-format selected')
    parser.add_argument("--length-data", type=Path, nargs="+",
                        help="Raw lengths as .npy (N,) or (N,1), one matching file per frequency shard; required for fisher/six-state")
    parser.add_argument("--architecture", choices=("b1", "b3"))
    parser.add_argument("--length-transform", choices=("none", "raw", "log10", "both"),
                        default=None)
    parser.add_argument("--selector-output", type=Path)
    parser.add_argument("--model-prefix", help="Optional output-name prefix")
    parser.add_argument("--output-dir", type=Path, default=Path('.'))
    parser.add_argument("--validation-fraction", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--score-batch-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--l2", type=float)
    parser.add_argument("--activation", choices=("relu", "tanh"))
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--threads", type=int,
                        default=max(1, int(os.environ.get('NSLOTS', os.cpu_count() or 1))))
    args = parser.parse_args(argv)
    if args.selected_patterns is None and args.input_format != 'selected':
        args.selected_patterns = 10000
    if args.length_transform is None:
        args.length_transform = 'none' if args.method == 'raw' else 'log10'
    if args.method == 'raw':
        if args.length_transform != 'none' or args.length_data:
            parser.error('--method raw is fixed-length and does not accept length inputs')
    elif args.length_transform == 'none' or not args.length_data:
        parser.error('Fisher Fisher and six-state-LA require --length-data and a length transform')
    aware = args.length_transform != 'none'
    modern = args.method == 'fisher' or (args.method == 'six-state' and aware)
    if args.architecture is None:
        args.architecture = 'b1' if args.method == 'fisher' else 'b3'
    defaults = {
        'validation_fraction': 0.2 if modern else 0.04,
        'seed': 13 if modern else 42,
        'patience': 20 if modern or aware else 50,
        'dropout': 0.4 if args.method == 'fisher' else 0.2 if aware else 0.3,
        'l2': 1e-4 if modern else 0.0,
        'activation': 'relu' if modern or aware else 'tanh',
    }
    for name, default in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, default)
    if args.selected_patterns is not None and not 1 <= args.selected_patterns <= PATTERN_COLUMNS:
        parser.error("--selected-patterns must be between 1 and 160000")
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between 0 and 1")
    if not 0 <= args.dropout < 1 or args.l2 < 0:
        parser.error("--dropout must be in [0,1) and --l2 must be non-negative")
    if min(args.score_batch_size, args.batch_size, args.epochs, args.patience, args.threads) < 1:
        parser.error("Batch sizes, epochs, patience, and threads must be positive")
    if args.learning_rate <= 0:
        parser.error('Learning rate must be positive')
    if args.method == 'six-state' and args.recoding is None:
        parser.error("--method six-state requires --recoding")
    if args.recoding and args.method != 'six-state':
        parser.error("--recoding is only valid for six-state training")
    if args.input_format == 'recoded' and args.method != 'six-state':
        parser.error('Recoded input requires --method six-state')
    if args.input_format == 'selected' and (args.method != 'fisher' or not args.selector_model):
        parser.error('Selected input requires --method fisher and --selector-model')
    if args.selector_model and args.input_format != 'selected':
        parser.error('--selector-model is only used with --input-format selected')
    if args.selector_output and (args.method != 'fisher' or args.selector_output.suffix != '.npz'):
        parser.error('--selector-output requires Fisher selection and a .npz suffix')
    return args



def use_keras2():
    """The training code uses the Keras 2 API and saves SavedModel folders, like the released networks.

    TensorFlow 2.15 has Keras 2 built in. Newer TensorFlow comes with Keras 3, but provides Keras 2 as
    the package tf-keras, selected with TF_USE_LEGACY_KERAS=1 before TensorFlow is imported.
    """
    import importlib.util
    from importlib import metadata
    try:
        keras_major = int(metadata.version("keras").split(".")[0])
    except (metadata.PackageNotFoundError, ValueError):
        keras_major = 2
    if keras_major < 3:
        return
    if importlib.util.find_spec("tf_keras") is None:
        raise SystemExit("phyloai train needs Keras 2. With TensorFlow >=2.16 install the matching tf-keras "
                         "(conda install -c conda-forge tf-keras), or use TensorFlow 2.15.")
    os.environ["TF_USE_LEGACY_KERAS"] = "1"


def main(argv=None, prog=None):
    args = arguments(argv, prog)
    # Thread limits must be set before numpy and TensorFlow are imported.
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = str(args.threads)
    use_keras2()
    from phyloai import _training
    _training.args = args
    _training.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
