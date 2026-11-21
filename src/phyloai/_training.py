"""Training code for `phyloai train` (arguments and entry point: phyloai.train).

Train fixed-length full-pattern, Fisher, or six-state-LA networks.

Frequency .npy shards contain 160,000 alphabetical-order site-pattern columns.
Fisher and six-state-LA require matching --length-data .npy files containing raw
alignment lengths. No PCA, random projection, or shallow architecture is used.
Feature selection and normalization use training rows only.
Raw Fisher/six-state runs save transformed .features.npy arrays. Reuse them
with --input-format selected (plus --selector-model) or recoded respectively.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from phyloai.train import AA_ORDER, PATTERN_COLUMNS, SCHEMES  # noqa: E402

# Set by phyloai.train.main() before main() runs.
args = None

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402


class ShardedDataset:
    """Row-addressable view over memory-mapped .npy shards."""
    def __init__(self, paths, pattern_columns=PATTERN_COLUMNS):
        self.paths = list(paths)
        self.arrays = [np.load(path, mmap_mode="r") for path in self.paths]
        if not self.arrays:
            raise ValueError("No training-data shards supplied")
        if self.arrays[0].ndim != 2:
            raise ValueError('Frequency shards must be two-dimensional')
        columns = self.arrays[0].shape[1]
        self.pattern_columns = pattern_columns
        if columns not in (pattern_columns, pattern_columns + 1):
            raise ValueError(f'Expected {pattern_columns} pattern columns, optionally followed by length; got {columns}')
        for path, array in zip(self.paths, self.arrays):
            if array.ndim != 2 or array.shape[1] != columns:
                raise ValueError(f"Incompatible training shard {path}: {array.shape}")
        self.columns = columns
        self.offsets = np.concatenate(([0], np.cumsum([len(a) for a in self.arrays])))
        self.shape = (int(self.offsets[-1]), columns)
        self.ndim = 2
        self.selected = None
        self.recoding_indices = None
        self.feature_width = pattern_columns
        self.feature_cache = None
        self.length_values = None

    def __len__(self):
        return self.shape[0]

    def patterns(self, rows, selected=None):
        rows = np.asarray(rows, dtype=np.int64)
        width = self.pattern_columns if selected is None else len(selected)
        output = np.empty((len(rows), width), dtype=np.float32)
        for shard, array in enumerate(self.arrays):
            mask = (rows >= self.offsets[shard]) & (rows < self.offsets[shard + 1])
            if not np.any(mask):
                continue
            local = rows[mask] - self.offsets[shard]
            if selected is None:
                output[mask] = array[local, :self.pattern_columns]
            else:
                output[mask] = array[np.ix_(local, selected)]
        if not np.isfinite(output).all():
            raise ValueError('Training features contain non-finite values')
        return output

    def features(self, rows):
        if self.feature_cache is not None:
            return np.asarray(self.feature_cache[rows], dtype=np.float32)
        values = self.patterns(rows, self.selected)
        if self.recoding_indices is not None:
            values = recode_batch(values, self.recoding_indices)
        return np.asarray(values, dtype=np.float32)

    def lengths(self, rows=None):
        if self.length_values is None and self.columns != self.pattern_columns + 1:
            raise ValueError("This sharded dataset has no alignment-length column")
        if self.length_values is None:
            self.length_values = np.concatenate([
                np.asarray(array[:, self.pattern_columns]) for array in self.arrays
            ]).astype(np.float32)
        output = self.length_values if rows is None else self.length_values[rows]
        if not np.isfinite(output).all() or np.any(output <= 0):
            raise ValueError('Alignment lengths must be finite and positive')
        return output


def read_labels(paths, rows: int) -> np.ndarray:
    pieces = []
    for path in paths:
        labels = np.load(path)
        if labels.ndim == 2 and labels.shape[1] == 3:
            if not np.isfinite(labels).all() or not np.all(np.isin(labels, (0, 1))) or not np.all(labels.sum(axis=1) == 1):
                raise ValueError(f'Expected one-hot topology labels in {path}')
            labels = labels.argmax(axis=1)
        else:
            labels = labels.reshape(-1)
        if not np.all(np.isin(labels, (0, 1, 2))):
            raise ValueError(f'Invalid topology labels in {path}')
        pieces.append(labels.astype(np.int64))
    labels = np.concatenate(pieces)
    if len(labels) != rows or not np.all(np.isin(labels, (0, 1, 2))):
        raise ValueError(f"Expected {rows} topology labels in classes 0,1,2")
    return labels


def recode_batch(frequencies, indices):
    """Same axis-wise aggregation used by recode_aa_frequencies_6state.py."""
    values = frequencies.reshape((-1, 20, 20, 20, 20))
    for axis in range(1, 5):
        values = np.stack([np.take(values, group, axis=axis).sum(axis=axis)
                           for group in indices], axis=axis)
    values = values.reshape((-1, 1296))
    sums = values.sum(axis=1, keepdims=True)
    if np.any(~np.isfinite(sums)) or np.any(sums <= 0):
        raise ValueError('Six-state recoding requires positive finite row sums')
    return values / sums


def cache_features(data, path):
    """Save selected/recoded frequencies once and train from the compact array."""
    destination = np.lib.format.open_memmap(
        path, mode='w+', dtype=np.float32,
        shape=(len(data), data.feature_width),
    )
    for start in range(0, len(data), args.score_batch_size):
        rows = np.arange(start, min(start + args.score_batch_size, len(data)))
        destination[rows] = data.features(rows)
    destination.flush()
    del destination
    data.feature_cache = np.load(path, mmap_mode='r')


def length_class_weights(lengths, labels, indices):
    """Give every observed (length, topology) stratum total weight one."""
    pairs = np.column_stack((lengths[indices], labels[indices]))
    unique, inverse, counts = np.unique(
        pairs, axis=0, return_inverse=True, return_counts=True
    )
    row_weights = 1.0 / counts[inverse].astype(np.float64)
    return row_weights, unique, counts


def fisher_scores(data, labels, train_indices, has_length):
    ordered = np.sort(train_indices)
    lengths = (
        data.lengths()
        if has_length else np.ones(len(data), dtype=np.uint8)
    )
    row_weights, strata, counts = length_class_weights(
        lengths, labels, ordered
    )
    sums = np.zeros((3, PATTERN_COLUMNS), dtype=np.float64)
    squared_sums = np.zeros_like(sums)
    weights = np.zeros(3, dtype=np.float64)

    for start in range(0, len(ordered), args.score_batch_size):
        stop = min(start + args.score_batch_size, len(ordered))
        rows = ordered[start:stop]
        block = data.patterns(rows)
        block_labels = labels[rows]
        block_weights = row_weights[start:stop]
        for topology in range(3):
            mask = block_labels == topology
            if not np.any(mask):
                continue
            w = block_weights[mask]
            x = block[mask]
            sums[topology] += np.einsum("i,ij->j", w, x, optimize=True)
            squared_sums[topology] += np.einsum(
                "i,ij,ij->j", w, x, x, optimize=True
            )
            weights[topology] += w.sum()
        print(f"Fisher statistics: {stop:,}/{len(ordered):,} training rows", flush=True)

    means = sums / weights[:, None]
    overall = np.average(means, axis=0, weights=weights)
    between = np.sum(weights[:, None] * (means - overall) ** 2, axis=0)
    within = np.sum(
        np.maximum(0.0, squared_sums - weights[:, None] * means ** 2), axis=0
    )
    scores = between / np.maximum(within, np.finfo(np.float64).eps)
    scores[~np.isfinite(scores)] = 0.0
    count = args.selected_patterns
    selected = np.argpartition(scores, -count)[-count:]
    selected = selected[np.argsort(scores[selected])[::-1]].astype(np.int64)
    return selected, scores[selected], strata, counts


class PatternSequence(tf.keras.utils.Sequence):
    def __init__(self, data, labels, indices, length_transform,
                 batch_size, shuffle, seed):
        self.data, self.labels = data, labels
        self.indices = np.asarray(indices).copy()
        self.length_transform = length_transform
        self.batch_size, self.shuffle = batch_size, shuffle
        self.rng = np.random.default_rng(seed)
        self.on_epoch_end()

    def __len__(self):
        return (len(self.indices) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, item):
        rows = self.indices[item * self.batch_size:(item + 1) * self.batch_size]
        patterns = np.asarray(
            self.data.features(rows), dtype=np.float32
        )
        targets = tf.keras.utils.to_categorical(self.labels[rows], 3)
        if self.length_transform == "none":
            return patterns, targets
        length = np.asarray(
            self.data.lengths(rows), dtype=np.float32
        ).reshape(-1, 1)
        if self.length_transform == "log10":
            length = np.log10(length)
        return (patterns, length), targets

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)


def normalization_layers(data, train_indices, length_transform):
    pattern_norm = tf.keras.layers.Normalization(name="normalize_patterns")
    pattern_ds = tf.data.Dataset.from_generator(
        lambda: (
            data.features(rows)
            for rows in np.array_split(
                np.sort(train_indices),
                max(1, int(np.ceil(len(train_indices) / args.batch_size)))
            ) if len(rows)
        ),
        output_signature=tf.TensorSpec(shape=(None, data.feature_width), dtype=tf.float32),
    )
    pattern_norm.adapt(pattern_ds)
    if length_transform == "none":
        return pattern_norm, None
    length = data.lengths(train_indices).astype(np.float32).reshape(-1, 1)
    if length_transform == "log10":
        length = np.log10(length)
    length_norm = tf.keras.layers.Normalization(
        name=f"normalize_alignment_length_{length_transform}"
    )
    length_norm.adapt(length, batch_size=args.batch_size)
    return pattern_norm, length_norm


def build_model(features, pattern_norm, length_norm):
    pattern_input = tf.keras.Input((features,), name="selected_patterns")
    normalized = pattern_norm(pattern_input)
    regularizer = tf.keras.regularizers.l2(args.l2) if args.l2 else None
    def dense(units, name):
        return tf.keras.layers.Dense(
            units, activation=args.activation, kernel_regularizer=regularizer, name=name
        )
    branches = 1 if args.architecture == "b1" else 3
    outputs = []
    for branch_number in range(branches):
        x = dense(256, f"pattern_branch_{branch_number + 1}_dense_1")(normalized)
        x = tf.keras.layers.Dropout(args.dropout)(x)
        x = dense(128, f"pattern_branch_{branch_number + 1}_dense_2")(x)
        x = tf.keras.layers.Dropout(args.dropout)(x)
        x = dense(64, f"pattern_branch_{branch_number + 1}_projection")(x)
        outputs.append(x)
    if length_norm is None:
        inputs = pattern_input
        merged = outputs[0] if len(outputs) == 1 else tf.keras.layers.Concatenate(
            name="combine_pattern_branches"
        )(outputs)
    else:
        length_input = tf.keras.Input((1,), name="alignment_length")
        length = length_norm(length_input)
        length = dense(16, "length_dense")(length)
        inputs = [pattern_input, length_input]
        merged = tf.keras.layers.Concatenate(name="combine_patterns_and_length")(
            outputs + [length]
        )
    merged = dense(128, "combined_dense_1")(merged)
    merged = tf.keras.layers.Dropout(args.dropout)(merged)
    merged = dense(64, "combined_dense_2")(merged)
    topology = tf.keras.layers.Dense(
        3, activation="softmax", kernel_regularizer=regularizer, name="topology"
    )(merged)
    model = tf.keras.Model(inputs, topology,
                           name=f"topology_aware_{args.architecture}")
    model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate), loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def build_other_model(features, pattern_norm, length_norm):
    """Preserve the fixed-length and six-state-LA B1/B3 architectures."""
    layers = tf.keras.layers
    regularizer = tf.keras.regularizers.l2(args.l2) if args.l2 else None
    input_name = 'six_state_patterns' if args.method == 'six-state' else 'site_patterns'
    patterns = tf.keras.Input((features,), name=input_name)
    normalized = pattern_norm(patterns)

    def dense(x, units, name):
        return layers.Dense(units, activation=args.activation,
                            kernel_regularizer=regularizer, name=name)(x)

    six_la = args.method == 'six-state' and length_norm is not None
    widths = [(256, 256, 256, 128)] if args.architecture == 'b1' else [
        (256, 256, 128), (256, 256, 128),
        (256, 128, 128) if length_norm is None else (256, 256, 128),
    ]
    if six_la and args.architecture == 'b1':
        widths = [(256, 256, 128)]
    branches = []
    for number, branch_widths in enumerate(widths, 1):
        x = normalized
        for index, width in enumerate(branch_widths, 1):
            x = dense(x, width, f'pattern_branch_{number}_dense_{index}')
            if index <= (3 if args.architecture == 'b1' and length_norm is not None and not six_la else 2):
                x = layers.Dropout(args.dropout)(x)
        branches.append(x)
    inputs = patterns
    if length_norm is not None:
        length_input = tf.keras.Input((1,), name='alignment_length')
        length = length_norm(length_input)
        if six_la:
            length = dense(length, 32, 'length_dense_1')
        length = dense(length, 16, 'length_projection')
        branches.append(length)
        inputs = [patterns, length_input]
    x = branches[0] if len(branches) == 1 else layers.Concatenate()(branches)
    head = (64, 32) if args.architecture == 'b1' and not six_la else (256, 128, 64)
    for index, width in enumerate(head, 1):
        x = dense(x, width, f'combined_dense_{index}')
        if index < len(head):
            x = layers.Dropout(args.dropout)(x)
    topology = layers.Dense(3, activation='softmax', name='topology')(x)
    model = tf.keras.Model(inputs, topology,
                           name=f"phyloai_{args.method.replace('-', '_')}_{args.architecture}")
    model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(min(2, args.threads))
    if len(args.training_data) != len(args.topology_data):
        raise ValueError("Frequency and topology shard counts differ")
    selector_archive = None
    if args.input_format == 'selected':
        with np.load(args.selector_model, allow_pickle=False) as archive:
            selector_archive = {key: archive[key].copy() for key in archive.files}
        required = {'indices', 'training_order', 'training_rows', 'validation_rows'}
        if not required.issubset(selector_archive):
            raise ValueError('Selector must include indices, training_order and original train/validation rows')
        indices = selector_archive['indices']
        if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
                or not 1 <= len(indices) <= PATTERN_COLUMNS
                or len(np.unique(indices)) != len(indices)
                or indices.min() < 0 or indices.max() >= PATTERN_COLUMNS
                or str(selector_archive['training_order']) != AA_ORDER):
            raise ValueError('Invalid selector indices or amino-acid pattern order')
        if args.selected_patterns is not None and args.selected_patterns != len(indices):
            raise ValueError('--selected-patterns disagrees with the supplied selector')
        args.selected_patterns = len(indices)
    input_width = (args.selected_patterns if args.input_format == 'selected' else
                   1296 if args.input_format == 'recoded' else PATTERN_COLUMNS)
    data = ShardedDataset(args.training_data, input_width)
    has_length = args.method != 'raw'
    if has_length:
        if len(args.length_data) != len(data.arrays):
            raise ValueError('Provide one --length-data file per frequency shard')
        pieces = []
        for array, path in zip(data.arrays, args.length_data):
            values = np.load(path)
            if values.ndim not in (1, 2) or (values.ndim == 2 and values.shape[1] != 1):
                raise ValueError(f'{path}: expected raw lengths with shape (N,) or (N,1)')
            values = values.reshape(-1).astype(np.float32)
            if len(values) != len(array) or not np.isfinite(values).all() or np.any(values <= 0):
                raise ValueError(f'{path}: lengths must match shard rows and be finite and positive')
            if data.columns == input_width + 1 and not np.array_equal(values, array[:, input_width]):
                raise ValueError(f'{path}: separate lengths disagree with the embedded length column')
            pieces.append(values)
        data.length_values = np.concatenate(pieces)
    elif data.columns == input_width + 1:
        fixed_lengths = data.lengths()
        if not np.all(fixed_lengths == fixed_lengths[0]):
            raise ValueError('Fixed-length raw training received multiple alignment lengths')
    for array, label_path in zip(data.arrays, args.topology_data):
        if len(array) != len(np.load(label_path, mmap_mode='r')):
            raise ValueError(f"Frequency/label shard row counts disagree: {label_path}")
    labels = read_labels(args.topology_data, len(data))
    if set(labels.tolist()) != {0, 1, 2}:
        raise ValueError("Training requires all three topology classes")
    if selector_archive is not None:
        # Never resplit rows after supervised feature selection: that could leak
        # original training labels into a newly created validation subset.
        train_indices = selector_archive['training_rows']
        validation_indices = selector_archive['validation_rows']
        for rows in (train_indices, validation_indices):
            if rows.ndim != 1 or not len(rows) or not np.issubdtype(rows.dtype, np.integer):
                raise ValueError('Invalid original train/validation rows in selector')
        all_rows = np.concatenate((train_indices, validation_indices))
        if not np.array_equal(np.sort(all_rows), np.arange(len(data))):
            raise ValueError('Selector split does not match the transformed dataset rows')
        args.validation_fraction = len(validation_indices) / len(data)
    else:
        train_indices, validation_indices = train_test_split(
            np.arange(len(data)), test_size=args.validation_fraction,
            random_state=args.seed, stratify=labels,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aware = args.length_transform != 'none'
    recoding_name = {'dayhoff-6': 'Dayhoff-6', 'sr-6': 'SR-6', 'kgb-6': 'KGB-6'}
    family = ('Fisher' if args.method == 'fisher' else
              recoding_name[args.recoding] if args.method == 'six-state' else args.method.upper())
    if aware and args.method != 'fisher':
        family += '-LA'
    default_prefix = (f'PhyloAI-{family}' if args.method == 'six-state' else
                      f'PhyloAI-{family}-{args.substitution_model}')
    prefix = args.output_dir / (args.model_prefix or default_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    transforms = ("raw", "log10") if args.length_transform == "both" else (args.length_transform,)
    width = args.selected_patterns if args.method == 'fisher' else 1296 if args.method == 'six-state' else PATTERN_COLUMNS
    feature_tag = f'-k{width}' if args.method == 'fisher' else ''
    model_paths = [Path(f"{prefix}{feature_tag}-{args.architecture}-{'no-length' if t == 'none' else t}")
                   for t in transforms]
    selector_path = (args.selector_output or Path(f'{prefix}{feature_tag}-{args.architecture}-selector.npz')
                     if args.method == 'fisher' else None)
    split_path = Path(f'{prefix}{feature_tag}-{args.architecture}-split.npz')
    cache_path = (Path(f'{prefix}{feature_tag}.features.npy') if args.input_format == 'raw'
                  and args.method in ('fisher', 'six-state') else None)
    artifacts = model_paths + [Path(str(p) + '.training.json') for p in model_paths]
    artifacts += [Path(str(p) + '.history.json') for p in model_paths]
    artifacts += [p for p in (selector_path, split_path, cache_path) if p is not None]
    if any(p.exists() for p in artifacts):
        raise FileExistsError("Output already exists; choose a new --output-dir or --model-prefix")
    selected = None
    if args.method == 'fisher' and selector_archive is not None:
        selector_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.selector_model, selector_path)
    elif args.method == 'fisher':
        selected, scores, strata, counts = fisher_scores(data, labels, train_indices, has_length)
        data.selected = np.sort(selected)
        data.feature_width = len(selected)
        selector_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            selector_path, indices=data.selected, ranked_indices=selected,
            fisher_scores=scores, pattern_columns=PATTERN_COLUMNS,
            training_order=AA_ORDER, seed=args.seed,
            training_rows=train_indices, validation_rows=validation_indices,
            length_topology_strata=strata, stratum_counts=counts,
        )
    elif args.method == 'six-state':
        data.feature_width = 1296
        if args.input_format == 'raw':
            data.recoding_indices = tuple(
                np.asarray([AA_ORDER.index(c) for c in group], dtype=np.intp)
                for group in SCHEMES[args.recoding]
            )
    if cache_path:
        cache_features(data, cache_path)
    np.savez_compressed(split_path, training_rows=train_indices,
                        validation_rows=validation_indices, seed=args.seed)
    for transform, model_path in zip(transforms, model_paths):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(args.seed)
        pattern_norm, length_norm = normalization_layers(data, train_indices, transform)
        builder = build_model if args.method == 'fisher' else build_other_model
        model = builder(data.feature_width, pattern_norm, length_norm)
        model.summary()
        training = PatternSequence(data, labels, train_indices, transform,
                                   args.batch_size, True, args.seed)
        validation = PatternSequence(data, labels, validation_indices, transform,
                                     args.batch_size, False, args.seed)
        # Six-state-LA originally saved the restored best-loss weights.
        # Fisher and full-pattern networks save the best-accuracy checkpoint.
        best_loss = args.method == 'six-state' and transform != 'none'
        monitor = 'val_loss' if best_loss else 'val_accuracy'
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=args.patience, min_delta=1e-4,
                restore_best_weights=True, verbose=1,
            ),
        ]
        if not best_loss:
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(
                str(model_path), monitor=monitor, mode='min' if best_loss else 'max',
                save_best_only=True, verbose=1,
            ))
        history = model.fit(training, validation_data=validation, epochs=args.epochs,
                            callbacks=callbacks, verbose=2, workers=1)
        if best_loss:
            model.save(str(model_path))
        # Report the actual saved checkpoint, not different restored weights.
        saved = tf.keras.models.load_model(str(model_path))
        loss, accuracy = saved.evaluate(validation, verbose=0)
        metadata = {
            "model": model_path.name, "substitution_model": args.substitution_model,
            "method": args.method, "architecture": args.architecture,
            "length_aware": transform != 'none', "length_transform": transform,
            "length_input": 'separate' if transform != 'none' else None,
            "pattern_columns": PATTERN_COLUMNS, "training_order": AA_ORDER,
            "input_format": args.input_format, "features": data.feature_width,
            "recoding": args.recoding, "recoded_pattern_count": 1296 if args.recoding else None,
            "selector": os.path.relpath(selector_path.resolve(), model_path.parent.resolve()) if selector_path else None,
            "selected_patterns": data.feature_width if args.method == 'fisher' else None,
            "pretrained_selector": str(args.selector_model) if args.selector_model else None,
            "split_source": 'selector' if selector_archive is not None else 'stratified_random',
            "feature_cache": cache_path.name if cache_path else None,
            "checkpoint_monitor": monitor,
            "split": split_path.name, "training_rows": int(len(train_indices)),
            "validation_rows": int(len(validation_indices)), "seed": args.seed,
            "validation_fraction": args.validation_fraction,
            "validation_loss": float(loss), "validation_accuracy": float(accuracy),
            "epochs_completed": len(history.history['loss']),
            "dropout": args.dropout, "l2": args.l2, "activation": args.activation,
            "learning_rate": args.learning_rate, "batch_size": args.batch_size,
            "patience": args.patience,
            "training_data": [str(p) for p in args.training_data],
            "topology_data": [str(p) for p in args.topology_data],
            "length_data": [str(p) for p in args.length_data] if args.length_data else None,
        }
        Path(f"{model_path}.training.json").write_text(json.dumps(metadata, indent=2) + "\n")
        Path(f"{model_path}.history.json").write_text(json.dumps(history.history, indent=2) + "\n")
        print(f"Saved {model_path}; validation accuracy={accuracy:.6f}", flush=True)


if __name__ == "__main__":
    from phyloai.train import main as train_main
    raise SystemExit(train_main())
