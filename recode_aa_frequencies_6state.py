#!/usr/bin/env python3
"""Collapse 20-state quartet frequencies into Dayhoff-, S&R-, or KGB-6."""

import argparse
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap


POLYMOSIM_AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_ALPHABET = frozenset(POLYMOSIM_AA_ORDER)
INPUT_FREQUENCIES = 20 ** 4
OUTPUT_FREQUENCIES = 6 ** 4

SCHEMES = {
    "dayhoff-6": ("AGPST", "DENQ", "HKR", "MIVL", "WFY", "C"),
    "sr-6": ("APST", "DENG", "QKR", "MIVL", "WC", "FYH"),
    "kgb-6": ("AGPS", "DENQHKRT", "MIL", "W", "FY", "CV"),
}


def group_indices(scheme, aa_order=POLYMOSIM_AA_ORDER):
    """Return indices for a recoding scheme in the supplied amino-acid order."""
    groups = SCHEMES[scheme]
    flattened = "".join(groups)
    if len(flattened) != 20 or set(flattened) != AA_ALPHABET:
        raise RuntimeError(f"Invalid recoding scheme definition: {scheme}")
    if len(aa_order) != 20 or set(aa_order) != AA_ALPHABET:
        raise ValueError(f"Invalid amino-acid order: {aa_order!r}")
    aa_to_index = {amino_acid: index for index, amino_acid in enumerate(aa_order)}
    return tuple(
        np.asarray([aa_to_index[amino_acid] for amino_acid in group], dtype=np.intp)
        for group in groups
    )


def recode_batch(frequencies, indices):
    recoded = frequencies.reshape((-1, 20, 20, 20, 20))
    # After each reduction, the next unreduced amino-acid axis remains at
    # positions 2, 3, and 4 respectively.
    for axis in range(1, 5):
        recoded = np.stack(
            [np.take(recoded, group, axis=axis).sum(axis=axis) for group in indices],
            axis=axis,
        )
    recoded = recoded.reshape((-1, OUTPUT_FREQUENCIES))
    row_sums = recoded.sum(axis=1, keepdims=True)
    if np.any(~np.isfinite(row_sums)) or np.any(row_sums <= 0):
        bad_rows = np.flatnonzero((~np.isfinite(row_sums) | (row_sums <= 0)).ravel())
        raise ValueError(f"Cannot normalize rows with invalid sums: {bad_rows[:10]}")
    recoded /= row_sums
    return recoded


def default_output_path(input_path, scheme):
    return input_path.with_name(f"{input_path.stem}_{scheme}{input_path.suffix}")


def convert(input_path, output_path, scheme, batch_size, aa_order):
    source = np.load(input_path, mmap_mode="r")
    if source.ndim != 2 or source.shape[1] not in (INPUT_FREQUENCIES, INPUT_FREQUENCIES + 1):
        raise ValueError(
            "Input must have shape (rows, 160000) or (rows, 160001); "
            f"received {source.shape}"
        )
    has_length = source.shape[1] == INPUT_FREQUENCIES + 1
    output_columns = OUTPUT_FREQUENCIES + int(has_length)
    output_dtype = source.dtype if np.issubdtype(source.dtype, np.floating) else np.float64
    destination = open_memmap(
        output_path, mode="w+", dtype=output_dtype,
        shape=(source.shape[0], output_columns),
    )
    indices = group_indices(scheme, aa_order)

    for start in range(0, source.shape[0], batch_size):
        stop = min(start + batch_size, source.shape[0])
        frequencies = np.asarray(source[start:stop, :INPUT_FREQUENCIES], dtype=output_dtype)
        destination[start:stop, :OUTPUT_FREQUENCIES] = recode_batch(frequencies, indices)
        if has_length:
            destination[start:stop, OUTPUT_FREQUENCIES] = source[
                start:stop, INPUT_FREQUENCIES
            ]
        print(f"Processed rows {start + 1}-{stop} of {source.shape[0]}")
    destination.flush()
    return destination.shape, destination.dtype, has_length


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input .npy frequency array")
    parser.add_argument(
        "--scheme", required=True, choices=tuple(SCHEMES),
        help="Six-state amino-acid recoding scheme",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Output .npy path; defaults to '<input>_<scheme>.npy'",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Rows processed simultaneously (default: 8)",
    )
    parser.add_argument(
        "--aa-order", default=POLYMOSIM_AA_ORDER,
        help=("Amino-acid order used by the 160,000 input columns "
              f"(default: {POLYMOSIM_AA_ORDER})"),
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    output = args.output or default_output_path(args.input, args.scheme)
    if output.resolve() == args.input.resolve():
        raise ValueError("Input and output paths must be different")
    output.parent.mkdir(parents=True, exist_ok=True)
    shape, dtype, kept_length = convert(
        args.input, output, args.scheme, args.batch_size, args.aa_order
    )
    print("Saved:", output)
    print("Output shape:", shape)
    print("Output dtype:", dtype)
    print("Alignment-length column preserved:", kept_length)
    print("Input amino-acid order:", args.aa_order)


if __name__ == "__main__":
    main()
