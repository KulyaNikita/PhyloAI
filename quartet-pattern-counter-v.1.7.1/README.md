# Quartet Pattern Counter 1.7.1

QPC 1.7.1 combines the upstream v1.7 counter and standalone program with the
safety and Python-interface fixes used by PhyloAI. It retains alphabetical
protein pattern order; no feature-order conversion is required.

## Python extension

Build with the Python interpreter used to run PhyloAI:

```bash
python -m pip install --no-deps --no-build-isolation .
```

The environment must provide NumPy, setuptools, wheel, pybind11, and a C++17
compiler. The build uses optimization and threading without `-march=native`,
so the resulting extension does not intentionally target only the build CPU.

```python
import numpy as np
import QuartetPatternCounterModule as qpc

counter = qpc.QuartetPatternCounter('alignment.fasta', 'P', 1, 0)
frequencies = np.empty((8, 160000), dtype=np.float32)
lengths = np.empty(8, dtype=np.uint32)
while True:
    n = counter.next_quartet_pattern_frequencies_with_lengths(frequencies, lengths)
    if n == 0:
        break
    # Consume frequencies[:n] and lengths[:n] before the next call.
```

- Protein (`P`): 160,000 columns in `ACDEFGHIKLMNPQRSTVWY` Cartesian-product order.
- DNA (`D`): 256 columns in `ACGT` Cartesian-product order.
- Frequencies are normalized over sites valid in all four sequences. Effective
  lengths count those sites; zero-site quartets return zero frequencies and length.
- All-quartet rows follow combinations of FASTA sequence positions. Four-cluster
  rows retain the supplied taxon order, including for strided input arrays.
- Output buffers must be writable, C-contiguous, and exactly `float32`
  frequencies, `uint32` lengths, or `uint16` quartet indices of shape `(batch, 4)`.
- The module exposes `__version__`, `AA_PATTERN_ORDER`, `DNA_PATTERN_ORDER`,
  `get_num_cols()`, and the four existing batched counting methods.

## Standalone program

```bash
make standalone
./quartet-pattern-counter -p four_proteins.fasta frequencies.npy
```

Omit `-p` for DNA. The standalone program requires exactly four sequences and
writes a one-dimensional float32 NumPy array. Its command format matches v1.1,
but sites with gaps or ambiguous symbols are excluded from normalization.
The Python interface supports alignments with more than four taxa.

## Regression tests

After building the extension and standalone program:

```bash
python -m unittest discover -s tests -v
```

Tests cover pattern order, effective lengths, batching, threads, four-cluster
indices, invalid buffers/configurations/FASTA files, and standalone output.
