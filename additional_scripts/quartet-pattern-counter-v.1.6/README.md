# Quartet Pattern Counter 1.6

This source distribution contains QPC 1.6 with local correctness fixes, identified
as `1.6.1`. Build it for the Python interpreter used to run PhyloAI:

```bash
python -m pip install --no-deps --no-build-isolation .
python -m unittest discover -s tests -v
```

The environment must provide NumPy, setuptools, wheel, pybind11, and a C++17
compiler. Installation compiles the extension; no prebuilt binary is included.
Builds use optimization and threading, without `-march=native`.

## Interface and ordering

```python
import numpy as np
import QuartetPatternCounterModule as qpc

counter = qpc.QuartetPatternCounter('alignment.fasta', 'P', 1, 0)
freq = np.empty((8, 160000), dtype=np.float32)
lengths = np.empty(8, dtype=np.uint32)
while True:
    n = counter.next_quartet_pattern_frequencies_with_lengths(freq, lengths)
    if n == 0:
        break
    # Consume freq[:n] and lengths[:n] before requesting the next batch.
    # Exclude rows with lengths[:n] == 0 from neural inference.
```

- DNA: data type `D`, 256 columns, alphabet `ACGT`.
- Protein: data type `P`, 160,000 columns, alphabet `ACDEFGHIKLMNPQRSTVWY`.
- Patterns follow the Cartesian product of the alphabet in taxon order; the
  fourth taxon varies fastest within a pattern index.
- All-quartet rows follow combinations of zero-based FASTA sequence positions.
- Four-cluster construction accepts four one-dimensional integer arrays before
  the data type, thread count, and verbosity. Rows follow their Cartesian
  product, retaining the supplied sequence order within each quartet.
- Frequencies sum to one over sites containing valid states in all four taxa.
  Effective length is the number of these sites, rather than necessarily the
  full alignment length. Zero-site quartets return all-zero frequencies and
  length zero, while still occupying a row in the enumeration.
- Output buffers must be writable, C-contiguous, and have exactly the documented
  dtypes and dimensions. The optional quartet-index buffer has shape
  `(batch_size, 4)` and dtype `uint16`.

The original method names remain available: `next_quartet_pattern_frequencies`,
`next_quartet_pattern_frequencies_with_patterns`,
`next_quartet_pattern_frequencies_with_lengths`, and
`next_quartet_pattern_frequencies_with_lengths_with_patterns`.

The fixes correct the quartet-index buffer offset, initialize zero-site output,
validate output buffers, reject malformed alignments and invalid configuration
with exceptions, and retain ownership of the filename. Tests cover counting,
effective lengths, batching, threads, four-cluster order, and buffer boundaries.

## Migration warning

QPC 1.4/1.5 protein frequencies use PAML order `ARNDCQEGHILKMFPSTWYV`.
QPC 1.6 emits alphabetical order. Older networks and reducers must receive their
training order; changing the counter does not convert model weights. QPC 1.6
does not supply the old command-line programs or automatically save `.npy`
files. See `../QPC_MIGRATION.md` for the callers awaiting migration.
