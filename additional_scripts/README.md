# Additional scripts and detailed training documentation

Run the commands below from the **project root**, not from this directory.
The supported workflow is documented in the [main README](../README.md).
Training and training-data simulation entry points remain in the project root;
the simulated-data benchmark is in this directory.

## Contents

- `vis_tree.py`: interactive visualization of a Newick tree: `python additional_scripts/vis_tree.py tree.nwk`.
- `run_analysis_on_test_datasets_aa.py`: simulated-data benchmark; see the example below.
- `pca_plot_variance_explanation.py`: plot cumulative variance from a fitted PCA model: `python additional_scripts/pca_plot_variance_explanation.py model.pkl LG`.
- `how_much_variances_explained_by_first_8_components.py`: fit PCA-8 to a frequency array and report retained variance: `python additional_scripts/how_much_variances_explained_by_first_8_components.py frequencies.npy`. This historical utility loads the array into RAM; provide frequencies without an appended length column.
- `quartet-pattern-counter-v.1.4/` and `quartet-pattern-counter-v.1.5/`: historical sources, not installed by the current installer.

The simulation test runner is `additional_scripts/run_analysis_on_test_datasets_aa.py`.
`simulate_training_data.py` and `train_phyloai.py` remain in the project root.
Use `run_phyloai.py` for empirical tree reconstruction. Old QPC sources are
retained for reference, not used by the current workflow.

## Training protein quartet models

Use **`train_phyloai.py`** as the single training entry point for the protein
models. It replaces the older `TopologyPredictorTraining.py` and
`ModelPredictorTraining.py`. ModelDetector is supplied as an external pretrained
model and is not trained by this script.

### Training data

Prepare matching NumPy arrays:

- Frequencies: `(N, 160000)`, containing only relative site-pattern
  frequencies in `ACDEFGHIKLMNPQRSTVWY` Cartesian-product order.
- Topology labels: `(N,)`, `(N, 1)`, or one-hot `(N, 3)`, with classes
  0 = AB|CD, 1 = AC|BD, and 2 = AD|BC.
- Fisher and six-state-LA additionally require `--length-data`: one `.npy`
  array of raw lengths, shaped `(N,)` or `(N, 1)`, per frequency file.
  Lengths must be finite and positive. Older arrays with an embedded length
  column remain accepted for compatibility; it must match the separate file.
- Multiple frequency, label and length files are accepted in matching order.
  Fixed-length `--method raw` ignores an embedded constant length column and
  does not accept separate length inputs.

These are the saved frequency arrays used for network training. Alignment
simulation is a separate preparation step using PolyMoSim. 

The supplied [training scheme](../training_scheme.txt) contains two columns:
alignment length and replicates **per topology and branch regime**. All rows
are simulated, without sharding. Three quartet topologies and two branch regimes
give `6 × sum(replicates)` examples. The supplied scheme spans 42 lengths from
50 to 1,000,000 sites and sums to 178,000 replicates, producing **1,068,000**
examples. The four-batch training subset used in the study contained 284,784
examples; that historical partition is no longer applied by this simulator.

```bash
python simulate_training_data.py -substitution_model LG \
  --training-scheme training_scheme.txt -processes 4 -output_dir simulation_output
```

For fixed-length training without a scheme:

```bash
python simulate_training_data.py -substitution_model LG \
  --alignment-length 1000000 --replicates 50000 \
  -processes 4 -output_dir simulation_output_fixed
```

The fixed-length example generates 300,000 quartets, not 50,000.
Use PolyMoSim model names such as `DAY`, `Qplant`, and `Qinsect` (without dots).
Override the simulator with `--poly-mosim /path/to/binary` if needed.

Each run saves three arrays, with matching row order:

- `site_pattern_frequencies_<model>_scheme_<scheme-name>.npy`: float32 `(N,160000)`,
  containing only original frequencies in alphabetical amino-acid order.
- `trainTopoClass_<model>_scheme_<scheme-name>.npy`: int64 `(N,1)`, labels 0–2.
- `trainLength_<model>_scheme_<scheme-name>.npy`: float32 `(N,1)`, raw lengths
  ready for the trainer's `--length-data` option.

For fixed-length runs, the suffix is `L<length>_R<replicates>` instead of
`scheme_<scheme-name>`. Raw length is stored only in its separate file.

Results are written in bounded batches to disk-backed NumPy arrays; the full
dataset is not retained in RAM. `--batch-size` defaults to 1,000; its frequency
buffer alone is about 640 MB, with additional temporary copies and worker memory.
`-processes` controls simulation workers. Disk space still scales with all rows:
the supplied scheme needs about 684 GB for the frequency file, and the
300,000-row fixed-length example needs about 192 GB. 

Training internal branches span 0.02–0.5 substitutions/site in one regime,
and 0.001–0.02 in the short regime. Below 1,000 sites, the short-regime lower
bound is multiplied by `1000 / length`. Terminal branches range from 0.02 to
0.5; gamma shape ranges from 0.01 to 4 and invariant proportion from 0 to 0.5.
Seeds are deterministic and unique within a run. Changing worker count or
buffer size does not change results. Temporary simulation files are removed.

### Default PhyloAI-Fisher training

```bash
python train_phyloai.py \
  --training-data simulation_output/site_pattern_frequencies_LG_scheme_training_scheme.npy \
  --topology-data simulation_output/trainTopoClass_LG_scheme_training_scheme.npy \
  --length-data simulation_output/trainLength_LG_scheme_training_scheme.npy \
  --substitution-model LG \
  --method fisher \
  --selected-patterns 10000 \
  --architecture b1 \
  --length-transform log10 \
  --output-dir trained/LG \
  --threads 8
```

The selected  frequencies and log10 alignment length enter separate normalized branches.
The defaults retain the Fisher trainer's dropout of 0.4, L2 penalty of 0.0001,
20% validation split, seed 13, and maximum of 200 epochs.
Frequency arrays are memory-mapped and read in batches; only the much smaller
label and length arrays are loaded in full.

### Supported families

| Model | Options |
|---|---|
| Full-pattern, fixed-length | `--method raw --architecture b3` (no `--length-data`) |
| Fisher | `--method fisher --selected-patterns 10000 --architecture b1 --length-data LENGTHS.npy` |
| Dayhoff-6-LA | `--method six-state --recoding dayhoff-6 --architecture b3 --length-data LENGTHS.npy` |
| SR-6-LA | `--method six-state --recoding sr-6 --architecture b3 --length-data LENGTHS.npy` |
| KGB-6-LA | `--method six-state --recoding kgb-6 --architecture b3 --length-data LENGTHS.npy` |

Only B1 and B3 architectures are supported. PCA, random projections, shallow
networks, non-length-aware Fisher models, and non-length-aware six-state models
are no longer trained by this script. 

Six-state recoding replaces Fisher selection, producing 1,296 features.
The original comparisons used Dayhoff, JTT, and WAG for Dayhoff-6, SR-6, and
KGB-6 respectively. No PCA or Fisher selection is applied to six-state inputs.
Pre-recoded inputs may use `--input-format recoded`.

Raw full-pattern networks default to B3/tanh without length; use fixed-length
training data. Fisher defaults to B1/ReLU, and six-state-LA to B3/ReLU. Both
length-aware families receive length through a separate normalized branch;
`--length-transform log10` is the default. `raw` uses untransformed lengths,
and `both` trains separate raw-length and log10-length networks.

New simulations save separate raw-length arrays automatically.

Each run saves a TensorFlow SavedModel, `.training.json` metadata, history,
and train/validation row indices. Fisher also saves an `.npz` Fisher selector.
Fisher selection and six-state recoding are saved as float32 `.features.npy`
arrays: `(N,K)` for Fisher-selected patterns and `(N,1296)` for six-state patterns.
These contain all rows in their original order, before network normalization;
length remains separate. Training reads these arrays through memory mapping.
Feature selection and normalization use training rows only. Existing outputs
are not overwritten; choose a new output directory or model prefix.

```bash
python train_phyloai.py --help
```

### Reusing transformed arrays

For Fisher-selected input, provide the matching selector; feature columns must
follow its `indices` order. No Fisher scoring or selection is repeated:

```bash
python train_phyloai.py \
  --training-data trained/LG/PhyloAI-Fisher-LG-k10000.features.npy \
  --topology-data simulation_output/trainTopoClass_LG_scheme_training_scheme.npy \
  --length-data simulation_output/trainLength_LG_scheme_training_scheme.npy \
  --substitution-model LG --method fisher --input-format selected \
  --selector-model trained/LG/PhyloAI-Fisher-LG-k10000-b1-selector.npz \
  --output-dir retrained/LG --threads 8
```

The feature count is inferred from the selector. Its original train/validation
row indices are reused instead of creating a new split, avoiding validation
leakage from supervised feature selection. Keep exactly the same examples,
labels, lengths and row order; `--validation-fraction` does not change this split.
The selector must contain `training_rows`, `validation_rows` and `training_order`.
It is copied beside the new network for use by the workflow.

For six-state input, specify the recoding scheme that produced the array:

```bash
python train_phyloai.py \
  --training-data trained/Dayhoff/PhyloAI-Dayhoff-6-LA.features.npy \
  --topology-data simulation_output/trainTopoClass_DAY_scheme_training_scheme.npy \
  --length-data simulation_output/trainLength_DAY_scheme_training_scheme.npy \
  --substitution-model Dayhoff --method six-state --recoding dayhoff-6 \
  --input-format recoded --output-dir retrained/Dayhoff --threads 8
```

Six-state recoding is not repeated, and no Fisher selector is required.
When transformed arrays are supplied, the trainer uses them directly without
creating another copy. Supply raw, untransformed lengths as before.

## Benchmarking on simulated protein alignments

`run_analysis_on_test_datasets_aa.py` is the amino-acid quartet benchmark used
for the simulation comparisons.  It simulates each of the three quartet topologies with PolyMoSim, counts
site-pattern frequencies with QPC 1.6, and evaluates the supplied networks.
It supports TA/Fisher selectors, PCA/GRP/SRP reducers, raw-pattern networks,
and Dayhoff/SR/KGB 6-state models, with model-specific length inputs when
required. QPC 1.6 frequencies already follow the training pattern order;
no order conversion is performed. The included
`recode_aa_frequencies_6state.py` supplies the 6-state recoding functions.

For example, evaluate the released LG model without bootstrap or IQ-TREE:

```bash
python additional_scripts/run_analysis_on_test_datasets_aa.py \
  --models PhyloAI-Fisher-Kaggle \
  --model-name PhyloAI-Fisher-LG \
  --substitution-model LG --branch-mode intermedbr \
  --alignment-length 300 --replicates 1000 \
  --bootstrap-replicates 0 --skip-iqtree \
  --processes 1 --seed 20260806
```

The release uses the matching `.json` metadata and `.npz` selector. Historical
model collections with `.training.json` metadata are also supported. To test
all networks for one substitution model, point `--models` to that model's
directory and omit `--model-name`. Keep each network with its own reducer or
selector. For nonstandard layouts, use `model_manifest.json`; generate an
example with `--write-manifest-template`.

The benchmark uses 1,000 replicates **per topology** in the example above
(3,000 alignments in total). The main comparison conditions are:

| Internal-branch condition | `--branch-mode` | Unrooted internal-branch length |
|---|---|---|
| Short | `shortint` | 0.001–0.020 substitutions/site |
| Intermediate | `intermedbr` | 0.020–0.100 substitutions/site |
| Long | `normbr` | 0.100–0.500 substitutions/site |

The internal edge is the sum of the two equal root-adjacent branches in the
simulated rooted representation. Tested alignment lengths were 300, 1,000,
and 10,000 amino-acid sites. Use `DAY` for Dayhoff simulations and `Q.plant`,
`Q.insect`, etc. for the Q-series models.

`PolyMoSim` is the executable installed by `install.sh`; override it with
`--poly-mosim /path/to/binary` if necessary. QPC must be installed in the same
Python environment. To include maximum-likelihood predictions, omit
`--skip-iqtree` and supply `--iqtree /path/to/iqtree3` if it is not on PATH.
IQ-TREE is optional and is not installed by `install.sh`. Its model is fixed
to the simulation's substitution matrix with invariant sites and gamma rate
variation, rather than selected by ModelFinder. Bootstrap defaults to 100;
explicitly use `--bootstrap-replicates 0` for ordinary predictions only.

Results are saved under `aa_analysis_results/` by default, including
`summary.csv`, `model_ranking_percent.csv`, `MODEL_RANKING.txt`, compressed
per-quartet predictions, model inventory, and simulation configuration.


```bash
python additional_scripts/run_analysis_on_test_datasets_aa.py --help
```
