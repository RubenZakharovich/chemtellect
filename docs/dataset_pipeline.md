# Reusable local acquisition workflow

`notebooks/07_dataset_pipeline.ipynb` coordinates `chemtellect.raw_pipeline` and
displays compact, verified results. It defaults to **load** mode. Heavy logic is in
modules; the notebook neither edits nor executes notebooks 00–06.

Run from the repository root using the existing locked environment:

```bash
.venv/bin/python -m chemtellect.raw_pipeline --mode execute \
  --config experiments/configs/chemicalspace_new_scale_1m_pipeline.json \
  --directory experiments/results/pipeline/chemicalspace_new/scale_1m_00/experiment_20261112_0333 \
  --audit-path experiments/results/raw_pipeline/acquisition_audit
```

This is the actual experiment command used for the new acquisition. The audit
cache was freshly generated with `dataset_overview.audit` and its raw linkage,
chemistry counts, eligible rows and file checksums are verified before reuse.
Omit `--directory` and `--audit-path` to start a new immutable experiment and fresh
audit. Use the same explicit paths to resume. Unfinished stage outputs are retained
and require inspection; the runner never automatically repairs or deletes them.
A controller lock prevents simultaneous execution of the same experiment. A lock
left after interruption must be inspected along with the live worker and staging
files; it is never automatically broken. Load mode can still verify published runs.
A deliberate new timing run uses a **new experiment directory**. Completed sizes
are verified rather than retimed on resume. No client server is contacted.

Verify and export saved reports without searching:

```bash
.venv/bin/python -m chemtellect.raw_pipeline --mode load \
  --directory experiments/results/pipeline/chemicalspace_new/scale_1m_00/experiment_20261112_0333
.venv/bin/jupyter lab notebooks/07_dataset_pipeline.ipynb
```

Each export gets a unique `report_<id>` directory with Markdown, CSV, PNG/SVG and
checksums; `latest_report.json` points to its verified manifest. Large matrices,
response/native-score evidence, orders, raw timings, databases and graphs remain
in the established HNSW run directories. `state.json` is a mutable controller journal;
stage markers, fingerprint artifacts and benchmark runs are immutable. A nonzero
CLI execute exit (2) means blocked/partial, not a completed experiment.

New parents are saved by split identity under `data/processed/<source>/<package>/splits/`.
If a matching artifact already exists, its exact split path is adopted only after
full artifact/configuration/content verification. This also supports the initial
1M parent's saved path inside its original experiment directory. New experiment
directories reuse that verified parent and artifact while producing fresh run IDs;
changing a report directory never changes or reinterprets the parent.

The package-specific JSON config selects 100 queries, seed 42, all-eligible
descriptors (`descriptor_limit: null`) and the registered nested sizes. It references
the complete approved Morgan and HNSW configurations. Historical configs and saved
reports retain their original meaning. Only explicitly registered local acquisitions
are accepted; `dataset_overview.inventory` intentionally retains its historical nine
pool display, while `dataset()` additionally accepts this new 1M acquisition.

The workflow verifies acquisition count/SHA, input integrity, sanitized canonical
isomeric structures and first physical duplicate representatives. Integrity failures
and chemical exclusions stay separate. Fewer than 1,000,100 eligible structures blocks
the requested parent; no substitute split, query overlap, resampling or smaller
index labeled 1M is permitted. Descriptors use batches of 256 and cover every
eligible structure. Unused structures are saved separately from the selected parent.

Fingerprint preparation uses `fingerprint_store.prepare_artifact` once for the full
parent and validates configuration, hashes, dimensions, little packed-bit order,
ordered identities, RDKit linkage and nonzero fingerprints. Distinct canonical
structures with colliding fingerprints retain distinct identities. Artifact
generation timings and reuse/verification timings are recorded separately.

Each size runs in a separate worker through the existing
`hnsw_benchmark.run_comparison`, with synthetic compatibility gates and real-data
preflight. RDKit/FPSim2 remain exact with tie-aware membership; HNSW misses lower
recall. Native-score tolerance is absolute 1e-6, relative zero, never membership.
Timing retains the original protocol, k values, thresholds, one warmup and five
measured passes, seeded query shuffles, rotating backend order, timer boundaries
and one concurrent query. RDKit always recomputes scores during timed requests.
One fresh graph is built per size. The NMSLIB binary's CPU-instruction warning is
recorded in logs; no backend replacement or dependency upgrade is performed.

Resource gates record RAM, swap, disk and planning estimates before large work.
Smaller completed sizes provide RSS snapshot calibration before the next size.
Estimates and snapshots are not measured peaks. Preparation now writes the exact
float64 reference row by row into an `.npy` memory map, reads/verifies it by rows,
reconstructs RDKit vectors directly from the verified binary bytes, streams NMSLIB
text registration and batches FPSim2 conversion/round trips/database appends.
These changes preserve search semantics and timing. Matrix persistence is included
in the reference-construction cost; the later reference-evidence stage covers
top-k evidence. Hashing, publication and controller overhead are not all individually
timed. The original FPSim2/native NMSLIB graph and database paths remain in memory.

For the 1M preparation, NMSLIB text encoding uses an exhaustive 256-entry byte
lookup table to produce exactly the original 4,095-character row string. The
first three sizes used NumPy scalar-to-string encoding. This changes the conversion
cost, which should not be interpreted as one constant implementation scaling curve;
native graph construction and all timed search operations/configurations are
unchanged. Exhaustive byte equivalence and native compatibility are verified.

Comparison report schema 2 adds verified query and prefix identity digests and
explicit registered sizes; the existing loader still accepts schema 1. Parent
charts reject different sources, parents, queries, configurations and reordered
prefixes. The historical pilot plus three-size overview remains supported.

Tests use synthetic/small fixtures, including a real unattended three-backend
workflow, checksum tampering, cross-parent/prefix rejection, collision/tie/empty
threshold cases and bit-for-bit bounded-reference equivalence:

```bash
.venv/bin/python -m pytest -q
```

Interpret latency as median/mean/linear p95 of 100 per-query five-pass medians;
speedups are ratios of headline medians. Quality uses only the designated first
measured pass. CSVs preserve per-query threshold ground-truth counts and empty-set
denominators. This is one acquisition and fixed queries, one graph per size, and
machine-specific sequential performance; repetitions are not independent graphs.
There is no measured 100M or biological validity claim and no combined speed/quality
score. A skipped 1M run means the requested four-size experiment is incomplete.
