# RDKit exhaustive retrieval checkpoint

`chemtellect.exact_retrieval` and notebook 04 benchmark an existing verified
fingerprint artifact. The existing [fingerprint configuration](fingerprints.md),
canonicalization, saved split and [nonzero policy](nonzero_fingerprint_policy.md)
remain authoritative. There is no acquisition, molecular processing, fingerprint
regeneration, deduplication, FPSim2 or HNSW graph construction.

## Inputs and reusable APIs

`load_selection(root, artifact_path, index_size)` locates the linked split from
explicitly selected artifact metadata. It reuses `validate_split`,
`verify_artifact`, `index_records` and `mapping`. Complete-parent verification
covers configuration, provenance, checksums, shapes and ordered identities.
Selection retains all saved queries and an explicitly registered index prefix.
Mappings derive through the existing API from the verified split. Missing files,
invalid linkage, unsupported sizes, corruption and RDKit version mismatch stop
execution without choosing alternatives or generating data.

`run_benchmark(root, artifact_path, index_size, expected_selection=None)` verifies
inputs itself. Optional expected selection asserts source, package, query count
and index size before preparation. The shared `validate_nonzero_fingerprints`
scans **every selected row** before existing `unpack_fingerprints` reconstructs
RDKit vectors from verified little-bit-order bytes. Unselected parent index rows
receive integrity verification but are outside this run's nonzero scan. Zero
fingerprints stop the run without record removal, replacement or backend repair.

`exact_top_k(scores, k)`, `threshold_members(scores, threshold)` and
`validate_scores(scores, shape)` implement numerical policy. `measure_exact`
measures one fresh request. `benchmark_queries` orchestrates already guarded,
reconstructed inputs; custom candidate counts/clocks/scorers support small tests.
Production `run_benchmark` uses native RDKit and the fixed protocol below.
`load_run(directory)` verifies completion, inventory, checksums, internal linkage,
matrix contract, top-k identities/order/scores, request order and summaries.
Checksums detect changes; they do not prove scientific correctness or authenticate
externally supplied evidence.

## Exact numerical policy

Every query calls `DataStructs.BulkTanimotoSimilarity(query, index)` on the entire
selected index. Python float results become a float64 row. The saved
`exact_scores.npy`, shape `(query_count, index_size)`, is authoritative and retains
original role ordering. Wrong dimensions/dtype, nonfinite values and scores
outside [0, 1] are rejected. Values are never clipped, rounded or adjusted.
Subsequent computations must agree by direct array equality outside retrieval.
Backend compatibility tolerances never enter these definitions.

Ties use direct equality. Displayed top-k is descending similarity, then ascending
original index position. `numpy.partition` finds the exact kth boundary; all
higher scores plus the earliest original positions at that boundary fill k
slots. Only those k positions are ordered. An oversized boundary tie therefore
does not trigger a full-index sort. Partition/scans require linear work and
space; selected ordering costs O(k log k). When k equals index size, ordering
necessarily covers all rows. Valid k is a Python integer in 1..index size;
booleans are rejected. Distinct structures with identical bits stay distinct.

`threshold_members` uses `score >= np.float64(threshold)` and returns original
index positions, including an empty array when none qualify. A configured
threshold must be a finite Python float or NumPy float64 in [0, 1]. There is no
threshold sweep or default scientific cutoff in this run. These definitions
measure agreement with a lossy representation, not independent chemical relevance.

## Timing protocol and boundaries

Candidate counts are **10, 25, 50, 100**, in that order. Each k has one **untimed**
warmup pass over all queries, then **five measured** passes. Queries execute
sequentially. A dedicated local `random.Random(42)` shuffles a fresh original
query-position list for each pass, including warmups. Split generation and
Python/NumPy global random state are untouched. Seed, implementation, Python
version and all realized query permutations are recorded.

Every request recomputes exhaustive scores and selects/orders its own k results.
Cached scores never answer retrieval requests. The first measured pass for k=10
captures ground truth once per original query. All subsequent computations,
including later k warmups, must match it exactly after the timer closes. The
first measured pass for **each k** supplies designated top-k evidence; repetitions
are never pooled or used to choose favorable results.

`time.perf_counter_ns()` records:

| Stage | Boundary |
| --- | --- |
| `scoring_ns` | Immediately before BulkTanimotoSimilarity through its returned Python list |
| `selection_ordering_ns` | Float64 conversion, partition, boundary ties and ordering of selected positions |
| `retrieval_total_ns` | Encloses scoring through ordered positions, including conversion and clock overhead |

Argument/protocol checks precede timing. Full score validation, equality checks,
ground-truth capture/persistence, identity-dictionary materialization, evaluation,
reconstruction, printing and report writing are excluded. Warmups score, convert
and select without reading the timing clock.

All raw measured timings retain k, pass index, original query position and
execution-order position. Each query/k/stage is summarized by its median across
five passes. Median, mean and p95 are then computed across those per-query
medians, using recorded NumPy percentile `method="linear"`. Disk units are integer
nanoseconds for raw records and float nanoseconds for summaries; the notebook
displays microseconds. This is a **repeated in-memory benchmark**, not cold-start
latency or concurrent-service performance. OS scheduling and CPU behavior remain
uncontrolled; hardware records include platform, CPU model/count and affinity.

## Preparation, provenance and publication

Selection metadata/split loading and validation is separately timed. The existing
artifact API combines full-array loading and integrity verification, so that
stage is accurately labeled as combined, including prefix selection. Nonzero
validation and RDKit reconstruction are separate. Result assembly, serialization
and writing of five payload files is a combined stage; reloading/validating them
is separately measured.
Metadata/manifest writing, checksums, final completion validation and directory
publication are explicitly excluded from those payload-writing measurements.
Stages need not sum to whole-run elapsed time.

Unique runs use:

```text
experiments/results/exact_retrieval/<source>/<package>/<artifact_id>/run_<uuid>/
  exact_scores.npy
  records.json
  top_k.json
  timings.json
  summary.json
  metadata.json
  manifest.json
```

The matrix loads with `allow_pickle=False`. Ordered records retain original row,
source ID and physical raw line number. Designated top-k stores query identity
and ordered matches with index position, record identity and exact similarity.
Raw timings and every realized warmup/measured query order are retained, together
with all per-query medians and aggregates.

Metadata records inherited raw SHA-256 (raw pools are not rehashed here), split
path/hash/identity, artifact path/manifest hash/identity, selected counts, complete
fingerprint identity/configuration, Python/RDKit/NumPy versions, Git revision and
dirty state, hardware, numerical definitions, timing settings/units and stage
boundaries. Artifact reuse is explicitly labeled; it is never reported as
fingerprint generation. Original artifact creation/environment/execution metadata
is retained. Available original `operation="generated"` reports are referenced
with checksums; reuse reports do not count as generation evidence.

Files are exclusively created in a unique hidden staging directory. Payloads are
reloaded and validated against in-memory scores and identities, metadata is
round-tripped, and checksums are checked before writing `manifest.json` last as
the completion marker. The complete run is checked again before directory
publication. Completed runs are never overwritten. Failures retain staging for
explicit inspection; there is no automatic recovery, resumability, caching or
cleanup. Hidden staging is not a published run, even if failure occurred after a
manifest was written. This is not a power-loss durability guarantee.

## Validation and execution

```bash
.venv/bin/python -m pytest tests/test_exact_retrieval.py tests/test_fingerprint_store.py tests/test_fingerprints.py tests/test_nmslib_adapter.py
.venv/bin/python -m pytest
.venv/bin/jupyter lab notebooks/04_exact_retrieval.ipynb
```

Synthetic nonzero tests cover scalar/bulk agreement, identity alignment,
independent ordering references, oversized ties, adjacent float64 thresholds,
distinct-record collisions, invalid inputs, controlled clocks/scorers, local
shuffling, designated-pass retention, median aggregation, storage-contract
fixtures, output round trips, corruption and failed publication. No test asserts
machine-dependent latency. Notebook outputs exclude the score matrix.

## Initial local evidence

The authorized initial run on 2026-11-10 verified `chemicalspace_new/package_00`,
100 queries and the registered 1,000-row index. Artifact identity was
`628c36671feeb38e1a97a34dd664bb54f43851374fcea4190a474479a12df427` and split identity
was `f85552ea5b17e4829a09df82c79ad3bb2d762ee672c0f52de234dc7fc668e908`.
The published directory is `run_1494eeec3c0f489c9b67f7352dc573b7` under the
source/package/artifact path described above. The executed notebook retains its
compact report. The `(100, 1000)` float64 matrix, 2,000 raw measured requests and
all other saved evidence passed `load_run`; selected artifact and split file
checksums were unchanged before/after execution.

Observed total retrieval summaries, in microseconds, aggregate **per-query
medians across five passes**:

| k | Median | Mean | p95 (linear) |
| --- | ---: | ---: | ---: |
| 10 | 121.757 | 122.002 | 123.618 |
| 25 | 122.572 | 123.038 | 124.522 |
| 50 | 122.823 | 123.455 | 128.613 |
| 100 | 126.377 | 126.698 | 129.581 |

Selection metadata/split loading-validation took 3.760 ms; combined full artifact
loading/verification/selection took 8.145 ms. Selected nonzero validation took
0.380 ms, reconstruction 331.581 ms, payload assembly/serialization/writing
57.625 ms, and payload reload-validation 93.959 ms. These are separate preparation
or output measurements, excluded from retrieval latency. The metadata references
the available original generated fingerprint report with its checksum. This run
reused the artifact; it did not generate fingerprints.

Validation before execution: 97 focused tests and 174 full-suite tests passed;
all five notebooks passed schema validation and code-cell syntax checks.

## Reporting additions in the FPSim2 checkpoint

New exact runs record synchronous named RAM/RSS snapshots, including total and
available system RAM in bytes, process RSS in bytes, UTC and sampling duration.
These are separate quantities and neither is true peak memory. No background
polling occurs during retrieval. Metadata explicitly records sequential RDKit
execution, and that outside-timer validation/evaluation has no separate aggregate
timing. Historical run files are unchanged; absent fields display “not recorded”.
The loader accepts legacy runs and new runs with optional checksummed chart files.

Notebook 04 presents provenance/protocol, scoring/selection/total medians and
total p95 in tables, readable preparation/output units, labeled memory snapshots,
and median/p95 latency curves. The p95 label is explicitly “p95 of per-query
median latency”. Optional notebook-owned report writers save PNG/SVG files before
publication, outside retrieval and payload-writing timers. `EXISTING_RUN_PATH`
can select a historical run for read-only display; `None` creates a new run.

[FPSim2 compatibility and freshly paired timing](fpsim2.md) now have their own
checkpoint. True peak memory, HNSW and cross-backend break-even analysis remain
deferred. The full matrix and reconstructed index reside in memory. These pilot
data do not establish population representativeness, general performance scaling
or chemical relevance.
