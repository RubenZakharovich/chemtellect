# Single-configuration HNSW and three-backend comparison

`notebooks/06_hnsw_comparison.ipynb` selects four explicit comparisons:

| Experiment | Package | Held-out queries | Index sizes |
| --- | --- | ---: | --- |
| Independent pilot | chemicalspace_new/package_00 | 100 | 1,000 |
| Nested parent scaling | chemicalspace_new/scale_100k_00 | the same 100 at every size | 1,000 / 10,000 / 100,000 |

Packages remain separate. Prefix membership is nested inside the parent; graph
topology is not. Every selected size gets a new graph. No parameter sweep,
fallback pool, 100M measurement or reliable 100M prediction is provided.

## Parent preparation and verified inputs

The parent raw pool and acquisition manifest already existed. Its absent
processed inputs were prepared through the existing notebook 01/02 workflow:
`dataset_overview.audit`, `make_split`, `descriptor_rows`, `save_json`, then
`fingerprint_store.prepare_artifact` and `verify_artifact`. Configuration files
`dataset_overview_v1.json` and `morgan_r2_2048_chiral_v1.json` were unchanged.
No raw data was modified or reacquired. The overview config selects this parent's
registered prefixes; no independent per-size split is generated.

The fresh audit found 102,000 raw records, 101,994 parsed records, six invalid
molecular records and 101,991 unique canonical structures (three surplus duplicate
records). First-physical-line representatives, canonical/isomeric sanitized SMILES
without standardization, seed 42 and `sha256-json-seed-source-package-id-v1` are
unchanged. All 101,991 eligible unique structures received the existing descriptor
set, with `descriptor_limit: null`. The split has 100 queries, a 100,000-row ordered
index and 1,891 unused eligible structures. Canonical query/index structures are
disjoint; fingerprint collisions remain separate records.

Parent identities:

- Raw SHA-256: `b77428937326c99f80cb28b1d0b3be549d9663fbf11cdee26b0d55f90dc96e1d`
- Split: `4141f1864e80bc91e9d064df766dd038eae7210792107f52c534f317849097de`
- Fingerprint artifact: `d2b04c1e24357f16ff9a8609e0aa2349eccade282d596da005afc8563edd4d3e`

The split is under `data/processed/chemicalspace_new/scale_100k_00/`, and the
artifact under its `fingerprints/<artifact_id>/` directory. Audit and descriptor
summary files are under `experiments/results/overview/chemicalspace_new/scale_100k_00/`;
the original generation report is under `experiments/results/fingerprints/`.
All 100 queries and 100,000 parent fingerprints passed the shared nonzero guard.
The pilot's original raw/split/artifact identities remain unchanged.

`run_comparison` reuses `exact_retrieval.load_selection` to verify complete parent
artifacts, linked split identities/checksums, configuration and ordered mappings,
then selects only a registered prefix. It hashes the current raw pool and checks
its physical count against acquisition metadata and inherited split hash.
Expected source/package/query/index counts are asserted by the notebook. Missing
or inconsistent inputs stop before backend preparation. Every selected query and
index row passes the shared nonzero guard before reconstruction or indexing.

## Installed NMSLIB integration

The installed NMSLIB 2.1.2 binary's docstrings and native synthetic probes were
inspected. `addDataPoint(id, value)` returns an insertion position; `knnQuery`
returns explicit IDs, observed as int32, and float32 distances. This distinction
was checked with nonconsecutive IDs. `getDistance` still uses insertion positions.
No batch-query behavior from another release is assumed.

`hnsw_adapter` reuses the existing 2,048-token, space-separated bit serialization
and original-order explicit registration from `nmslib_adapter`. The approved
configuration in `experiments/configs/hnsw_v1.json` is fixed:

```text
space=bit_jaccard, data_type=OBJECT_AS_STRING, dtype=FLOAT, method=hnsw
createIndex({M:16, efConstruction:200, indexThreadQty:1})
setQueryTimeParams({efSearch:200})
knnQuery(prepared_query_string, k=k), one query at a time
```

This build accepted one construction thread. No batch calls, parallel queries,
parameter tuning or graph reuse occur. Graph construction is measured once per
size. No exposed graph random seed is set; byte-identical graph topology and
build-to-build variance are not established. Saved graph/data files preserve the
actual constructed graph for inspection.

## Correctness and recall

Each run computes and saves a **fresh full float64 RDKit score matrix** from
verified bits before comparative timing. Each timed RDKit request independently
recomputes exhaustive scores and selects top-k; saved scores never answer a timed
query. The existing exact selection, exact-equality ties, original-position
ordering and inclusive float64 threshold definitions remain authoritative.

FPSim2 uses the existing saved-bit adapter and strict synthetic/real top-k and
threshold gates. HNSW's synthetic/real gates check exactly k unique valid explicit
IDs, original identity mapping, finite distances in [0,1], and agreement of each
returned distance with `1 - RDKit_score` within absolute 1e-6, relative zero.
Distances are not clipped; `1-distance` is retained only as a diagnostic similarity.
Returned neighbors are never replaced, padded or repaired using ground truth.
An approximate recall below one is a quality observation, distinct from malformed
responses or incompatible distances. Required correctness failures block speed
publication and preserve failure evidence.

For kth authoritative score b, let A be positions with score > b and B positions
with score == b. For returned candidate set C of size k, tie-aware top-k recall is:

```text
(|C ∩ A| + min(k - |A|, |C ∩ B|)) / k
```

Extra boundary ties cannot compensate for missing strict winners. There is no
epsilon. For tau=0.3,0.5,0.7,0.9, threshold recall is the fraction of the complete
`score >= float64(tau)` qualifying set present in that same designated candidate
set. Empty qualifying sets have null recall and explicit counts; reported means
use defined queries only. This is top-k-limited threshold recall, separate from
FPSim2's native threshold-search compatibility gate. Native tolerances never enter
these metrics. Recall measures agreement with a lossy fingerprint representation,
not independent chemical relevance.

## Timing, costs and evidence

For k=10,25,50,100 independently: one untimed warmup and five sequential measured
passes. A local `random.Random(42)` produces a fresh permutation per k/pass,
shared across all three backends. Global RNG/split state is untouched. Backend
order cycles through three orders, then two reversed orders, and is recorded.
Each request executes its requested k; no max-k search is truncated. First measured
pass candidates per backend/k/query are retained for every recall calculation.
Later repetitions supply timing and diagnostics, never pooled or best-pass recall.

Existing RDKit scoring/selection timers and FPSim2 search/normalization timers are
reused. HNSW separately times single `knnQuery`, distance/original-row ordering,
and their enclosing total. String preparation is outside search latency; NMSLIB's
native parsing inside `knnQuery` remains included. Normalization never consults
ground truth. Primary speedup uses total latency, including required normalization.

Query conversion is separately measured for all three backends. FPSim2's combined
database preparation retains component conversion timings; components must not
be added to their enclosing total. HNSW conversion/registration, graph construction,
query settings and graph saving are separate. Fresh reference computation,
preflight checks, outside-timer validation/result assembly, designated recall,
plots, serialization and reload validation have separately labeled costs.
Stages need not sum to elapsed time.

Each query's five-pass median is computed first. Across-query median, mean and p95
use the existing linear percentile convention. Raw integer ns, per-query medians,
absolute time saved and speedups vs fresh RDKit and FPSim2 totals are retained.
All warmup and measured responses are checked outside retrieval timers.

Versions, Git revision/dirty state, source-file and installed-binary checksums,
hardware, actual thread settings and synchronous total/available system RAM and
process RSS snapshots are retained. Snapshots are **not peak memory**; no memory
polling runs during retrieval. OS scheduling, caches and CPU behavior are uncontrolled.

Unique runs are under:

```text
experiments/results/hnsw/<source>/<package>/<artifact_id>/index_<size>/run_<uuid>/
  exact_scores.npy, ground_truth_top_k.json, records.json, configuration.json
  index_fpsim2.h5, synthetic_fpsim2.h5, prepared_queries_fpsim2.npy
  index_hnsw.bin, index_hnsw.bin.dat, prepared_queries_hnsw.json
  report.json, manifest.json
  three_backend_latency.png/.svg, three_backend_speedup.png/.svg
  recall_vs_latency.png/.svg
```

Exclusive writes to new hidden staging directories, payload reload/scientific
validation and file checksums precede the final manifest and directory publication.
Completed runs are never overwritten. Failed correctness runs publish
`failed_validation` evidence and no comparative summaries or charts; unexpected
output failures retain staging for inspection. There is no automatic recovery or
power-loss durability guarantee. `load_comparison_run(path, root)` rechecks saved
responses, designated passes, recall, normalized ordering, timing protocol and
summaries, inventory/checksums and current artifact/raw/identity linkage.

## Checks and notebook

```bash
.venv/bin/python -m pytest tests/test_hnsw.py tests/test_fpsim2.py tests/test_exact_retrieval.py tests/test_nmslib_adapter.py tests/test_fingerprint_store.py tests/test_fingerprints.py
.venv/bin/python -m pytest
.venv/bin/jupyter lab notebooks/06_hnsw_comparison.ipynb
```

Synthetic checks cover actual installed bit conversion, explicit IDs, collisions,
ties, zero overlap, malformed/short responses, distance diagnostics, tie-slot
accounting, undefined threshold recall, independent requests, clocks, shared orders,
first-pass retention, failure evidence and publication/corruption behavior.
The thin notebook validates four explicit inputs and shows provenance, configuration,
correctness, recall, per-size/per-k latency, graph/preparation costs, memory, speedup
and recall-versus-latency plots. Completed runs can be inspected without writes.
