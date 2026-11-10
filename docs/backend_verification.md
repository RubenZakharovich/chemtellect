# Step 3: backend compatibility verification

This checkpoint checks bit preservation and binary Jaccard/Tanimoto agreement
within the researcher-approved [nonzero fingerprint domain](nonzero_fingerprint_policy.md).
All saved query and index rows must pass the backend-independent guard before
real backend preparation. Supported synthetic checks and the deterministic real
fixture must then run and pass. Unsupported zero-input observations are retained
separately and do not determine supported-domain success.

## Representation and installed interface

Existing Morgan radius-2, 2048-bit, chirality-aware artifacts are authoritative.
Scientific settings, splits, arrays and manifests are unchanged.
`fingerprint_store.verify_artifact` loads full arrays while verifying provenance,
checksums, shapes and ordered record mapping. `unpack_fingerprints` reconstructs
RDKit vectors. The adapter never parses SMILES, generates fingerprints,
deduplicates records or uses source rfp. Strings exist only in memory.

Packed storage is uint8 `(n,256)`, bit 0 in the least-significant position of byte
0. NMSLIB accepts **2048 space-separated `0` or `1` tokens**, with token i equal
to RDKit bit i. Leading/trailing zeros and row order are preserved. Contiguous
text such as `01000000` is rejected by the installed parser. Every packed byte
value 0–255 is valid. Empty conversion returns []; empty pairwise fixtures fail.

Initialization uses `space="bit_jaccard"`,
`data_type=nmslib.DataType.OBJECT_AS_STRING`, `dtype=nmslib.DistType.FLOAT`,
`method="hnsw"`. No graph is built. Registration accepts unique signed 32-bit
integer IDs, excluding booleans. Installed `getDistance` addresses **insertion
positions**, not explicit IDs. The adapter maps each ID to its insertion position;
tests use nonconsecutive IDs. Molecular identity stays in records.json. Future
index IDs correspond to original index rows.

## Supported checks and unsupported diagnostics

For a nonempty union, binary Jaccard similarity is intersection/union, equal to
binary Tanimoto. We compare NMSLIB distance with `1 - RDKit TanimotoSimilarity`
using absolute tolerance **1e-6**, relative tolerance **0**. Required values must
be finite and within [0,1], allowing only that tolerance at boundaries. Reverse
calls check symmetry; self-pairs and independently counted intersections/unions
are included. Values are never clipped or adjusted.

In the tested NMSLIB 2.1.2 / RDKit 2026.09.1 environment:

| Unsupported input pair | RDKit Tanimoto | 1 − T | NMSLIB distance |
| --- | ---: | ---: | ---: |
| zero / zero | 0 | 1 | NaN |
| zero / nonzero | 0 | 1 | 1 |

These native observations remain version-specific diagnostic evidence. They are
outside the approved input domain. NaN is encoded as null with explicit `NaN`
classification; it is never normalized. These pairs do not count toward
supported equivalence or its maximum error. The original unsuccessful report is
preserved; new reports record the approved domain and new outcome.

## APIs and fixture scope

`fingerprints.validate_nonzero_fingerprints(queries, index, records)` is the
shared backend-independent full-input guard. It validates shape/dtype/mapping,
scans all packed bytes, returns counts or raises `NonzeroFingerprintError` with
all affected original identities. The runner checks full manifest counts and
original row order before invoking it. Future exact and HNSW entry points must
invoke the guard on all selected rows before preparing either backend.

`nmslib_adapter` exposes conversion (`to_nmslib_strings`), explicit registration
(`register_rows`), low-level controlled comparison (`compare_pairs`), supported
synthetic checks (`verify_synthetic`), native unsupported diagnostics
(`unsupported_zero_diagnostics`), and the guarded real fixture
(`verify_real_fixture`). Low-level comparison intentionally permits diagnostic
zeros; the production real-data entry points always enforce the guard.

The synthetic fixture independently checks token positions, including 0, 1, 7,
8, 31, 32 and 2047, plus empty/full representations. Required distances use only
six nonzero rows (21 unordered pairs including self): duplicates, disjoint sets
and hand-computable partial overlap are retained. Zero probes are diagnostic only.

The real fixture selects exactly query rows 0–9 and index rows 0–9 **after the
full scan**. Temporary IDs 0–19 map to role, original row, source ID and line
number. All 210 unordered pairs including self are compared, with reverse calls
for symmetry. Fewer than ten rows in either role fails. The small fixture is
not representative; full validation is not exhaustive dataset retrieval.

## Execution, reports and timing

```bash
uv run pytest tests/test_nmslib_adapter.py tests/test_fingerprints.py
uv run pytest
uv run jupyter lab notebooks/03_backend_verification.ipynb
```

Edit the single artifact-path selection cell in notebook 03 for another existing
artifact. Split linkage is derived from its manifest and verified by the storage
module. Source, package, artifact/split identities and output paths derive from
verified metadata. Full arrays are loaded, never a 1K prefix in place of a larger
parent. Missing input stops without fallback or generation. Packages stay separate.

`run_verification(root, manifest, queries, index, records, loading_timings_ns)`
accepts full arrays and mappings from verified loading. It exclusively creates
unique `run_<uuid>.json` reports under
`experiments/results/backend_verification/<source>/<package>/<artifact_id>/`.
Report schema 2 records policy/domain, full checked counts, zeros/affected records,
provenance/environment/Git state, exact serialization, tolerance, required check
statuses, selected identities, pair observations, supported maximum error and
separate unsupported diagnostics. Reports exclude arrays, strings and SMILES.

Guard failure saves the report before `VerificationFailure` is raised. Subsequent
checks are `not_run` with null pass status. A failed or incomplete required check
cannot yield overall success. The exception exposes `report_path` and `report`;
its message identifies the failed policy and affected row examples. Unique runs
preserve historical reports. Valid JSON never emits nonfinite numeric constants.

Timing uses perf_counter_ns. Full artifact loading/integrity verification remains
a combined I/O stage under the existing storage API; metadata loading is separate.
Nonzero validation, conversion/registration, RDKit reconstruction and distance
verification have separate measurements. The guarded fixture also independently
validates its full inputs for safe standalone use. Diagnostics have separate
measurements. These are preparation diagnostics, not benchmark latency results.

There is no HNSW graph, nearest-neighbor search, exhaustive dataset comparison,
top-k/threshold ground truth, recall/tie evaluation or parameter sweep.
Compatibility checks do not establish retrieval recall or performance.
