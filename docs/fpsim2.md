# FPSim2 compatibility and paired top-k checkpoint

This checkpoint consumes verified saved fingerprints and an explicit completed
RDKit run. Saved RDKit float64 scores remain authoritative. Acquisition,
canonicalization, splits, fingerprint configuration and raw/processed artifacts
are unchanged. No molecules are parsed, no fingerprints are generated, and no
MongoDB, FPSim2 molecular writer or HNSW search is invoked.

## Dependency and inspected integration

FPSim2 was absent. `uv add FPSim2==0.7.4` with constraints for all installed
packages, followed by `uv sync --locked`, added the pinned backend and required
transitive dependencies. No existing locked versions changed, including RDKit
2026.9.1, NumPy 2.5.4 and NMSLIB 2.1.2. The release is documented on
[PyPI](https://pypi.org/project/FPSim2/0.7.4/). Installed Python and C++ sources,
not assumptions about a newer upstream branch, determined this adapter. Their
checksums and runtime versions of FPSim2, PyTables, Blosc2, NumExpr, SciPy, psutil,
RDKit, NumPy and NMSLIB are recorded per run.

FPSim2 0.7.4's public molecular database writer generates fingerprints, so it is
not used. The narrow version-specific adapter writes precomputed rows using
FPSim2's `create_schema(2048)` and config layout. `process_fp` converts RDKit
vectors reconstructed by the existing storage API; it serializes existing bits
and computes population counts without molecular generation. HDF5 records have
ID, 32 uint64 words and enabled-bit population count. Words follow ascending
RDKit 64-bit blocks, with bit `64*j` at word j's most significant bit and bit
`64*j+63` at its least significant bit. This differs from the artifact's little
bit ordering inside bytes; simply viewing packed bytes as uint64 is incorrect.

The ordinary FPSim2 in-memory PyTables loader runs with `fps_sort=True`. It sorts
by population count and derives native population-count bins. Every loaded row
is decoded back to original packed bytes by its explicit search ID; every bit
and stored population count must match. Search IDs are distinct uint32 integers,
mapped to original artifact index positions and source identities. Production
uses IDs 0..999; synthetic tests deliberately use nonconsecutive IDs. Internal
sorted row positions are never interpreted as search IDs. Identical fingerprints
remain separate records.

Prepared queries use the same native conversion and full bit/count round trips.
A small `PreparedEngine` subclass overrides **only** `load_query` to return the
already prepared uint64 query. It does not change native scoring, heap selection,
pruning, bounds, search or threading. All query preparation is before timing.
The adapter rejects versions other than the inspected 0.7.4. This is a restricted
internal prepared-array interface, not a general molecular-input API.

## Input and correctness gate

`run_comparison(root, artifact_path, exact_run_path, index_size, ...)` calls the
existing `exact_retrieval.load_selection` and `load_run`. It checks the explicit
exact-run directory/run ID, every saved file checksum, complete artifact/split/raw
linkage, current artifact-manifest checksum, fingerprint identity, counts and
ordered query/index mappings. Missing or invalid input stops without fallback.
The shared nonzero guard scans all selected queries and index rows **before**
backend preparation. `verify_ground_truth`, `prepare_backend`, `verify_queries`,
`check_response`, `paired_benchmark` and `load_comparison_run` are reusable APIs.
Low-level preparation requires already guarded inputs; the public runner enforces
that requirement. Passing `root` to the comparison loader also rechecks linked
historical ground truth and saved scientific diagnostics.

The required real gate checks every query at k=10,25,50,100 and every query at
thresholds 0.3,0.5,0.7,0.9. A response must have unique valid IDs, finite scores
in [0,1], and exactly k results for top-k (no padding or truncation from another
request). Top-k explicitly sets `threshold=0.0`, including zero-similarity
neighbors when needed. For authoritative kth score b, a valid set contains all
records with RDKit score > b, fills remaining slots with score == b, and contains
nothing below b. Alternative boundary-tied IDs are valid; identical deterministic
IDs are not demanded.

Threshold responses are compared against the **complete** authoritative set
`score >= np.float64(threshold)`. Membership and ties use direct comparisons,
without epsilon, rounding or widened search thresholds. Native C++ uses float32
scores, thresholds and dynamic pruning; returned scores are float32. Every
returned native score, authoritative float64 score, signed discrepancy and
absolute error are recorded separately. Finite score diagnostics use absolute
tolerance 1e-6, relative tolerance zero. That tolerance never changes membership
or tie checks. A membership failure blocks comparative timing; nothing is patched
or corrected with ground truth or another backend.

Native integration fixtures use nonzero synthetic bits, nonconsecutive IDs,
word/byte boundaries including bit 2047, internal sorting, collisions, zero-overlap
neighbors and nested bit counts with scores exactly at each configured threshold
and 0.001 on either side. Validator tests additionally use adjacent float64
values to prove that even sub-tolerance membership errors are rejected. These
checks do not establish compatibility for every possible float64 threshold or
all future datasets. Required real-query failures remain evidence and stop speed
claims rather than being waived as float32 noise.

## Paired sequential timing

Both RDKit and FPSim2 use prepared in-memory inputs. RDKit exhaustive scoring and
selection run **afresh**, using existing `measure_exact`; historical timing is
never a speedup denominator. FPSim2 uses inherited `top_k(..., threshold=0.0,
n_workers=1)`. One query executes at a time; no concurrent queries or background
memory polling occur.

For each k, one untimed warmup per backend precedes five measured passes. A local
`random.Random(42)` shuffles original query positions once per k/pass. Both
backends use that same permutation. Each backend completes that pass sequentially;
measured passes alternate RDKit-first/FPSim2-first, starting RDKit-first. All
permutations and backend orders are saved. Split generation and global RNG states
are untouched. Every query/k request is independent, not a larger-k truncation.

RDKit retains scoring, selection/ordering and enclosing total nanoseconds.
FPSim2 records search time around the inherited engine call (including bounds,
dispatch, native pruning/heap/scoring), separate output-normalization time, and
an enclosing total. Normalization orders the returned k native scores descending,
then original index positions ascending for deterministic presentation. It uses
no ground-truth lookup, score replacement or candidate filling. Primary FPSim2
latency is the **combined total**, equivalent to delivering ordered positions.
Native search alone is also shown, but is not the speedup denominator.

Input conversion/preparation, full validation, ground-truth lookups, evaluation,
display formatting, plotting and report writing are excluded. Outside-timer
validation/evaluation overhead currently has **no separate aggregate timing**.
All warmup and measured results are checked outside timers. Each fresh full RDKit
row must equal the historical authoritative row exactly. Every measured FPSim2
response receives the same strict top-k gate as preflight. A later failure blocks
successful comparative publication even after earlier passes succeeded.

Measured pass zero per backend/k supplies designated candidates. Every threshold
evaluation reuses that one set; no additional retrieval or pooling occurs.
Top-k-limited threshold recall uses authoritative scores; empty qualifying sets
have null recall with an explicit empty-set flag. Native threshold-search
compatibility is a separate preflight check. Later passes only provide timing
and diagnostics; changes in candidate membership are explicitly reported, with
valid boundary alternatives allowed.

All raw timings retain backend/k/query/pass/execution-order positions. The existing
summary API computes each query's median across five passes, then across-query
median, mean and **p95 of per-query median latency**, using `method='linear'`.
Units on disk are ns, display µs (preparation ms). Time saved is RDKit total
median minus FPSim2 combined total median. Speedup is RDKit total median divided
by FPSim2 combined total median. This is a repeated in-memory benchmark, not
cold-start latency, production concurrency, a confidence interval or a 100M
projection.

## New files, publication and reporting

All backend preparation and evidence files are new, under:

```text
experiments/results/fpsim2/<source>/<package>/<artifact_id>/run_<uuid>/
  index_fpsim2.h5          # verified saved real bits, not regenerated fingerprints
  synthetic_fpsim2.h5     # controlled nonzero integration fixture
  prepared_queries.npy   # precomputed native queries, no object data
  records.json           # ordered artifact identities and ground-truth linkage
  report.json            # metadata, native results/checks, timings, summaries,
                         # designated results, threshold evaluations and orders
  paired_latency.png/.svg
  speedup.png/.svg
  manifest.json          # completion/failure status and file checksums
```

Unique hidden staging directories, exclusive creation, reload validation and
checksum checks precede publication. The manifest is written last. Only runs
passing every required gate and measured check have `status='complete'`.
`failed_compatibility` runs retain actual responses, diagnostics and preparation
evidence; they do not claim a successful benchmark or comparative speed. Chart
writers omit comparative charts on failure. Unfinished staging is retained for
explicit inspection; no automatic recovery/cache/resumption is implemented.
Completed runs and historical ground truth are never overwritten. This is not
a power-loss durability guarantee.

Preparation is separately recorded: combined artifact loading/verification,
exact-run loading/verification, selected nonzero validation, synthetic preparation
and checks, FPSim2 conversion/database writing/loading/round-trip checks (combined),
RDKit reconstruction and real preflight checks. The FPSim2 combined preparation
stage also writes the prepared-query file. Report serialization/writing is
measured separately in the final manifest. Plotting, checksums and final
publication are outside that report-write measurement and retrieval latency.

Synchronous snapshots record UTC, perf-counter sampling start/duration, **total
and available system RAM in bytes**, and **process RSS in bytes**, at named
loading/preparation/post-run stages. Available system RAM is not process RSS;
neither is a true peak. Snapshot calls do not occur inside retrieval passes.
New exact runs receive the same metadata improvements. Historical runs remain
readable, are not edited, and notebook displays use “not recorded” for missing
legacy fields.

Notebooks own presentation; modules own execution. Notebook 04 presents provenance,
protocol, scoring/selection/total medians, total p95, preparation/output stages,
memory snapshots and latency curves. It can read an explicit old run or create
a new one with charts. Notebook 05 presents gate counts/discrepancies/native score
errors, paired latency/time-saved/speedup tables, grouped latency and 1×-referenced
speedup charts, preparation times and clearly labeled memory snapshots. Charts
are saved as reusable PNG/SVG files with checksums in new runs. All axes start at
zero; p95 labels describe percentiles, not uncertainty intervals.

## Verification

```bash
.venv/bin/python -m pytest tests/test_fpsim2.py tests/test_exact_retrieval.py tests/test_fingerprint_store.py tests/test_fingerprints.py tests/test_nmslib_adapter.py
.venv/bin/python -m pytest
.venv/bin/jupyter lab notebooks/05_fpsim2_benchmark.ipynb
```

Tests exercise the actual installed native backend, saved-bit round trips,
nonconsecutive IDs and sorting, exact ties/collisions, zero-score neighbors,
threshold and empty-set boundaries, malformed/short/oversized responses,
controlled clocks, paired order, designated retention, changing boundary
alternatives, provenance, immutable history and success/failure publication.
No test asserts machine-dependent latency or speed.

Remaining limits include float32 native arithmetic, a pilot 1K index and its query
distribution, uncontrolled scheduling/cache effects, full in-memory preparation,
no true peak-memory measurement, and no population/chemical-relevance claim.
HNSW and cross-backend break-even analysis remain deferred.

## Authorized local observations (2026-11-10)

The explicit historical reference was
`run_1494eeec3c0f489c9b67f7352dc573b7`, with artifact
`628c36671feeb38e1a97a34dd664bb54f43851374fcea4190a474479a12df427` and split
`f85552ea5b17e4829a09df82c79ad3bb2d762ee672c0f52de234dc7fc668e908`.
All input linkage/checksums and exactly 100 queries / 1,000 index rows were
verified before preparation. The completed comparison is
`experiments/results/fpsim2/chemicalspace_new/package_00/<artifact_id>/run_a46310ed08c74b9492dbf38e96a638a9/`.

Native integration and every required real check passed: 400 top-k and 400
threshold responses, with no missing/unexpected members or invalid/duplicate
identities. Threshold responses at 0.3, 0.5, 0.7 and 0.9 contained respectively
184, 1, 0 and 0 records across the 100 queries, exactly matching the complete
qualifying sets. Empty responses have no native-score error observation; notebook
tables label those errors not applicable. Maximum real native-score absolute
error was 1.4712538909389394e-8. No membership definition was relaxed.
All 4,000 measured responses passed outside-timer validation (2,000 per backend),
and no repeated candidate-set changes were observed.

Observed µs values below aggregate per-query five-pass medians. FPSim2 totals
include normalization, and the RDKit denominator was freshly timed in this paired
run, not taken from historical evidence.

| k | RDKit total median | FPSim2 total median | Time saved | Speedup | RDKit p95 | FPSim2 p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 137.953 | 71.722 | 66.231 | 1.923× | 143.563 | 73.042 |
| 25 | 140.938 | 80.727 | 60.212 | 1.746× | 146.031 | 85.104 |
| 50 | 144.588 | 92.777 | 51.811 | 1.558× | 156.425 | 98.282 |
| 100 | 150.618 | 115.417 | 35.201 | 1.305× | 157.428 | 121.814 |

Both p95 columns mean **p95 of per-query median latency**, linear method.
FPSim2 search medians ranged 60.271–81.622 µs; normalization medians ranged
11.281–33.921 µs. Combined FPSim2 conversion/database/loading/verification took
392.929 ms; RDKit reconstruction 351.451 ms; exact-run loading/verification
102.669 ms; real preflight checks 153.644 ms. None is retrieval latency.
System total RAM snapshots recorded 16,146,571,264 bytes; available RAM and
process RSS were separately sampled at all named stages. No peak is inferred.

Notebook 04 produced new exact report runs `run_c3116abc111c441a88f60fde50c48869`
and `run_0b54e3da1454423b88279253f6e32179` under the corresponding exact-retrieval
artifact path. Visual inspection found clipped p95 markers in the first plot;
its completed files were preserved and the latter run saved the corrected chart
with explicit headroom. The paired latency and 1×-referenced speedup charts and
the corrected exact chart were visually inspected. Notebook 05 can explicitly
inspect the completed paired run without writes, or select `None` for a new run.
The historical reference, source artifact/split, and pre-existing notebook 03
changes remained unchanged. No raw data, molecular fingerprints or splits were
regenerated.

Final verification: 132 focused tests and 209 full-suite tests passed, including
native integration and controlled-clock tests. All six notebooks passed schema
and code-cell syntax checks; notebooks 04 and 05 contain compact executed outputs.
The completed comparison was reloaded and its saved responses rechecked against
the authoritative historical matrix after the final identity-uniqueness test.
