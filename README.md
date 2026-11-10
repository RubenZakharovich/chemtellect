# Chemtellect

Experimental molecular retrieval with binary Morgan fingerprints.

## Molecular validation

Run the independent RDKit diagnostic stage with:

```bash
uv run python scripts/validate_molecules.py \
  data/raw/otava_chemical/package_00_pool.jsonl \
  --report experiments/results/validation/otava_chemical_package_00.json \
  --min-unique 1100
uv run pytest tests/test_molecular_validation.py
```

The reusable API is `chemtellect.molecular_validation.validate_molecules(Path(...),
min_unique=1100)`. Reporting is separate: `format_summary(report)` and
`write_report(report, Path(...))`.

Each physical line counts as an input record, including blank lines. Malformed
JSON/UTF-8, non-object records, and missing/blank/non-string or repeated source
IDs are **input-integrity errors**. Each such line is rejected once before
chemical validation. IDs are compared exactly; the first occurrence reserves an
ID even if its SMILES is invalid. Missing/blank/non-string SMILES, RDKit parsing
or sanitization failures, and empty molecular graphs are **chemical findings**.
All findings retain line numbers and available valid string IDs.

Total input records = parsed molecular records + invalid molecular records +
integrity-rejected records. Parsed records = unique canonical structures +
surplus duplicate records. Duplicate-group record counts include every member;
surplus counts exclude one member per group. The JSON report retains every
duplicate source ID and every issue; terminal examples are bounded.

Structures are equal only when their canonical isomeric SMILES match, using
standard RDKit sanitization and explicit `canonical=True, isomericSmiles=True`.
There is no tautomer normalization, salt stripping, charge neutralization, or
invention of missing stereochemistry. Results depend on the recorded RDKit
version. Raw data and source fingerprint fields are never changed or used.
The checksum is computed from the same streamed bytes that are analyzed. Reports
contain no timestamps and are deterministic for an unchanged input path, file,
RDKit version, and acceptance setting. Structure groups and issues are retained
in memory; the input file itself is streamed.

Exit codes: **0** means no integrity errors and the optional minimum passed;
**1** means integrity errors or a failed minimum; **2** means usage or file/report
I/O errors. Chemical findings and duplicates alone do not fail the process.
The 1,100-structure threshold is specific to the initial benchmark; only the
configurable `--min-unique` controls acceptance. Sampling has no fixed minimum.
The proposed 1,000-index / 100-query split belongs to future experiment
configuration. Reports are still written when integrity or acceptance checks fail.

## Pilot data acquisition

`chemtellect.mongo_sampling` uses existing authorized SSH configuration and the
legacy `mongo` executable; it installs nothing remotely. Preview locally first:

```bash
uv run python scripts/sample_mongo.py --ssh-target chemtellect-server \
  --database otava_chemical --collection compounds --package package_01 --dry-run
```

To acquire, execute the same command without `--dry-run`. The SSH alias is a
placeholder for your authorized configuration. No credentials or SSH target are
stored in manifests. Parameters include `--size` (default 1200, any positive
integer), `--server-timeout-ms` (30000), `--client-timeout` (120 seconds), and
`--raw-root` (`data/raw`). Samples above 10000 records require
`--allow-large-sample`. `--max-transfer-bytes` defaults to 100 MiB for combined
stdout/stderr; this is monitored periodically, so a brief overshoot is possible.
Increasing either limit must be deliberate. Dry run validates and prints the
command/destinations without creating files or making network requests.

The read-only aggregation starts with `$sample`, then projects only `_id`,
`code`, `smiles`, `inchikey`, and `formula`. Missing or invalid SMILES are retained.
New exports use the versioned `bson-id-v1` format recorded in each manifest:
ObjectId becomes `objectid:<24 lowercase hex digits>`; a BSON string becomes
`string:<JSON string literal>` (including quotes and JSON escapes). Other BSON
ID types fail extraction explicitly. ObjectId and string IDs never collapse.
Compare exact IDs only within the same database, collection, and encoding.
The earlier sampler used shell-dependent `tojson(_id)` text; those IDs are also
not directly comparable to this format.

The original `package_00_pool.jsonl` used `String(_id)`. That can lose BSON type
information: an ObjectId and a string containing its hex text can coincide.
Do not infer types from hex-looking text or compare it directly with new IDs.
Cross-format overlap assessment requires independently verified original types;
no automatic conversion or cross-package deduplication is performed.
The original file and its identifiers remain unchanged.

Commands target legacy `mongo`, not `mongosh`: ObjectId `valueOf()` returns hex
in the legacy shell ([MongoDB compatibility documentation](https://www.mongodb.com/docs/mongodb-shell/reference/compatibility/)).
`--norc` suppresses the user's `.mongorc.js`, avoiding user startup code and output.
It also bypasses authentication or connection setup kept there; use authorized
SSH/server configuration instead. A system-wide shell startup file may still run;
read-only credentials and trusted remote configuration remain necessary.
Arguments are validated locally and the remote command is shell-quoted.
No writes or automatic retries are issued by the sampling script.

Manual non-mutating connectivity check (not run during development):

```bash
ssh -T -o BatchMode=yes -- chemtellect-server \
  'mongo --quiet --norc --eval "var r = db.getSiblingDB(\"admin\").runCommand({ping: 1}); if (r.ok !== 1) quit(1); print(\"ping ok\");"'
```

This uses the remote shell's default local MongoDB endpoint. Replace the SSH
alias with your authorized host alias; confirm the intended remote configuration
before any live extraction.

Output pairs follow this pattern (both are excluded from Git):

```text
data/raw/otava_chemical/package_01_pool.jsonl
data/raw/otava_chemical/package_01_pool.meta.json
data/raw/chemicalspace_new/package_00_pool.jsonl
data/raw/chemicalspace_new/package_00_pool.meta.json
```

Schema version 1 manifests contain database, collection, package, method and
limitations, requested/actual counts, UTC extraction start/completion times,
SHA-256, output filename, retrieved fields, identity encoding, integrity status,
missing-SMILES count, server/client timeouts and byte limit. Collection size and
snapshot identifier are `null` (unknown). No metadata is added to individual
records. The pre-existing `package_00_pool.jsonl` is left unchanged and receives
no manufactured manifest.

The method is `mongodb_sample_first_stage`: MongoDB normally uses its optimized
pseudo-random cursor when `$sample` is first, the collection exceeds 100 records,
and sample size is below 5% of the collection. Planner behavior is not verified
by this utility. Selection bias is possible; these pilot samples are **not
guaranteed uniform or statistically representative**, and do not justify
unbiased population estimates. No collection-scan fallback is implemented.
Independent packages may overlap and are not a frozen database snapshot.
Representativeness, query distributions, composition and index-size effects
require separate assessment.

Extraction streams to a hidden `.partial` file. Process success and exact count,
JSON/UTF-8, object and unique nonblank string ID checks must pass before
publication. `chemtellect.data_integrity.validate_jsonl(path, expected=...)`
returns structured results and hashes the same bytes it checks, without RDKit.
SMILES absence is a quality observation. The backward-compatible raw CLI remains
strict unless explicitly permitted:

```bash
uv run python scripts/validate_raw_sample.py \
  data/raw/otava_chemical/package_01_pool.jsonl --expected 1200 --allow-missing-smiles
```

Failures never trigger automatic retries: `$sample` could return different data.
Staging paths are reported for inspection and explicit cleanup. On launch,
nonzero exit, timeout, interruption, or transfer-limit failure, up to 64 KiB of
stderr is retained in a uniquely named `.stderr.log` beside staging data, mode
0600. Its location is reported without printing server diagnostics. It may
contain credentials or other sensitive details: inspect locally, redact before
sharing, and explicitly remove when no longer needed. Successful transfers
remove temporary stderr storage. Legacy mongo may put errors on stdout; invalid
or incomplete output fails integrity checks and remains only in private staging.
The combined transfer limit is polled, so transient disk use can exceed it. Publication uses
no-overwrite hard links on the local filesystem; the manifest is published last
and is the completion marker. Treat a final JSONL without its manifest as
incomplete publication. If metadata publication fails, verify the retained
staging manifest and JSONL checksum/count, then publish that manifest without
overwriting any existing file; no new sampling is needed. If metadata generation
itself failed, retain the raw staging file for diagnosis and choose a new package
for a deliberate fresh acquisition. Do not use partial files as datasets.
This is not a transactional two-file commit or a power-loss durability guarantee.

Extraction integrity establishes a complete export; RDKit validation separately
assesses molecular quality and canonical duplicates. Neither implies population
representativeness. Both validators stream input, but retain identifiers/issues
(and RDKit structure groups) in memory. Very large acquisitions will require
streaming molecular preprocessing and further resource planning.

## Fingerprint preparation checkpoint

After `notebooks/01_dataset_overview.ipynb` saves an audited split,
`notebooks/02_fingerprint_generation.ipynb` generates or verifies one artifact.
It defaults to the existing Chemicalspace package_00 split and never creates a
missing split. Select another approved saved pool by explicitly changing the
split path in notebook 02. The overview now describes all eligible structures
with `descriptor_limit: null`.

`chemtellect.fingerprints` validates scientific configuration and generates
sequential batches. `chemtellect.fingerprint_store` validates splits, packs,
persists, verifies, reconstructs and loads registered index prefixes. Morgan
connectivity fingerprints use radius 2, 2048 bits and chirality, bond types and
ring membership, without molecular standardization. Storage is uint8 `(n, 256)`
NumPy arrays with explicit little bit ordering. See [methodology](docs/fingerprints.md)
for all scientific choices, limitations, APIs and provenance requirements.

```bash
uv run pytest
uv run jupyter lab notebooks/02_fingerprint_generation.ipynb
```

Run notebook cells in order using the project's Python kernel. Artifacts go under
`data/processed/<source>/<package>/fingerprints/<artifact_id>/`; unique run reports
under `experiments/results/fingerprints/<source>/<package>/<artifact_id>/`.
Both are excluded from Git. Rerunning verifies all persisted data and linkage,
without generating molecules again. Corruption fails rather than regenerating.
Interrupted staging and stale locks require explicit inspection and preservation
or removal; they are never automatically resumed or published. See recovery
instructions in the methodology. Raw pools, audits and splits remain unchanged.

This checkpoint implements preparation only. Exhaustive retrieval, exact ground
truth, recall measurements and NMSLIB/HNSW integration remain future work.

## Backend compatibility verification (Step 3)

`chemtellect.nmslib_adapter` converts verified packed fingerprints in memory to
NMSLIB `bit_jaccard` input and compares direct pairwise distances with RDKit
`1 - TanimotoSimilarity`. Notebook 03 selects the existing artifact and checks
synthetic cases plus query/index rows 0–9 (210 real-fixture pairs). No fingerprints
are regenerated and no HNSW graph or nearest-neighbor search is performed.

```bash
uv run pytest tests/test_nmslib_adapter.py
uv run jupyter lab notebooks/03_backend_verification.ipynb
```

Edit the single artifact-path cell in notebook 03 to select another existing
pool. Split linkage and output location derive from verified metadata. The
backend-independent `nonzero_fingerprints_v1` guard scans **all saved query and
index rows** before real backend preparation, then the small fixture checks
compatibility. Future exact and HNSW runs must reuse this guard on all selected
inputs. See the [nonzero policy](docs/nonzero_fingerprint_policy.md) and
[backend methodology](docs/backend_verification.md).

Reports use unique filenames under
`experiments/results/backend_verification/<source>/<package>/<artifact_id>/`.
Required representation and nonzero distance checks determine success. Native
zero-input probes are unsupported diagnostics: in the tested NMSLIB 2.1.2 /
RDKit 2026.09.1 environment, zero/zero gives NaN versus Tanimoto 0 (distance 1).
No result is patched and no records are removed. The original failure report
is preserved. A failed full-input guard saves a report and stops before real
backend preparation. Compatibility checks do not establish recall or performance.
