# Nonzero fingerprint input policy

The researcher-approved policy `nonzero_fingerprints_v1` requires at least one
enabled bit in every selected query and index fingerprint before either backend
uses it. An all-zero packed fingerprint has all 256 bytes equal to zero: it
encodes no enabled environments. This is unexpected for ordinary nonempty
molecules under the approved Morgan connectivity settings, which generate atom
environments and set their hashed bits. The guard checks stored bits directly;
it does not infer correctness from molecular assumptions or reprocess molecules.

In the tested environment (NMSLIB 2.1.2, RDKit 2026.09.1), two all-zero vectors
produce NMSLIB binary Jaccard distance NaN, while RDKit returns Tanimoto 0,
giving `1 - T = 1`. Zero versus nonzero produces distance 1 in both libraries.
These are version-specific observations, not a universal mathematical convention.
The benchmark explicitly excludes zero fingerprints rather than imposing an
empty-union convention or modifying either library's calculations.

`fingerprints.validate_nonzero_fingerprints(queries, index, records)` checks
packed uint8 `(n,256)` arrays and their corresponding ordered mappings, then
scans every supplied row using its bytes. It is independent of NMSLIB, datasets
and artifact sizes. On success it returns the policy ID, checked row counts and
zero counts. Mappings must have one valid original row identity per supplied
row. Original row indices may be nonconsecutive for a future selected subset;
callers must supply the corresponding mapping from verified artifact loading.
The full-artifact runner additionally requires the complete original row order
and manifest counts, so an index prefix cannot masquerade as a full scan.

Notebook 03 loads and verifies full arrays through `fingerprint_store`, then
calls the runner. The runner applies the guard before real-data RDKit
reconstruction or NMSLIB registration. A 100K parent is scanned across its
entire saved index, even though distance comparisons use only ten index rows.
Checking only the 20-fingerprint compatibility fixture was insufficient: zero
rows elsewhere would remain undetected. Full-artifact validation establishes
absence of zero rows; sampled distance checks establish compatibility on the
specified fixture. Neither constitutes exhaustive retrieval or establishes recall.

If zeros are found, `NonzeroFingerprintError` carries the validation summary
and **all** affected records: role, original artifact row index, source ID and
line number. The runner saves a failure report, marks subsequent checks not run,
and raises a concise error containing the report path and identity examples.
No records are removed, replaced, regenerated or renumbered. Silently removing
records would change approved selection and identity alignment. No distances
are patched; native results remain evidence rather than substituted values.

Edit the single artifact-path selection cell in notebook 03 to use another
pool's existing artifact. Split/source/package/identity/output linkage comes
from artifact metadata verified by the existing storage module. Missing input
stops without fallback or generation. Packages remain separate.

Report schema 2 separates full nonzero validation, required nonzero synthetic
and real checks, and unsupported zero-input diagnostics. All required checks
must run and pass. Unsupported diagnostics retain zero-zero NaN as JSON null
plus a `NaN` classification; zero/nonzero diagnostics are also unsupported.
These probes deliberately bypass the production guard only in diagnostic code.
They neither count as successful equivalence checks nor enter the supported
maximum-error calculation. Tolerances remain absolute 1e-6, relative zero.
The guard has its own timing. Unique new reports preserve the original failure
report unchanged.

Future exact-search and HNSW entry points **must call the same guard on all rows
selected for that run before backend preparation or graph construction**.
Those entry points remain unimplemented. See [backend verification](backend_verification.md)
for APIs, report locations and execution commands.
