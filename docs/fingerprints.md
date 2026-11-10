# Saved-split Morgan fingerprint methodology

The researcher-approved configuration is
`experiments/configs/morgan_r2_2048_chiral_v1.json`. The library validates this
version exactly; changing scientific policy requires explicit review and a new
supported configuration. Identity includes the complete configuration.

Standard Morgan connectivity invariants encode local connectivity, rather than
pharmacophore features. Radius 2 and 2048 bits fix the representation. Chirality,
bond types and ring membership are enabled to retain these structural features.
Count simulation is disabled, count bounds are null, only-nonzero invariants and
redundant environments are disabled. No custom atom or bond invariants are used.
All atoms participate, none are ignored. `GetFingerprint` returns a fixed-length
`ExplicitBitVect`; count and sparse variants are excluded. Effective options and
the generator information string are checked and the string is recorded.

Input is exclusively the saved split's canonical SMILES. Sanitization is enabled.
The existing SMILES parser handles hydrogens; there is no AddHs or RemoveHs stage.
There is no salt stripping, fragment selection, tautomer normalization, charge
neutralization or protonation change. Disconnected components are retained.
Specified stereochemistry is preserved and unspecified stereochemistry remains
unspecified. These choices retain the molecular preparation used by the split.
Source `rfp` is never used. Any parsing/generation failure stops with role, source
ID and original line number; nothing is skipped, repaired, replaced or resampled.

Chirality-aware fingerprints can distinguish specified and unspecified
stereochemistry. They are lossy molecular representations: different structures
can have identical bits. Such records remain separate because fingerprint
collisions do not establish molecular identity. Future recall will quantify
agreement with exact Tanimoto under this representation, not independently
established chemical relevance. Acquisition is not proven population-representative.

## Reproducibility and storage

Saved splits remain authoritative and unchanged. Validation recomputes the
producer's sorted, compact UTF-8 JSON SHA-256 digest and checks schema, counts,
unique IDs/structures, disjoint roles, approved seed-42 SHA-256 ordering policy
and first-raw-occurrence representative policy. Generation requires exactly the
split's RDKit version. Cross-version bit reproducibility is not promised.
The raw hash is inherited provenance; this checkpoint does not rehash raw pools.

Generate only queries and the entire saved index order, never reserves. A 100K
parent generates one 100-query set and one 100,000-row index; registered 1K/10K/100K
prefixes share that ordering. Packages remain separate without cross-package
caching. `records.json` maps each zero-based role row to source ID and original
line number; the split retains SMILES authority.

Uncompressed NumPy arrays have uint8 dtype and shape `(n, 256)`. RDKit indices
0 through 2047 are packed with `bitorder="little"`: bit 0 is the least-significant
bit of byte 0. Unpacking uses the same convention, with exact bit preservation.
Arrays load with `allow_pickle=False`. Reconstruction operates row by row;
normal generation holds only batch molecule objects and full packed arrays.

Artifact directories use full SHA-256 identities over a versioned object with
split ID, complete configuration, current RDKit and storage contract. JSON uses
sorted keys, compact separators, UTF-8 and rejects non-finite numbers. Time and
execution measurements are excluded. Manifests record split linkage/checksum,
identity, generator information, environment, Git revision/dirty state, counts,
storage, execution settings, UTC creation time and data-file checksums/sizes.
Checksums detect changes; they do not independently prove scientific correctness.

## APIs and execution

`fingerprints.load_configuration(path)`, `build_generator(config)` and
`generate_batches(records, role, generator, batch_size)` provide the scientific
path. `fingerprint_store.validate_split(split)`, `pack_fingerprints(vectors)`,
`unpack_fingerprints(array)`, `prepare_artifact(root, split_path, config, ...)`,
`verify_artifact(directory, split_path, config, root)` and
`load_artifact(directory, split_path, config, root, index_size)` are notebook-independent.
The loader returns queries and an explicitly registered packed index prefix;
unregistered sizes fail. Reconstruction is an explicit separate operation.

Execution uses one thread and sequential batches of 1–1000 molecules. Parsing
and generation are separate loops. Optional progress callbacks receive role,
completed count and total after each batch. Stage measurements use
`perf_counter_ns`: verification, generator initialization, role-specific parsing,
generation and packing, writing, reload verification and total preparation.
Progress display is outside stage timers. Total includes unallocated overhead,
such as callbacks, allocations, locking, provenance and publication; sums need
not equal total. Report serialization occurs after the measured preparation.
These are preparation times, not retrieval latency. Unique run reports retain
UTC start/end, operation, environment, execution counts and timings. On reuse,
generation stages are null; original manifest provenance remains unchanged.

## Publication and recovery

Artifacts live in `data/processed/<source>/<package>/fingerprints/<artifact_id>/`;
run reports live in `experiments/results/fingerprints/<source>/<package>/<artifact_id>/`.
Both are Git-ignored. A per-artifact exclusive lock coordinates cooperating writers.
A unique sibling staging directory receives arrays and mapping, verifies reload
and bit preservation, then receives its completed manifest last. It is verified
before directory publication. Existing artifacts are verified, never overwritten.
This is not a power-loss durability guarantee.

On failure, diagnostics identify retained staging. Normal loading selects only
the explicit final artifact, never discovers or resumes staging. After an
interruption, inspect retained files and lock ownership. A hard termination may
leave a stale lock; it is never automatically broken. Conflicting or damaged
artifacts require explicit researcher review and preservation/removal before
regeneration. Do not manually publish unverified staging remnants.

## Next checkpoint boundary

Exact retrieval will consume reconstructed RDKit fingerprints. NMSLIB will
receive a lossless encoding of those same bits; its later adapter will map index
row identifiers to source records. It must not parse SMILES or regenerate
fingerprints. Later controlled tests will compare binary Jaccard distance with
one minus Tanimoto using a tolerance selected at that checkpoint. No serialization
choice or tolerance is selected here. Exhaustive comparisons, similarity matrices,
ground truth, ranking/recall metrics, NMSLIB adapters, HNSW construction/search and
parameter sweeps remain unimplemented. No benchmark results are claimed.
