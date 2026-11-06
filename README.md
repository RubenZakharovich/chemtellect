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
Every report also states whether 1,100 unique valid structures are available,
independently of the optional `--min-unique` acceptance criterion. Reports are
still written when integrity or acceptance checks fail.
