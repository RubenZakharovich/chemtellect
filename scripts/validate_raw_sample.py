
import argparse
import hashlib
import json
from pathlib import Path

# script usage from the therminal             -> pass the jsonl argument
# uv run python scripts/validate_raw_sample.py data/raw/otava_chemical/package_00_pool.jsonl --expected 1200

def validate(path: Path, expected: int | None) -> bool:
    total = 0
    identifiers = set()
    missing_smiles = 0
    errors = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            total += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"Line {line_number}: Invalid JSON")
                continue

            if not isinstance(record, dict):
                errors.append(f"Line {line_number}: Not a JSON object")
                continue

            source_id = record.get("source_id")

            if not isinstance(source_id, str) or not source_id.strip():
                errors.append(f"Line {line_number}: Missing source ID")
            elif source_id in identifiers:
                errors.append(f"Line {line_number}: Duplicate source ID")
            else:
                identifiers.add(source_id)

            smiles = record.get("smiles")

            if not isinstance(smiles, str) or not smiles.strip():
                missing_smiles += 1

    if expected is not None and total != expected:
        errors.append(f"Expected {expected} records, found {total}")

    if missing_smiles:
        errors.append(f"{missing_smiles} missing/blank SMILES")

    with path.open("rb") as file:
        checksum = hashlib.file_digest(file, "sha256").hexdigest()

    print(f"File: {path}")
    print(f"Total records: {total}")
    print(f"Unique source IDs: {len(identifiers)}")
    print(f"Missing/blank SMILES: {missing_smiles}")
    print(f"SHA-256: {checksum}")

    if errors:
        print("\nValidation FAILED:")
        for error in errors[:10]:
            print(f" - {error}")
        if len(errors) > 10:
            print(f" ... and {len(errors) - 10} more errors")
        return False

    print("\nBasic extraction checks PASSED.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate an extracted molecular JSONL sample."
    )

    parser.add_argument("path", type=Path)
    parser.add_argument("--expected", type=int, default=None)

    args = parser.parse_args()

    if not validate(args.path, args.expected):
        raise SystemExit(1)

