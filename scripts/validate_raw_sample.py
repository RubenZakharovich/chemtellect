"""Backward-compatible CLI for lightweight extraction validation."""
import argparse
from pathlib import Path
from chemtellect.data_integrity import validate_jsonl


def validate(
    path: Path, expected: int | None = None, *, allow_missing_smiles: bool = False
) -> bool:
    report = validate_jsonl(path, expected)
    errors = list(report.errors)
    if report.missing_smiles and not allow_missing_smiles:
        errors.append(f"{report.missing_smiles} missing/blank SMILES")
    print(f"File: {path}\nTotal records: {report.total_records}\n"
          f"Unique source IDs: {report.unique_source_ids}\n"
          f"Missing/blank SMILES: {report.missing_smiles}\nSHA-256: {report.sha256}")
    if errors:
        print("\nValidation FAILED:")
        for error in errors[:10]:
            print(f" - {error}")
        if len(errors) > 10:
            print(f" ... and {len(errors) - 10} more errors")
    else:
        print("\nBasic extraction checks PASSED.")
    return not errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--expected", type=int)
    parser.add_argument("--allow-missing-smiles", action="store_true")
    args = parser.parse_args()
    try:
        passed = validate(
            args.path, args.expected, allow_missing_smiles=args.allow_missing_smiles
        )
        return 0 if passed else 1
    except (OSError, ValueError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
