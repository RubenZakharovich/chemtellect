"""Command-line entry point for diagnostic molecular validation."""

import argparse
from pathlib import Path
import sys

from chemtellect.molecular_validation import format_summary, validate_molecules, write_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--min-unique", type=int)
    args = parser.parse_args()
    if args.min_unique is not None and args.min_unique < 0:
        parser.error("--min-unique must be nonnegative")
    try:
        report = validate_molecules(args.path, args.min_unique)
        print(format_summary(report))
        if args.report is not None:
            write_report(report, args.report)
    except (OSError, ValueError) as error:
        print(f"Validation error: {error}", file=sys.stderr)
        return 2
    return 0 if report.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
