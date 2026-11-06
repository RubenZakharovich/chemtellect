"""Acquire a pilot MongoDB sample without modifying the remote database."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from chemtellect.mongo_sampling import SamplingConfig, sample_mongo


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ssh-target", "database", "collection", "package"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--size", type=int, default=1200)
    parser.add_argument("--server-timeout-ms", type=int, default=30_000)
    parser.add_argument("--client-timeout", type=float, default=120)
    parser.add_argument("--max-transfer-bytes", type=int, default=100 * 1024 * 1024)
    parser.add_argument("--allow-large-sample", action="store_true")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--dry-run", action="store_true")
    args = vars(parser.parse_args())
    dry_run = args.pop("dry_run")
    try:
        result = sample_mongo(SamplingConfig(**args), dry_run=dry_run)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Sampling failed: {error}", file=sys.stderr)
        for note in getattr(error, "__notes__", []):
            print(note, file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
