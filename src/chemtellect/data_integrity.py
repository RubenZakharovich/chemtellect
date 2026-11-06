"""Streaming extraction checks, independent of molecular interpretation."""
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path


def decode_record(raw: bytes, identifiers: set[str]) -> tuple[dict | None, str | None, str | None]:
    """Decode one physical line and reserve its exact string source identity."""
    try:
        record = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        return None, None, f"Invalid JSON/UTF-8: {error}"
    if not isinstance(record, dict):
        return None, None, "Record is not a JSON object"
    source_id = record.get("source_id")
    if not isinstance(source_id, str) or not source_id.strip():
        return record, None, "Missing, blank, or non-string source_id"
    if source_id in identifiers:
        return record, source_id, "Duplicate source_id"
    identifiers.add(source_id)
    return record, source_id, None


@dataclass
class IntegrityReport:
    total_records: int = 0
    unique_source_ids: int = 0
    missing_smiles: int = 0
    sha256: str = ""
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return not self.errors


def validate_jsonl(path: Path, expected: int | None = None) -> IntegrityReport:
    """Count physical lines and hash exactly the bytes inspected; never write input.

    Missing/non-string/blank SMILES are observations, not integrity failures.
    A missing final newline is accepted; truncated JSON is rejected by decoding.
    Identifier storage grows with sample size, but records are never accumulated.
    """
    if expected is not None and (type(expected) is not int or expected <= 0):
        raise ValueError("expected must be a positive integer")
    report = IntegrityReport()
    identifiers: set[str] = set()
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for number, raw in enumerate(stream, 1):
            checksum.update(raw)
            report.total_records += 1
            record, _, error = decode_record(raw, identifiers)
            if error:
                report.errors.append(f"Line {number}: {error}")
            if record is not None:
                smiles = record.get("smiles")
                if not isinstance(smiles, str) or not smiles.strip():
                    report.missing_smiles += 1
    report.unique_source_ids = len(identifiers)
    report.sha256 = checksum.hexdigest()
    if expected is not None and report.total_records != expected:
        report.errors.append(f"Expected {expected} records, found {report.total_records}")
    return report
