"""Diagnostic RDKit validation without molecular standardization.

Every physical input line is counted. Integrity rejections (bad JSON/object or
missing/repeated IDs) are separate from chemical findings (bad/missing SMILES).
Only records passing integrity checks contribute to molecular counts.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from rdkit import Chem, rdBase


@dataclass(frozen=True)
class RecordIssue:
    line_number: int
    source_id: str | None
    reason: str


@dataclass
class ValidationReport:
    source_file: str
    sha256: str
    rdkit_version: str
    total_input_records: int
    successfully_parsed_records: int
    invalid_records: list[RecordIssue]
    integrity_errors: list[RecordIssue]
    structures: dict[str, list[str]]
    min_unique: int | None

    @property
    def duplicate_groups(self) -> dict[str, list[str]]:
        return {smiles: ids for smiles, ids in self.structures.items() if len(ids) > 1}

    @property
    def acceptance_passed(self) -> bool:
        return self.min_unique is None or len(self.structures) >= self.min_unique

    @property
    def success(self) -> bool:
        return not self.integrity_errors and self.acceptance_passed

    def to_dict(self) -> dict[str, object]:
        """Return deterministic JSON-compatible diagnostics and count definitions."""
        duplicates = self.duplicate_groups
        duplicate_records = sum(map(len, duplicates.values()))
        return {
            "source_file": self.source_file,
            "sha256": self.sha256,
            "rdkit_version": self.rdkit_version,
            "canonicalization": {"canonical": True, "isomericSmiles": True,
                                 "sanitize": True, "standardization": "none"},
            "total_input_records": self.total_input_records,
            "successfully_parsed_records": self.successfully_parsed_records,
            "invalid_molecular_records": len(self.invalid_records),
            "integrity_rejected_records": len(self.integrity_errors),
            "unique_canonical_structures": len(self.structures),
            "duplicate_structure_groups": len(duplicates),
            "records_in_duplicate_groups": duplicate_records,
            "surplus_duplicate_records": duplicate_records - len(duplicates),
            "at_least_1100_unique_valid_structures": len(self.structures) >= 1100,
            "min_unique": self.min_unique,
            "acceptance_passed": self.acceptance_passed,
            "success": self.success,
            "invalid_records": [asdict(issue) for issue in self.invalid_records],
            "integrity_errors": [asdict(issue) for issue in self.integrity_errors],
            "duplicate_groups": dict(sorted(duplicates.items())),
        }


def canonicalize_smiles(smiles: object) -> tuple[str | None, str | None]:
    """Parse with standard sanitization; return canonical isomeric SMILES or reason."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None, "Missing, empty, or non-string SMILES"
    try:
        molecule = Chem.MolFromSmiles(smiles, sanitize=True)
        if molecule is None:
            return None, "RDKit SMILES parsing or sanitization failed"
        if molecule.GetNumAtoms() == 0:
            return None, "Empty molecular graph"
        return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True), None
    except (ValueError, RuntimeError) as error:
        return None, f"RDKit parsing or canonicalization failed: {error}"


def validate_molecules(path: Path, min_unique: int | None = None) -> ValidationReport:
    """Stream JSONL, retaining findings; propagate file errors to the caller.

    IDs must be nonblank strings and are compared exactly. Repeated IDs are
    rejected even when their molecular structures differ. Total lines equal
    parsed + chemically invalid + integrity rejected records.
    """
    if min_unique is not None and min_unique < 0:
        raise ValueError("min_unique must be nonnegative")
    report = ValidationReport(str(path), "", rdBase.rdkitVersion, 0, 0, [], [], {}, min_unique)
    identifiers: set[str] = set()
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for line_number, raw_line in enumerate(source, 1):
            checksum.update(raw_line)
            report.total_input_records += 1
            try:
                record = json.loads(raw_line.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                report.integrity_errors.append(RecordIssue(line_number, None, f"Invalid JSON/UTF-8: {error}"))
                continue
            if not isinstance(record, dict):
                report.integrity_errors.append(RecordIssue(line_number, None, "Record is not a JSON object"))
                continue
            source_id = record.get("source_id")
            if not isinstance(source_id, str) or not source_id.strip():
                report.integrity_errors.append(RecordIssue(line_number, None, "Missing, blank, or non-string source_id"))
                continue
            if source_id in identifiers:
                report.integrity_errors.append(RecordIssue(line_number, source_id, "Duplicate source_id"))
                continue
            identifiers.add(source_id)
            canonical, reason = canonicalize_smiles(record.get("smiles"))
            if canonical is None:
                report.invalid_records.append(RecordIssue(line_number, source_id, reason or "Invalid molecule"))
            else:
                report.successfully_parsed_records += 1
                report.structures.setdefault(canonical, []).append(source_id)
    report.sha256 = checksum.hexdigest()
    return report


def write_report(report: ValidationReport, path: Path) -> None:
    """Write a JSON report, refusing to overwrite the raw input (including aliases)."""
    source = Path(report.source_file)
    if path.resolve() == source.resolve() or (path.exists() and path.samefile(source)):
        raise ValueError("Report path must not overwrite the source file")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def format_summary(report: ValidationReport, max_examples: int = 5) -> str:
    """Format bounded examples alongside complete aggregate statistics."""
    data = report.to_dict()
    lines = [f"File: {report.source_file}", f"SHA-256: {report.sha256}",
             f"RDKit: {report.rdkit_version} (sanitize=True; canonical=True; isomericSmiles=True)"]
    for key in ("total_input_records", "successfully_parsed_records", "invalid_molecular_records",
                "integrity_rejected_records", "unique_canonical_structures", "duplicate_structure_groups",
                "records_in_duplicate_groups", "surplus_duplicate_records",
                "at_least_1100_unique_valid_structures"):
        lines.append(f"{key.replace('_', ' ').capitalize()}: {data[key]}")
    for label, issues in (("Integrity", report.integrity_errors), ("Chemical", report.invalid_records)):
        for issue in issues[:max_examples]:
            lines.append(f"{label}, line {issue.line_number}, ID {issue.source_id!r}: {issue.reason}")
    for smiles, ids in list(report.duplicate_groups.items())[:max_examples]:
        lines.append(f"Duplicate {smiles}: {ids[:max_examples]} ({len(ids)} records)")
    lines.append(f"Minimum unique: {report.min_unique}; acceptance passed: {report.acceptance_passed}")
    lines.append(f"Validation {'PASSED' if report.success else 'FAILED'}")
    return "\n".join(lines)
