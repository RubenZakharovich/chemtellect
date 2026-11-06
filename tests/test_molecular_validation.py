"""Synthetic molecular and input-integrity checks using real RDKit behavior."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from chemtellect.molecular_validation import (
    canonicalize_smiles, format_summary, validate_molecules, write_report,
)


def make_sample(tmp_path, records):
    path = tmp_path / "sample.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


def test_valid_duplicates_counts_and_source_preserved(tmp_path):
    path = make_sample(tmp_path, [
        {"source_id": str(i), "smiles": smiles}
        for i, smiles in enumerate(["CCO", "OCC", "C(C)O", "C", "N", "N"])
    ])
    original = path.read_bytes()
    report = validate_molecules(path)
    data = report.to_dict()
    assert data["total_input_records"] == data["successfully_parsed_records"] == 6
    assert data["unique_canonical_structures"] == 3
    assert data["duplicate_structure_groups"] == 2
    assert data["records_in_duplicate_groups"] == 5
    assert data["surplus_duplicate_records"] == 3
    assert report.duplicate_groups["CCO"] == ["0", "1", "2"]
    assert report.success
    assert report.sha256 == hashlib.sha256(original).hexdigest()
    output = tmp_path / "reports" / "report.json"
    write_report(report, output)
    first = output.read_bytes()
    write_report(validate_molecules(path), output)
    assert output.read_bytes() == first
    assert json.loads(first)["duplicate_groups"]["N"] == ["4", "5"]
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="overwrite"):
        write_report(report, path)
    alias = tmp_path / "alias.jsonl"
    alias.symlink_to(path)
    with pytest.raises(ValueError, match="overwrite"):
        write_report(report, alias)
    assert path.read_bytes() == original


@pytest.mark.parametrize("smiles", ["not-a-smiles", "C1CC", "[CH5]", "", "  ", None, 123, " ", ">>"])
def test_invalid_smiles(tmp_path, smiles):
    report = validate_molecules(make_sample(tmp_path, [{"source_id": "a", "smiles": smiles}]))
    assert len(report.invalid_records) == 1
    assert report.invalid_records[0].line_number == 1
    assert report.invalid_records[0].source_id == "a"
    assert report.invalid_records[0].reason
    assert report.successfully_parsed_records == 0
    assert report.success  # Chemical findings alone are diagnostic.


def test_missing_smiles(tmp_path):
    report = validate_molecules(make_sample(tmp_path, [{"source_id": "a"}]))
    assert "Missing" in report.invalid_records[0].reason


def test_empty_graph(monkeypatch):
    from rdkit import Chem

    # Exercise the defensive check with a real zero-atom RDKit molecule.
    def empty_molecule(smiles, *, sanitize):
        assert sanitize is True
        return Chem.Mol()

    monkeypatch.setattr(Chem, "MolFromSmiles", empty_molecule)
    canonical, reason = canonicalize_smiles("C")
    assert canonical is None
    assert reason == "Empty molecular graph"


def test_stereochemistry(tmp_path):
    report = validate_molecules(make_sample(tmp_path, [
        {"source_id": str(i), "smiles": smiles}
        for i, smiles in enumerate(["F[C@H](Cl)Br", "F[C@@H](Cl)Br", "FC(Cl)Br"])
    ]))
    assert len(report.structures) == 3
    assert not report.duplicate_groups
    assert "FC(Cl)Br" in report.structures


def test_integrity_and_count_partition(tmp_path):
    path = make_sample(tmp_path, [
        {"source_id": "a", "smiles": "C"},
        {"source_id": "a", "smiles": "N"},
        {"smiles": "C"}, {"source_id": " "}, {"source_id": 12},
        None, [], "", {}, {"source_id": "b", "smiles": "bad"},
    ])
    with path.open("a") as stream:
        stream.write("{broken\n\n")
    report = validate_molecules(path)
    assert report.total_input_records == 12
    assert report.successfully_parsed_records == 1
    assert len(report.invalid_records) == 1
    assert len(report.integrity_errors) == 10
    assert report.integrity_errors[0].source_id == "a"
    assert report.integrity_errors[0].reason == "Duplicate source_id"
    assert report.integrity_errors[-1].line_number == 12
    assert not report.success
    assert "Integrity, line 2" in format_summary(report)


def test_minimum_and_empty_file(tmp_path):
    path = make_sample(tmp_path, [])
    report = validate_molecules(path, 1)
    assert report.total_input_records == 0
    assert not report.acceptance_passed
    assert not report.success
    assert "at_least_1100_unique_valid_structures" not in report.to_dict()
    assert validate_molecules(path, 0).success
    with pytest.raises(ValueError, match="nonnegative"):
        validate_molecules(path, -1)


def test_cli_exit_statuses(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "validate_molecules.py"
    path = make_sample(tmp_path, [{"source_id": "a", "smiles": "C"}])
    def run(*args):
        return subprocess.run([sys.executable, str(script), str(path), *args], capture_output=True, text=True)
    assert run().returncode == 0
    output = tmp_path / "report.json"
    result = run("--min-unique", "2", "--report", str(output))
    assert result.returncode == 1
    assert "Validation FAILED" in result.stdout
    assert not json.loads(output.read_text())["acceptance_passed"]
    assert run("--report", str(path)).returncode == 2
    path.write_text("broken\n")
    assert run().returncode == 1
    path.unlink()
    assert run().returncode == 2
