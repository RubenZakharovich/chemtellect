import hashlib
from pathlib import Path
import subprocess
import sys

import pytest
from chemtellect.data_integrity import validate_jsonl


def test_success_checksum_quality_and_preservation(tmp_path):
    path = tmp_path / 'sample.jsonl'
    raw = b'{"source_id":"a","smiles":"C"}\n{"source_id":"b"}\n'
    path.write_bytes(raw)
    report = validate_jsonl(path, 2)
    assert report.success
    assert report.total_records == report.unique_source_ids == 2
    assert report.missing_smiles == 1
    assert report.sha256 == hashlib.sha256(raw).hexdigest()
    assert path.read_bytes() == raw
    assert not validate_jsonl(path, 3).success


@pytest.mark.parametrize('raw,reason', [
    (b'{broken\n', 'Invalid JSON'), (b'\xff\n', 'UTF-8'),
    (b'[]\n', 'not a JSON object'), (b'{}\n', 'source_id'),
    (b'{"source_id":" "}\n', 'source_id'),
    (b'{"source_id":"a"}\n{"source_id":"a"}\n', 'Duplicate'),
    (b'{"source_id":', 'Invalid JSON'),
])
def test_failures(tmp_path, raw, reason):
    path = tmp_path / 'sample'
    path.write_bytes(raw)
    report = validate_jsonl(path)
    assert not report.success
    assert reason in ' '.join(report.errors)
    assert report.sha256 == hashlib.sha256(raw).hexdigest()


def test_cli_strict_and_allow_missing(tmp_path):
    path = tmp_path / 'sample'
    path.write_text('{"source_id":"a"}\n')
    script = Path(__file__).resolve().parents[1] / 'scripts/validate_raw_sample.py'
    def run(*args):
        return subprocess.run([sys.executable, str(script), str(path), *args], capture_output=True)
    assert run('--expected', '1').returncode == 1
    assert run('--expected', '1', '--allow-missing-smiles').returncode == 0
    assert run('--expected', '0').returncode == 2
    path.write_bytes(b'\xff\n')
    assert run().returncode == 1
