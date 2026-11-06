from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
from unittest.mock import MagicMock

import pytest
import chemtellect.mongo_sampling as sampling
from chemtellect.mongo_sampling import SamplingConfig, build_command, sample_mongo


@pytest.fixture
def config(tmp_path):
    return SamplingConfig('chemtellect-server', 'otava_chemical', 'compounds', 'package_01', size=2, raw_root=tmp_path / 'raw')


RAW = b'{"source_id":"ObjectId(abc)","smiles":"C"}\n{"source_id":"42"}\n'


def mock_process(monkeypatch, raw=RAW, statuses=(0,)):
    process = MagicMock()
    process.__enter__.return_value = process
    process.poll.side_effect = statuses
    def start(command, *, stdout, stderr, shell):
        assert shell is False
        stdout.write(raw)
        stdout.flush()
        return process
    monkeypatch.setattr(sampling.subprocess, 'Popen', start)
    return process


@pytest.mark.parametrize('changes', [
    {'size': 0}, {'size': -1}, {'size': True}, {'size': 10001},
    {'database': '../bad'}, {'collection': 'x;drop()'}, {'package': '../p'},
    {'ssh_target': '-oProxyCommand=evil'}, {'ssh_target': 'host;touch bad'},
    {'ssh_target': 'host\nwhoami'}, {'server_timeout_ms': 0},
    {'client_timeout': float('nan')}, {'max_transfer_bytes': 0},
])
def test_reject_unsafe(config, changes):
    with pytest.raises(ValueError):
        sample_mongo(replace(config, **changes), dry_run=True)
    assert not config.raw_root.exists()


def test_query(config):
    command = build_command(config)
    assert command[:6] == ['ssh', '-T', '-o', 'BatchMode=no', '--', 'chemtellect-server']
    remote = shlex.split(command[-1])
    assert remote[:4] == ['mongo', '--quiet', '--norc', '--eval']
    js = remote[4]
    pipeline = json.loads(js.split('.aggregate(', 1)[1].split(', {maxTimeMS:', 1)[0])
    assert pipeline == [{'$sample': {'size': 2}}, {'$project': {field: 1 for field in sampling.FIELDS}}]
    assert 'getSiblingDB("otava_chemical")' in js
    assert 'getCollection("compounds")' in js
    assert 'maxTimeMS: 30000' in js
    assert "encoded = 'string:' + JSON.stringify(id)" in js
    assert "'objectid:' + id.valueOf().toLowerCase()" in js
    assert 'Unsupported BSON source identifier type' in js
    assert 'JSON.stringify(row)' in js
    for forbidden in ('rfp', 'stdmol', '$match', '$out', '$merge', '$rand', '$sampleRate', 'insert(', 'update(', 'remove('):
        assert forbidden not in js
    assert build_command(replace(config, size=10001, allow_large_sample=True))


def test_success_metadata_and_preservation(config, monkeypatch):
    directory = config.raw_root / config.database
    directory.mkdir(parents=True)
    original = directory / 'package_00_pool.jsonl'
    original.write_bytes(b'original')
    mock_process(monkeypatch)
    manifest = sample_mongo(config)
    output = directory / 'package_01_pool.jsonl'
    metadata = output.with_suffix('.meta.json')
    assert output.read_bytes() == RAW
    assert json.loads(metadata.read_text()) == manifest
    assert manifest['sha256'] == hashlib.sha256(RAW).hexdigest()
    assert manifest['requested_record_count'] == manifest['actual_extracted_record_count'] == 2
    assert manifest['missing_smiles_count'] == 1
    assert manifest['sampling_method'] == 'mongodb_sample_first_stage'
    assert 'not a guaranteed uniform' in manifest['sampling_method_limitations']
    assert manifest['source_collection_size'] is manifest['snapshot_identifier'] is None
    assert manifest['fields_retrieved'] == sampling.FIELDS
    assert 'chemtellect-server' not in metadata.read_text()
    assert manifest['extraction_integrity_status'] == 'passed'
    assert original.read_bytes() == b'original'
    assert not list(directory.glob('*.partial'))
    with pytest.raises(FileExistsError):
        sample_mongo(config)
    assert output.read_bytes() == RAW


@pytest.mark.parametrize('raw', [b'{broken', b'{"source_id":"a"}\n', b'{"source_id":"a"}\n{"source_id":"a"}\n', b'\xff\n'])
def test_bad_transfers(config, monkeypatch, raw):
    mock_process(monkeypatch, raw)
    with pytest.raises(ValueError, match='integrity') as exc:
        sample_mongo(config)
    assert 'Recovery files' in exc.value.__notes__[0]
    directory = config.raw_root / config.database
    assert not list(directory.glob('*.jsonl'))
    assert not list(directory.glob('*.json'))
    assert list(directory.glob('*.partial'))[0].read_bytes() == raw


@pytest.mark.parametrize('kind', ['status', 'timeout', 'bytes', 'missing', 'interrupt'])
def test_process_failures(config, monkeypatch, kind):
    process = mock_process(monkeypatch, statuses=(7,) if kind == 'status' else (None,))
    if kind == 'timeout':
        config = replace(config, client_timeout=0.001)
        times = iter([0, 1])
        monkeypatch.setattr(sampling.time, 'monotonic', lambda: next(times))
    elif kind == 'bytes':
        config = replace(config, max_transfer_bytes=1)
    elif kind == 'missing':
        def missing(*args, **kwargs):
            raise FileNotFoundError('ssh')
        monkeypatch.setattr(sampling.subprocess, 'Popen', missing)
    elif kind == 'interrupt':
        process.poll.side_effect = KeyboardInterrupt
    with pytest.raises((RuntimeError, subprocess.TimeoutExpired, FileNotFoundError, KeyboardInterrupt)):
        sample_mongo(config)
    assert not list(config.raw_root.rglob('*.jsonl'))
    if kind in ('timeout', 'bytes', 'interrupt'):
        process.kill.assert_called_once()
        process.wait.assert_called_once()


def test_dry_run(config, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('network invoked')
    monkeypatch.setattr(sampling.subprocess, 'Popen', forbidden)
    result = sample_mongo(config, dry_run=True)
    assert result['requested_record_count'] == 2
    assert not config.raw_root.exists()


@pytest.mark.parametrize('existing', ['package_01_pool.jsonl', 'package_01_pool.meta.json'])
def test_existing_outputs(config, existing):
    directory = config.raw_root / config.database
    directory.mkdir(parents=True)
    path = directory / existing
    path.write_bytes(b'preserve')
    with pytest.raises(FileExistsError):
        sample_mongo(config)
    assert path.read_bytes() == b'preserve'


def test_manifest_publication_failure_recoverable(config, monkeypatch):
    mock_process(monkeypatch)
    real_link = sampling.os.link
    def link(source, target):
        if str(target).endswith('.meta.json'):
            raise OSError('manifest publication failure')
        real_link(source, target)
    monkeypatch.setattr(sampling.os, 'link', link)
    with pytest.raises(OSError, match='manifest') as exc:
        sample_mongo(config)
    assert 'Recovery files' in exc.value.__notes__[0]
    directory = config.raw_root / config.database
    assert (directory / 'package_01_pool.jsonl').read_bytes() == RAW
    assert not (directory / 'package_01_pool.meta.json').exists()
    staged = next(directory.glob('*.meta.partial'))
    assert json.loads(staged.read_text())['sha256'] == hashlib.sha256(RAW).hexdigest()
    with pytest.raises(FileExistsError):
        sample_mongo(config)


def test_metadata_generation_failure(config, monkeypatch):
    mock_process(monkeypatch)
    def fail(*args, **kwargs):
        raise OSError('metadata write failed')
    monkeypatch.setattr(sampling.json, 'dump', fail)
    with pytest.raises(OSError, match='metadata write'):
        sample_mongo(config)
    assert not list(config.raw_root.rglob('*.jsonl'))
    assert not list(config.raw_root.rglob('*.json'))
    assert list(config.raw_root.rglob('*.partial'))


def test_publication_race_does_not_overwrite(config, monkeypatch):
    mock_process(monkeypatch)
    real_link = sampling.os.link
    def competing_link(source, target):
        Path(target).write_bytes(b'competing acquisition')
        real_link(source, target)
    monkeypatch.setattr(sampling.os, 'link', competing_link)
    with pytest.raises(FileExistsError):
        sample_mongo(config)
    output = config.raw_root / config.database / 'package_01_pool.jsonl'
    assert output.read_bytes() == b'competing acquisition'
    assert not output.with_suffix('.meta.json').exists()


def test_cli_dry_run_and_invalid_size(config):
    import sys
    script = Path(__file__).resolve().parents[1] / 'scripts/sample_mongo.py'
    base = [sys.executable, str(script), '--ssh-target', config.ssh_target,
            '--database', config.database, '--collection', config.collection,
            '--package', config.package, '--raw-root', str(config.raw_root), '--dry-run']
    result = subprocess.run(base, capture_output=True, text=True)
    assert result.returncode == 0
    assert json.loads(result.stdout)['requested_record_count'] == 1200
    result = subprocess.run(base + ['--size', '0'], capture_output=True, text=True)
    assert result.returncode == 1
    assert 'positive integer' in result.stderr
    assert not config.raw_root.exists()


@pytest.mark.parametrize('failure', [None, 'ssh', 'mongo'])
def test_offline_ssh_integration(config, tmp_path, monkeypatch, failure):
    """Run a real child process; PATH resolves ssh only to this local fixture."""
    executable = tmp_path / 'ssh'
    captured = tmp_path / 'arguments.json'
    executable.write_text(
        '#!/usr/bin/python3\n'
        'import json, os, sys\n'
        'from pathlib import Path\n'
        'Path(os.environ["CAPTURE"]).write_text(json.dumps(sys.argv[1:]))\n'
        f'sys.stdout.buffer.write({RAW!r})\n'
        'failure = os.environ.get("FAKE_FAILURE")\n'
        'if failure:\n'
        '    sys.stderr.write(failure + ": private-token\\n" + "x" * 100000)\n'
        '    sys.exit(255 if failure == "ssh" else 1)\n'
    )
    executable.chmod(0o700)
    monkeypatch.setenv('PATH', str(tmp_path))
    monkeypatch.setenv('CAPTURE', str(captured))
    monkeypatch.setenv('FAKE_FAILURE', failure or '')
    directory = config.raw_root / config.database
    if failure:
        with pytest.raises(RuntimeError) as caught:
            sample_mongo(config)
        diagnostic = next(directory.glob('*.stderr.log'))
        assert diagnostic.stat().st_mode & 0o777 == 0o600
        assert diagnostic.stat().st_size == sampling.DIAGNOSTIC_LIMIT
        assert diagnostic.read_text().startswith(failure + ': private-token')
        assert str(diagnostic) in '\n'.join(caught.value.__notes__)
        assert 'private-token' not in str(caught.value)
        assert not list(directory.glob('*.jsonl'))
        assert not list(directory.glob('*.meta.json'))
    else:
        manifest = sample_mongo(config)
        output = directory / 'package_01_pool.jsonl'
        assert output.read_bytes() == RAW
        assert json.loads(output.with_suffix('.meta.json').read_text()) == manifest
        assert manifest['source_identity_encoding'] == sampling.SOURCE_ID_ENCODING
        assert not list(directory.glob('*.stderr.log'))
        assert not list(directory.glob('*.partial'))
        assert json.loads(captured.read_text()) == build_command(config)[1:]
        captured.unlink()
        with pytest.raises(FileExistsError):
            sample_mongo(config)
        assert not captured.exists()  # Existing data blocks even spawning ssh.
        assert output.read_bytes() == RAW
        return
    assert json.loads(captured.read_text()) == build_command(config)[1:]


def test_source_id_javascript(config, tmp_path):
    """Execute generated JS with a local legacy-shell stub, never a database."""
    import shutil

    node = shutil.which('node')
    if node is None:
        pytest.skip('Optional JavaScript runtime unavailable')
    js = shlex.split(build_command(config)[-1])[-1]
    hex_id = '507f191e810c19729de860ea'
    harness = '''
function ObjectId(value) { this.valueOf = function() { return value; }; }
var rows = [
    {_id: new ObjectId('507F191E810C19729DE860EA')},
    {_id: '507f191e810c19729de860ea'},
    {_id: 'quoted"\\\\value'}
];
var print = console.log;
var db = {getSiblingDB: function() {return {
    getCollection: function() {return {
        aggregate: function() {return {
            forEach: function(callback) {rows.forEach(callback);}
        };}
    };}
};}};
'''
    result = subprocess.run([node, '-e', harness + js], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    ids = [json.loads(line)['source_id'] for line in result.stdout.splitlines()]
    assert ids[0] == 'objectid:' + hex_id
    assert ids[1] == 'string:' + json.dumps(hex_id)
    assert len(set(ids)) == 3
    assert json.loads(ids[2].removeprefix('string:')) == 'quoted"\\value'
    result = subprocess.run(
        [node, '-e', harness + 'rows = [{_id: 42}];' + js],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert 'Unsupported BSON source identifier type' in result.stderr
