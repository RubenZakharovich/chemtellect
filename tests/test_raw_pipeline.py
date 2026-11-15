"""Small regression fixtures; never use the acquisition as a unit test."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from rdkit import DataStructs

from chemtellect import fingerprint_store as store
from chemtellect.exact_retrieval import write_reference, exact_top_k, threshold_members
from chemtellect.hnsw_reporting import verify_parent_series
from chemtellect.raw_pipeline import verify_audit, configuration, load_pipeline, pipeline_lock

ROOT = Path(__file__).resolve().parents[1]


def test_concurrent_or_interrupted_controller_lock_is_never_bypassed(tmp_path):
    with pipeline_lock(tmp_path):
        with pytest.raises(RuntimeError, match='controller lock exists'):
            with pipeline_lock(tmp_path): pytest.fail('bypassed lock')
        assert (tmp_path/'.pipeline.lock').exists()
    assert not (tmp_path/'.pipeline.lock').exists()


def test_bounded_reference_exact_equivalence_collision_ties_empty(tmp_path):
    def vector(bits):
        fp = DataStructs.ExplicitBitVect(2048)
        for bit in bits: fp.SetBit(bit)
        return fp
    index = [vector([0, 2047]), vector([0, 2047]), vector([7])]
    queries = [vector([0, 2047]), vector([999])]
    before = [f.ToBitString() for f in index]
    saved = write_reference(tmp_path/'scores.npy', queries, index)
    assert isinstance(saved, np.memmap) and saved.dtype == np.float64
    expected = np.asarray([DataStructs.BulkTanimotoSimilarity(q, index) for q in queries])
    np.testing.assert_array_equal(saved, expected)
    assert exact_top_k(saved[0], 1).tolist() == [0]
    assert threshold_members(saved[0], 1.0).tolist() == [0, 1]
    assert threshold_members(saved[1], .3).tolist() == []
    assert [f.ToBitString() for f in index] == before
    with pytest.raises(FileExistsError): write_reference(tmp_path/'scores.npy', queries, index)


def test_binary_reconstruction_preserves_all_bit_positions():
    packed = np.zeros((4, 256), dtype=np.uint8)
    packed[0] = np.arange(256, dtype=np.uint8)
    packed[1] = 255; packed[2, [0, 255]] = [1, 128]
    vectors = store.unpack_fingerprints(packed)
    for row, vector in zip(packed, vectors):
        assert vector.ToBitString() == ''.join(map(str, np.unpackbits(row, bitorder='little')))
    np.testing.assert_array_equal(store.pack_fingerprints(vectors), packed)


def test_byte_lookup_nmslib_encoding_is_identical_for_every_byte():
    from chemtellect.nmslib_adapter import to_nmslib_strings
    rows = np.stack([np.arange(256, dtype=np.uint8), np.full(256, 255, dtype=np.uint8),
                     np.zeros(256, dtype=np.uint8)])
    expected = [' '.join(map(str, np.unpackbits(row, bitorder='little'))) for row in rows]
    assert to_nmslib_strings(rows) == expected
    assert all(len(s) == 4095 for s in expected)


def series():
    runs = []
    sizes = [1000, 10000, 100000, 1000000]
    for n in sizes:
        runs.append({'passed': True, 'source': 'chemicalspace_new', 'package': 'scale_1m_00',
            'query_count': 100, 'index_size': n, 'configuration': {'approved': True},
            'fingerprint_identity': {'fixed': True}, 'settings': {'five_passes': True},
            'linkage': dict(split_id='parent', artifact_id='artifact', raw_sha256_inherited='raw', split_file_sha256='split'),
            'selection_metadata': {'index_sizes': sizes, 'query_identity_sha256': 'queries',
                                   'prefix_identity_sha256': {str(k): f'prefix{k}' for k in sizes if k <= n}}})
    return runs


@pytest.mark.parametrize('damage', ['parent', 'raw', 'queries', 'prefix', 'config', 'order', 'missing_prefix'])
def test_parent_join_rejects_mixed_identities(damage):
    runs = series(); assert verify_parent_series(runs) == runs
    if damage == 'parent': runs[1]['linkage']['split_id'] = 'another'
    elif damage == 'raw': runs[1]['linkage']['raw_sha256_inherited'] = 'another'
    elif damage == 'queries': runs[1]['selection_metadata']['query_identity_sha256'] = 'other'
    elif damage == 'prefix': runs[-1]['selection_metadata']['prefix_identity_sha256']['1000'] = 'other'
    elif damage == 'missing_prefix': runs[-1]['selection_metadata']['prefix_identity_sha256'].pop('1000')
    elif damage == 'config': runs[1]['configuration'] = {'approved': False}
    else: runs.reverse()
    with pytest.raises(ValueError): verify_parent_series(runs)


def test_legacy_parent_identity_still_supported():
    runs = series()[:3]
    for r in runs: r.pop('selection_metadata')
    assert verify_parent_series(runs) == runs
    runs[1]['linkage']['artifact_id'] = 'different'
    with pytest.raises(ValueError): verify_parent_series(runs)


@pytest.mark.parametrize('size', [100, 1000])
def test_unattended_small_workflow_and_verified_resume(tmp_path, size):
    # Isotopes supply unique canonical molecules with deliberately colliding bits.
    raw = tmp_path/'data/raw/chemicalspace_new/package_00_pool.jsonl'
    raw.parent.mkdir(parents=True)
    raw.write_text(''.join(json.dumps({'source_id': f'id:{i}', 'smiles': f'[{i}CH4]'})+'\n'
                           for i in range(1, 221)))
    store.write_json(raw.with_suffix('.meta.json'), {
        'manifest_schema_version': 1, 'source_database': 'chemicalspace_new', 'source_collection': 'compounds',
        'package_name': 'package_00', 'output_filename': raw.name, 'extraction_integrity_status': 'passed',
        'requested_record_count': 220, 'actual_extracted_record_count': 220,
        'missing_smiles_count': 0, 'sha256': store.file_info(raw)['sha256']})
    configs = tmp_path/'experiments/configs'; configs.mkdir(parents=True)
    config = json.loads((ROOT/'experiments/configs/chemicalspace_new_scale_1m_pipeline.json').read_text())
    config.update(package='package_00', index_sizes=[size])
    for name in ('morgan_r2_2048_chiral_v1.json', 'hnsw_v1.json', 'dataset_overview_v1.json'):
        (configs/name).write_bytes((ROOT/'experiments/configs'/name).read_bytes())
    config_path = configs/'pipeline.json'; store.write_json(config_path, config)
    directory = tmp_path/'experiments/results/fixture'
    command = [sys.executable, '-m', 'chemtellect.raw_pipeline', '--root', str(tmp_path),
               '--config', str(config_path), '--directory', str(directory), '--mode', 'execute']
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == (0 if size == 100 else 2), result.stdout+result.stderr
    state, summary, reports = load_pipeline(tmp_path, directory)
    assert summary['counts']['unique_canonical_structures'] == 220
    assert state['stages']['descriptors']['rows'] == 220
    if size == 1000:
        assert state['status'] == 'blocked' and not reports
        assert state['eligibility']['deficit'] == 880
        assert not (directory/'parent_split.json').exists()
        assert (directory/'latest_report.json').exists()
        return
    assert state['status'] == 'complete' and len(reports) == 1
    assert state['stages']['split']['unused_count'] == 20
    run_path = tmp_path/state['sizes']['100']['path']
    before = {p.name: store.file_info(p) for p in run_path.iterdir()}
    load_command = command[:-1]+['load']
    load_command[load_command.index('--directory')+1] = str(directory.relative_to(tmp_path))
    load_result = subprocess.run(load_command, cwd=tmp_path, capture_output=True, text=True)
    assert load_result.returncode == 0, load_result.stdout+load_result.stderr
    assert before == {p.name: store.file_info(p) for p in run_path.iterdir()}
    resumed = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert resumed.returncode == 0, resumed.stdout+resumed.stderr
    restored = json.loads((directory/'state.json').read_text())
    assert restored['sizes']['100']['run_id'] == state['sizes']['100']['run_id']
    assert before == {p.name: store.file_info(p) for p in run_path.iterdir()}
    # A deliberate new experiment reuses the verified parent/artifact and gets
    # a fresh immutable timing run, rather than failing on a different split path.
    rerun_command = list(command)
    rerun_directory = tmp_path/'experiments/results/intentional_rerun'
    rerun_command[rerun_command.index('--directory')+1] = str(rerun_directory)
    rerun = subprocess.run(rerun_command, cwd=tmp_path, capture_output=True, text=True)
    assert rerun.returncode == 0, rerun.stdout+rerun.stderr
    rerun_state, _, _ = load_pipeline(tmp_path, rerun_directory)
    assert rerun_state['stages']['fingerprints']['artifact_id'] == state['stages']['fingerprints']['artifact_id']
    assert rerun_state['stages']['fingerprints']['operation'] == 'reused'
    assert rerun_state['sizes']['100']['run_id'] != state['sizes']['100']['run_id']
    assert before == {p.name: store.file_info(p) for p in run_path.iterdir()}
    # Config changes and audit tampering are rejected instead of silently reused.
    with (directory/'acquisition_audit/eligible.jsonl').open('a') as f: f.write('{}\n')
    with pytest.raises(ValueError, match='checksum'):
        load_pipeline(tmp_path, directory)
