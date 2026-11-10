"""Exact numerical policy, controlled timing, and verified output publication."""
import json
import random

import numpy as np
import pytest
from rdkit import DataStructs, rdBase

from chemtellect import exact_retrieval as exact
from chemtellect import fingerprint_store as store
from chemtellect.dataset_overview import make_split
from chemtellect.fingerprints import APPROVED, NonzeroFingerprintError


def vectors(enabled):
    result = []
    for bits in enabled:
        fp = DataStructs.ExplicitBitVect(2048)
        fp.SetBitsFromList(bits)
        result.append(fp)
    return result


def test_scalar_bulk_alignment_and_distinct_records():
    queries = vectors([[1, 2], [3], [1, 2, 3]])
    index = vectors([[3], [1, 2], [1, 2], [9], [1]])
    matrix = np.array([DataStructs.BulkTanimotoSimilarity(q, index) for q in queries], dtype=np.float64)
    for q in range(3):
        for i in range(5):
            assert matrix[q, i] == DataStructs.TanimotoSimilarity(queries[q], index[i])
    np.testing.assert_array_equal(matrix[0], [0, 1, 1, 0, .5])
    result = exact.benchmark_queries(queries, index, candidate_counts=(2,))
    np.testing.assert_array_equal(result['scores'], matrix)
    assert result['positions']['2'][0] == [1, 2]  # Identical bits, separate original positions.
    records = {role: [dict(row_index=i, source_id=f'{role}:{i}', line_number=i+1)
                      for i in range(n)] for role, n in [('queries', 3), ('index', 5)]}
    saved = exact._results(result, records)
    assert [r['record']['source_id'] for r in saved['by_k']['2'][0]['matches']] == ['index:1', 'index:2']
    assert saved['by_k']['2'][1]['query']['source_id'] == 'queries:1'


def test_partial_selection_oversized_boundary_ties(monkeypatch):
    scores = np.full(10000, .5, dtype=np.float64)
    scores[[100, 8000]] = 1
    scores[4] = np.nextafter(.5, 1)
    original = scores.copy()
    # Full-index ordering would violate this checkpoint's selection policy.
    lexsort = np.lexsort
    def checked(keys, *args, **kwargs):
        assert len(keys[0]) <= 10
        return lexsort(keys, *args, **kwargs)
    monkeypatch.setattr(np, 'lexsort', checked)
    expected = [100, 8000, 4, 0, 1, 2, 3, 5, 6, 7]
    assert exact.exact_top_k(scores, 10).tolist() == expected
    np.testing.assert_array_equal(scores, original)
    assert exact.exact_top_k(np.ones(10000), 10).tolist() == list(range(10))


def test_selection_agrees_with_full_order_reference():
    rng = np.random.default_rng(192)
    for count in (1, 5, 100, 1000):
        scores = rng.integers(0, 8, count).astype(np.float64) / 8
        reference = sorted(range(count), key=lambda i: (-scores[i], i))
        for k in {1, min(10, count), count}:
            assert exact.exact_top_k(scores, k).tolist() == reference[:k]


@pytest.mark.parametrize('k', [0, -1, True, 1.0, 4, '2', np.int64(2)])
def test_invalid_k(k):
    with pytest.raises(ValueError, match='k must'):
        exact.exact_top_k(np.ones(3), k)


@pytest.mark.parametrize('scores', [np.array([np.nan]), np.array([np.inf]), np.array([-.01]),
    np.array([1.01]), np.ones(3, dtype=np.float32), np.ones((1, 3)), np.ones((1, 1, 3)), [1.]])
def test_invalid_scores(scores):
    with pytest.raises(ValueError):
        exact.exact_top_k(scores, 1)
    with pytest.raises(ValueError):
        exact.validate_scores(scores, (1,))


def test_inclusive_threshold_and_empty_sets():
    boundary = np.float64(.5)
    scores = np.array([0, np.nextafter(boundary, 0), boundary, np.nextafter(boundary, 1), .75])
    assert exact.threshold_members(scores, boundary).tolist() == [2, 3, 4]
    assert exact.threshold_members(scores, 1.).tolist() == []
    assert exact.threshold_members(scores, 0.).tolist() == list(range(5))
    assert exact.threshold_members(np.array([], dtype=np.float64), .5).tolist() == []
    for threshold in (True, np.nan, np.inf, -.1, 1.1, '0.5', np.float32(.5)):
        with pytest.raises(ValueError):
            exact.threshold_members(scores, threshold)


def test_clock_stage_boundaries_and_conversion():
    events = []
    ticks = iter([100, 130, 180, 185])
    def clock():
        events.append('clock')
        return next(ticks)
    class Raw:
        def __array__(self, dtype=None, copy=None):
            events.append('convert')
            return np.array([.5, 1., 1.], dtype=dtype)
    def scorer(query, index):
        events.append(('score', query, index))
        return Raw()
    scores, positions, ns = exact.measure_exact('q', ['a', 'b', 'c'], 2, clock=clock, scorer=scorer)
    assert events == ['clock', ('score', 'q', ['a', 'b', 'c']), 'clock', 'convert', 'clock', 'clock']
    assert ns == dict(scoring_ns=30, selection_ordering_ns=50, retrieval_total_ns=85)
    assert scores.dtype == np.float64 and positions.tolist() == [1, 2]


def test_timing_passes_shuffle_and_no_cached_retrieval():
    queries, index = list(range(7)), list(range(100))
    calls, ticks = [], []
    def scorer(q, index):
        calls.append(q)
        return [float(q % 2)] * len(index)
    def clock():
        ticks.append(len(ticks) * 10)
        return ticks[-1]
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    result = exact.benchmark_queries(queries, index, clock=clock, scorer=scorer)
    assert random.getstate() == python_state
    state_after = np.random.get_state()
    assert numpy_state[0] == state_after[0] and numpy_state[2:] == state_after[2:]
    np.testing.assert_array_equal(numpy_state[1], state_after[1])
    assert len(calls) == 7 * 6 * 4  # Every warmup and measured request scores afresh.
    assert len(ticks) == 7 * 5 * 4 * 4  # Warmups do not call the clock.
    assert len(result['raw_timings']) == 7 * 5 * 4
    assert len(result['query_orders']) == 6 * 4
    expected_rng = random.Random(42)
    for row in result['query_orders']:
        order = list(range(7)); expected_rng.shuffle(order)
        assert row['query_positions'] == order
    assert calls == [q for row in result['query_orders'] for q in row['query_positions']]
    again = exact.benchmark_queries(queries, index, clock=lambda: 0, scorer=scorer)
    assert again['query_orders'] == result['query_orders']
    np.testing.assert_array_equal(result['scores'][:, 0], np.arange(7) % 2)
    for k in exact.CANDIDATE_COUNTS:
        assert result['positions'][str(k)][6] == list(range(k))
        summary = result['summary']['by_k'][str(k)]['across_query_medians_ns']
        assert summary['retrieval_total_ns'] == dict(median=30., mean=30., p95=30.)


def test_designated_pass_is_not_pooled(monkeypatch):
    real = exact.measure_exact
    calls = 0
    def request(*args, **kwargs):
        nonlocal calls
        scores, positions, ns = real(*args, **kwargs)
        calls += 1
        if calls > 1:
            positions = positions[::-1]  # Deliberately different later-pass stub evidence.
        return scores, positions, ns
    monkeypatch.setattr(exact, 'measure_exact', request)
    result = exact.benchmark_queries([1], [1, 2, 3], candidate_counts=(2,),
                                     scorer=lambda q, index: [1., 1., .5])
    assert result['positions']['2'] == [[0, 1]]


def test_changed_scores_rejected_outside_timer():
    calls = 0
    ticking = 0
    def clock():
        nonlocal ticking
        ticking += 1
        return ticking
    def scorer(q, i):
        nonlocal calls
        calls += 1
        return [1. if calls <= 2 else .5]
    with pytest.raises(ValueError, match='scores changed'):
        exact.benchmark_queries([1], [1], candidate_counts=(1,), clock=clock, scorer=scorer)
    assert ticking == 8  # warmup, capture, changed computation; full timer closed before rejection.


@pytest.mark.parametrize('row', [[np.nan], [np.inf], [-.1], [1.1], [1., .5]])
def test_computed_scores_validated_before_capture(row):
    with pytest.raises(ValueError):
        exact.benchmark_queries([1], [1], candidate_counts=(1,), scorer=lambda *a: row)


def test_summary_aggregates_per_query_medians():
    raw = [dict(k=10, query_position=q, pass_index=p,
                **{s: value for s in exact.STAGES})
           for q, values in enumerate(([0, 1, 2, 100, 1000], [10, 11, 12, 100, 1000]))
           for p, value in enumerate(values)]
    summary = exact.summarize_timings(raw, 2, (10,))
    assert summary['percentile_method'] == 'linear'
    assert summary['by_k']['10']['across_query_medians_ns']['retrieval_total_ns'] == {
        'median': 7., 'mean': 7., 'p95': 11.5}
    with pytest.raises(ValueError, match='five'):
        exact.summarize_timings(raw[:-1], 2, (10,))


@pytest.fixture
def artifact(tmp_path):
    """Valid storage fixture with synthetic bits and separate structure identities."""
    rows = [dict(source_id=f'source:{i}', line_number=i+1, smiles=f'fixture:{i}',
                 canonical_smiles=f'distinct-structure:{i}') for i in range(122)]
    summary = dict(source='chemicalspace_new', package='package_00', sha256='a'*64,
                   rdkit_version=rdBase.rdkitVersion,
                   representative_policy='first raw physical line per canonical structure')
    split = make_split(summary, rows, (100, 120), query_count=2)
    split_path = tmp_path/'split.json'
    store.write_json(split_path, split)
    ident = store.identity(split, APPROVED)
    aid = store.digest(ident)
    directory = tmp_path/'data/processed/chemicalspace_new/package_00/fingerprints'/aid
    directory.mkdir(parents=True)
    q = store.pack_fingerprints(vectors([[1, 2], [3]]))
    i = store.pack_fingerprints(vectors([[1, 2], [1, 2], [3], [8]] * 30))
    np.save(directory/'queries.npy', q, allow_pickle=False)
    np.save(directory/'index.npy', i, allow_pickle=False)
    store.write_json(directory/'records.json', store.mapping(split))
    manifest = {'manifest_schema_version': 1, 'status': 'complete', 'artifact_id': aid,
        'identity': ident, **store.linkage(tmp_path, split_path, split), 'configuration': APPROVED,
        'storage': store.STORAGE, 'query_count': 2, 'largest_index_count': 120, 'index_sizes': [100, 120],
        'dimensions': {'queries': [2, 256], 'index': [120, 256]}, 'created_utc': store.utc(),
        'environment': store.environment(tmp_path), 'execution': {'batch_size': 1000, 'thread_count': 1},
        'files': {name: store.file_info(directory/name) for name in ('queries.npy', 'index.npy', 'records.json')}}
    store.write_json(directory/'manifest.json', manifest)
    return directory, manifest


def test_verified_selection_and_roundtrip(tmp_path, artifact, monkeypatch):
    directory, manifest = artifact
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    monkeypatch.setattr(store, 'prepare_artifact', lambda *a, **k: pytest.fail('generated fingerprints'))
    runs = []
    for _ in range(2):
        path, metadata, summary = exact.run_benchmark(tmp_path, directory, 100)
        runs.append(path)
        restored = exact.load_run(path)
        assert metadata == restored['metadata'] and summary == restored['summary']
        assert path.parent.parts[-3:] == ('chemicalspace_new', 'package_00', manifest['artifact_id'])
        assert restored['scores'].shape == (2, 100) and restored['scores'].dtype == np.float64
        records = restored['records']
        assert records['linkage'] == metadata['linkage']
        assert records['index'] == json.loads((directory/'records.json').read_text())['index'][:100]
        assert metadata['nonzero_validation']['index_rows_checked'] == 100
        assert metadata['linkage']['raw_sha256_inherited'] == 'a'*64
        q, i = store.unpack_fingerprints(np.load(directory/'queries.npy')), store.unpack_fingerprints(np.load(directory/'index.npy')[:100])
        expected = np.array([DataStructs.BulkTanimotoSimilarity(fp, i) for fp in q])
        np.testing.assert_array_equal(restored['scores'], expected)
        first = restored['top_k']['by_k']['10'][0]['matches']
        assert [r['index_position'] for r in first] == [0, 1, 4, 5, 8, 9, 12, 13, 16, 17]
        assert first[0]['record']['source_id'] != first[1]['record']['source_id']
        assert first[0]['similarity'] == first[1]['similarity'] == 1
        assert len(restored['timings']['raw']) == 2*4*5
        for name, info in restored['manifest']['files'].items():
            assert store.file_info(path/name) == info
    assert runs[0] != runs[1]
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}


@pytest.mark.parametrize('size', [99, 101, True, 100.0])
def test_unregistered_size_fails_before_reconstruction(tmp_path, artifact, monkeypatch, size):
    monkeypatch.setattr(store, 'unpack_fingerprints', lambda *a: pytest.fail('reconstructed'))
    with pytest.raises(ValueError):
        exact.run_benchmark(tmp_path, artifact[0], size)
    assert not (tmp_path/'experiments/results/exact_retrieval').exists()


@pytest.mark.parametrize('damage', ['missing', 'checksum', 'mapping', 'zero', 'expected_selection'])
def test_invalid_input_stops_before_backend(tmp_path, artifact, monkeypatch, damage):
    directory, manifest = artifact
    if damage == 'missing':
        directory = directory/'absent'
    elif damage in ('checksum', 'zero'):
        index = np.load(directory/'index.npy'); index[99] = 0
        np.save(directory/'index.npy', index)
        if damage == 'zero':
            manifest['files']['index.npy'] = store.file_info(directory/'index.npy')
    elif damage == 'mapping':
        records = json.loads((directory/'records.json').read_text()); records['index'].reverse()
        (directory/'records.json').write_text(json.dumps(records))
        manifest['files']['records.json'] = store.file_info(directory/'records.json')
    if damage != 'missing':
        (directory/'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(store, 'unpack_fingerprints', lambda *a: pytest.fail('backend before validation'))
    expected = {'source': 'wrong'} if damage == 'expected_selection' else None
    with pytest.raises((ValueError, FileNotFoundError, NonzeroFingerprintError)):
        exact.run_benchmark(tmp_path, directory, 100, expected_selection=expected)
    assert not (tmp_path/'experiments/results/exact_retrieval').exists()


@pytest.mark.parametrize('damage', ['scores', 'results', 'provenance', 'mapping', 'summary', 'timings', 'metadata'])
def test_saved_output_corruption_rejected(tmp_path, artifact, damage):
    path, _, _ = exact.run_benchmark(tmp_path, artifact[0], 100)
    filename = {'scores': 'exact_scores.npy', 'results': 'top_k.json', 'provenance': 'records.json',
                'mapping': 'records.json', 'summary': 'summary.json', 'timings': 'timings.json',
                'metadata': 'metadata.json'}[damage]
    if damage == 'scores':
        np.save(path/filename, np.zeros((2, 100), dtype=np.float32))
    else:
        value = json.loads((path/filename).read_text())
        if damage == 'results': value['by_k']['10'][0]['matches'].reverse()
        if damage == 'provenance': value['linkage']['artifact_id'] = 'b'*64
        if damage == 'mapping': value['queries'].reverse()
        if damage == 'summary': value['by_k']['10']['across_query_medians_ns']['scoring_ns']['mean'] = -1
        if damage == 'timings': value['raw'][0]['query_position'] = 100
        if damage == 'metadata': value['linkage']['split_id'] = 'b'*64
        (path/filename).write_text(json.dumps(value))
    with pytest.raises(ValueError, match='checksum'):
        exact.load_run(path)
    # Recomputed hashes cannot hide inconsistent numerical/provenance evidence.
    manifest = json.loads((path/'manifest.json').read_text())
    manifest['files'][filename] = store.file_info(path/filename)
    (path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        exact.load_run(path)


def test_output_failure_retains_staging_without_complete_run(tmp_path, artifact, monkeypatch):
    original = store.write_json
    def fail(path, value):
        if path.name == 'top_k.json':
            raise OSError('injected output failure')
        original(path, value)
    monkeypatch.setattr(store, 'write_json', fail)
    with pytest.raises(OSError):
        exact.run_benchmark(tmp_path, artifact[0], 100)
    parent = tmp_path/'experiments/results/exact_retrieval/chemicalspace_new/package_00'/artifact[1]['artifact_id']
    assert list(parent.glob('.*.staging-*'))
    assert not list(parent.glob('run_*'))
    assert not list(parent.glob('*/manifest.json'))


def test_completed_run_cannot_be_overwritten(tmp_path, artifact, monkeypatch):
    class FixedID:
        hex = 'fixed-publication-conflict'
    monkeypatch.setattr(exact, 'uuid4', FixedID)
    path, _, _ = exact.run_benchmark(tmp_path, artifact[0], 100)
    before = {p.name: p.read_bytes() for p in path.iterdir()}
    with pytest.raises(FileExistsError, match='Publication conflict'):
        exact.run_benchmark(tmp_path, artifact[0], 100)
    assert before == {p.name: p.read_bytes() for p in path.iterdir()}
    assert exact.load_run(path)['manifest']['status'] == 'complete'
