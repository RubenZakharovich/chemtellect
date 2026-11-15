"""Installed NMSLIB integration, approximate metrics, timing, and immutable publication."""
import json
import random

import numpy as np
import pytest
from rdkit import DataStructs, rdBase

from chemtellect import exact_retrieval as exact
from chemtellect import fingerprint_store as store
from chemtellect import fpsim2_adapter as fpsim2
from chemtellect import hnsw_adapter as hnsw
from chemtellect import hnsw_benchmark as bench
from chemtellect import nmslib_adapter as binary
from chemtellect.dataset_overview import make_split
from chemtellect.fingerprints import APPROVED, validate_nonzero_fingerprints
from test_exact_retrieval import vectors


def response(ids, distances):
    return np.asarray(ids, dtype=np.int32), np.asarray(distances, dtype=np.float32)


def test_actual_native_strings_explicit_ids_collisions_and_zero_overlap():
    rows, enabled = binary.synthetic_fixture()
    rows = rows[1:]
    strings = binary.to_nmslib_strings(rows)
    for value, bits in zip(strings, enabled[1:]):
        assert [i for i, token in enumerate(value.split()) if token == '1'] == bits
    records = {role: [dict(row_index=i, source_id=f'{role}:{i}', line_number=i+1)
                      for i in range(len(rows))] for role in ('queries', 'index')}
    validate_nonzero_fingerprints(rows, rows, records)
    ids = [901, 7, 42, 88, 1234, 3001]
    engine, queries, mapping, prep = hnsw.prepare_backend(rows, rows, ids)
    assert mapping == dict(zip(ids, range(len(ids))))
    assert prep['insertion_positions'] == list(range(len(ids)))
    assert prep['build_count'] == prep['construction_threads'] == 1
    assert prep['query_conversion_ns'] >= 0 and prep['graph_construction_ns'] >= 0
    found, distances = engine.knnQuery(queries[1], k=6)
    assert found.dtype == np.int32 and distances.dtype == np.float32
    assert set(found) == set(ids)
    assert set(found[distances == 0]) == {7, 42}  # Collision records remain distinct.
    assert found[distances == 1].tolist() == [1234]
    qv = store.unpack_fingerprints(rows)
    reference = np.asarray(DataStructs.BulkTanimotoSimilarity(qv[1], qv))
    check = hnsw.check_response((found, distances), reference, mapping, 6, records['index'])
    assert check['passed'] and check['maximum_absolute_distance_error'] <= binary.ATOL
    ordered = hnsw.normalize_results((found, distances), mapping)
    assert ordered[0][:2].tolist() == [7, 42]
    assert hnsw.verify_synthetic()['passed']


def test_fixed_build_once_and_no_batch_api(monkeypatch):
    calls = []
    class Engine:
        def createIndex(self, params, print_progress): calls.append(('build', params, print_progress))
        def setQueryTimeParams(self, params): calls.append(('query_params', params))
    def register(rows, ids):
        calls.append(('register', list(ids)))
        return Engine(), {i: p for p, i in enumerate(ids)}
    monkeypatch.setattr(binary, 'register_rows', register)
    packed = store.pack_fingerprints(vectors([[0, 2047], [1]]))
    _, queries, mapping, _ = hnsw.prepare_backend(packed[:1], packed, [42, 900])
    assert calls == [('register', [42, 900]),
        ('build', {'M': 16, 'efConstruction': 200, 'indexThreadQty': 1}, False),
        ('query_params', {'efSearch': 200})]
    assert len(queries) == 1 and mapping == {42: 0, 900: 1}


@pytest.mark.parametrize('bad', [None, [], [np.ones(1)], (np.ones(2), np.ones(2)),
    (np.zeros((1, 2), dtype=np.int32), np.ones(2)),
    (np.array([0, 1]), np.ones((1, 2))), (np.array([0, 1]), np.ones(1)),
    response([0], [0]), response([0, 0], [0, 0]), response([0, 9], [0, .5]),
    response([-1, 1], [0, .5]), response([0, 1], [np.nan, .5]),
    response([0, 1], [np.inf, .5]), response([0, 1], [-.1, .5]),
    response([0, 1], [1.1, .5]), response([0, 1], [.1, .5]),
    response([0, 1, 2], [0, .5, 1])])
def test_malformed_short_ids_and_distances_are_preserved_and_rejected(bad):
    ordered = hnsw.normalize_results(bad, {i: i for i in range(3)})
    check = hnsw.check_response(bad, np.array([1., .5, 0.]), {i: i for i in range(3)}, 2)
    assert not check['passed']
    json.dumps(check, allow_nan=False)


def test_aliasing_ids_rejected_and_native_errors_do_not_repair_scores():
    check = hnsw.check_response(response([42, 88], [.5, .5]), np.array([.5, .5]), {42: 0, 88: 0}, 2)
    assert not check['passed'] and check['aliased_positions'] == [0]
    check = hnsw.check_response(response([0], [.5+2e-6]), np.array([.5]), {0: 0}, 1)
    assert not check['passed'] and check['maximum_absolute_distance_error'] > 1e-6


def test_approximate_misses_pass_identity_gate_but_reduce_recall():
    reference = np.array([1., .5, .25, 0.])
    check = hnsw.check_response(response([1, 2], [.5, .75]), reference, {i: i for i in range(4)}, 2)
    assert check['passed']
    assert bench.tie_aware_recall(reference, [1, 2], 2)['recall'] == .5


def test_exact_boundary_ties_never_replace_missing_strict_winners():
    reference = np.array([1., .5, .5, .5, np.nextafter(.5, 0)])
    assert bench.tie_aware_recall(reference, [0, 2], 2)['recall'] == 1
    assert bench.tie_aware_recall(reference, [0, 1], 2)['recall'] == 1
    assert bench.tie_aware_recall(reference, [0, 4], 2)['recall'] == .5
    missing = bench.tie_aware_recall(reference, [1, 2, 3], 3)
    assert missing['recall'] == 2/3 and missing['credited_boundary_hits'] == 2
    for positions in ([0, 0], [0], [0, 9]):
        with pytest.raises(ValueError): bench.tie_aware_recall(reference, positions, 2)


def test_threshold_recall_inclusive_empty_undefined_and_same_designated_set():
    reference = np.array([[.3, np.nextafter(.3, 0), .7, .9], [0., .1, .2, .25]])
    designated = {'hnsw': {'2': [{'native_results': [{'index_position': p} for p in positions]}
                                for positions in ([0, 2], [2, 3])]}}
    quality = bench.evaluate_designated(designated, reference)
    rows = quality['threshold_evaluations']['rows']
    low = next(r for r in rows if r['query_position'] == 0 and r['threshold'] == .3)
    assert low['qualifying_count'] == 3 and low['recall'] == 2/3
    high = next(r for r in rows if r['query_position'] == 0 and r['threshold'] == .9)
    assert high['recall'] == 0
    assert all(r['recall'] is None and r['empty_qualifying_set'] for r in rows if r['query_position'] == 1)
    assert quality['by_backend']['hnsw']['2']['thresholds']['0.3']['empty_queries'] == 1


def test_controlled_clock_covers_knn_and_normalization():
    calls = []
    class Engine:
        def knnQuery(self, query, k):
            calls.append((query, k))
            return response([42, 7], [.5, .5])
    ticks = iter([10, 30, 50, 55])
    native, ordered, ns = hnsw.measure_query(Engine(), 'prepared', 2, {42: 1, 7: 0}, lambda: next(ticks))
    assert calls == [('prepared', 2)]
    assert native[0].tolist() == [42, 7] and ordered[0].tolist() == [7, 42]
    assert ns == dict(native_search_ns=20, normalization_ns=20, retrieval_total_ns=45)


def test_three_native_backends_shared_orders_rng_warmups_and_designation(tmp_path):
    packed = store.pack_fingerprints(vectors([[1], [1, 2], [3], [3], [4], [5]]))
    qv, iv = store.unpack_fingerprints(packed[:2]), store.unpack_fingerprints(packed)
    reference = np.array([DataStructs.BulkTanimotoSimilarity(q, iv) for q in qv])
    fe, fq, fm, _ = fpsim2.prepare_backend(tmp_path/'fps.h5', packed[:2], packed, APPROVED)
    he, hq, hm, _ = hnsw.prepare_backend(packed[:2], packed)
    ticks = []
    def clock():
        ticks.append(len(ticks))
        return ticks[-1]
    state = random.getstate()
    result = bench.benchmark_three(qv, iv, fe, fq, fm, he, hq, hm, reference,
                                    candidate_counts=(2, 6), clock=clock)
    assert result['passed'] and random.getstate() == state
    assert len(ticks) == 2*2*5*3*4  # No retrieval clock calls during warmups.
    assert len(result['measured_results']) == 2*2*5*3
    for order in result['query_orders']:
        assert order['backend_order'] == bench.backend_order(order['pass_index'])
        if order['pass_index'] >= 0:
            for b in bench.BACKENDS:
                rows = [r for r in result['raw_timings'][b] if r['k'] == order['k'] and r['pass_index'] == order['pass_index']]
                assert [r['query_position'] for r in rows] == order['query_positions']
    for b in bench.BACKENDS:
        assert result['summaries'][b]['by_k']['2']['across_query_medians_ns']['retrieval_total_ns']['median'] == 3
    quality = bench.evaluate_designated(result['designated'], reference)
    assert quality['by_backend']['rdkit']['2']['top_k']['mean'] == 1
    assert quality['by_backend']['fpsim2']['2']['top_k']['mean'] == 1


def test_short_measured_hnsw_response_stops_with_partial_evidence():
    qv, iv = vectors([[1]]), vectors([[1], [1, 2]])
    reference = np.array([[1., .5]])
    class FPS:
        def top_k(self, *a, **kw):
            return np.array([(0, 1.), (1, .5)], dtype=[('mol_id', 'i4'), ('coeff', 'f4')])
    class Short:
        calls = 0
        def knnQuery(self, *a, **kw):
            self.calls += 1
            return response([0, 1], [0, .5]) if self.calls == 1 else response([0], [0])
    result = bench.benchmark_three(qv, iv, FPS(), [None], {0: 0, 1: 1}, Short(), [None],
                                   {0: 0, 1: 1}, reference, candidate_counts=(2,), clock=lambda: 0)
    assert not result['passed'] and result['summaries'] is None
    assert result['failure']['backend'] == 'hnsw' and result['failure']['pass_index'] == 0
    assert result['failure']['check']['actual_count'] == 1
    assert len(result['raw_timings']['hnsw']) == 1


def test_repeated_candidates_are_not_pooled_or_best_pass_selected():
    qv, iv = vectors([[1]]), vectors([[1], [1, 2], [1, 2, 3, 4]])
    reference = np.array([[1., .5, .25]])
    class FPS:
        def top_k(self, *a, **kw):
            return np.array([(0, 1.), (1, .5)], dtype=[('mol_id', 'i4'), ('coeff', 'f4')])
    class Changing:
        calls = 0
        def knnQuery(self, *a, **kw):
            self.calls += 1
            return response([0, 2], [0, .75]) if self.calls == 2 else response([0, 1], [0, .5])
    mapping = {i: i for i in range(3)}
    result = bench.benchmark_three(qv, iv, FPS(), [None], mapping, Changing(), [None], mapping,
                                   reference, candidate_counts=(2,), clock=lambda: 0)
    assert result['passed'] and result['repeated_candidate_membership_changes']
    quality = bench.evaluate_designated(result['designated'], reference)
    assert quality['by_backend']['hnsw']['2']['top_k']['mean'] == .5
    assert [r['index_position'] for r in result['designated']['hnsw']['2'][0]['native_results']] == [0, 2]


@pytest.fixture
def artifact(tmp_path):
    rows = [dict(source_id=f'source:{i}', line_number=i+1, smiles=f'fixture:{i}',
                 canonical_smiles=f'structure:{i}') for i in range(122)]
    raw = tmp_path/'data/raw/chemicalspace_new/package_00_pool.jsonl'
    raw.parent.mkdir(parents=True)
    raw.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    digest = store.file_info(raw)['sha256']
    store.write_json(raw.with_name('package_00_pool.meta.json'), {
        'manifest_schema_version': 1, 'source_database': 'chemicalspace_new', 'source_collection': 'compounds',
        'package_name': 'package_00', 'output_filename': raw.name, 'extraction_integrity_status': 'passed',
        'sha256': digest, 'actual_extracted_record_count': 122, 'requested_record_count': 122})
    summary = dict(source='chemicalspace_new', package='package_00', sha256=digest,
                   rdkit_version=rdBase.rdkitVersion, representative_policy='first raw physical line per canonical structure')
    split = make_split(summary, rows, (100, 120), query_count=2)
    split_path = tmp_path/'split.json'; store.write_json(split_path, split)
    ident = store.identity(split, APPROVED); aid = store.digest(ident)
    directory = tmp_path/'data/processed/chemicalspace_new/package_00/fingerprints'/aid
    directory.mkdir(parents=True)
    arrays = {'queries': store.pack_fingerprints(vectors([[1, 2], [3]])),
              'index': store.pack_fingerprints(vectors([[1, 2], [1, 2], [3], [8]]*30))}
    for role, a in arrays.items(): np.save(directory/f'{role}.npy', a, allow_pickle=False)
    store.write_json(directory/'records.json', store.mapping(split))
    manifest = {'manifest_schema_version': 1, 'status': 'complete', 'artifact_id': aid,
        'identity': ident, **store.linkage(tmp_path, split_path, split), 'configuration': APPROVED,
        'storage': store.STORAGE, 'query_count': 2, 'largest_index_count': 120, 'index_sizes': [100, 120],
        'dimensions': {r: list(a.shape) for r, a in arrays.items()},
        'files': {name: store.file_info(directory/name) for name in ('queries.npy', 'index.npy', 'records.json')}}
    store.write_json(directory/'manifest.json', manifest)
    return directory, manifest


def test_run_roundtrip_fresh_truth_graph_files_and_unchanged_inputs(tmp_path, artifact):
    directory, _ = artifact
    before = {p.name: store.file_info(p) for p in directory.iterdir()}
    path, report = bench.run_comparison(tmp_path, directory, 100)
    assert report['passed'] and bench.load_comparison_run(path, tmp_path) == report
    assert report['experiment'] == 'nested parent scaling'  # metadata, despite package_00 name
    assert before == {p.name: store.file_info(p) for p in directory.iterdir()}
    scores = np.load(path/'exact_scores.npy', allow_pickle=False)
    assert scores.shape == (2, 100) and scores.dtype == np.float64
    assert (path/'index_hnsw.bin').is_file() and (path/'index_hnsw.bin.dat').is_file()
    assert report['hnsw_preparation']['build_count'] == 1
    assert len(report['benchmark']['measured_results']) == 2*4*5*3
    for b in ('rdkit', 'fpsim2'):
        assert all(v['top_k']['mean'] == 1 for v in report['quality']['by_backend'][b].values())
    assert all(s['system_ram_total_bytes'] > 0 and s['process_rss_bytes'] > 0 for s in report['memory_snapshots'])


@pytest.mark.parametrize('damage', ['raw', 'selection', 'unregistered', 'artifact_checksum'])
def test_invalid_inputs_stop_before_backend_preparation(tmp_path, artifact, monkeypatch, damage):
    directory, _ = artifact
    monkeypatch.setattr(hnsw, 'prepare_backend', lambda *a, **k: pytest.fail('prepared invalid input'))
    if damage == 'raw':
        (tmp_path/'data/raw/chemicalspace_new/package_00_pool.jsonl').write_text('bad\n')
    if damage == 'artifact_checksum':
        (directory/'index.npy').write_bytes(b'bad')
    with pytest.raises(ValueError):
        bench.run_comparison(tmp_path, directory, 99 if damage == 'unregistered' else 100,
                             expected_selection={'source': 'wrong'} if damage == 'selection' else None)
    assert not (tmp_path/'experiments/results/hnsw').exists()


def test_full_nonzero_guard_failure_preserves_report_before_any_backend(tmp_path, artifact, monkeypatch):
    directory, manifest = artifact
    index = np.load(directory/'index.npy'); index[99] = 0
    np.save(directory/'index.npy', index, allow_pickle=False)
    manifest['files']['index.npy'] = store.file_info(directory/'index.npy')
    (directory/'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(store, 'unpack_fingerprints', lambda *a: pytest.fail('reconstructed before guard'))
    with pytest.raises(bench.ComparisonFailure) as caught:
        bench.run_comparison(tmp_path, directory, 100)
    report = bench.load_comparison_run(caught.value.report_path, tmp_path)
    assert report['nonzero_validation']['summary']['index_rows_checked'] == 100
    assert report['nonzero_validation']['affected_records'][0]['row_index'] == 99
    assert report['benchmark'] is None and report['latency_comparisons'] is None


def test_failed_preflight_never_times_or_plots_and_retains_checks(tmp_path, artifact, monkeypatch):
    original = hnsw.verify_queries
    def fail(*a, **k):
        result = original(*a, **k)
        if len(a[2]) != 2:  # Leave the separately required synthetic gate intact.
            return result
        result['checks'][0]['native_results'][0]['native_distance'] = .123
        # Produce actual validator evidence, not a fabricated inconsistent flag.
        c = result['checks'][0]
        response_ = response([r['search_id'] for r in c['native_results']],
                             [r['native_distance'] for r in c['native_results']])
        result['checks'][0] = {'query_position': c['query_position'], **hnsw.check_response(
            response_, a[2][c['query_position']], a[3], c['k'], a[4])}
        result['passed'] = False
        return result
    monkeypatch.setattr(hnsw, 'verify_queries', fail)
    monkeypatch.setattr(bench, 'benchmark_three', lambda *a, **k: pytest.fail('timed after failed preflight'))
    with pytest.raises(bench.ComparisonFailure) as caught:
        bench.run_comparison(tmp_path, artifact[0], 100,
            report_writer=lambda *a: pytest.fail('plotted comparative speed after failure'))
    report = bench.load_comparison_run(caught.value.report_path, tmp_path)
    assert not report['passed'] and report['preflight']['hnsw']['checks'][0]['native_results']
    assert report['benchmark'] is None and report['latency_comparisons'] is None


@pytest.mark.parametrize('damage', ['score', 'recall', 'order', 'designated', 'timing', 'identity', 'prefix'])
def test_saved_evidence_corruption_rejected_even_with_rehashed_files(tmp_path, artifact, damage):
    path, report = bench.run_comparison(tmp_path, artifact[0], 100)
    filename = 'report.json'
    if damage == 'score':
        filename = 'exact_scores.npy'
        np.save(path/filename, np.zeros((2, 100), dtype=np.float32), allow_pickle=False)
    else:
        if damage == 'recall': report['quality']['top_k_rows'][0]['recall'] = -1
        if damage == 'order': report['benchmark']['query_orders'][0]['query_positions'].reverse()
        if damage == 'designated': report['benchmark']['designated']['hnsw']['10'].reverse()
        if damage == 'timing': report['benchmark']['raw_timings']['hnsw'][0]['query_position'] = 9
        if damage == 'identity': report['benchmark']['measured_results'][0]['check']['native_results'][0]['index_position'] = 99
        if damage == 'prefix': report['selection_metadata']['prefix_identity_sha256']['100'] = 'wrong'
        (path/filename).write_text(json.dumps(report))
    with pytest.raises(ValueError, match='checksum'):
        bench.load_comparison_run(path)
    marker = json.loads((path/'manifest.json').read_text())
    marker['files'][filename] = store.file_info(path/filename)
    (path/'manifest.json').write_text(json.dumps(marker))
    with pytest.raises(ValueError): bench.load_comparison_run(path)


def test_failed_writer_retains_staging_and_never_publishes(tmp_path, artifact):
    def fail(*a): raise OSError('injected plot failure')
    with pytest.raises(OSError, match='injected plot failure'):
        bench.run_comparison(tmp_path, artifact[0], 100, report_writer=fail)
    parent = tmp_path/'experiments/results/hnsw/chemicalspace_new/package_00'/artifact[1]['artifact_id']/'index_100'
    assert list(parent.glob('.*.staging-*')) and not list(parent.glob('run_*'))


def test_completed_run_cannot_be_overwritten(tmp_path, artifact, monkeypatch):
    class Fixed:
        hex = 'fixed-publication-test'
    monkeypatch.setattr(bench, 'uuid4', Fixed)
    path, _ = bench.run_comparison(tmp_path, artifact[0], 100)
    before = {p.name: store.file_info(p) for p in path.iterdir()}
    with pytest.raises(FileExistsError, match='Publication conflict'):
        bench.run_comparison(tmp_path, artifact[0], 100)
    assert before == {p.name: store.file_info(p) for p in path.iterdir()}


def test_reporting_uses_one_run_and_omits_failed_speed_charts(tmp_path, artifact):
    from chemtellect.hnsw_reporting import tables, write_charts
    path, report = bench.run_comparison(tmp_path, artifact[0], 100, report_writer=write_charts)
    restored = bench.load_comparison_run(path, tmp_path)
    assert len(restored['charts']) == 6
    assert len(tables(restored)['Sequential latency']) == 12
    assert len(tables(restored)['Top-k-limited threshold recall']) == 48
    assert write_charts(tmp_path, {'passed': False}) == []
