import json
import random

import numpy as np
import pytest
from rdkit import DataStructs

from chemtellect import exact_retrieval as exact
from chemtellect import fingerprint_store as store
from chemtellect import fpsim2_adapter as adapter
from chemtellect import fpsim2_benchmark as bench
from chemtellect.fingerprints import APPROVED, validate_nonzero_fingerprints
from test_exact_retrieval import artifact, vectors


def response(ids, scores):
    return np.array(list(zip(ids, scores)), dtype=[('mol_id', '<i8'), ('coeff', '<f8')])


def test_actual_backend_words_sorting_ids_and_zero_neighbors(tmp_path, monkeypatch):
    enabled = [[0, 7, 8, 31, 32, 63, 64, 65, 127, 128, 2047], [2046], [0, 63], [0, 63]]
    fp = vectors(enabled)
    packed = store.pack_fingerprints(fp)
    records = {role: [dict(row_index=i, source_id=f'{role}:{i}', line_number=i+1)
                      for i in range(n)] for role, n in [('queries', 1), ('index', 4)]}
    validate_nonzero_fingerprints(packed[:1], packed, records)
    # Any invocation of FPSim2 molecular generation or parsing must fail.
    import FPSim2.base as base
    import FPSim2.io.chem as chem
    def forbidden(*a, **k): pytest.fail('molecular processing invoked')
    monkeypatch.setattr(base, 'build_fp', forbidden)
    monkeypatch.setattr(base, 'load_molecule', forbidden)
    monkeypatch.setattr(chem, 'build_fp', forbidden)
    engine, queries, mapping, representation = adapter.prepare_backend(
        tmp_path/'native.h5', packed[:1], packed, APPROVED, [901, 3, 42, 88])
    native = adapter.packed_to_native(packed, [901, 3, 42, 88])
    assert int(native[0, 1]) & (1 << 63)  # bit 0 -> word 0 MSB
    assert int(native[0, 1]) & 1  # bit 63 -> word 0 LSB
    assert int(native[0, 2]) & (1 << 63)  # bit 64 -> word 1 MSB
    assert int(native[0, 32]) & 1  # bit 2047 -> word 31 LSB
    np.testing.assert_array_equal(adapter.native_to_packed(native), packed)
    assert representation['internal_row_search_ids'] != [901, 3, 42, 88]
    assert queries[0].dtype == np.uint64 and not queries.flags.writeable
    result = engine.top_k(queries[0], k=4, threshold=0.0, n_workers=1)
    assert set(result['mol_id']) == {901, 3, 42, 88}
    assert result['coeff'].dtype == np.float32
    assert result[result['mol_id'] == 3]['coeff'].tolist() == [0.]
    reference = np.array(DataStructs.BulkTanimotoSimilarity(fp[0], fp))
    check = adapter.check_response(result, reference, mapping, k=4)
    assert check['passed'] and check['actual_count'] == 4
    # Both identical fingerprints are distinct results, after internal reordering.
    assert mapping[42] == 2 and mapping[88] == 3
    first = engine.top_k(queries[0], k=2, threshold=0.0, n_workers=1)
    assert adapter.check_response(first, reference, mapping, k=2)['passed']
    bad = native.copy(); bad[0, -1] += 1
    with pytest.raises(ValueError, match='population'):
        adapter.native_to_packed(bad)
    with pytest.raises(FileExistsError):
        adapter.prepare_backend(tmp_path/'native.h5', packed[:1], packed, APPROVED)


@pytest.mark.parametrize('ids', [[1, 1], [True, 2], [1., 2], [-1, 2], [2**32, 1], [1]])
def test_bad_search_ids(ids):
    with pytest.raises(ValueError): adapter.validate_ids(ids, 2)


def test_actual_synthetic_gate_all_thresholds(tmp_path):
    result = bench.verify_synthetic(tmp_path/'synthetic.h5')
    assert result['passed'] and len(result['checks']) == 16
    assert result['maximum_native_score_error'] <= 1e-6
    assert any(c['k'] == 100 and any(r['native_score'] == 0 for r in c['native_results'])
               for c in result['checks'])
    for threshold in adapter.THRESHOLDS:
        check = next(c for c in result['checks'] if c['query_position'] == 0 and c['threshold'] == threshold)
        assert any(r['rdkit_reference_score'] == threshold for r in check['native_results'])
        assert not check['missing_positions'] and not check['unexpected_positions']


def test_exact_ties_allow_alternatives_without_epsilon():
    reference = np.array([1., .5, .5, np.nextafter(.5, 0), 0])
    mapping = {10*i+7: i for i in range(5)}
    assert adapter.check_response(response([7, 27], [1., .5]), reference, mapping, k=2)['passed']
    assert adapter.check_response(response([7, 17], [1., .5]), reference, mapping, k=2)['passed']
    below = adapter.check_response(response([7, 37], [1., .5]), reference, mapping, k=2)
    assert not below['passed'] and below['unexpected_positions'] == [3]
    missing = adapter.check_response(response([17, 27], [.5, .5]), reference, mapping, k=2)
    assert not missing['passed'] and missing['missing_positions'] == [0]


@pytest.mark.parametrize('threshold', adapter.THRESHOLDS)
def test_threshold_near_boundaries_membership_is_not_tolerant(threshold):
    reference = np.array([np.nextafter(threshold, 0), threshold, np.nextafter(threshold, 1)])
    mapping = {i: i for i in range(3)}
    good = adapter.check_response(response([1, 2], [threshold, threshold]), reference, mapping, threshold=threshold)
    assert good['passed']
    bad = adapter.check_response(response([0, 1, 2], [threshold]*3), reference, mapping, threshold=threshold)
    assert not bad['passed'] and bad['unexpected_positions'] == [0]
    assert bad['maximum_native_score_error'] < adapter.ATOL
    assert adapter.check_response(response([], []), np.array([0., .1, .2]), mapping, threshold=threshold)['passed']


@pytest.mark.parametrize('result', [response([0, 0], [1, 1]), response([0, 9], [1, .5]),
    response([0], [1]), response([0, 1, 2], [1, .5, 0]), response([0, 1], [np.nan, .5]),
    response([0, 1], [np.inf, .5]), response([0, 1], [1.1, .5]), response([-1, 1], [1, .5]),
    np.ones(2), np.zeros((1, 2)), np.array([(0., .5)], dtype=[('mol_id', 'f8'), ('coeff', 'f8')])])
def test_malformed_responses(result):
    report = adapter.check_response(result, np.array([1., .5, 0]), {i: i for i in range(3)}, k=2)
    assert not report['passed']
    json.dumps(report, allow_nan=False)


def test_native_score_errors_do_not_change_membership():
    result = adapter.check_response(response([0], [.5+2e-6]), np.array([.5]), {0: 0}, k=1)
    assert not result['passed'] and result['maximum_native_score_error'] > 1e-6
    assert result['missing_positions'] == result['unexpected_positions'] == []


def test_distinct_search_ids_cannot_alias_one_index_identity():
    check = adapter.check_response(response([5, 9], [.5, .5]), np.array([.5, .5]), {5: 0, 9: 0}, k=2)
    assert not check['passed']


def test_controlled_fpsim2_clock_includes_normalization():
    calls = []
    class Engine:
        def top_k(self, query, **kwargs):
            calls.append((query, kwargs))
            return response([42, 8], [.5, .5])
    ticks = iter([10, 30, 40, 45])
    native, ordered, ns = bench.measure_fpsim2(Engine(), 'prepared', 2, {42: 1, 8: 0}, lambda: next(ticks))
    assert calls == [('prepared', dict(k=2, threshold=0., n_workers=1))]
    assert native['mol_id'].tolist() == [42, 8] and ordered['mol_id'].tolist() == [8, 42]
    assert ns == dict(native_search_ns=20, normalization_ns=10, retrieval_total_ns=35)


def test_paired_orchestration_same_orders_no_global_rng_and_designation(tmp_path):
    packed = store.pack_fingerprints(vectors([[1], [1, 2], [3], [3]]))
    engine, prepared, mapping, _ = adapter.prepare_backend(tmp_path/'db.h5', packed[:2], packed, APPROVED)
    qv, iv = store.unpack_fingerprints(packed[:2]), store.unpack_fingerprints(packed)
    truth = np.array([DataStructs.BulkTanimotoSimilarity(q, iv) for q in qv])
    state = random.getstate()
    result = bench.paired_benchmark(qv, iv, engine, prepared, truth, mapping,
                                    candidate_counts=(2, 4), clock=lambda: 0)
    assert result['passed'] and random.getstate() == state
    assert len(result['measured_results']) == 2*2*5*2
    for order in result['query_orders']:
        if order['pass_index'] >= 0:
            assert order['backend_order'][0] == ('rdkit' if order['pass_index'] % 2 == 0 else 'fpsim2')
            for backend in ('rdkit', 'fpsim2'):
                requests = [r for r in result['raw_timings'][backend]
                            if r['k'] == order['k'] and r['pass_index'] == order['pass_index']]
                assert [r['query_position'] for r in requests] == order['query_positions']
    evaluation = bench.evaluate_designated(result['designated'], truth)
    assert len(evaluation['rows']) == 2*2*2*4
    assert all(r['recall'] in (0., 1., None) for r in evaluation['rows'])


def test_repeated_boundary_alternatives_are_recorded_without_pooling():
    class Alternate:
        calls = 0
        def top_k(self, query, **kwargs):
            self.calls += 1
            return response([0, 1 if self.calls % 2 else 2], [1., .5])
    qv, iv = vectors([[1]]), vectors([[1], [1, 2], [1, 3]])
    reference = np.array([[1., .5, .5]])
    result = bench.paired_benchmark(qv, iv, Alternate(), [None], reference,
                                    {0: 0, 1: 1, 2: 2}, candidate_counts=(2,), clock=lambda: 0)
    assert result['passed']
    assert result['repeated_candidate_membership_changes']
    first = result['designated']['fpsim2']['2'][0]
    assert {r['index_position'] for r in first['native_results']} == {0, 2}
    assert len(first['native_results']) == 2


def test_measured_gate_rejects_short_response_after_preflight():
    class Short:
        calls = 0
        def top_k(self, query, **kwargs):
            self.calls += 1
            return response([0, 1], [1., .5]) if self.calls == 1 else response([0], [1.])
    qv, iv = vectors([[1]]), vectors([[1], [1, 2]])
    result = bench.paired_benchmark(qv, iv, Short(), [None], np.array([[1., .5]]),
                                    {0: 0, 1: 1}, candidate_counts=(2,), clock=lambda: 0)
    assert not result['passed'] and result['summaries'] is None
    assert result['failure']['check']['actual_count'] == 1


def test_existing_exact_run_has_missing_metadata_and_remains_readable():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    path = root/'experiments/results/exact_retrieval/chemicalspace_new/package_00/628c36671feeb38e1a97a34dd664bb54f43851374fcea4190a474479a12df427/run_1494eeec3c0f489c9b67f7352dc573b7'
    if not path.exists():
        pytest.skip('local legacy artifact not distributed with repository')
    saved = exact.load_run(path)
    assert 'rdkit_execution' not in saved['metadata']['timing']
    assert not isinstance(saved['metadata']['memory'], dict)


def test_new_exact_run_chart_and_snapshot_roundtrip(tmp_path, artifact):
    # Minimal writer fixture exercises artifact publication; actual charts are
    # rendered in notebook execution and visually inspected separately.
    def writer(directory, saved):
        assert isinstance(saved['metadata']['memory'], dict)
        (directory/'chart.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
        return ['chart.svg']
    path, metadata, _ = exact.run_benchmark(tmp_path, artifact[0], 100, report_writer=writer)
    restored = exact.load_run(path)
    assert 'chart.svg' in restored['manifest']['files']
    assert metadata['timing']['outside_timer_overhead'].endswith('aggregate timing recorded')


def test_comparison_roundtrip_provenance_and_memory(tmp_path, artifact):
    directory, manifest = artifact
    truth_path, _, _ = exact.run_benchmark(tmp_path, directory, 100)
    historical_before = {p.name: store.file_info(p) for p in truth_path.iterdir()}
    path, report = bench.run_comparison(tmp_path, directory, truth_path, 100)
    saved = bench.load_comparison_run(path, tmp_path)
    assert saved == report and saved['passed']
    assert report['linkage']['exact_run']['run_id'] == truth_path.name.removeprefix('run_')
    assert report['compatibility']['query_count'] == 2
    assert len(report['compatibility']['checks']) == 16
    assert len(report['benchmark']['measured_results']) == 80
    assert historical_before == {p.name: store.file_info(p) for p in truth_path.iterdir()}
    for snapshot in report['memory_snapshots']:
        assert snapshot['system_ram_total_bytes'] > 0 and snapshot['system_ram_available_bytes'] >= 0
        assert snapshot['process_rss_bytes'] > 0 and snapshot['sampling_duration_ns'] >= 0
    prepared = np.load(path/'prepared_queries.npy', allow_pickle=False)
    np.testing.assert_array_equal(adapter.native_to_packed(prepared), np.load(directory/'queries.npy'))
    with (path/'report.json').open('ab') as stream: stream.write(b' ')
    with pytest.raises(ValueError, match='checksum'):
        bench.load_comparison_run(path)


def test_linkage_failure_stops_before_preparation(tmp_path, artifact, monkeypatch):
    directory, manifest = artifact
    truth_path, _, _ = exact.run_benchmark(tmp_path, directory, 100)
    marker_before = store.file_info(truth_path/'manifest.json')
    (truth_path/'exact_scores.npy').write_bytes(b'bad')
    monkeypatch.setattr(adapter, 'prepare_backend', lambda *a, **kw: pytest.fail('backend prepared'))
    with pytest.raises(ValueError, match='checksum'):
        bench.run_comparison(tmp_path, directory, truth_path, 100)
    assert store.file_info(truth_path/'manifest.json') == marker_before


def test_failure_evidence_without_timing_or_speed(tmp_path, artifact, monkeypatch):
    directory, _ = artifact
    truth_path, _, _ = exact.run_benchmark(tmp_path, directory, 100)
    original = adapter.verify_queries
    def fail(*args, **kwargs):
        result = original(*args, **kwargs)
        result['passed'] = False
        return result
    monkeypatch.setattr(adapter, 'verify_queries', fail)
    monkeypatch.setattr(bench, 'paired_benchmark', lambda *a, **kw: pytest.fail('timed after failed gate'))
    with pytest.raises(bench.CompatibilityFailure) as caught:
        bench.run_comparison(tmp_path, directory, truth_path, 100)
    report = bench.load_comparison_run(caught.value.report_path, tmp_path)
    assert not report['passed'] and report['benchmark'] is None
    assert json.loads((caught.value.report_path/'manifest.json').read_text())['status'] == 'failed_compatibility'
