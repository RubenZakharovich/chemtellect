"""Gated FPSim2/RDKit comparison against one explicit saved exact run."""
import json
from pathlib import Path
import random
import tempfile
from time import perf_counter_ns
from uuid import uuid4

import numpy as np
from rdkit import DataStructs

from chemtellect import exact_retrieval as exact
from chemtellect import fingerprint_store as store
from chemtellect import fpsim2_adapter as adapter
from chemtellect.benchmark_context import memory_snapshot
from chemtellect.fingerprints import APPROVED, validate_nonzero_fingerprints

FPSIM2_STAGES = ('native_search_ns', 'normalization_ns', 'retrieval_total_ns')


class CompatibilityFailure(RuntimeError):
    def __init__(self, path, report):
        self.report_path, self.report = path, report
        super().__init__(f'FPSim2 compatibility/comparison failed; preserved evidence: {path}')


def verify_ground_truth(root, path, artifact_path, manifest, records, index_size):
    root, path, artifact_path = map(lambda p: Path(p).resolve(), (root, path, artifact_path))
    path.relative_to(root)
    saved = exact.load_run(path)
    md = saved['metadata']
    if path.name != f"run_{md['run_id']}":
        raise ValueError('Explicit exact run directory does not match recorded run ID')
    for key in ('source', 'package', 'artifact_id', 'split_id', 'split_path',
                'split_file_sha256', 'raw_sha256_inherited'):
        if md['linkage'][key] != manifest[key]:
            raise ValueError(f'Exact-run/artifact linkage mismatch: {key}')
    if (md['linkage']['artifact_path'] != str(artifact_path.relative_to(root))
            or md['linkage']['artifact_manifest'] != store.file_info(artifact_path/'manifest.json')
            or md['fingerprint_identity'] != manifest['identity']
            or md['index_size'] != index_size or md['query_count'] != manifest['query_count']
            or {k: v for k, v in saved['records'].items() if k != 'linkage'} != records):
        raise ValueError('Exact-run selection/ordered identity linkage mismatch')
    return saved, {'path': str(path.relative_to(root)), 'run_id': md['run_id'],
                   'manifest': store.file_info(path/'manifest.json'),
                   'files': saved['manifest']['files']}


def measure_fpsim2(engine, query, k, id_to_position, clock=None):
    clock = perf_counter_ns if clock is None else clock
    start = clock()
    native = engine.top_k(query, k=k, threshold=0.0, n_workers=1)
    searched = clock()
    ordered = adapter.normalize_results(native, id_to_position)
    normalized = clock()
    end = clock()
    return native, ordered, dict(zip(FPSIM2_STAGES, (searched-start, normalized-searched, end-start)))


def _rdkit_response(positions, scores):
    response = np.empty(len(positions), dtype=[('mol_id', '<u4'), ('coeff', '<f8')])
    response['mol_id'], response['coeff'] = positions, scores[positions]
    return response


def paired_benchmark(query_vectors, index_vectors, engine, prepared_queries, reference,
                     id_to_position, records=None, candidate_counts=exact.CANDIDATE_COUNTS, clock=None):
    """Same permutation per backend/pass; alternate full-backend pass order.

    All validation and evaluation is outside timers and has no aggregate timer.
    Even warmup responses are checked. No score/candidate cache serves requests.
    """
    nq = len(query_vectors)
    exact.validate_scores(reference, (nq, len(index_vectors)))
    rng = random.Random(exact.TIMING_ORDER_SEED)
    raw = {'rdkit': [], 'fpsim2': []}
    designated = {b: {str(k): [None]*nq for k in candidate_counts} for b in raw}
    measured_results, changes, orders = [], [], []
    original_map = {i: i for i in range(len(index_vectors))}
    failure = None
    for k in candidate_counts:
        for pass_index in range(-1, exact.MEASURED_PASSES):
            query_order = list(range(nq)); rng.shuffle(query_order)
            backend_order = ['rdkit', 'fpsim2'] if pass_index < 0 or pass_index % 2 == 0 else ['fpsim2', 'rdkit']
            orders.append({'k': k, 'pass_index': pass_index, 'query_positions': query_order,
                           'backend_order': backend_order})
            for backend in backend_order:
                for order_position, q in enumerate(query_order):
                    if backend == 'rdkit':
                        if pass_index < 0:
                            scores = np.asarray(DataStructs.BulkTanimotoSimilarity(query_vectors[q], index_vectors), dtype=np.float64)
                            positions = exact._top_k(scores, k)
                        else:
                            scores, positions, ns = exact.measure_exact(query_vectors[q], index_vectors, k, clock=clock)
                        agrees = np.array_equal(scores, reference[q])
                        response = ordered = _rdkit_response(positions, scores)
                        mapping = original_map
                    else:
                        if pass_index < 0:
                            response = engine.top_k(prepared_queries[q], k=k, threshold=0.0, n_workers=1)
                            ordered = adapter.normalize_results(response, id_to_position)
                        else:
                            response, ordered, ns = measure_fpsim2(engine, prepared_queries[q], k, id_to_position, clock)
                        agrees, mapping = True, id_to_position
                    check = adapter.check_response(response, reference[q], mapping, k=k, records=records)
                    check['full_rdkit_row_agrees'] = agrees if backend == 'rdkit' else None
                    check['passed'] = check['passed'] and agrees
                    context = {'backend': backend, 'k': k, 'pass_index': pass_index,
                               'query_position': q, 'order_position': order_position,
                               'backend_order': backend_order}
                    if pass_index >= 0:
                        raw[backend].append({**context, **ns})
                        measured_results.append({**context, 'check': check,
                            'ordered_search_ids': [int(i) for i in ordered['mol_id']]})
                        if pass_index == 0:
                            designated[backend][str(k)][q] = check
                        else:
                            previous = designated[backend][str(k)][q]
                            if {r['search_id'] for r in previous['native_results']} != {int(i) for i in response['mol_id']}:
                                changes.append(context)
                    if not check['passed']:
                        failure = {**context, 'check': check}
                        break
                if failure: break
            if failure: break
        if failure: break
    summaries = None if failure else {b: exact.summarize_timings(
        rows, nq, candidate_counts, exact.STAGES if b == 'rdkit' else FPSIM2_STAGES) for b, rows in raw.items()}
    return {'passed': failure is None, 'failure': failure, 'raw_timings': raw,
            'designated': designated, 'measured_results': measured_results, 'query_orders': orders,
            'repeated_candidate_membership_changes': changes, 'summaries': summaries}


def evaluate_designated(designated, reference):
    """Reuse each designated candidate set for every threshold; never search here."""
    rows = []
    for backend in sorted(designated):
        by_k = designated[backend]
        for k in sorted(by_k, key=int):
            checks = by_k[k]
            for q, check in enumerate(checks):
                positions = {r['index_position'] for r in check['native_results']}
                for threshold in adapter.THRESHOLDS:
                    qualifying = set(exact.threshold_members(reference[q], threshold).tolist())
                    selected = positions & qualifying
                    rows.append({'backend': backend, 'k': int(k), 'query_position': q,
                        'threshold': threshold, 'qualifying_count': len(qualifying),
                        'designated_qualifying_count': len(selected),
                        'recall': len(selected)/len(qualifying) if qualifying else None,
                        'empty_qualifying_set': not qualifying})
    return {'definition': 'top-k-limited threshold recall using authoritative scores; empty-set recall is null',
            'designated_measured_pass_index': 0, 'rows': rows}


def verify_synthetic(path):
    """Native integration fixture: exact/near boundaries, collisions and zero overlap."""
    enabled = [list(range(n)) for n in (299, 300, 301, 499, 500, 501, 699, 700, 701, 899, 900, 901)]
    enabled += [[2047], [2047], [0, 7, 8, 63, 64, 127, 128, 2047]]
    index_bits = [enabled[i % len(enabled)] for i in range(100)]
    query_bits = [list(range(1000)), [2047]]
    def packed(groups):
        vectors = []
        for group in groups:
            fp = DataStructs.ExplicitBitVect(2048); fp.SetBitsFromList(group); vectors.append(fp)
        return store.pack_fingerprints(vectors), vectors
    queries, qv = packed(query_bits); index, iv = packed(index_bits)
    records = {role: [dict(row_index=i, source_id=f'synthetic:{role}:{i}', line_number=i+1)
                      for i in range(n)] for role, n in [('queries', 2), ('index', 100)]}
    validate_nonzero_fingerprints(queries, index, records)
    ids = [1009 + 17*i for i in range(100)]
    engine, prepared, mapping, representation = adapter.prepare_backend(path, queries, index, APPROVED, ids)
    reference = np.asarray([DataStructs.BulkTanimotoSimilarity(q, iv) for q in qv], dtype=np.float64)
    result = adapter.verify_queries(engine, prepared, reference, mapping, records['index'])
    return {**result, 'representation': representation, 'definition':
            'nonzero synthetic saved bits; nested counts straddle thresholds by 0.001, exact decimal boundaries, duplicate bits and disjoint rows'}


def load_comparison_run(path, root=None):
    """Validate immutable file inventory/checksums and success/failure linkage."""
    path = Path(path)
    marker = json.loads((path/'manifest.json').read_text())
    if marker.get('schema_version') != 1 or marker.get('status') not in ('complete', 'failed_compatibility'):
        raise ValueError('Unpublished comparison run')
    for name, info in marker['files'].items():
        if Path(name).name != name or store.file_info(path/name) != info:
            raise ValueError(f'Comparison file/checksum mismatch: {name}')
    if set(marker['files']) != {p.name for p in path.iterdir() if p.is_file() and p.name != 'manifest.json'}:
        raise ValueError('Comparison file inventory mismatch')
    report = json.loads((path/'report.json').read_text())
    if report['run_id'] != marker['run_id'] or report['linkage'] != marker['linkage']:
        raise ValueError('Comparison report linkage mismatch')
    records = json.loads((path/'records.json').read_text())
    if records['linkage'] != report['linkage']:
        raise ValueError('Comparison identity linkage mismatch')
    artifact = report['linkage']['artifact']
    if (store.digest(report['fingerprint_identity']) != artifact['artifact_id']
            or report['fingerprint_identity']['split_id'] != artifact['split_id']
            or any(records[k] != artifact[k] or report[k] != artifact[k] for k in ('source', 'package'))):
        raise ValueError('Comparison fingerprint/source identity mismatch')
    if any(len(records[role]) != report[count] for role, count in [('queries', 'query_count'), ('index', 'index_size')]):
        raise ValueError('Comparison selected identity counts mismatch')
    if report['passed'] != (marker['status'] == 'complete'):
        raise ValueError('Comparison completion/gate status mismatch')
    if report['passed'] and (not report['synthetic']['passed'] or not report['compatibility']['passed']
                             or not report['benchmark']['passed']):
        raise ValueError('Completed comparison has failed required checks')
    if root is not None:
        truth_path = Path(root)/report['linkage']['exact_run']['path']
        truth = exact.load_run(truth_path)
        if (store.file_info(truth_path/'manifest.json') != report['linkage']['exact_run']['manifest']
                or truth['manifest']['files'] != report['linkage']['exact_run']['files']
                or truth['metadata']['linkage'] != artifact
                or truth['records']['queries'] != records['queries'] or truth['records']['index'] != records['index']):
            raise ValueError('Comparison saved exact-run linkage mismatch')
        if report['compatibility']:
            for check in report['compatibility']['checks']:
                _recheck_saved(check, truth['scores'][check['query_position']], records['index'])
        if report['passed']:
            benchmark = report['benchmark']
            for item in benchmark['measured_results']:
                _recheck_saved(item['check'], truth['scores'][item['query_position']], records['index'])
            if evaluate_designated(benchmark['designated'], truth['scores']) != report['threshold_evaluations']:
                raise ValueError('Designated threshold evaluations differ')
    if report['passed']:
        benchmark = report['benchmark']
        for backend, rows in benchmark['raw_timings'].items():
            stages = exact.STAGES if backend == 'rdkit' else FPSIM2_STAGES
            if any(any(type(r[s]) is not int or r[s] < 0 for s in stages)
                   or r['retrieval_total_ns'] < sum(r[s] for s in stages[:-1]) for r in rows):
                raise ValueError('Invalid comparative timing stages')
            if exact.summarize_timings(rows, report['query_count'], stages=stages) != benchmark['summaries'][backend]:
                raise ValueError('Comparative timing summary mismatch')
    return report


def _recheck_saved(check, reference, records):
    if check['malformed'] is not None:
        if check['passed']:
            raise ValueError('Malformed saved response marked passed')
        return
    response = np.array([(r['search_id'], float(r['nonfinite']) if r['native_score'] is None else r['native_score'])
                         for r in check['native_results']], dtype=[('mol_id', '<i8'), ('coeff', '<f8')])
    mapping = {r['search_id']: r['index_position'] for r in check['native_results'] if r['index_position'] is not None}
    restored = adapter.check_response(response, reference, mapping, k=check['k'],
                                      threshold=check['threshold'], records=records)
    # Extra context fields are kept separately from the scientific check.
    if any(check[key] != value for key, value in restored.items()):
        raise ValueError('Persisted response diagnostics differ from authoritative scores')


def run_comparison(root, artifact_path, exact_run_path, index_size, *, expected_selection=None, report_writer=None):
    """Create new preparation files and evidence only; failure cannot claim speed."""
    root, artifact_path = Path(root).resolve(), Path(artifact_path).resolve()
    started = store.utc()
    snapshots = [memory_snapshot('before_loading')]
    manifest, queries, index, records, timings = exact.load_selection(root, artifact_path, index_size)
    selected = {k: manifest[k] for k in ('source', 'package', 'query_count')}
    selected['index_size'] = index_size
    if expected_selection is not None and selected != expected_selection:
        raise ValueError(f'Unexpected selection: {selected}')
    t = perf_counter_ns()
    truth, truth_link = verify_ground_truth(root, exact_run_path, artifact_path, manifest, records, index_size)
    timings['exact_run_loading_and_complete_verification'] = perf_counter_ns()-t
    snapshots.append(memory_snapshot('verified_inputs_and_ground_truth'))
    t = perf_counter_ns()
    nonzero = validate_nonzero_fingerprints(queries, index, records)
    timings['selected_nonzero_validation'] = perf_counter_ns()-t
    link = {'artifact': truth['metadata']['linkage'], 'exact_run': truth_link,
            'query_count': len(queries), 'index_size': len(index)}
    parent = root/'experiments/results/fpsim2'/manifest['source']/manifest['package']/manifest['artifact_id']
    parent.mkdir(parents=True, exist_ok=True)
    run_id = uuid4().hex
    final = parent/f'run_{run_id}'
    staging = Path(tempfile.mkdtemp(prefix=f'.run_{run_id}.staging-', dir=parent))
    report = {'schema_version': 1, 'run_id': run_id, 'started_utc': started, 'passed': False,
        'linkage': link, **selected, 'environment': {**store.environment(root), **adapter.versions()},
        'hardware': exact._hardware(), 'implementation_files': adapter.implementation_evidence(),
        'nonzero_validation': nonzero, 'fingerprint_identity': manifest['identity'],
        'timings_ns': timings, 'memory_snapshots': snapshots, 'synthetic': None,
        'representation': None, 'compatibility': None, 'benchmark': None, 'error': None,
        'settings': {'candidate_counts': list(exact.CANDIDATE_COUNTS), 'thresholds': list(adapter.THRESHOLDS),
            'top_k_threshold': 0.0, 'n_workers': 1, 'rdkit_execution': 'sequential', 'concurrent_queries': 1,
            'timing_order_seed': 42, 'warmup_passes': 1, 'measured_passes': 5,
            'percentile_method': 'linear', 'p95_definition': 'p95 of per-query median latency',
            'timing_units': 'ns', 'query_preparation': 'verified saved bits reconstructed/converted before retrieval',
            'normalization': 'descending native score then original index position; no reference lookup',
            'native_search_boundary': 'inherited FPSim2Engine.top_k including bounds/dispatch/native C++ search; prepared query bypass',
            'primary_latency': 'retrieval_total_ns: search and required selection/normalization',
            'outside_timer_overhead': 'validation/evaluation excluded; no separate aggregate timing recorded',
            'scope': 'repeated in-memory sequential benchmark; not cold-start or production concurrency',
            'memory': 'synchronous system RAM and process RSS snapshots; no background polling; not true peak memory'},
        'charts': []}
    try:
        t = perf_counter_ns()
        report['synthetic'] = verify_synthetic(staging/'synthetic_fpsim2.h5')
        timings['synthetic_preparation_and_compatibility'] = perf_counter_ns()-t
        t = perf_counter_ns()
        engine, prepared, mapping, representation = adapter.prepare_backend(
            staging/'index_fpsim2.h5', queries, index, manifest['configuration'])
        report['representation'] = representation
        with (staging/'prepared_queries.npy').open('xb') as stream:
            np.save(stream, prepared, allow_pickle=False)
        if not np.array_equal(np.load(staging/'prepared_queries.npy', allow_pickle=False), prepared):
            raise ValueError('Saved native queries differ on reload')
        timings['fpsim2_conversion_database_writing_loading_and_round_trip_verification'] = perf_counter_ns()-t
        snapshots.append(memory_snapshot('fpsim2_prepared'))
        t = perf_counter_ns()
        qv, iv = store.unpack_fingerprints(queries), store.unpack_fingerprints(index)
        timings['rdkit_reconstruction'] = perf_counter_ns()-t
        snapshots.append(memory_snapshot('both_backends_prepared'))
        t = perf_counter_ns()
        report['compatibility'] = adapter.verify_queries(engine, prepared, truth['scores'], mapping, records['index'])
        timings['real_preflight_compatibility'] = perf_counter_ns()-t
        if report['synthetic']['passed'] and report['compatibility']['passed']:
            report['benchmark'] = paired_benchmark(qv, iv, engine, prepared, truth['scores'], mapping, records['index'])
            if report['benchmark']['passed']:
                report['threshold_evaluations'] = evaluate_designated(report['benchmark']['designated'], truth['scores'])
                report['passed'] = True
        snapshots.append(memory_snapshot('after_gate_and_optional_benchmark'))
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
    # Report generation never runs inside retrieval timers. Failure plots must
    # not show comparative speed; the notebook writer follows this contract.
    store.write_json(staging/'records.json', {**records, 'linkage': link})
    if report_writer is not None:
        report['charts'] = report_writer(staging, report)
        if any(Path(n).name != n or Path(n).suffix not in ('.png', '.svg') for n in report['charts']):
            raise ValueError('Report writer must return simple PNG/SVG filenames')
    report['finished_utc'] = store.utc()
    t = perf_counter_ns()
    # Assembly/serialization/write time of the report is measured in the marker,
    # avoiding a self-referential duration inside the report being written.
    store.write_json(staging/'report.json', report)
    writing_ns = perf_counter_ns()-t
    if json.loads((staging/'report.json').read_text()) != report:
        raise ValueError('Comparison report round trip differs')
    files = {p.name: store.file_info(p) for p in staging.iterdir() if p.is_file()}
    for name, info in files.items():
        if store.file_info(staging/name) != info:
            raise ValueError('Comparison output changed before completion')
    marker = {'schema_version': 1, 'status': 'complete' if report['passed'] else 'failed_compatibility',
        'run_id': run_id, 'linkage': link, 'files': files, 'report_serialization_writing_ns': writing_ns}
    store.write_json(staging/'manifest.json', marker)
    load_comparison_run(staging, root)
    if final.exists():
        raise FileExistsError(f'Publication conflict: {final}')
    staging.rename(final)
    if not report['passed']:
        raise CompatibilityFailure(final, report)
    return final, report
