"""Fixed HNSW / FPSim2 / fresh RDKit comparison, one selected artifact prefix per run."""
import json
from importlib.metadata import version
from pathlib import Path
import random
import tempfile
from time import perf_counter_ns
from uuid import uuid4

import numpy as np
import numexpr
from rdkit import DataStructs
from threadpoolctl import threadpool_info

from chemtellect import exact_retrieval as exact
from chemtellect import fingerprint_store as store
from chemtellect import fpsim2_adapter as fpsim2
from chemtellect import fpsim2_benchmark as paired
from chemtellect import hnsw_adapter as hnsw
from chemtellect.benchmark_context import memory_snapshot
from chemtellect.dataset_overview import dataset
from chemtellect.fingerprints import validate_nonzero_fingerprints, validate_configuration, NonzeroFingerprintError

BACKENDS = ('rdkit', 'fpsim2', 'hnsw')


class ComparisonFailure(RuntimeError):
    def __init__(self, path, report):
        self.report_path, self.report = path, report
        super().__init__(f'Three-backend validation failed; preserved evidence: {path}; {report.get("error")}')


def verify_raw(root, manifest):
    """Check current raw bytes against both acquisition metadata and split linkage."""
    item = dataset(root, manifest['source'], manifest['package'])
    raw = store.file_info(item.raw)
    acquisition = json.loads(item.manifest.read_text())
    expected = {'manifest_schema_version': 1, 'source_database': manifest['source'],
        'source_collection': 'compounds', 'package_name': manifest['package'],
        'output_filename': item.raw.name, 'extraction_integrity_status': 'passed',
        'sha256': raw['sha256']}
    if any(acquisition.get(key) != value for key, value in expected.items()):
        raise ValueError('Raw acquisition manifest/hash mismatch')
    with item.raw.open('rb') as stream:
        count = sum(1 for _ in stream)
    if (raw['sha256'] != manifest['raw_sha256_inherited']
            or type(acquisition.get('actual_extracted_record_count')) is not int
            or count != acquisition['actual_extracted_record_count']
            or count != acquisition.get('requested_record_count')):
        raise ValueError('Raw count/hash differs from acquisition manifest or saved split')
    return {'path': str(item.raw.relative_to(root)), **raw, 'raw_count': count,
            'acquisition_manifest': {'path': str(item.manifest.relative_to(root)), **store.file_info(item.manifest)}}


def tie_aware_recall(reference, positions, k):
    """Credit all strict winners and at most the available exact boundary-tie slots."""
    exact.validate_scores(reference, (len(reference),))
    exact._validate_k(k, len(reference))
    positions = list(positions)
    if (len(positions) != k or len(set(positions)) != k
            or any(type(p) is not int or not 0 <= p < len(reference) for p in positions)):
        raise ValueError('Recall requires exactly k unique valid original positions')
    boundary = float(np.partition(reference, len(reference)-k)[len(reference)-k])
    strict_count = int(np.count_nonzero(reference > boundary))
    returned = reference[positions]
    strict_hits = int(np.count_nonzero(returned > boundary))
    boundary_hits = int(np.count_nonzero(returned == boundary))
    credited_ties = min(k-strict_count, boundary_hits)
    return {'recall': (strict_hits+credited_ties)/k, 'boundary': boundary,
            'strict_winners': strict_count, 'strict_hits': strict_hits,
            'boundary_hits': boundary_hits, 'credited_boundary_hits': credited_ties}


def evaluate_designated(designated, reference):
    threshold = paired.evaluate_designated(designated, reference)
    rows = []
    summaries = {}
    for backend in sorted(designated):
        by_k = designated[backend]
        summaries[backend] = {}
        for key in sorted(by_k, key=int):
            checks = by_k[key]
            k = int(key)
            group = []
            for q, check in enumerate(checks):
                positions = [r['index_position'] for r in check['native_results']]
                row = {'backend': backend, 'k': k, 'query_position': q,
                       **tie_aware_recall(reference[q], positions, k)}
                group.append(row); rows.append(row)
            summaries[backend][key] = {
                'top_k': {'mean': float(np.mean([r['recall'] for r in group])),
                          'median': float(np.median([r['recall'] for r in group])),
                          'minimum': min(r['recall'] for r in group), 'query_count': len(group)},
                'thresholds': {}}
            for tau in fpsim2.THRESHOLDS:
                selected = [r for r in threshold['rows']
                            if r['backend'] == backend and r['k'] == k and r['threshold'] == tau]
                defined = [r['recall'] for r in selected if r['recall'] is not None]
                summaries[backend][key]['thresholds'][str(tau)] = {
                    'mean_defined_recall': float(np.mean(defined)) if defined else None,
                    'defined_queries': len(defined), 'empty_queries': len(selected)-len(defined),
                    'total_qualifying_records': sum(r['qualifying_count'] for r in selected)}
    return {'top_k_definition': '(strict-winner hits + min(boundary slots, exact-boundary hits)) / k',
        'ties': 'direct float64 equality; no epsilon', 'designated_measured_pass_index': 0,
        'top_k_rows': rows, 'threshold_evaluations': threshold, 'by_backend': summaries}


def backend_order(pass_index):
    if pass_index < 0:
        return list(BACKENDS)
    offset = pass_index % 3
    order = list(BACKENDS[offset:]+BACKENDS[:offset])
    return order[::-1] if pass_index >= 3 else order


def benchmark_three(qv, iv, fps_engine, fps_queries, fps_mapping, hns_engine, hns_queries,
                    hns_mapping, reference, records=None, *, candidate_counts=exact.CANDIDATE_COUNTS, clock=None):
    """Three explicit sequential paths, shared permutations; no cached timed search."""
    nq, ni = len(qv), len(iv)
    exact.validate_scores(reference, (nq, ni))
    candidate_counts = tuple(candidate_counts)
    if not nq or not candidate_counts or len(set(candidate_counts)) != len(candidate_counts):
        raise ValueError('Nonempty queries and distinct candidate counts required')
    for k in candidate_counts:
        exact._validate_k(k, ni)
    raw = {b: [] for b in BACKENDS}
    designated = {b: {str(k): [None]*nq for k in candidate_counts} for b in BACKENDS}
    measured, changes, orders = [], [], []
    identity_map = {i: i for i in range(ni)}
    rng = random.Random(exact.TIMING_ORDER_SEED)
    failure, validation_ns = None, 0
    for k in candidate_counts:
        for pass_index in range(-1, exact.MEASURED_PASSES):
            order = list(range(nq)); rng.shuffle(order)
            backends = backend_order(pass_index)
            orders.append({'k': k, 'pass_index': pass_index,
                           'query_positions': order, 'backend_order': backends})
            for backend in backends:
                for order_position, q in enumerate(order):
                    context = {'backend': backend, 'k': k, 'pass_index': pass_index,
                               'query_position': q, 'order_position': order_position}
                    try:
                        if backend == 'rdkit':
                            if pass_index < 0:
                                scores = np.asarray(DataStructs.BulkTanimotoSimilarity(qv[q], iv), dtype=np.float64)
                                positions = exact._top_k(scores, k)
                            else:
                                scores, positions, ns = exact.measure_exact(qv[q], iv, k, clock=clock)
                            t = perf_counter_ns()
                            response = paired._rdkit_response(positions, scores)
                            check = fpsim2.check_response(response, reference[q], identity_map, k=k, records=records)
                            check['full_rdkit_row_agrees'] = bool(np.array_equal(scores, reference[q]))
                            check['passed'] = check['passed'] and check['full_rdkit_row_agrees']
                            ordered_ids = positions.tolist()
                        elif backend == 'fpsim2':
                            if pass_index < 0:
                                response = fps_engine.top_k(fps_queries[q], k=k, threshold=0.0, n_workers=1)
                                ordered = fpsim2.normalize_results(response, fps_mapping)
                            else:
                                response, ordered, ns = paired.measure_fpsim2(
                                    fps_engine, fps_queries[q], k, fps_mapping, clock)
                            t = perf_counter_ns()
                            check = fpsim2.check_response(response, reference[q], fps_mapping, k=k, records=records)
                            ordered_ids = [int(i) for i in ordered['mol_id']]
                        else:
                            if pass_index < 0:
                                response = hns_engine.knnQuery(hns_queries[q], k=k)
                                ordered = hnsw.normalize_results(response, hns_mapping)
                            else:
                                response, ordered, ns = hnsw.measure_query(hns_engine, hns_queries[q], k, hns_mapping, clock)
                            t = perf_counter_ns()
                            check = hnsw.check_response(response, reference[q], hns_mapping, k, records)
                            ordered_ids = [int(i) for i in ordered[0]] if hnsw._well_formed(ordered) else None
                        item = {**context, 'check': check, 'ordered_search_ids': ordered_ids}
                        if pass_index >= 0:
                            raw[backend].append({**context, **ns})
                            measured.append(item)
                            if pass_index == 0:
                                designated[backend][str(k)][q] = check
                            elif check['passed']:
                                previous = designated[backend][str(k)][q]
                                if {r['search_id'] for r in previous['native_results']} != set(ordered_ids):
                                    changes.append(context)
                        validation_ns += perf_counter_ns()-t
                        if not check['passed']:
                            failure = item
                            break
                    except Exception as error:
                        failure = {**context, 'error': f'{type(error).__name__}: {error}'}
                        break
                if failure: break
            if failure: break
        if failure: break
    summaries = None if failure else {b: exact.summarize_timings(
        raw[b], nq, candidate_counts, exact.STAGES if b == 'rdkit' else hnsw.STAGES) for b in BACKENDS}
    return {'passed': failure is None, 'failure': failure, 'raw_timings': raw,
            'measured_results': measured, 'designated': designated, 'query_orders': orders,
            'repeated_candidate_membership_changes': changes,
            'validation_and_result_assembly_ns': validation_ns, 'summaries': summaries}


def latency_comparisons(summaries):
    rows = []
    for key in sorted(summaries['rdkit']['by_k'], key=int):
        totals = {b: summaries[b]['by_k'][key]['across_query_medians_ns']['retrieval_total_ns'] for b in BACKENDS}
        for b in BACKENDS:
            denominator = totals[b]['median']
            rows.append({'backend': b, 'k': int(key), **totals[b],
                'units': 'ns', 'rdkit_time_saved_ns': totals['rdkit']['median']-denominator,
                'rdkit_speedup': totals['rdkit']['median']/denominator if denominator else None,
                'fpsim2_time_saved_ns': totals['fpsim2']['median']-denominator,
                'fpsim2_speedup': totals['fpsim2']['median']/denominator if denominator else None})
    return rows


def _restore_check(check, backend, reference, mapping, records):
    if check['malformed'] is not None:
        if check['passed']:
            raise ValueError('Malformed saved response marked passed')
        return None
    if backend == 'hnsw':
        response = (np.asarray([r['search_id'] for r in check['native_results']], dtype=np.int64),
                    np.asarray([r['native_distance'] if r['native_distance'] is not None else float(r['nonfinite'])
                                for r in check['native_results']], dtype=np.float64))
        restored = hnsw.check_response(response, reference, mapping, check['k'], records)
        ordered_ids = hnsw.normalize_results(response, mapping)[0].tolist()
    else:
        response = np.asarray([(r['search_id'], r['native_score'] if r['native_score'] is not None else float(r['nonfinite']))
                               for r in check['native_results']], dtype=[('mol_id', '<i8'), ('coeff', '<f8')])
        restored = fpsim2.check_response(response, reference, mapping, k=check['k'], threshold=check['threshold'], records=records)
        ordered_ids = fpsim2.normalize_results(response, mapping)['mol_id'].tolist()
        if backend == 'rdkit':
            restored['full_rdkit_row_agrees'] = check.get('full_rdkit_row_agrees')
            restored['passed'] = restored['passed'] and restored['full_rdkit_row_agrees'] is True
    if any(check[key] != value for key, value in restored.items()):
        raise ValueError('Persisted response diagnostics differ from authoritative scores/identities')
    return ordered_ids


def _validate_payloads(path, report):
    if report.get('schema_version') not in (1, 2):
        raise ValueError('Unsupported comparison report schema')
    records = json.loads((path/'records.json').read_text())
    if records['linkage'] != report['linkage']:
        raise ValueError('Saved ordered identity linkage mismatch')
    link = report['linkage']
    if report.get('schema_version') == 2:
        selection = report['selection_metadata']
        expected_prefixes = {str(n): store.digest(records['index'][:n])
                             for n in selection['index_sizes'] if n <= report['index_size']}
        if (selection['query_identity_sha256'] != store.digest(records['queries'])
                or selection['prefix_identity_sha256'] != expected_prefixes
                or report['index_size'] not in selection['index_sizes']
                or report['experiment'] != ('nested parent scaling' if len(selection['index_sizes']) > 1
                                            else 'independent pilot')):
            raise ValueError('Saved selection metadata differs from ordered identities')
    validate_configuration(report['fingerprint_identity']['configuration'])
    if (store.digest(report['fingerprint_identity']) != link['artifact_id']
            or report['fingerprint_identity']['split_id'] != link['split_id']
            or report['configuration'] != hnsw.CONFIGURATION
            or json.loads((path/'configuration.json').read_text()) != hnsw.CONFIGURATION
            or any(records[key] != report[key] or report[key] != link[key] for key in ('source', 'package'))):
        raise ValueError('Saved fingerprint/HNSW configuration mismatch')
    for role, count in [('queries', report['query_count']), ('index', report['index_size'])]:
        if (len(records[role]) != count
                or [r['row_index'] for r in records[role]] != list(range(count))):
            raise ValueError('Saved original record ordering/count mismatch')
    if not (path/'exact_scores.npy').exists():
        if report['passed']:
            raise ValueError('Completed run lacks authoritative matrix')
        return
    scores = np.load(path/'exact_scores.npy', allow_pickle=False, mmap_mode='r')
    exact.validate_scores(scores, (report['query_count'], report['index_size']))
    if not (path/'ground_truth_top_k.json').exists() and not report['passed']:
        return
    top_k = json.loads((path/'ground_truth_top_k.json').read_text())
    expected = exact._results({'scores': scores, 'positions': {
        str(k): [exact.exact_top_k(row, k).tolist() for row in scores] for k in exact.CANDIDATE_COUNTS}}, records)
    if top_k != expected:
        raise ValueError('Saved top-k ground truth differs from authoritative scores')
    fps_mapping = dict(zip(report['fpsim2_representation']['internal_row_search_ids'],
                           report['fpsim2_representation']['internal_row_original_positions'])) if report['fpsim2_representation'] else {}
    hns_mapping = {i: p for p, i in enumerate(report['hnsw_preparation']['registered_ids'])} if report['hnsw_preparation'] else {}
    identity = {i: i for i in range(report['index_size'])}
    mappings = {'rdkit': identity, 'fpsim2': fps_mapping, 'hnsw': hns_mapping}
    for backend, gate in report['preflight'].items():
        if gate is not None:
            for check in gate['checks']:
                _restore_check(check, backend, scores[check['query_position']], mappings[backend], records['index'])
            if gate['passed'] != all(c['passed'] for c in gate['checks']):
                raise ValueError('Preflight gate summary differs from saved checks')
    benchmark = report['benchmark']
    if benchmark is None:
        return
    if benchmark['failure'] and 'check' in benchmark['failure']:
        failure = benchmark['failure']
        _restore_check(failure['check'], failure['backend'], scores[failure['query_position']],
                       mappings[failure['backend']], records['index'])
    for item in benchmark['measured_results']:
        ordered = _restore_check(item['check'], item['backend'], scores[item['query_position']],
                                 mappings[item['backend']], records['index'])
        if ordered is not None and ordered != item['ordered_search_ids']:
            raise ValueError('Saved normalized ordering mismatch')
    if not report['passed']:
        if benchmark['summaries'] is not None or report['latency_comparisons'] is not None:
            raise ValueError('Failed validation cannot publish comparative speed')
        return
    expected_orders = []
    rng = random.Random(exact.TIMING_ORDER_SEED)
    for k in exact.CANDIDATE_COUNTS:
        for p in range(-1, exact.MEASURED_PASSES):
            order = list(range(report['query_count'])); rng.shuffle(order)
            expected_orders.append({'k': k, 'pass_index': p, 'query_positions': order, 'backend_order': backend_order(p)})
    if benchmark['query_orders'] != expected_orders:
        raise ValueError('Saved three-backend order differs from fixed protocol')
    expected_measured = [(b, o['k'], o['pass_index'], q, i)
                        for o in expected_orders if o['pass_index'] >= 0
                        for b in o['backend_order'] for i, q in enumerate(o['query_positions'])]
    actual_measured = [tuple(r[key] for key in ('backend', 'k', 'pass_index', 'query_position', 'order_position'))
                       for r in benchmark['measured_results']]
    if actual_measured != expected_measured or any(not r['check']['passed'] for r in benchmark['measured_results']):
        raise ValueError('Saved measured request inventory/validation mismatch')
    for backend in BACKENDS:
        stages = exact.STAGES if backend == 'rdkit' else hnsw.STAGES
        rows = benchmark['raw_timings'][backend]
        actual = [tuple(r[key] for key in ('backend', 'k', 'pass_index', 'query_position', 'order_position')) for r in rows]
        if actual != [r for r in expected_measured if r[0] == backend]:
            raise ValueError('Saved timing request inventory mismatch')
        if any(any(type(r[s]) is not int or r[s] < 0 for s in stages)
               or r['retrieval_total_ns'] < sum(r[s] for s in stages[:-1]) for r in rows):
            raise ValueError('Invalid saved retrieval timing stages')
        if benchmark['summaries'][backend] != exact.summarize_timings(rows, report['query_count'], stages=stages):
            raise ValueError('Saved per-query latency summary mismatch')
        for key, checks in benchmark['designated'][backend].items():
            first = [None]*report['query_count']
            for item in benchmark['measured_results']:
                if item['backend'] == backend and item['k'] == int(key) and item['pass_index'] == 0:
                    first[item['query_position']] = item['check']
            if first != checks:
                raise ValueError('Designated candidates differ from first measured pass')
    if evaluate_designated(benchmark['designated'], scores) != report['quality']:
        raise ValueError('Saved recall differs from authoritative float64 scores')
    if latency_comparisons(benchmark['summaries']) != report['latency_comparisons']:
        raise ValueError('Saved time-saved/speedup summaries differ')


def load_comparison_run(path, root=None):
    path = Path(path)
    marker = json.loads((path/'manifest.json').read_text())
    if marker.get('schema_version') != 1 or marker.get('status') not in ('complete', 'failed_validation'):
        raise ValueError('Unpublished three-backend run')
    inventory = {p.name for p in path.iterdir() if p.is_file() and p.name != 'manifest.json'}
    if set(marker['files']) != inventory:
        raise ValueError('Comparison file inventory mismatch')
    for name, info in marker['files'].items():
        if Path(name).name != name or store.file_info(path/name) != info:
            raise ValueError(f'Comparison checksum/size mismatch: {name}')
    report = json.loads((path/'report.json').read_text())
    if (report['run_id'] != marker['run_id'] or report['linkage'] != marker['linkage']
            or report['passed'] != (marker['status'] == 'complete')):
        raise ValueError('Completion/report linkage mismatch')
    if report['passed'] and (not report['synthetic']['passed'] or not report['fpsim2_synthetic']['passed']
            or not all(report['preflight'][b]['passed'] for b in ('fpsim2', 'hnsw'))
            or not report['benchmark']['passed']):
        raise ValueError('Completed run contains failed required checks')
    if not report['passed'] and (report['quality'] is not None or report['latency_comparisons'] is not None or report['charts']):
        raise ValueError('Failed run contains comparative quality/speed publication')
    _validate_payloads(path, report)
    if root is not None:
        root = Path(root).resolve()
        link = report['linkage']
        manifest, q, i, records, _ = exact.load_selection(root, root/link['artifact_path'], report['index_size'])
        if (manifest['identity'] != report['fingerprint_identity']
                or store.file_info(root/link['artifact_path']/'manifest.json') != link['artifact_manifest']
                or verify_raw(root, manifest) != report['raw_verification']):
            raise ValueError('Current raw/artifact linkage differs from saved run')
        if report.get('schema_version') == 2 and report['selection_metadata']['index_sizes'] != manifest['index_sizes']:
            raise ValueError('Saved registered sizes differ from verified split')
        saved_records = json.loads((path/'records.json').read_text())
        if any(saved_records[key] != records[key] for key in records):
            raise ValueError('Current ordered identities differ from saved run')
    return report


def run_comparison(root, artifact_path, index_size, *, expected_selection=None, report_writer=None, progress=None):
    """Verify raw/split/artifact, guard all rows, score afresh, build once, gate and time."""
    root, artifact_path = Path(root).resolve(), Path(artifact_path).resolve()
    started = store.utc()
    snapshots = [memory_snapshot('before_loading')]
    def observe(stage):
        snapshot = memory_snapshot(stage)
        snapshots.append(snapshot)
        if progress is not None:
            progress(snapshot)
    manifest, queries, index, records, timings = exact.load_selection(root, artifact_path, index_size)
    selected = {key: manifest[key] for key in ('source', 'package', 'query_count')}
    selected['index_size'] = index_size
    if expected_selection is not None and selected != expected_selection:
        raise ValueError(f'Unexpected selected dataset: {selected}')
    for k in exact.CANDIDATE_COUNTS:
        exact._validate_k(k, index_size)
    t = perf_counter_ns()
    raw_verification = verify_raw(root, manifest)
    timings['raw_hash_and_count_verification'] = perf_counter_ns()-t
    link = {key: manifest[key] for key in ('source', 'package', 'artifact_id', 'split_id',
                                          'split_path', 'split_file_sha256', 'raw_sha256_inherited')}
    link.update(artifact_path=str(artifact_path.relative_to(root)),
                artifact_manifest=store.file_info(artifact_path/'manifest.json'))
    run_id = uuid4().hex
    parent = root/'experiments/results/hnsw'/manifest['source']/manifest['package']/manifest['artifact_id']/f'index_{index_size}'
    parent.mkdir(parents=True, exist_ok=True)
    final = parent/f'run_{run_id}'
    staging = Path(tempfile.mkdtemp(prefix=f'.run_{run_id}.staging-', dir=parent))
    selection_metadata = {'index_sizes': manifest['index_sizes'],
        'query_identity_sha256': store.digest(records['queries']),
        'prefix_identity_sha256': {str(n): store.digest(records['index'][:n])
                                  for n in manifest['index_sizes'] if n <= index_size}}
    report = {'schema_version': 2, 'run_id': run_id, 'started_utc': started, 'passed': False,
        **selected, 'linkage': link, 'raw_verification': raw_verification,
        'experiment': 'nested parent scaling' if len(manifest['index_sizes']) > 1 else 'independent pilot',
        'selection_metadata': selection_metadata,
        'fingerprint_identity': manifest['identity'], 'configuration': hnsw.CONFIGURATION,
        'environment': {**store.environment(root), **fpsim2.versions(),
                        **{name: version(name) for name in ('matplotlib', 'pandas', 'threadpoolctl')},
                        'ambient_threadpools': threadpool_info(), 'numexpr_threads': numexpr.get_num_threads()},
        'hardware': exact._hardware(),
        'nmslib_implementation': hnsw.implementation_evidence(),
        'fpsim2_implementation': fpsim2.implementation_evidence(),
        'source_files': {name: store.file_info(Path(__file__).parent/name)
                         for name in ('hnsw_benchmark.py', 'hnsw_adapter.py', 'exact_retrieval.py',
                                      'fpsim2_adapter.py', 'fpsim2_benchmark.py', 'nmslib_adapter.py', 'hnsw_reporting.py')},
        'configuration_files': {name: store.file_info(Path(__file__).resolve().parents[2]/'experiments/configs'/name)
            for name in ('hnsw_v1.json', 'dataset_overview_v1.json', 'morgan_r2_2048_chiral_v1.json')},
        'nonzero_validation': None, 'synthetic': None, 'fpsim2_synthetic': None,
        'fpsim2_representation': None, 'hnsw_preparation': None,
        'preflight': {'fpsim2': None, 'hnsw': None}, 'benchmark': None,
        'quality': None, 'latency_comparisons': None, 'timings_ns': timings,
        'memory_snapshots': snapshots, 'error': None, 'charts': [],
        'settings': {'candidate_counts': list(exact.CANDIDATE_COUNTS), 'thresholds': list(fpsim2.THRESHOLDS),
            'warmup_passes_per_k': 1, 'measured_passes_per_k': 5, 'timing_order_seed': 42,
            'shuffle': 'local random.Random.shuffle; same permutation across all three backends',
            'backend_order': 'three cyclic orders then two reversed cyclic orders; recorded per pass',
            'concurrent_queries': 1, 'fpsim2_n_workers': 1, 'hnsw_construction_threads': 1,
            'hnsw_query_api': 'single knnQuery; no batch queries or query-thread option',
            'percentile_method': 'linear', 'p95_definition': 'p95 of per-query five-pass median latency',
            'ground_truth': 'fresh full RDKit float64 matrix in this run, before comparative timing',
            'ties': 'exact equality', 'threshold_membership': 'score >= float64 threshold; no epsilon',
            'distance_diagnostic_absolute_tolerance': 1e-6, 'distance_diagnostic_relative_tolerance': 0,
            'designated_measured_pass_index': 0, 'retrieval_units': 'ns',
            'scope': 'sequential repeated in-memory pilot; no cold-start, service concurrency, or 100M prediction',
            'memory': 'synchronous total/available system RAM and process RSS; not peak memory'},
        'stage_definitions': {'rdkit_scoring_ns': 'BulkTanimotoSimilarity including Python list',
            'rdkit_selection_ordering_ns': 'float64 conversion, partition, exact ties and selected ordering',
            'fpsim2_native_search_ns': 'prepared-query inherited top_k(threshold=0.0,n_workers=1)',
            'hnsw_native_search_ns': 'knnQuery of prepared string; includes native parsing of that string',
            'normalization_ns': 'native score/distance ordering then original row, without ground truth',
            'retrieval_total_ns': 'encloses scoring/search plus selection/normalization and clock overhead',
            'excluded': 'query conversion, graph build, reference computation, validation/evaluation, plotting and writing',
            'graph_construction_ns': 'one createIndex call; excludes encoding, registration and graph saving',
            'validation_and_result_assembly_ns': 'outside-timer per-response checks, row agreement, evidence assembly',
            'report_serialization_writing_ns': 'stored in manifest; excludes plotting, checksums and publication'}}
    report['reference_storage'] = {'format': 'float64 npy memory map', 'construction': 'one fresh RDKit score row at a time',
        'fresh_exhaustive_reference_computation_and_validation': 'includes mmap writing, flush and row-wise round trip',
        'reference_persistence_and_top_k_evidence': 'top-k evidence only; matrix writing is in reference construction',
        'reuse': False}
    store.write_json(staging/'records.json', {**records, 'linkage': link})
    store.write_json(staging/'configuration.json', hnsw.CONFIGURATION)
    try:
        t = perf_counter_ns()
        report['nonzero_validation'] = validate_nonzero_fingerprints(queries, index, records)
        timings['selected_nonzero_validation'] = perf_counter_ns()-t
        observe('verified_and_guarded_inputs')
        t = perf_counter_ns(); qv = store.unpack_fingerprints(queries)
        timings['rdkit_query_reconstruction'] = perf_counter_ns()-t
        t = perf_counter_ns(); iv = store.unpack_fingerprints(index)
        timings['rdkit_index_reconstruction'] = perf_counter_ns()-t
        t = perf_counter_ns()
        reference = exact.write_reference(staging/'exact_scores.npy', qv, iv)
        timings['fresh_exhaustive_reference_computation_and_validation'] = perf_counter_ns()-t
        t = perf_counter_ns()
        truth = exact._results({'scores': reference, 'positions': {
            str(k): [exact.exact_top_k(row, k).tolist() for row in reference] for k in exact.CANDIDATE_COUNTS}}, records)
        store.write_json(staging/'ground_truth_top_k.json', truth)
        timings['reference_persistence_and_top_k_evidence'] = perf_counter_ns()-t
        observe('rdkit_and_fresh_reference_prepared')
        t = perf_counter_ns()
        report['synthetic'] = hnsw.verify_synthetic()
        report['fpsim2_synthetic'] = paired.verify_synthetic(staging/'synthetic_fpsim2.h5')
        timings['synthetic_preparation_and_validation'] = perf_counter_ns()-t
        if not report['synthetic']['passed'] or not report['fpsim2_synthetic']['passed']:
            raise ValueError('Required synthetic integration gate failed')
        t = perf_counter_ns()
        fps_engine, fpq, fmap, report['fpsim2_representation'] = fpsim2.prepare_backend(
            staging/'index_fpsim2.h5', queries, index, manifest['configuration'])
        timings['fpsim2_conversion_database_loading_and_verification'] = perf_counter_ns()-t
        observe('fpsim2_prepared')
        hns_engine, hnq, hmap, report['hnsw_preparation'] = hnsw.prepare_backend(queries, index)
        observe('fresh_hnsw_graph_built')
        t = perf_counter_ns()
        with (staging/'prepared_queries_fpsim2.npy').open('xb') as stream:
            np.save(stream, fpq, allow_pickle=False)
        store.write_json(staging/'prepared_queries_hnsw.json', hnq)
        timings['prepared_query_persistence'] = perf_counter_ns()-t
        t = perf_counter_ns()
        hns_engine.saveIndex(str(staging/'index_hnsw.bin'), save_data=True)
        timings['hnsw_graph_and_data_saving'] = perf_counter_ns()-t
        t = perf_counter_ns()
        report['preflight']['fpsim2'] = fpsim2.verify_queries(fps_engine, fpq, reference, fmap, records['index'])
        report['preflight']['hnsw'] = hnsw.verify_queries(hns_engine, hnq, reference, hmap, records['index'])
        timings['real_preflight_search_and_validation'] = perf_counter_ns()-t
        if not all(gate['passed'] for gate in report['preflight'].values()):
            raise ValueError('Required real preflight correctness gate failed')
        report['benchmark'] = benchmark_three(qv, iv, fps_engine, fpq, fmap,
            hns_engine, hnq, hmap, reference, records['index'])
        if not report['benchmark']['passed']:
            raise ValueError('A warmup or measured response failed validation')
        t = perf_counter_ns()
        report['quality'] = evaluate_designated(report['benchmark']['designated'], reference)
        report['latency_comparisons'] = latency_comparisons(report['benchmark']['summaries'])
        timings['designated_recall_and_latency_comparison_evaluation'] = perf_counter_ns()-t
        report['passed'] = True
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        if isinstance(error, NonzeroFingerprintError):
            report['nonzero_validation'] = {'passed': False, 'summary': error.summary,
                                           'affected_records': error.affected_records}
    observe('after_gate_and_optional_comparative_passes')
    if report['passed'] and report_writer is not None:
        t = perf_counter_ns()
        report['charts'] = report_writer(staging, report)
        timings['plotting'] = perf_counter_ns()-t
        if any(Path(n).name != n or Path(n).suffix not in ('.png', '.svg') for n in report['charts']):
            raise ValueError('Report writer must return simple PNG/SVG filenames')
    report['finished_utc'] = store.utc()
    t = perf_counter_ns(); store.write_json(staging/'report.json', report)
    writing_ns = perf_counter_ns()-t
    restored_report = json.loads((staging/'report.json').read_text())
    if restored_report != report:
        raise ValueError('Comparison report round trip differs')
    t = perf_counter_ns(); _validate_payloads(staging, restored_report)
    reload_validation_ns = perf_counter_ns()-t
    files = {p.name: store.file_info(p) for p in staging.iterdir() if p.is_file()}
    marker = {'schema_version': 1, 'status': 'complete' if report['passed'] else 'failed_validation',
        'run_id': run_id, 'linkage': link, 'files': files,
        'report_serialization_writing_ns': writing_ns, 'payload_reload_validation_ns': reload_validation_ns}
    store.write_json(staging/'manifest.json', marker)
    if final.exists():
        raise FileExistsError(f'Publication conflict: {final}')
    staging.rename(final)
    if not report['passed']:
        raise ComparisonFailure(final, report)
    return final, report
