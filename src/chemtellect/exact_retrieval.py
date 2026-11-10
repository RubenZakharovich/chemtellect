"""Sequential RDKit exhaustive retrieval and immutable benchmark evidence."""
import json
import os
from pathlib import Path
import platform
import random
import tempfile
from time import perf_counter_ns
from uuid import uuid4

import numpy as np
from rdkit import DataStructs

from chemtellect import fingerprint_store as store
from chemtellect.dataset_overview import index_records
from chemtellect.fingerprints import validate_nonzero_fingerprints

CANDIDATE_COUNTS = (10, 25, 50, 100)
MEASURED_PASSES = 5
TIMING_ORDER_SEED = 42
PERCENTILE_METHOD = 'linear'
STAGES = ('scoring_ns', 'selection_ordering_ns', 'retrieval_total_ns')
PAYLOADS = ('exact_scores.npy', 'records.json', 'top_k.json', 'timings.json', 'summary.json')


def validate_scores(scores, shape):
    """The authoritative representation is float64; no coercion or tolerance."""
    if (not isinstance(scores, np.ndarray) or scores.dtype != np.float64
            or scores.shape != tuple(shape) or not np.isfinite(scores).all()
            or np.any(scores < 0) or np.any(scores > 1)):
        raise ValueError('Expected finite float64 scores of the required shape in [0, 1]')


def _validate_k(k, count):
    if type(k) is not int or not 1 <= k <= count:
        raise ValueError('k must be an integer in 1..index size')


def _top_k(scores, k):
    # Partition locates the exact boundary without sorting the full index.
    # flatnonzero retains ascending original index position, even for huge ties.
    boundary = np.partition(scores, len(scores) - k)[len(scores) - k]
    above = np.flatnonzero(scores > boundary)
    tied = np.flatnonzero(scores == boundary)[:k - len(above)]
    selected = np.concatenate((above, tied))
    return selected[np.lexsort((selected, -scores[selected]))]


def exact_top_k(scores, k):
    """Descending similarity, then ascending original index position; exact ties."""
    if not isinstance(scores, np.ndarray) or scores.ndim != 1:
        raise ValueError('Expected one score row')
    validate_scores(scores, (len(scores),))
    _validate_k(k, len(scores))
    return _top_k(scores, k)


def threshold_members(scores, threshold):
    """Inclusive float64 threshold membership in original index order."""
    if not isinstance(scores, np.ndarray) or scores.ndim != 1:
        raise ValueError('Expected one score row')
    validate_scores(scores, (len(scores),))
    if (isinstance(threshold, (bool, np.bool_))
            or not isinstance(threshold, (float, np.float64))
            or not np.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError('threshold must be a finite float64 value in [0, 1]')
    return np.flatnonzero(scores >= np.float64(threshold))


def measure_exact(query, index, k, *, clock=None, scorer=None):
    """One fresh exhaustive request. Validation/equality checks belong to caller.

    Scoring measures BulkTanimotoSimilarity, including its Python result list.
    Selection includes float64 conversion, partition, ties and result ordering.
    The enclosing total ends after selection, before evaluation or persistence.
    """
    _validate_k(k, len(index))
    clock = perf_counter_ns if clock is None else clock
    scorer = DataStructs.BulkTanimotoSimilarity if scorer is None else scorer
    start = clock()
    raw = scorer(query, index)
    scored = clock()
    scores = np.asarray(raw, dtype=np.float64)
    positions = _top_k(scores, k)
    selected = clock()
    end = clock()
    return scores, positions, dict(zip(STAGES, (scored-start, selected-scored, end-start)))


def summarize_timings(raw, query_count, candidate_counts=CANDIDATE_COUNTS, stages=STAGES):
    """Median per query across five passes, then aggregate those medians."""
    summaries = {}
    for k in candidate_counts:
        per_query = []
        for q in range(query_count):
            rows = [r for r in raw if r['k'] == k and r['query_position'] == q]
            if sorted(r['pass_index'] for r in rows) != list(range(MEASURED_PASSES)):
                raise ValueError('Expected exactly five measured passes per query/k')
            per_query.append({'query_position': q, **{
                stage: float(np.median([r[stage] for r in rows])) for stage in stages}})
        summaries[str(k)] = {'per_query_medians_ns': per_query, 'across_query_medians_ns': {
            stage: {'median': float(np.median(values)), 'mean': float(np.mean(values)),
                    'p95': float(np.percentile(values, 95, method=PERCENTILE_METHOD))}
            for stage in stages for values in [[r[stage] for r in per_query]]}}
    return {'units': 'ns', 'percentile_method': PERCENTILE_METHOD, 'by_k': summaries}


def benchmark_queries(queries, index, *, candidate_counts=CANDIDATE_COUNTS,
                      clock=None, scorer=None):
    """Already reconstructed/guarded inputs; no concurrent requests or cached retrieval.

    Custom candidate counts/clock/scorer support small controlled tests. The public
    artifact runner always uses the checkpoint's fixed protocol and native RDKit.
    """
    candidate_counts = tuple(candidate_counts)
    if not queries or not index or not candidate_counts or len(set(candidate_counts)) != len(candidate_counts):
        raise ValueError('Nonempty queries/index and distinct candidate counts required')
    for k in candidate_counts:
        _validate_k(k, len(index))
    scorer = DataStructs.BulkTanimotoSimilarity if scorer is None else scorer
    rng = random.Random(TIMING_ORDER_SEED)  # Does not touch split/global RNG state.
    matrix = np.empty((len(queries), len(index)), dtype=np.float64)
    saved = np.zeros(len(queries), dtype=bool)
    evidence, raw, orders = {}, [], []
    for k in candidate_counts:
        evidence[str(k)] = [None] * len(queries)
        # One untimed warmup per k, followed by five measured passes.
        for pass_index in range(-1, MEASURED_PASSES):
            order = list(range(len(queries)))
            rng.shuffle(order)
            orders.append({'k': k, 'pass_index': pass_index, 'query_positions': order})
            for order_position, q in enumerate(order):
                if pass_index == -1:
                    scores = np.asarray(scorer(queries[q], index), dtype=np.float64)
                    positions = _top_k(scores, k)
                else:
                    scores, positions, elapsed = measure_exact(queries[q], index, k, clock=clock, scorer=scorer)
                    raw.append({'k': k, 'pass_index': pass_index, 'order_position': order_position,
                                'query_position': q, **elapsed})
                # Evaluation, persistence preparation and equality checks are outside retrieval.
                validate_scores(scores, (len(index),))
                if saved[q]:
                    if not np.array_equal(matrix[q], scores):
                        raise ValueError(f'RDKit scores changed for original query position {q}')
                elif pass_index == 0:
                    matrix[q] = scores
                    saved[q] = True
                if pass_index == 0:
                    evidence[str(k)][q] = positions.tolist()
    validate_scores(matrix, (len(queries), len(index)))
    return {'scores': matrix, 'positions': evidence, 'raw_timings': raw, 'query_orders': orders,
            'summary': summarize_timings(raw, len(queries), candidate_counts)}


def load_selection(root, artifact_path, index_size):
    """Verify existing artifact, then select only an explicitly registered prefix."""
    root, artifact_path = Path(root).resolve(), Path(artifact_path).resolve()
    artifact_path.relative_to(root)
    t = perf_counter_ns()
    selection = json.loads((artifact_path/'manifest.json').read_text())
    split_path = (root/selection['split_path']).resolve()
    split_path.relative_to(root)
    split = store.validate_split(json.loads(split_path.read_text()))
    metadata_ns = perf_counter_ns() - t
    t = perf_counter_ns()
    manifest, arrays = store.verify_artifact(artifact_path, split_path, selection['configuration'], root)
    if type(index_size) is not int:
        raise ValueError('Index size must be a registered integer')
    selected_index_records = index_records(split, index_size)
    records = store.mapping(split)
    records['index'] = records['index'][:len(selected_index_records)]
    if split['split_id'] != manifest['split_id']:
        raise ValueError('Selected split identity changed during verification')
    return manifest, arrays['queries'], arrays['index'][:index_size], records, {
        'selection_metadata_and_split_loading_validation': metadata_ns,
        'full_artifact_loading_integrity_verification_and_selection': perf_counter_ns()-t}


def _results(benchmark, records):
    return {'designated_measured_pass_index': 0, 'by_k': {
        k: [{'query_position': q, 'query': records['queries'][q], 'matches': [
            {'index_position': p, 'record': records['index'][p],
             'similarity': float(benchmark['scores'][q, p])} for p in positions]}
            for q, positions in enumerate(rows)] for k, rows in benchmark['positions'].items()}}


def _read_payloads(directory, metadata):
    """Validate saved scientific evidence and summaries before completion."""
    scores = np.load(directory/'exact_scores.npy', allow_pickle=False)
    nq, ni = metadata['query_count'], metadata['index_size']
    validate_scores(scores, (nq, ni))
    values = {name: json.loads((directory/name).read_text()) for name in PAYLOADS[1:]}
    records, results, timing, summary = (values[name] for name in PAYLOADS[1:])
    if records['linkage'] != metadata['linkage']:
        raise ValueError('Output record/provenance linkage mismatch')
    link = metadata['linkage']
    if (store.digest(metadata['fingerprint_identity']) != link['artifact_id']
            or metadata['fingerprint_identity']['split_id'] != link['split_id']
            or metadata['fingerprint_identity']['configuration'] != metadata['configuration']
            or any(records[key] != link[key] or metadata[key] != link[key]
                   for key in ('source', 'package'))):
        raise ValueError('Output fingerprint/source provenance mismatch')
    for role, count in [('queries', nq), ('index', ni)]:
        if (len(records[role]) != count
                or [r['row_index'] for r in records[role]] != list(range(count))):
            raise ValueError('Saved ordered record mapping mismatch')
    expected = _results({'scores': scores, 'positions': {
        str(k): [exact_top_k(row, k).tolist() for row in scores] for k in CANDIDATE_COUNTS}}, records)
    if results != expected:
        raise ValueError('Saved top-k differs from authoritative scores/identities')
    if timing['query_orders'] != metadata['timing']['query_orders']:
        raise ValueError('Saved timing order mismatch')
    expected_requests = []
    for order in timing['query_orders']:
        if sorted(order['query_positions']) != list(range(nq)):
            raise ValueError('Invalid timing query permutation')
        if order['pass_index'] >= 0:
            expected_requests.extend((order['k'], order['pass_index'], i, q)
                                     for i, q in enumerate(order['query_positions']))
    actual_requests = []
    for row in timing['raw']:
        actual_requests.append(tuple(row[key] for key in ('k', 'pass_index', 'order_position', 'query_position')))
        if (any(type(row[s]) is not int or row[s] < 0 for s in STAGES)
                or row['retrieval_total_ns'] < row['scoring_ns'] + row['selection_ordering_ns']):
            raise ValueError('Invalid timing stage values')
    if actual_requests != expected_requests or len(actual_requests) != nq * len(CANDIDATE_COUNTS) * MEASURED_PASSES:
        raise ValueError('Raw timing requests do not match protocol order')
    if summary != summarize_timings(timing['raw'], nq):
        raise ValueError('Saved timing summary mismatch')
    return {'scores': scores, 'records': records, 'top_k': results, 'timings': timing, 'summary': summary}


def load_run(directory):
    """Read a completed run, checking every payload checksum and internal linkage."""
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest.get('schema_version') != 1 or manifest.get('status') != 'complete':
        raise ValueError('Run is not complete')
    for name, info in manifest['files'].items():
        if Path(name).name != name:
            raise ValueError('Invalid run filename')
        if store.file_info(directory/name) != info:
            raise ValueError(f'Run checksum/size mismatch: {name}')
    metadata = json.loads((directory/'metadata.json').read_text())
    if set(manifest['files']) != {*PAYLOADS, 'metadata.json', *metadata.get('charts', [])}:
        raise ValueError('Unexpected run file inventory')
    if manifest['run_id'] != metadata['run_id'] or manifest['linkage'] != metadata['linkage']:
        raise ValueError('Run metadata/provenance linkage mismatch')
    return {'metadata': metadata, 'manifest': manifest, **_read_payloads(directory, metadata)}


def _generation_evidence(root, manifest):
    """Reference original generated reports when present; reuse is not generation."""
    parent = root/'experiments/results/fingerprints'/manifest['source']/manifest['package']/manifest['artifact_id']
    evidence = []
    for path in sorted(parent.glob('run_*.json')):
        report = json.loads(path.read_text())
        if report.get('artifact_id') == manifest['artifact_id'] and report.get('operation') == 'generated':
            evidence.append({'path': str(path.relative_to(root)), **store.file_info(path)})
    return evidence


def _hardware():
    cpu = None
    path = Path('/proc/cpuinfo')
    if path.is_file():
        cpu = next((line.split(':', 1)[1].strip() for line in path.read_text().splitlines()
                    if line.startswith('model name')), None)
    return {'platform': platform.platform(), 'machine': platform.machine(),
            'processor': cpu or platform.processor(), 'logical_cpu_count': os.cpu_count(),
            'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None}


def run_benchmark(root, artifact_path, index_size, *, expected_selection=None, report_writer=None):
    """Verify, guard, reconstruct, benchmark and publish one unique completed run.

    expected_selection optionally asserts source/package/query_count/index_size
    before any backend preparation (used by the authorized initial real-data run).
    Missing/corrupt inputs never trigger generation, resampling or fallback.
    """
    root, artifact_path = Path(root).resolve(), Path(artifact_path).resolve()
    from chemtellect.benchmark_context import memory_snapshot
    started = store.utc()
    memory = [memory_snapshot('before_input_loading')]
    manifest, queries, index, records, preparation = load_selection(root, artifact_path, index_size)
    memory.append(memory_snapshot('verified_packed_inputs'))
    selected = {key: manifest[key] for key in ('source', 'package', 'query_count')}
    selected['index_size'] = index_size
    if expected_selection is not None and selected != expected_selection:
        raise ValueError(f'Unexpected real-data selection: {selected}')
    for k in CANDIDATE_COUNTS:
        _validate_k(k, index_size)
    t = perf_counter_ns()
    nonzero = validate_nonzero_fingerprints(queries, index, records)
    preparation['selected_nonzero_validation'] = perf_counter_ns()-t
    t = perf_counter_ns()
    query_vectors, index_vectors = store.unpack_fingerprints(queries), store.unpack_fingerprints(index)
    preparation['rdkit_reconstruction'] = perf_counter_ns()-t
    memory.append(memory_snapshot('rdkit_prepared_before_retrieval'))
    benchmark = benchmark_queries(query_vectors, index_vectors)
    memory.append(memory_snapshot('after_retrieval_and_outside_timer_validation'))
    linkage = {key: manifest[key] for key in ('source', 'package', 'artifact_id', 'split_id',
                                            'split_path', 'split_file_sha256', 'raw_sha256_inherited')}
    linkage.update(artifact_path=str(artifact_path.relative_to(root)),
                   artifact_manifest=store.file_info(artifact_path/'manifest.json'))
    records = {**records, 'linkage': linkage}
    run_id = uuid4().hex
    metadata = {'schema_version': 1, 'run_id': run_id, 'started_utc': started,
        'linkage': linkage, **selected, 'fingerprint_identity': manifest['identity'],
        'configuration': manifest['configuration'], 'environment': store.environment(root),
        'hardware': _hardware(), 'nonzero_validation': nonzero,
        'artifact_operation': 'verified reuse; no fingerprint generation',
        'original_generation': {'artifact_created_utc': manifest['created_utc'],
            'artifact_environment': manifest['environment'], 'artifact_execution': manifest['execution'],
            'reports': _generation_evidence(root, manifest)},
        'numerical_policy': {'authority': 'saved RDKit float64 scores', 'ties': 'direct equality',
            'top_k_order': 'descending score, ascending original index position',
            'threshold_membership': 'score >= configured float64 threshold', 'backend_tolerance': None},
        'timing': {'clock': 'time.perf_counter_ns', 'units': 'ns', 'candidate_counts': list(CANDIDATE_COUNTS),
            'warmup_passes_per_k': 1, 'measured_passes_per_k': MEASURED_PASSES,
            'timing_order_seed': TIMING_ORDER_SEED, 'shuffle': 'local random.Random.shuffle per pass, including warmup',
            'concurrent_queries': 1, 'designated_measured_pass_index': 0,
            'rdkit_execution': 'sequential; one query at a time',
            'p95_definition': 'p95 of per-query median latency',
            'outside_timer_overhead': 'validation/evaluation excluded; no separate aggregate timing recorded',
            'ground_truth_capture': 'first measured pass of first k, once per original query',
            'agreement': 'all subsequent computations checked by direct array equality outside retrieval',
            'percentile_method': PERCENTILE_METHOD, 'query_orders': benchmark['query_orders'],
            'scope': 'repeated in-memory benchmark; excludes cold start and concurrent service',
            'boundaries': {'scoring_ns': 'BulkTanimotoSimilarity including Python list production',
                'selection_ordering_ns': 'float64 conversion, partition, boundary ties and selected-row ordering',
                'retrieval_total_ns': 'encloses scoring through ordered positions; includes conversion and clock overhead',
                'excluded': 'reconstruction, ground-truth capture/persistence, validation/evaluation, printing and report writing'}},
        'preparation_timings_ns': preparation,
        'output_timing_boundaries': {'payload_output_writing': 'result assembly, serialization and writing of five payload files',
            'payload_reload_validation': 'reload and scientific/timing validation of five payloads',
            'excluded': 'metadata/manifest writing, checksums, final completion validation and directory publication'},
        'memory': {'snapshots': memory, 'background_polling': False,
                   'interpretation': 'available system RAM is not process RSS; snapshots are not true peak memory'},
        'charts': []}
    parent = root/'experiments/results/exact_retrieval'/manifest['source']/manifest['package']/manifest['artifact_id']
    parent.mkdir(parents=True, exist_ok=True)
    final = parent/f'run_{run_id}'
    staging = Path(tempfile.mkdtemp(prefix=f'.run_{run_id}.staging-', dir=parent))
    try:
        t = perf_counter_ns()
        with (staging/'exact_scores.npy').open('xb') as stream:
            np.save(stream, benchmark['scores'], allow_pickle=False)
        store.write_json(staging/'records.json', records)
        store.write_json(staging/'top_k.json', _results(benchmark, records))
        store.write_json(staging/'timings.json', {'raw': benchmark['raw_timings'], 'query_orders': benchmark['query_orders']})
        store.write_json(staging/'summary.json', benchmark['summary'])
        preparation['payload_output_writing'] = perf_counter_ns()-t
        t = perf_counter_ns()
        restored = _read_payloads(staging, metadata)
        if not np.array_equal(restored['scores'], benchmark['scores']) or restored['records'] != records:
            raise ValueError('Saved scores or identities differ from verified in-memory evidence')
        preparation['payload_reload_validation'] = perf_counter_ns()-t
        if report_writer is not None:
            metadata['charts'] = report_writer(staging, {'metadata': metadata, 'summary': benchmark['summary']})
            if any(Path(name).name != name or Path(name).suffix not in ('.png', '.svg') for name in metadata['charts']):
                raise ValueError('Report writer must return simple PNG/SVG filenames')
        metadata['finished_utc'] = store.utc()
        store.write_json(staging/'metadata.json', metadata)
        files = {name: store.file_info(staging/name) for name in (*PAYLOADS, 'metadata.json', *metadata['charts'])}
        # Validate all output bytes before writing the completion marker last.
        for name, info in files.items():
            if store.file_info(staging/name) != info:
                raise ValueError(f'Output changed before completion: {name}')
        if json.loads((staging/'metadata.json').read_text()) != metadata:
            raise ValueError('Metadata round trip differs')
        store.write_json(staging/'manifest.json', {'schema_version': 1, 'status': 'complete',
            'run_id': run_id, 'linkage': linkage, 'files': files})
        load_run(staging)
        if final.exists():
            raise FileExistsError(f'Publication conflict: {final}')
        staging.rename(final)
        return final, metadata, benchmark['summary']
    except BaseException as error:
        error.add_note(f'Benchmark stopped; retained staging for inspection: {staging}. No automatic recovery.')
        raise
