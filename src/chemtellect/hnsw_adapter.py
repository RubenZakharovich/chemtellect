"""The single approved NMSLIB 2.1.2 HNSW configuration on verified saved bits."""
from pathlib import Path
from time import perf_counter_ns

import nmslib
import numpy as np

from chemtellect import fingerprint_store as store
from chemtellect import nmslib_adapter as binary
from chemtellect import exact_retrieval as exact

CONFIGURATION = {
    'space': 'bit_jaccard', 'data_type': 'OBJECT_AS_STRING', 'dtype': 'FLOAT',
    'method': 'hnsw', 'index_params': {'M': 16, 'efConstruction': 200, 'indexThreadQty': 1},
    'query_params': {'efSearch': 200}, 'query_api': 'knnQuery', 'concurrent_queries': 1}
STAGES = ('native_search_ns', 'normalization_ns', 'retrieval_total_ns')


def implementation_evidence():
    engine = nmslib.init(space='bit_jaccard', data_type=nmslib.DataType.OBJECT_AS_STRING,
                         dtype=nmslib.DistType.FLOAT, method='hnsw')
    return {'version': nmslib.__version__, 'binary': str(Path(nmslib.__file__).resolve()),
            'binary_info': store.file_info(Path(nmslib.__file__)),
            'api_docs': {name: getattr(engine, name).__doc__ for name in (
                'addDataPoint', 'createIndex', 'setQueryTimeParams', 'knnQuery', 'getDistance', 'saveIndex')},
            'serialization': binary.SERIALIZATION,
            'observed_query_contract': 'knnQuery returns explicit int32 IDs and float32 distances; see synthetic evidence'}


def prepare_backend(queries, index, ids=None):
    """Caller guards every selected row first. Build exactly one fresh graph.

    Reuse the existing binary encoding and original-order explicit registration.
    Returned insertion positions are checked; search IDs map to original rows.
    No batch query, graph reuse, parameter sweep or graph-seed claim is made.
    """
    if nmslib.__version__ != '2.1.2':
        raise ValueError('Only inspected NMSLIB 2.1.2 is supported')
    ids = list(range(len(index))) if ids is None else list(ids)
    t = perf_counter_ns()
    engine, insertion = binary.register_rows(index, ids)
    registration_ns = perf_counter_ns()-t
    if insertion != {int(i): p for p, i in enumerate(ids)}:
        raise ValueError('NMSLIB insertion positions differ from verified original order')
    t = perf_counter_ns()
    prepared = binary.to_nmslib_strings(queries)
    query_conversion_ns = perf_counter_ns()-t
    t = perf_counter_ns()
    engine.createIndex(dict(CONFIGURATION['index_params']), print_progress=False)
    construction_ns = perf_counter_ns()-t
    t = perf_counter_ns()
    engine.setQueryTimeParams(dict(CONFIGURATION['query_params']))
    query_settings_ns = perf_counter_ns()-t
    return engine, prepared, insertion, {
        'index_conversion_and_original_order_registration_ns': registration_ns,
        'query_conversion_ns': query_conversion_ns,
        'graph_construction_ns': construction_ns, 'query_settings_ns': query_settings_ns,
        'registered_ids': [int(i) for i in ids], 'insertion_positions': list(insertion.values()),
        'build_count': 1, 'graph_seed': 'not exposed or set; graph topology reproducibility not asserted',
        'construction_threads': 1, 'query_execution': 'single knnQuery, one query at a time'}


def _well_formed(response):
    return (isinstance(response, (tuple, list)) and len(response) == 2
            and all(isinstance(a, np.ndarray) and a.ndim == 1 for a in response)
            and response[0].dtype.kind in 'iu' and response[1].dtype.kind == 'f'
            and len(response[0]) == len(response[1]))


def normalize_results(response, id_to_position):
    """Ascending returned distance, then original row. Never reference ground truth.

    Malformed responses pass unchanged to the outside-timer validator, preserving
    evidence rather than failing before it can be recorded.
    """
    if not _well_formed(response):
        return response
    ids, distances = response
    positions = np.fromiter((id_to_position.get(int(i), 2**63-1) for i in ids), dtype=np.int64)
    order = np.lexsort((positions, distances))
    return ids[order], distances[order]


def measure_query(engine, query, k, id_to_position, clock=None):
    clock = perf_counter_ns if clock is None else clock
    start = clock()
    response = engine.knnQuery(query, k=k)
    searched = clock()
    ordered = normalize_results(response, id_to_position)
    normalized = clock()
    end = clock()
    return response, ordered, dict(zip(STAGES, (searched-start, normalized-searched, end-start)))


def check_response(response, reference, id_to_position, k, records=None):
    """Gate identity/distance correctness, separately from approximate recall.

    A valid approximate result may miss exact neighbors. The numerical tolerance
    only checks native distance against 1 - authoritative score for returned IDs.
    Neither recall nor threshold membership uses that tolerance.
    """
    exact.validate_scores(reference, (len(reference),))
    exact._validate_k(k, len(reference))
    report = {'passed': False, 'k': k, 'expected_count': k, 'actual_count': None,
        'malformed': None, 'duplicate_ids': [], 'invalid_ids': [], 'aliased_positions': [],
        'invalid_distances': [], 'native_results': [], 'maximum_absolute_distance_error': None}
    if not _well_formed(response):
        report['malformed'] = 'Expected equal-length 1D integer IDs and floating distances'
        report['raw_observation'] = repr(response)
        return report
    ids, distances = response
    report['actual_count'] = len(ids)
    seen, positions, errors = set(), set(), []
    for identifier, distance in zip(ids, distances):
        identifier, distance = int(identifier), float(distance)
        if identifier in seen:
            report['duplicate_ids'].append(identifier)
        seen.add(identifier)
        p = id_to_position.get(identifier)
        if type(p) is not int or not 0 <= p < len(reference):
            report['invalid_ids'].append(identifier)
            p = None
        if p is not None:
            if p in positions:
                report['aliased_positions'].append(p)
            positions.add(p)
        finite = bool(np.isfinite(distance))
        error = abs(distance-(1.0-float(reference[p]))) if finite and p is not None else None
        if error is not None:
            errors.append(error)
        if not finite or not 0 <= distance <= 1 or error is None or error > binary.ATOL:
            report['invalid_distances'].append(identifier)
        report['native_results'].append({'search_id': identifier, 'index_position': p,
            'record': records[p] if records is not None and p is not None else None,
            'native_distance': distance if finite else None,
            'nonfinite': None if finite else str(distance),
            'similarity_from_distance': 1.0-distance if finite else None,
            'rdkit_reference_score': float(reference[p]) if p is not None else None,
            'absolute_distance_error': error})
    report['maximum_absolute_distance_error'] = max(errors) if errors else None
    report['passed'] = (report['actual_count'] == k and not report['duplicate_ids']
        and not report['invalid_ids'] and not report['aliased_positions'] and not report['invalid_distances'])
    return report


def verify_queries(engine, queries, reference, mapping, records=None, candidate_counts=exact.CANDIDATE_COUNTS):
    checks = [{'query_position': q, **check_response(engine.knnQuery(query, k=k), reference[q],
               mapping, k, records)} for q, query in enumerate(queries) for k in candidate_counts]
    return {'passed': all(c['passed'] for c in checks), 'checks': checks,
        'maximum_absolute_distance_error': max((c['maximum_absolute_distance_error'] or 0 for c in checks), default=0),
        'distance_absolute_tolerance': binary.ATOL, 'distance_relative_tolerance': 0,
        'recall_gate': 'approximate recall is measured separately; not required to equal one'}


def verify_synthetic():
    rows, _ = binary.synthetic_fixture()
    rows = rows[1:]  # All synthetic inputs are nonzero; collisions remain separate.
    records = {role: [dict(row_index=i, source_id=f'synthetic:{role}:{i}', line_number=i+1)
                      for i in range(len(rows))] for role in ('queries', 'index')}
    from chemtellect.fingerprints import validate_nonzero_fingerprints
    from rdkit import DataStructs
    validate_nonzero_fingerprints(rows, rows, records)
    ids = [901, 7, 42, 88, 1234, 3001]
    engine, queries, mapping, preparation = prepare_backend(rows, rows, ids)
    vectors = store.unpack_fingerprints(rows)
    reference = np.asarray([DataStructs.BulkTanimotoSimilarity(q, vectors) for q in vectors], dtype=np.float64)
    check = verify_queries(engine, queries, reference, mapping, records['index'], (1, len(rows)))
    result_ids, distances = engine.knnQuery(queries[1], k=len(rows))
    observed_ids = set(result_ids.tolist()) == set(ids)
    return {**check, 'passed': check['passed'] and observed_ids,
        'preparation': preparation, 'explicit_ids_returned': observed_ids,
        'observed_ids': result_ids.tolist(), 'observed_distances': distances.tolist(),
        'id_dtype': str(result_ids.dtype), 'distance_dtype': str(distances.dtype),
        'definition': 'nonzero saved-bit fixture; nonconsecutive IDs, boundary bits, identical bits and zero overlap'}
