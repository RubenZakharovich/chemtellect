"""FPSim2 0.7.4 saved-bit adapter; no molecular parsing or generation."""
from importlib.metadata import version
from numbers import Integral
from pathlib import Path
from time import perf_counter_ns

import FPSim2
from FPSim2 import FPSim2Engine
from FPSim2.io.chem import process_fp
from FPSim2.io.backends.pytables import create_schema
import numpy as np
from rdkit import rdBase
import tables

from chemtellect import fingerprint_store as store
from chemtellect.exact_retrieval import validate_scores, threshold_members, _validate_k
from chemtellect.fingerprints import validate_packed, validate_configuration

SUPPORTED_VERSION = '0.7.4'
THRESHOLDS = (.3, .5, .7, .9)
ATOL = 1e-6


def versions():
    return {name: version(name) for name in ('FPSim2', 'tables', 'blosc2', 'numexpr',
                                            'scipy', 'psutil', 'numpy', 'rdkit', 'nmslib')}


def implementation_evidence():
    base = Path(FPSim2.__file__).parent
    names = [
        'FPSim2.py', 'base.py', 'io/chem.py', 'io/backends/pytables.py',
        'src/sim.cpp', 'src/utils.cpp', 'src/include/result.hpp']
    names += [p.name for p in base.glob('FPSim2lib*.so')]
    return {name: store.file_info(base/name) for name in names}


def validate_ids(ids, count):
    ids = list(ids)
    if (len(ids) != count or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral)
                                or not 0 <= i < 2**32 for i in ids)
            or len(set(ids)) != count):
        raise ValueError('Expected one distinct uint32 search ID per index record')
    return [int(i) for i in ids]


def packed_to_native(packed, ids):
    """Native process_fp: ID, 32 words in bit-string order, enabled-bit count."""
    validate_packed(packed)
    ids = validate_ids(ids, len(packed))
    result = np.empty((len(packed), 34), dtype=np.uint64)
    for start in range(0, len(packed), 1000):
        vectors = store.unpack_fingerprints(packed[start:start+1000])
        result[start:start+len(vectors)] = [process_fp(fp, i) for fp, i in zip(vectors, ids[start:start+1000])]
    return result


def native_to_packed(rows):
    if not isinstance(rows, np.ndarray) or rows.dtype != np.uint64 or rows.ndim != 2 or rows.shape[1] != 34:
        raise ValueError('Expected native uint64 (n,34) rows')
    # Native bit-string offset 0 is the MSB of word 0. Word order is unchanged.
    bits = np.unpackbits(rows[:, 1:33].astype('>u8').view(np.uint8), axis=1, bitorder='big')
    if not np.array_equal(bits.sum(axis=1), rows[:, -1]):
        raise ValueError('Stored population counts differ from enabled bits')
    return np.packbits(bits, axis=1, bitorder='little')


class PreparedEngine(FPSim2Engine):
    """Only bypass query conversion already performed/verified in preparation.

    All inherited search, pruning, heap and scoring code is unmodified. This
    class is internal: load_query receives only validated prepared native rows.
    """
    def load_query(self, query, full_sanitization=True):
        return query


def prepare_backend(path, queries, index, config, ids=None):
    """Caller must guard all selected packed rows first. Create exclusively.

    FPSim2 has no public precomputed database writer. Use its version-specific
    schema/config layout and native process_fp, then its normal in-memory loader.
    """
    if FPSim2.__version__ != SUPPORTED_VERSION:
        raise ValueError(f'Only inspected FPSim2 {SUPPORTED_VERSION} is supported')
    validate_configuration(config)
    ids = validate_ids(range(len(index)) if ids is None else ids, len(index))
    t = perf_counter_ns()
    db = packed_to_native(index, ids)
    index_conversion_ns = perf_counter_ns()-t
    t = perf_counter_ns()
    query_rows = packed_to_native(queries, range(len(queries)))
    query_conversion_ns = perf_counter_ns()-t
    if (any(not np.array_equal(native_to_packed(db[s:s+1000]), index[s:s+1000])
            for s in range(0, len(index), 1000))
            or not np.array_equal(native_to_packed(query_rows), queries)):
        raise ValueError('Native conversion round trip differs from saved bits')
    path = Path(path)
    # Reserve the name before PyTables opens it. Never replace any existing file.
    with path.open('xb'):
        pass
    with tables.open_file(path, mode='w') as out:
        table = out.create_table('/', 'fps', create_schema(2048))
        for start in range(0, len(db), 1000):
            table.append([tuple(int(v) for v in row) for row in db[start:start+1000]])
        cfg = out.create_vlarray('/', 'config', atom=tables.ObjectAtom())
        for item in ('Morgan', config['generator'], rdBase.rdkitVersion, SUPPORTED_VERSION, []):
            cfg.append(item)
    engine = PreparedEngine(str(path), in_memory_fps=True, fps_sort=True)
    id_to_position = {i: p for p, i in enumerate(ids)}
    loaded_ids = [int(v) for v in engine.fps[:, 0]]
    validate_ids(loaded_ids, len(ids))
    if set(loaded_ids) != set(ids):
        raise ValueError('Native loader changed search identities')
    original_positions = [id_to_position[i] for i in loaded_ids]
    if any(not np.array_equal(native_to_packed(engine.fps[s:s+1000]), index[original_positions[s:s+1000]])
           for s in range(0, len(index), 1000)):
        raise ValueError('Loaded/sorted backend rows differ from saved bits')
    if np.any(np.diff(engine.fps[:, -1].astype(np.int64)) < 0):
        raise ValueError('FPSim2 index is not population-count sorted')
    query_rows.setflags(write=False)
    return engine, query_rows, id_to_position, {
        'verified_query_rows': len(queries), 'verified_index_rows': len(index),
        'conversion_timings_ns': {'index_conversion': index_conversion_ns, 'query_conversion': query_conversion_ns},
        'conversion_round_trip': True, 'loaded_sorted_round_trip': True,
        'internal_row_search_ids': loaded_ids, 'internal_row_original_positions': original_positions,
        'popcount_bins': [[int(c), [int(a), int(b)]] for c, (a, b) in engine.popcnt_bins],
        'word_contract': '32 uint64 words in ascending RDKit bit-block order; bit 64*j is word j MSB',
        'stored_fields': 'uint64 view: ID, 32 fingerprint words, population count',
        'database': 'precomputed HDF5 schema; runtime population-count sorting; no generation',
        'query_path': 'PreparedEngine.load_query returns validated uint64 row unchanged',
        'native_precision': 'C++ float scores/threshold/dynamic pruning; returned coeff float32',
        'top_k_threshold': 0.0, 'n_workers': 1}


def normalize_results(response, id_to_position):
    """Deterministic presentation by native score, then original index position.

    No ground-truth lookup or replacement. Structural validation is outside the
    timer; unknown IDs sort last and are rejected by the subsequent validator.
    """
    positions = np.fromiter((id_to_position.get(int(i), 2**63-1) for i in response['mol_id']), dtype=np.int64)
    return response[np.lexsort((positions, -response['coeff']))]


def check_response(response, reference, id_to_position, *, k=None, threshold=None, records=None):
    """Exact membership gate; tolerance applies ONLY to native score diagnostics."""
    validate_scores(reference, (len(reference),))
    if (k is None) == (threshold is None):
        raise ValueError('Select exactly one top-k or threshold check')
    if k is not None:
        _validate_k(k, len(reference))
        boundary = float(np.partition(reference, len(reference)-k)[len(reference)-k])
        required = set(np.flatnonzero(reference > boundary).tolist())
        qualifying = set(np.flatnonzero(reference >= boundary).tolist())
        expected_count = k
    else:
        boundary = None
        required = qualifying = set(threshold_members(reference, threshold).tolist())
        expected_count = len(required)
    report = {'passed': False, 'actual_count': len(response) if isinstance(response, np.ndarray) and response.ndim else None,
        'expected_count': expected_count, 'boundary': boundary, 'k': k, 'threshold': threshold,
        'malformed': None, 'duplicate_ids': [], 'invalid_ids': [], 'missing_positions': [],
        'unexpected_positions': [], 'native_results': [], 'maximum_native_score_error': None}
    if (not isinstance(response, np.ndarray) or response.ndim != 1 or response.dtype.names is None
            or not {'mol_id', 'coeff'} <= set(response.dtype.names)
            or response.dtype['mol_id'].kind not in 'iu' or response.dtype['coeff'].kind != 'f'):
        report['malformed'] = 'Expected one-dimensional structured integer mol_id / floating coeff response'
        return report
    seen, positions, errors = set(), set(), []
    scores_valid = True
    for item in response:
        i, native = int(item['mol_id']), float(item['coeff'])
        if i in seen:
            report['duplicate_ids'].append(i)
        seen.add(i)
        p = id_to_position.get(i)
        if p is None or not 0 <= p < len(reference):
            report['invalid_ids'].append(i)
            p = None
        finite = bool(np.isfinite(native))
        valid = finite and 0 <= native <= 1
        error = abs(native-float(reference[p])) if finite and p is not None else None
        scores_valid = scores_valid and valid and error is not None and error <= ATOL
        if error is not None:
            errors.append(error)
        report['native_results'].append({'search_id': i, 'index_position': p,
            'record': records[p] if records is not None and p is not None else None,
            'native_score': native if finite else None,
            'nonfinite': None if finite else str(native),
            'rdkit_reference_score': float(reference[p]) if p is not None else None,
            'absolute_error': error,
            'native_minus_reference': native-float(reference[p]) if finite and p is not None else None})
        if p is not None:
            positions.add(p)
    report['missing_positions'] = sorted(required-positions)
    report['unexpected_positions'] = sorted(positions-qualifying)
    report['maximum_native_score_error'] = max(errors) if errors else None
    report['passed'] = (not report['duplicate_ids'] and not report['invalid_ids'] and scores_valid
        and report['actual_count'] == expected_count and len(positions) == report['actual_count']
        and not report['missing_positions']
        and not report['unexpected_positions'])
    # Empty threshold responses are valid when the complete qualifying set is empty.
    if len(response) == 0 and expected_count == 0:
        report['passed'] = True
    return report


def verify_queries(engine, queries, reference, id_to_position, records=None,
                   candidate_counts=(10, 25, 50, 100), thresholds=THRESHOLDS):
    validate_scores(reference, (len(queries), len(id_to_position)))
    checks = []
    for q, query in enumerate(queries):
        for k in candidate_counts:
            response = engine.top_k(query, k=k, threshold=0.0, n_workers=1)
            checks.append({'query_position': q, 'kind': 'top_k', **check_response(
                response, reference[q], id_to_position, k=k, records=records)})
        for threshold in thresholds:
            response = engine.similarity(query, threshold=threshold, n_workers=1)
            checks.append({'query_position': q, 'kind': 'threshold', **check_response(
                response, reference[q], id_to_position, threshold=threshold, records=records)})
    return {'passed': all(c['passed'] for c in checks), 'query_count': len(queries), 'checks': checks,
            'score_absolute_tolerance': ATOL, 'score_relative_tolerance': 0,
            'membership_tolerance': None,
            'maximum_native_score_error': max((c['maximum_native_score_error'] or 0 for c in checks), default=0)}
