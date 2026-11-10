"""In-memory bit_jaccard compatibility checks; no graph construction or search."""
import math
from numbers import Integral
from pathlib import Path
from time import perf_counter_ns
from uuid import uuid4

import nmslib
import numpy as np
from rdkit import DataStructs

from chemtellect.fingerprint_store import environment, unpack_fingerprints, utc, write_json

from chemtellect.fingerprints import (validate_packed, validate_nonzero_fingerprints,
                                     NonzeroFingerprintError, NONZERO_POLICY)

ATOL = 1e-6
RTOL = 0
SERIALIZATION = '2048 space-separated 0/1 tokens; token i is RDKit bit i; packed bytes unpacked little'
BACKEND = {'space': 'bit_jaccard', 'data_type': 'OBJECT_AS_STRING',
           'dtype': 'FLOAT', 'method': 'hnsw (initialization only; no graph)'}


def to_nmslib_strings(rows):
    """Preserve row/bit order, including zeros; empty (0,256) returns []. No writes."""
    validate_packed(rows)
    return [' '.join(map(str, np.unpackbits(row, bitorder='little'))) for row in rows]


def register_rows(rows, ids):
    """Return (NMSLIB object registry, ID -> insertion position).

    IDs must be unique signed 32-bit integers (NMSLIB's installed ID interface).
    Empty batches register nothing. getDistance addresses positions, not IDs.
    The caller retains molecular identity separately. No graph is constructed.
    """
    validate_packed(rows)
    ids = list(ids)
    if len(ids) != len(rows):
        raise ValueError('ID count must match rows')
    if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral) or not -(2**31) <= i < 2**31 for i in ids):
        raise ValueError('IDs must be signed 32-bit integers')
    ids = [int(i) for i in ids]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate registration IDs')
    strings = to_nmslib_strings(rows)
    registry = nmslib.init(space='bit_jaccard', data_type=nmslib.DataType.OBJECT_AS_STRING,
                          dtype=nmslib.DistType.FLOAT, method='hnsw')
    positions = {identifier: registry.addDataPoint(identifier, value)
                 for identifier, value in zip(ids, strings)}
    return registry, positions


def _finite(value):
    return value if math.isfinite(value) else None


def compare_pairs(rows, ids):
    """Check all unordered pairs including self, plus reverse distances.

    Empty fixtures raise. Nonfinite distances are retained as null with diagnostics
    and cause failure, never replaced. Returns observations even when checks fail.
    Conversion/registration, RDKit reconstruction and distance checks are timed
    separately. This function does not raise for an observed distance mismatch.
    """
    validate_packed(rows)
    if len(rows) == 0:
        raise ValueError('Cannot verify an empty fixture')
    ids = list(ids)
    start = perf_counter_ns()
    registry, positions = register_rows(rows, ids)
    conversion_ns = perf_counter_ns() - start
    start = perf_counter_ns()
    vectors = unpack_fingerprints(rows)
    reconstruction_ns = perf_counter_ns() - start
    start = perf_counter_ns()
    pairs = []
    for i, a in enumerate(ids):
        for j in range(i, len(ids)):
            b = ids[j]
            rdkit = 1.0 - DataStructs.TanimotoSimilarity(vectors[i], vectors[j])
            forward = float(registry.getDistance(positions[a], positions[b]))
            reverse = float(registry.getDistance(positions[b], positions[a]))
            intersection = len(set(vectors[i].GetOnBits()) & set(vectors[j].GetOnBits()))
            union = vectors[i].GetNumOnBits() + vectors[j].GetNumOnBits() - intersection
            expected = 1.0 - intersection / union if union else None
            finite = all(math.isfinite(v) for v in (rdkit, forward, reverse))
            error = abs(forward - rdkit) if finite else None
            passed = (finite and all(-ATOL <= v <= 1 + ATOL for v in (rdkit, forward, reverse))
                      and error <= ATOL and abs(reverse - rdkit) <= ATOL
                      and abs(forward - reverse) <= ATOL
                      and (expected is None or abs(rdkit - expected) <= ATOL))
            pairs.append({'id_a': int(a), 'id_b': int(b), 'rdkit_distance': _finite(rdkit),
                          'nmslib_distance': _finite(forward), 'reverse_nmslib_distance': _finite(reverse),
                          'absolute_error': error, 'intersection': intersection, 'union': union,
                          'count_derived_distance': expected, 'passed': bool(passed),
                          'diagnostic': None if finite else 'Nonfinite distance returned; not adjusted',
                          'nonfinite_classification': {name: ('NaN' if math.isnan(value) else
                              ('Infinity' if math.isinf(value) else None)) for name,value in
                              [('rdkit',rdkit),('nmslib',forward),('reverse_nmslib',reverse)]},
                          'zero_a': vectors[i].GetNumOnBits() == 0,
                          'zero_b': vectors[j].GetNumOnBits() == 0})
    errors = [p['absolute_error'] for p in pairs if p['absolute_error'] is not None]
    return {'passed': all(p['passed'] for p in pairs), 'comparisons': len(pairs), 'pairs': pairs,
            'maximum_absolute_error': max(errors) if errors else None,
            'maximum_error_excludes_nonfinite': any(p['absolute_error'] is None for p in pairs),
            'zero_vector_outcomes': [p for p in pairs if p['zero_a'] or p['zero_b']],
            'timings_ns': {'adapter_conversion_and_registration': conversion_ns,
                           'rdkit_reconstruction': reconstruction_ns,
                           'distance_verification': perf_counter_ns() - start}}


def synthetic_fixture():
    """Controlled bits, not molecular generation; includes duplicate and zero rows."""
    enabled = [[], list(range(2048)), [0, 1, 7, 8, 31, 32, 2047], [0, 1, 7, 8, 31, 32, 2047],
               [1, 8, 32], [100, 101], [0, 1, 8]]
    rows = np.zeros((len(enabled), 256), dtype=np.uint8)
    for row, bits in zip(rows, enabled):
        for bit in bits:
            row[bit // 8] |= np.uint8(1 << (bit % 8))
    return rows, enabled


def verify_synthetic():
    """Supported nonzero equivalence checks; native zero probes are diagnostic only."""
    rows, enabled = synthetic_fixture()
    result = compare_pairs(rows[1:], list(range(1, len(rows))))
    result['definition'] = {'enabled_positions_by_row': enabled[1:], 'ids': list(range(1, len(rows)))}
    result['representation_preserved'] = all(
        value.split(' ') == [str((int(row[bit // 8]) >> (bit % 8)) & 1) for bit in range(2048)]
        for row, value in zip(rows, to_nmslib_strings(rows)))
    result['passed'] = result['passed'] and result['representation_preserved']
    return result


def unsupported_zero_diagnostics():
    """Diagnostic-only bypass: observe native zero behavior without asserting equivalence."""
    rows, _ = synthetic_fixture()
    observations = compare_pairs(rows, list(range(len(rows))))
    return {'status': 'observed_unsupported_inputs', 'supported_domain_equivalence': False,
            'pairs': observations['zero_vector_outcomes'],
            'timings_ns': observations['timings_ns']}


def verify_real_fixture(queries, index, records):
    """Use exactly rows 0..9 of each loaded role; require ten, never shrink.

    The caller supplies full arrays from verified artifact loading and its verified
    records mapping. Temporary IDs 0..19 map to original role rows explicitly.
    """
    validate_nonzero_fingerprints(queries, index, records)
    for role, rows in [('queries', queries), ('index', index)]:
        validate_packed(rows)
        if len(rows) < 10 or len(records[role]) < 10:
            raise ValueError(f'{role}: requires rows 0 through 9')
    selected = []
    for role in ('queries', 'index'):
        for i, record in enumerate(records[role][:10]):
            if record['row_index'] != i:
                raise ValueError('Record mapping order mismatch')
            selected.append({'temporary_id': len(selected), 'role': role, 'row_index': i,
                             'source_id': record['source_id'], 'line_number': record['line_number']})
    rows = np.concatenate((queries[:10], index[:10]))
    result = compare_pairs(rows, [r['temporary_id'] for r in selected])
    result['selected_records'] = selected
    result['definition'] = 'query rows 0..9 then index rows 0..9; all unordered pairs including self'
    return result


class VerificationFailure(RuntimeError):
    """An unsuccessful run with its preserved report available to callers."""

    def __init__(self, path, report):
        self.report_path = path
        self.report = report
        super().__init__(f'Backend verification failed; report preserved: {path}. '
                         f"{report.get('diagnostic') or 'Required compatibility checks failed.'}")


def run_verification(root, manifest, queries, index, records, loading_timings_ns=None):
    """Save a unique report, including failures, then raise if incompatible.

    Inputs must be the full arrays already verified by fingerprint_store. No artifact writes.
    Loading measurements are caller observations; absent measurements remain null.
    """
    started = utc()
    run_id = uuid4().hex
    report = {'report_schema_version': 2, 'run_id': run_id, 'start_utc': started,
              'end_utc': None, 'passed': False, 'artifact_id': manifest['artifact_id'],
              'split_id': manifest['split_id'], 'dataset_sha256_inherited': manifest.get('raw_sha256_inherited'),
              'fingerprint_identity': manifest['identity'],
              'environment': {**environment(root), 'nmslib': nmslib.__version__},
              'backend': BACKEND, 'serialization': SERIALIZATION,
              'absolute_tolerance': ATOL, 'relative_tolerance': RTOL,
              'policy_identifier': NONZERO_POLICY,
              'supported_input_domain': 'Every query and index fingerprint has at least one enabled bit',
              'nonzero_validation': {'status': 'not_run', 'summary': None, 'affected_records': []},
              'nonzero_validation_ns': None,
              'loading_timings_ns': loading_timings_ns,
              'supported_checks': {name: {'status': 'not_run', 'passed': None} for name in ('synthetic','real')},
              'unsupported_diagnostics': {'status': 'not_run'}, 'diagnostic': None}
    failure = None
    try:
        # Prevent a prefix from masquerading as the full saved artifact.
        if len(queries) != manifest['query_count'] or len(index) != manifest['largest_index_count']:
            raise ValueError('Runner requires all saved query and index rows, not a prefix')
        for role in ('queries', 'index'):
            if [r['row_index'] for r in records[role]] != list(range(len(queries if role == 'queries' else index))):
                raise ValueError(f'{role}: full-artifact mapping must follow original row order')
        start = perf_counter_ns()
        try:
            summary = validate_nonzero_fingerprints(queries, index, records)
            report['nonzero_validation'] = {'status': 'passed', 'summary': summary, 'affected_records': []}
        except NonzeroFingerprintError as error:
            report['nonzero_validation'] = {'status': 'failed', 'summary': error.summary,
                                           'affected_records': error.affected_records}
            raise
        finally:
            report['nonzero_validation_ns'] = perf_counter_ns() - start
        for name, check in [('synthetic', verify_synthetic),
                            ('real', lambda: verify_real_fixture(queries, index, records))]:
            report['supported_checks'][name] = {'status': 'running', 'passed': None}
            result = check()
            report['supported_checks'][name] = {**result, 'status': 'passed' if result['passed'] else 'failed'}
        report['unsupported_diagnostics'] = unsupported_zero_diagnostics()
        report['passed'] = all(f['passed'] is True for f in report['supported_checks'].values())
        if not report['passed']:
            report['diagnostic'] = 'Supported-domain representation or distance checks failed'
    except Exception as error:
        failure = error
        report['diagnostic'] = f'{type(error).__name__}: {error}'
        for check in report['supported_checks'].values():
            if check['status'] == 'running': check['status'] = 'failed'
    report['end_utc'] = utc()
    report['number_of_supported_comparisons'] = sum(f.get('comparisons', 0) for f in report['supported_checks'].values())
    errors = [f['maximum_absolute_error'] for f in report['supported_checks'].values()
              if f.get('maximum_absolute_error') is not None]
    report['maximum_absolute_error_supported_domain'] = max(errors) if errors else None
    output = Path(root)/'experiments/results/backend_verification'/manifest['source']/manifest['package']/manifest['artifact_id']
    output.mkdir(parents=True, exist_ok=True)
    path = output/f'run_{run_id}.json'
    write_json(path, report)
    if not report['passed']:
        raise VerificationFailure(path, report) from failure
    return path, report
