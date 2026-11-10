"""Canonical sequential Morgan generation, independent of notebook display."""
import json
from time import perf_counter_ns
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

APPROVED = {'schema_version': 1, 'generator': {'radius': 2, 'fpSize': 2048, 'includeChirality': True, 'useBondTypes': True, 'includeRingMembership': True, 'countSimulation': False, 'countBounds': None, 'onlyNonzeroInvariants': False, 'includeRedundantEnvironments': False, 'atomInvariantsGenerator': None, 'bondInvariantsGenerator': None}, 'preprocessing': {'input': 'canonical_smiles', 'sanitize': True, 'hydrogens': 'SMILES parser handling; no AddHs or RemoveHs stage', 'standardization': 'none', 'disconnected_components': 'keep', 'stereochemistry': 'preserve specified; leave unspecified', 'atom_invariants': 'standard Morgan connectivity', 'atom_centers': 'all', 'ignored_atoms': [], 'custom_invariants': None, 'output': 'GetFingerprint ExplicitBitVect', 'source_rfp': 'never used', 'failure': 'stop with role, source_id and line_number', 'fingerprint_deduplication': False}, 'rationale': 'Fixed connectivity representation preserves saved molecular preparation and record identity; chirality may distinguish specified and unspecified stereochemistry. Future recall measures agreement with exact Tanimoto under this representation, not independently established chemical relevance.'}


def load_configuration(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_configuration(config)
    return config


def validate_configuration(config):
    if config != APPROVED:
        raise ValueError("Unsupported scientific configuration; requires researcher review")


def build_generator(config):
    validate_configuration(config)
    generator = rdFingerprintGenerator.GetMorganGenerator(**config['generator'])
    options = generator.GetOptions()
    for key in ('radius', 'fpSize', 'includeChirality', 'countSimulation',
                'onlyNonzeroInvariants', 'includeRedundantEnvironments'):
        if getattr(options, key) != config['generator'][key]:
            raise ValueError(f"Effective generator mismatch: {key}")
    info = generator.GetInfoString()
    for token in ('includeRingMembership=1', 'useBondTypes=1', 'useChirality=1',
                  'MorganInvariantGenerator', 'bitsPerFeature=1'):
        if token not in info:
            raise ValueError(f"Effective generator missing {token}")
    return generator


def generate_batches(records, role, generator, batch_size=1000):
    """Yield (offset, bit vectors, parse/generate nanoseconds); retain one batch."""
    if type(batch_size) is not int or not 1 <= batch_size <= 1000:
        raise ValueError('batch_size must be an integer from 1 through 1000')
    for offset in range(0, len(records), batch_size):
        batch = records[offset:offset + batch_size]
        molecules = []
        start = perf_counter_ns()
        for record in batch:
            try:
                mol = Chem.MolFromSmiles(record['canonical_smiles'], sanitize=True)
                if mol is None or not mol.GetNumAtoms():
                    raise ValueError('Invalid molecular graph')
                molecules.append(mol)
            except Exception as error:
                raise ValueError(f"{role}: {record['source_id']} line {record['line_number']}: parsing failed") from error
        parsing = perf_counter_ns() - start
        start = perf_counter_ns()
        vectors = []
        for record, mol in zip(batch, molecules):
            try:
                fp = generator.GetFingerprint(mol, fromAtoms=[], ignoreAtoms=[],
                                              customAtomInvariants=[], customBondInvariants=[])
                if not isinstance(fp, DataStructs.ExplicitBitVect) or fp.GetNumBits() != 2048:
                    raise ValueError('Expected 2048-bit ExplicitBitVect')
                vectors.append(fp)
            except Exception as error:
                raise ValueError(f"{role}: {record['source_id']} line {record['line_number']}: generation failed") from error
        yield offset, vectors, {'parsing': parsing, 'generation': perf_counter_ns() - start}


NONZERO_POLICY = 'nonzero_fingerprints_v1'


def validate_packed(rows):
    """Validate packed storage without unpacking or importing a backend."""
    import numpy as np
    if not isinstance(rows, np.ndarray) or rows.dtype != np.uint8 or rows.ndim != 2 or rows.shape[1] != 256:
        raise ValueError('Expected uint8 ndarray of shape (n, 256) packed bytes')


class NonzeroFingerprintError(ValueError):
    """Unsupported zero rows, retaining all identities and the validation summary."""

    def __init__(self, summary, affected_records):
        self.summary = summary
        self.affected_records = affected_records
        preview = '; '.join(f"{r['role']} row {r['row_index']} source_id={r['source_id']} line={r['line_number']}"
                            for r in affected_records[:5])
        super().__init__(f'{NONZERO_POLICY}: {len(affected_records)} all-zero fingerprints: {preview}'
                         + ('; further records in report' if len(affected_records) > 5 else ''))


def validate_nonzero_fingerprints(queries, index, records):
    """Scan every supplied packed row before either backend uses real inputs.

    records contains ordered queries/index lists of equal length to the arrays.
    Original row indices must be unique nonnegative integers within each role;
    they may be nonconsecutive for a future selected benchmark subset. The caller
    must supply the corresponding mapping from verified artifact loading.
    Empty roles check zero rows. Inputs are never modified or renumbered.
    """
    import numpy as np
    validate_packed(queries)
    validate_packed(index)
    summary = {'policy_identifier': NONZERO_POLICY, 'query_rows_checked': len(queries),
               'index_rows_checked': len(index), 'query_zero_rows': 0, 'index_zero_rows': 0}
    affected = []
    # Validate both mappings completely before attributing any zero rows.
    for role, rows in (('queries', queries), ('index', index)):
        validate_packed(rows)
        mapping = records.get(role) if isinstance(records, dict) else None
        if not isinstance(mapping, list) or len(mapping) != len(rows):
            raise ValueError(f'{role}: mapping count must match supplied rows')
        seen = set()
        for record in mapping:
            if (not isinstance(record, dict) or type(record.get('row_index')) is not int
                or record['row_index'] < 0 or record['row_index'] in seen
                or not isinstance(record.get('source_id'), str) or not record['source_id'].strip()
                or type(record.get('line_number')) is not int or record['line_number'] <= 0):
                raise ValueError(f'{role}: invalid or duplicate original record mapping')
            seen.add(record['row_index'])
    for role, rows in (('queries', queries), ('index', index)):
        zero_positions = np.flatnonzero(~np.any(rows, axis=1))
        summary['query_zero_rows' if role == 'queries' else 'index_zero_rows'] = len(zero_positions)
        for position in zero_positions:
            record = records[role][int(position)]
            affected.append({'role': role, **{key: record[key] for key in ('row_index','source_id','line_number')}})
    if affected:
        raise NonzeroFingerprintError(summary, affected)
    return summary
