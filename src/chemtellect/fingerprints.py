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
