"""Dataset audit and reproducible holdout selections; no acquisition or fingerprints."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

from chemtellect.data_integrity import validate_jsonl
from chemtellect.molecular_validation import validate_molecules

SOURCES = ('chemicalspace_new', 'otava_chemical')
PACKAGES = ('package_00', 'package_01', 'package_02', 'scale_10k_00', 'scale_100k_00')
POLICY = {'canonical': True, 'isomericSmiles': True, 'sanitize': True, 'standardization': 'none'}
COUNTS = ('total_input_records', 'successfully_parsed_records', 'invalid_molecular_records',
          'integrity_rejected_records', 'unique_canonical_structures',
          'duplicate_structure_groups', 'records_in_duplicate_groups', 'surplus_duplicate_records')


@dataclass(frozen=True)
class Dataset:
    source: str
    package: str
    raw: Path
    manifest: Path
    report: Path


def dataset(root: Path, source: str, package: str) -> Dataset:
    if source not in SOURCES or package not in PACKAGES or (source == 'otava_chemical' and package == 'scale_10k_00'):
        raise ValueError('Not an approved completed dataset')
    base = Path(root) / 'data/raw' / source
    return Dataset(source, package, base / f'{package}_pool.jsonl',
                   base / f'{package}_pool.meta.json',
                   Path(root) / 'experiments/results/validation' / f'{source}_{package}.json')


def inventory(root: Path) -> list[dict]:
    """Inventory only the nine approved pools; hashes are checked on selection."""
    rows = []
    for source in SOURCES:
        for package in PACKAGES:
            if source == 'otava_chemical' and package == 'scale_10k_00':
                continue
            item = dataset(root, source, package)
            rows.append({'source': source, 'package': package, 'raw_present': item.raw.is_file(),
                         'manifest_present': item.manifest.is_file(), 'report_present': item.report.is_file(),
                         'bytes': item.raw.stat().st_size if item.raw.is_file() else None})
    return rows


def audit(item: Dataset) -> tuple[dict, list[dict]]:
    """Recompute using the existing validator, compare historical report, select unique rows.

    Missing historical reports do not prevent fresh validation. Stale or inconsistent
    reports fail closed. A different RDKit version triggers a visible status; fresh
    chemistry counts must still match the historical report. Raw files are never written.
    """
    manifest = json.loads(item.manifest.read_text(encoding='utf-8'))
    required = {'manifest_schema_version': 1, 'source_database': item.source,
                'source_collection': 'compounds', 'package_name': item.package,
                'output_filename': item.raw.name, 'extraction_integrity_status': 'passed'}
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f'Manifest mismatch: {key}')
    expected = manifest.get('actual_extracted_record_count')
    if type(expected) is not int or expected <= 0 or manifest.get('requested_record_count') != expected:
        raise ValueError('Manifest counts are invalid or disagree')
    integrity = validate_jsonl(item.raw, expected=expected)
    if not integrity.success:
        raise ValueError(f'Integrity failed: {integrity.errors[:5]}')
    if integrity.sha256 != manifest.get('sha256'):
        raise ValueError('Raw SHA-256 does not match acquisition manifest')
    if integrity.missing_smiles != manifest.get('missing_smiles_count'):
        raise ValueError('Missing-SMILES count does not match manifest')
    validation = validate_molecules(item.raw)
    current = validation.to_dict()
    if not validation.success or validation.sha256 != integrity.sha256:
        raise ValueError('Fresh validation failed or input changed during audit')
    status = 'absent; fresh validation used'
    if item.report.is_file():
        old = json.loads(item.report.read_text(encoding='utf-8'))
        for key in ('sha256', 'canonicalization', *COUNTS):
            if old.get(key) != current[key]:
                raise ValueError(f'Historical validation report mismatch: {key}')
        if old.get('success') is not True:
            raise ValueError('Historical report did not pass')
        status = ('matched' if old.get('rdkit_version') == rdBase.rdkitVersion
                  else f"counts matched; RDKit changed from {old.get('rdkit_version')} to {rdBase.rdkitVersion}")
    # The existing validator inserts IDs in raw-line order. Reuse its canonical groups.
    representatives = {ids[0]: canonical for canonical, ids in validation.structures.items()}
    records = []
    digest = hashlib.sha256()
    with item.raw.open('rb') as stream:
        for line, raw in enumerate(stream, 1):
            digest.update(raw)
            record = json.loads(raw)
            sid = record['source_id']
            if sid in representatives:
                records.append({'source_id': sid, 'line_number': line, 'smiles': record.get('smiles'),
                                'canonical_smiles': representatives[sid]})
    if digest.hexdigest() != integrity.sha256:
        raise ValueError('Input changed during selection')
    summary = {'source': item.source, 'package': item.package, 'sha256': integrity.sha256,
               'rdkit_version': rdBase.rdkitVersion, 'canonicalization': POLICY,
               'representative_policy': 'first raw physical line per canonical structure',
               'historical_report_status': status, 'counts': {k: current[k] for k in COUNTS},
               'acquisition_manifest': manifest, 'fresh_validation': current}
    return summary, records


def make_split(summary: dict, records: list[dict], index_sizes=(1000,), query_count=100, seed=42) -> dict:
    """SHA-256-ranked selection, independent of Python/NumPy RNG implementation.

    A hash of seed, source, package and source ID defines one deterministic ordering.
    First query_count rows are queries; subsequent prefixes are nested indexes.
    This is a reproducible pseudorandom ordering, not a claim of unbiased acquisition.
    """
    sizes = sorted(set(index_sizes))
    if not sizes or any(type(n) is not int or n <= 0 for n in sizes):
        raise ValueError('Index sizes must be positive integers')
    if type(query_count) is not int or query_count <= 0 or type(seed) is not int:
        raise ValueError('query_count must be positive integer; seed must be integer')
    if len(records) < max(sizes) + query_count:
        raise ValueError(f'Need {max(sizes) + query_count} unique structures; have {len(records)}')
    if len({r['source_id'] for r in records}) != len(records) or len({r['canonical_smiles'] for r in records}) != len(records):
        raise ValueError('Records must have unique IDs and canonical structures')
    def rank(row):
        value = json.dumps([seed, summary['source'], summary['package'], row['source_id']],
                           ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        return hashlib.sha256(value).digest(), row['source_id']
    ordered = sorted(records, key=rank)
    payload = {'split_schema_version': 1, 'source': summary['source'], 'package': summary['package'],
               'raw_sha256': summary['sha256'], 'rdkit_version': summary['rdkit_version'],
               'canonicalization': POLICY, 'representative_policy': summary['representative_policy'],
               'ordering_algorithm': 'sha256-json-seed-source-package-id-v1', 'seed': seed,
               'query_count': query_count, 'index_sizes': sizes, 'eligible_unique_count': len(records),
               'unused_count': len(records) - query_count - max(sizes),
               'queries': ordered[:query_count], 'index_order': ordered[query_count:query_count + max(sizes)]}
    payload['split_id'] = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                                   separators=(',', ':')).encode('utf-8')).hexdigest()
    return payload


def index_records(split: dict, size: int) -> list[dict]:
    if size not in split['index_sizes']:
        raise ValueError('Size not registered in split')
    return split['index_order'][:size]


def descriptor_rows(records: list[dict]) -> list[dict]:
    """Descriptors for eligible unique structures, not raw rows; no standardization."""
    rows = []
    for record in records:
        mol = Chem.MolFromSmiles(record['canonical_smiles'], sanitize=True)
        if mol is None:
            raise ValueError('Previously valid canonical structure could not be parsed')
        rows.append({'source_id': record['source_id'], 'molecular_weight': Descriptors.MolWt(mol),
                     'logp': Descriptors.MolLogP(mol), 'tpsa': rdMolDescriptors.CalcTPSA(mol),
                     'heavy_atoms': mol.GetNumHeavyAtoms(), 'rotatable_bonds': Lipinski.NumRotatableBonds(mol)})
    return rows


def save_json(path: Path, value: dict) -> Path:
    """Exclusive create. An identical existing artifact is reusable; conflicts fail."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + '\n'
    try:
        with path.open('x', encoding='utf-8') as stream:
            stream.write(content)
    except FileExistsError:
        if path.read_text(encoding='utf-8') != content:
            raise ValueError(f'Refusing to replace different artifact: {path}')
    return path
