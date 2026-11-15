import hashlib
import json
from pathlib import Path

import pytest
from chemtellect.dataset_overview import (audit, dataset, inventory, make_split, index_records,
                                         descriptor_rows, save_json)
from chemtellect.molecular_validation import validate_molecules


def fixture_pool(root, smiles=('CCO', 'OCC', 'not smiles', 'CC', 'CCC', 'C', 'N', 'O')):
    item = dataset(root, 'chemicalspace_new', 'package_00')
    item.raw.parent.mkdir(parents=True)
    item.raw.write_text(''.join(json.dumps({'source_id': f'id:{i}', 'smiles': s, 'rfp': 'ignored'}) + '\n'
                                for i, s in enumerate(smiles)))
    manifest = {'manifest_schema_version': 1, 'source_database': item.source,
                'source_collection': 'compounds', 'package_name': item.package,
                'output_filename': item.raw.name, 'extraction_integrity_status': 'passed',
                'requested_record_count': len(smiles), 'actual_extracted_record_count': len(smiles),
                'missing_smiles_count': 0, 'sha256': hashlib.sha256(item.raw.read_bytes()).hexdigest()}
    item.manifest.write_text(json.dumps(manifest))
    item.report.parent.mkdir(parents=True)
    item.report.write_text(json.dumps(validate_molecules(item.raw).to_dict()))
    return item


def test_selection_preserves_raw_and_first_representative(tmp_path):
    item = fixture_pool(tmp_path)
    before = item.raw.read_bytes()
    summary, records = audit(item)
    assert item.raw.read_bytes() == before
    assert summary['counts']['unique_canonical_structures'] == 6
    assert summary['counts']['invalid_molecular_records'] == 1
    assert summary['counts']['surplus_duplicate_records'] == 1
    assert records[0]['source_id'] == 'id:0' and records[0]['line_number'] == 1
    assert all('rfp' not in r for r in records)


@pytest.mark.parametrize('field,value', [('sha256', 'bad'), ('package_name', 'package_01'),
    ('source_database', 'otava_chemical'), ('actual_extracted_record_count', 99),
    ('missing_smiles_count', 2), ('extraction_integrity_status', 'failed')])
def test_manifest_mismatch_fails(tmp_path, field, value):
    item = fixture_pool(tmp_path)
    manifest = json.loads(item.manifest.read_text()); manifest[field] = value
    item.manifest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError): audit(item)


@pytest.mark.parametrize('field,value', [('sha256', 'bad'), ('unique_canonical_structures', 99),
                                        ('canonicalization', {}), ('success', False)])
def test_stale_report_fails(tmp_path, field, value):
    item = fixture_pool(tmp_path)
    report = json.loads(item.report.read_text()); report[field] = value
    item.report.write_text(json.dumps(report))
    with pytest.raises(ValueError): audit(item)


def test_missing_report_recomputed_and_version_change_visible(tmp_path):
    item = fixture_pool(tmp_path)
    report = json.loads(item.report.read_text()); report['rdkit_version'] = 'old'
    item.report.write_text(json.dumps(report))
    assert 'RDKit changed' in audit(item)[0]['historical_report_status']
    item.report.unlink()
    assert 'absent' in audit(item)[0]['historical_report_status']


def test_duplicate_source_id_fails(tmp_path):
    item = fixture_pool(tmp_path)
    lines = item.raw.read_text().splitlines(); lines[1] = lines[0]
    item.raw.write_text('\n'.join(lines) + '\n')
    with pytest.raises(ValueError, match='Integrity'): audit(item)


def test_deterministic_disjoint_nested_split(tmp_path):
    summary, records = audit(fixture_pool(tmp_path))
    split = make_split(summary, records, (2, 4), query_count=2)
    assert split == make_split(summary, list(reversed(records)), (4, 2), query_count=2)
    assert index_records(split, 2) == index_records(split, 4)[:2]
    q = {r['canonical_smiles'] for r in split['queries']}
    assert q.isdisjoint(r['canonical_smiles'] for r in split['index_order'])
    assert len(split['queries']) == 2 and split['unused_count'] == 0
    assert split['split_id'] != make_split(summary, records, (2, 4), 2, 43)['split_id']
    with pytest.raises(ValueError): index_records(split, 3)


def test_insufficient_and_duplicate_structures_fail(tmp_path):
    summary, records = audit(fixture_pool(tmp_path))
    with pytest.raises(ValueError, match='Need'): make_split(summary, records)
    with pytest.raises(ValueError, match='unique'): make_split(summary, records + records, (2,), 1)


def test_inventory_only_approved_completed_names(tmp_path):
    fixture_pool(tmp_path)
    rows = inventory(tmp_path)
    assert len(rows) == 9 and sum(r['raw_present'] for r in rows) == 1
    assert dataset(tmp_path, 'chemicalspace_new', 'scale_1m_00').package == 'scale_1m_00'
    with pytest.raises(ValueError): dataset(tmp_path, 'otava_chemical', 'scale_1m_00')
    with pytest.raises(ValueError): dataset(tmp_path, 'otava_chemical', 'archive/package_00')


def test_descriptor_values_and_no_overwrite(tmp_path):
    rows = descriptor_rows([{'source_id': 'ethanol', 'canonical_smiles': 'CCO'}])
    assert rows[0]['heavy_atoms'] == 3
    assert rows[0]['molecular_weight'] == pytest.approx(46.069, abs=.01)
    path = tmp_path / 'out.json'
    save_json(path, {'a': 1}); save_json(path, {'a': 1})
    with pytest.raises(ValueError): save_json(path, {'a': 2})
    assert json.loads(path.read_text()) == {'a': 1}
