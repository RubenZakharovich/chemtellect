import copy
import json
import numpy as np
import pytest
from rdkit import DataStructs, rdBase
from chemtellect.dataset_overview import make_split
from chemtellect.fingerprints import APPROVED
from chemtellect import fingerprint_store as store


@pytest.fixture
def split_path(tmp_path):
    rows = [dict(source_id=str(i),line_number=i+1,smiles=s,canonical_smiles=s)
            for i,s in enumerate(['C','CC','CCC','N','O','CCO'])]
    summary = dict(source='chemicalspace_new',package='package_00',sha256='a'*64,
                   rdkit_version=rdBase.rdkitVersion,representative_policy='first raw physical line per canonical structure')
    split = make_split(summary,rows,(2,4),2)
    path = tmp_path/'split.json'; path.write_text(json.dumps(split))
    return path


def test_known_bits():
    mixed = DataStructs.ExplicitBitVect(2048)
    for i in (0,7,8,2047): mixed.SetBit(i)
    full = DataStructs.CreateFromBitString('1'*2048)
    empty = DataStructs.ExplicitBitVect(2048)
    packed = store.pack_fingerprints([mixed,full,empty])
    assert packed[0,0] == 129 and packed[0,1] == 1 and packed[0,-1] == 128
    assert packed[0,2:-1].sum() == 0
    assert store.unpack_fingerprints(packed) == [mixed,full,empty]


def test_generation_reuse_prefix(tmp_path,split_path,monkeypatch):
    events=[]
    result = store.prepare_artifact(tmp_path,split_path,APPROVED,2,lambda *e:events.append(e))
    directory=result['artifact_path']
    before={p.name:p.read_bytes() for p in directory.iterdir()}
    q,i=store.load_artifact(directory,split_path,APPROVED,tmp_path,2)
    q2,i2=store.load_artifact(directory,split_path,APPROVED,tmp_path,4)
    assert np.array_equal(q,q2) and np.array_equal(i,i2[:2])
    assert q.shape == (2,256) and events[-1] == ('index',4,4)
    split=json.loads(split_path.read_text())
    generator=store.build_generator(APPROVED)
    for role,key in [('queries','queries'),('index','index_order')]:
        expected=[v for _,batch,_ in store.generate_batches(split[key],role,generator,2) for v in batch]
        actual=np.load(directory/f'{role}.npy',allow_pickle=False)
        assert store.unpack_fingerprints(actual) == expected
    with pytest.raises(ValueError): store.load_artifact(directory,split_path,APPROVED,tmp_path,3)
    monkeypatch.setattr(store,'build_generator',lambda *_:pytest.fail('reuse generated'))
    reused=store.prepare_artifact(tmp_path,split_path,APPROVED)
    assert reused['report']['operation']=='reused'
    assert reused['report']['timings_ns']['queries_generation'] is None
    assert reused['report_path'] != result['report_path']
    assert before == {p.name:p.read_bytes() for p in directory.iterdir()}


@pytest.mark.parametrize('change', ['digest','duplicate','counts','version','sizes','policy'])
def test_invalid_split(split_path,change):
    s=json.loads(split_path.read_text())
    if change=='digest': s['split_id']='b'*64
    else:
        if change=='duplicate': s['index_order'][0]=s['queries'][0]
        if change=='counts': s['unused_count']=1
        if change=='version': s['rdkit_version']='old'
        if change=='sizes': s['index_sizes']=[2,2]
        if change=='policy': s['seed']=43
        s['split_id']=store.digest({k:v for k,v in s.items() if k!='split_id'})
    with pytest.raises(ValueError): store.validate_split(s)


@pytest.mark.parametrize('damage',['truncate','dtype','shape','mapping','incomplete'])
def test_corruption(tmp_path,split_path,damage):
    result=store.prepare_artifact(tmp_path,split_path,APPROVED)
    d=result['artifact_path']; m=json.loads((d/'manifest.json').read_text())
    if damage=='truncate': (d/'index.npy').write_bytes(b'bad')
    if damage in ('dtype','shape'):
        np.save(d/'index.npy',np.zeros((4,256) if damage=='dtype' else (3,256),dtype=np.int32 if damage=='dtype' else np.uint8))
        m['files']['index.npy']=store.file_info(d/'index.npy')
    if damage=='mapping':
        r=json.loads((d/'records.json').read_text()); r['queries'].reverse()
        (d/'records.json').write_text(json.dumps(r)); m['files']['records.json']=store.file_info(d/'records.json')
    if damage=='incomplete': m['status']='incomplete'
    (d/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError): store.prepare_artifact(tmp_path,split_path,APPROVED)


def test_failure_and_lock(tmp_path,split_path,monkeypatch):
    s=json.loads(split_path.read_text()); aid=store.digest(store.identity(s,APPROVED))
    parent=tmp_path/'data/processed/chemicalspace_new/package_00/fingerprints'
    original=store.write_json
    def fail(path,value):
        if path.name=='records.json': raise OSError('injected')
        original(path,value)
    monkeypatch.setattr(store,'write_json',fail)
    with pytest.raises(OSError): store.prepare_artifact(tmp_path,split_path,APPROVED)
    assert not (parent/aid).exists()
    assert list(parent.glob('*.staging-*'))
    (parent/f'.{aid}.lock').touch()
    with pytest.raises(RuntimeError,match='lock exists'): store.prepare_artifact(tmp_path,split_path,APPROVED)


def test_publication_conflict_preserves_existing(tmp_path,split_path,monkeypatch):
    original=store.verify_artifact
    def conflict(directory,*args):
        result=original(directory,*args)
        if '.staging-' in directory.name:
            final=directory.parent/result[0]['artifact_id']
            final.mkdir()
            (final/'sentinel').write_text('preserve')
        return result
    monkeypatch.setattr(store,'verify_artifact',conflict)
    with pytest.raises(FileExistsError,match='Publication conflict'):
        store.prepare_artifact(tmp_path,split_path,APPROVED)
    parent=tmp_path/'data/processed/chemicalspace_new/package_00/fingerprints'
    assert [p.read_text() for p in parent.glob('*/sentinel')] == ['preserve']
