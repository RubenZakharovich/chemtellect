"""Actual installed NMSLIB parser and direct-distance compatibility tests."""
import json
import numpy as np
import pytest
from chemtellect import nmslib_adapter as adapter


def test_representation_and_empty():
    rows, positions = adapter.synthetic_fixture()
    before = rows.copy()
    strings = adapter.to_nmslib_strings(rows)
    assert len(strings) == len(rows)
    for value, expected in zip(strings, positions):
        tokens = value.split(' ')
        assert len(tokens) == 2048
        assert [i for i, token in enumerate(tokens) if token == '1'] == expected
        assert set(tokens) <= {'0','1'}
    assert strings[0] == ' '.join(['0']*2048)
    assert strings[1] == ' '.join(['1']*2048)
    assert strings[4].startswith('0 ') and strings[4].endswith(' 0')
    # All packed byte values are valid, including values greater than one.
    all_bytes = np.arange(256,dtype=np.uint8).reshape(1,256)
    tokens = adapter.to_nmslib_strings(all_bytes)[0].split()
    assert tokens == [str((byte >> bit)&1) for byte in range(256) for bit in range(8)]
    assert np.array_equal(rows,before)
    assert adapter.to_nmslib_strings(np.empty((0,256),dtype=np.uint8)) == []
    with pytest.raises(ValueError,match='empty fixture'):
        adapter.compare_pairs(np.empty((0,256),dtype=np.uint8),[])


def test_actual_distances_ids_and_empty_union():
    rows,_ = adapter.synthetic_fixture()
    ids = [42,3,100,8,91,15,27]
    registry, positions = adapter.register_rows(rows,ids)
    assert positions == {identifier:i for i,identifier in enumerate(ids)}
    assert registry.getDistance(positions[100],positions[8]) == 0
    assert registry.getDistance(positions[100],positions[15]) == 1
    # Three bits overlap a seven-bit set: d = 1 - 3/7.
    assert registry.getDistance(positions[100],positions[91]) == pytest.approx(4/7,abs=1e-6,rel=0)
    result = adapter.compare_pairs(rows,ids)
    assert result['comparisons'] == 28
    zero = result['pairs'][0]
    assert zero['id_a'] == zero['id_b'] == 42
    assert zero['rdkit_distance'] == 1.0
    assert zero['nmslib_distance'] is None  # Observed NMSLIB 2.1.2 empty-union NaN.
    assert zero['diagnostic'] and not zero['passed']
    assert not result['passed']
    for pair in result['pairs'][1:]:
        assert pair['passed']
        assert pair['absolute_error'] <= 1e-6
        assert pair['nmslib_distance'] == pytest.approx(pair['reverse_nmslib_distance'],abs=1e-6,rel=0)
        if pair['union']:
            assert pair['rdkit_distance'] == pytest.approx(1-pair['intersection']/pair['union'],abs=1e-6,rel=0)
    assert adapter.compare_pairs(rows[1:],[3,100,8,91,15,27])['passed']


@pytest.mark.parametrize('rows', [np.zeros((1,256),dtype=float), np.zeros(256,dtype=np.uint8),
    np.zeros((1,255),dtype=np.uint8), np.zeros((1,1,256),dtype=np.uint8), [[0]*256]])
def test_bad_rows(rows):
    with pytest.raises(ValueError,match='uint8 ndarray'):
        adapter.to_nmslib_strings(rows)
    with pytest.raises(ValueError): adapter.register_rows(rows,[0])


@pytest.mark.parametrize('ids', [[1,1],[1],[1,2.0],[True,2],['1',2],[2**31,0]])
def test_bad_ids(ids):
    with pytest.raises(ValueError): adapter.register_rows(np.zeros((2,256),dtype=np.uint8),ids)


def real_inputs():
    rows,_ = adapter.synthetic_fixture()
    queries = np.tile(rows[2],(10,1)); index = np.tile(rows[4],(10,1))
    records = {role:[dict(row_index=i,source_id=f'{role}:{i}',line_number=i+1) for i in range(10)]
               for role in ('queries','index')}
    return queries,index,records


def test_real_fixture_mapping():
    queries,index,records = real_inputs()
    result = adapter.verify_real_fixture(queries,index,records)
    assert result['passed'] and result['comparisons'] == 210
    assert result['selected_records'][0] == dict(temporary_id=0,role='queries',row_index=0,source_id='queries:0',line_number=1)
    assert result['selected_records'][10] == dict(temporary_id=10,role='index',row_index=0,source_id='index:0',line_number=1)
    with pytest.raises(ValueError):
        adapter.verify_real_fixture(queries[:9],index,records)


def manifest_for(queries, index):
    return dict(artifact_id='a'*64,split_id='b'*64,identity={'scientific_configuration':'test'},
                source='arbitrary_source',package='independent_pool',raw_sha256_inherited='c'*64,
                query_count=len(queries),largest_index_count=len(index))


def test_reports_pass_nonzero_and_preserve_diagnostic(tmp_path):
    queries,index,records = real_inputs()
    paths=[]
    for _ in range(2):
        path,report = adapter.run_verification(tmp_path,manifest_for(queries,index),queries,index,records)
        paths.append(path)
        assert report['passed'] is True
        assert report['number_of_supported_comparisons'] == 231
        assert report['supported_checks']['real']['comparisons'] == 210
        diagnostic=report['unsupported_diagnostics']['pairs'][0]
        assert diagnostic['nmslib_distance'] is None
        assert diagnostic['nonfinite_classification']['nmslib'] == 'NaN'
        assert diagnostic['rdkit_distance'] == 1
        assert diagnostic['passed'] is False
        assert report['maximum_absolute_error_supported_domain'] <= 1e-6
        assert path.parent.parts[-3:] == ('arbitrary_source','independent_pool','a'*64)
        # Strict JSON disallows nonfinite numeric constants; textual classification is allowed.
        json.loads(path.read_text(), parse_constant=lambda v: pytest.fail(v))
    assert paths[0] != paths[1]


def test_nonzero_guard_valid_and_immutable():
    import copy
    from chemtellect.fingerprints import validate_nonzero_fingerprints
    queries,index,records=real_inputs()
    original=(queries.copy(),index.copy(),copy.deepcopy(records))
    summary=validate_nonzero_fingerprints(queries,index,records)
    assert summary == dict(policy_identifier='nonzero_fingerprints_v1',query_rows_checked=10,
                           index_rows_checked=10,query_zero_rows=0,index_zero_rows=0)
    assert np.array_equal(original[0],queries) and np.array_equal(original[1],index)
    assert original[2] == records


@pytest.mark.parametrize('query_zeros,index_zeros,index_count', [([2],[],10),([],[4],10),
    ([0,8],[1,5],10),([],[14],20),([],[1204],1205)])
def test_nonzero_guard_all_rows(query_zeros,index_zeros,index_count):
    import copy
    from chemtellect.fingerprints import validate_nonzero_fingerprints, NonzeroFingerprintError
    queries,index,records=real_inputs()
    index=np.tile(index[0],(index_count,1))
    records['index']=[dict(row_index=i,source_id=f'late:{i}',line_number=100+i) for i in range(index_count)]
    queries[query_zeros]=0; index[index_zeros]=0
    original=(queries.copy(),index.copy(),copy.deepcopy(records))
    with pytest.raises(NonzeroFingerprintError) as caught:
        validate_nonzero_fingerprints(queries,index,records)
    error=caught.value
    assert error.summary['query_zero_rows']==len(query_zeros)
    assert error.summary['index_zero_rows']==len(index_zeros)
    assert error.summary['index_rows_checked']==index_count
    assert error.affected_records == [dict(role=role,**records[role][i])
        for role,indices in [('queries',query_zeros),('index',index_zeros)] for i in indices]
    assert np.array_equal(original[0],queries) and np.array_equal(original[1],index)
    assert original[2]==records


@pytest.mark.parametrize('damage',['length','duplicate','negative','source','line','dtype','rank','width'])
def test_guard_rejects_mismatch(damage):
    from chemtellect.fingerprints import validate_nonzero_fingerprints
    q,i,r=real_inputs()
    if damage=='length': r['index'].pop()
    if damage=='duplicate': r['index'][1]['row_index']=0
    if damage=='negative': r['index'][0]['row_index']=-1
    if damage=='source': r['queries'][0]['source_id']=''
    if damage=='line': r['queries'][0]['line_number']=0
    if damage=='dtype': q=q.astype(float)
    if damage=='rank': q=q[0]
    if damage=='width': q=q[:,:255]
    with pytest.raises(ValueError): validate_nonzero_fingerprints(q,i,r)


def test_original_selected_row_identity():
    from chemtellect.fingerprints import validate_nonzero_fingerprints, NonzeroFingerprintError
    q,i,r=real_inputs()
    q=q[[4,7]]; i=i[[9]]
    r={'queries':[r['queries'][4],r['queries'][7]],'index':[r['index'][9]]}
    q[1]=0
    with pytest.raises(NonzeroFingerprintError) as caught: validate_nonzero_fingerprints(q,i,r)
    assert caught.value.affected_records == [dict(role='queries',row_index=7,source_id='queries:7',line_number=8)]


def test_guard_failure_stops_backend_work_and_reports(tmp_path,monkeypatch):
    q,i,r=real_inputs(); i[9]=0
    def forbidden(*args,**kwargs): pytest.fail('Backend preparation before full-input validation')
    monkeypatch.setattr(adapter,'unpack_fingerprints',forbidden)
    monkeypatch.setattr(adapter,'register_rows',forbidden)
    with pytest.raises(adapter.VerificationFailure) as caught:
        adapter.run_verification(tmp_path,manifest_for(q,i),q,i,r)
    report=json.loads(caught.value.report_path.read_text())
    assert report['passed'] is False
    assert report['nonzero_validation']['status']=='failed'
    assert report['nonzero_validation']['summary']['index_zero_rows']==1
    assert report['nonzero_validation']['affected_records']==[dict(role='index',**r['index'][9])]
    assert report['nonzero_validation_ns'] >= 0
    assert all(c==dict(status='not_run',passed=None) for c in report['supported_checks'].values())
    assert report['unsupported_diagnostics']['status']=='not_run'
    assert report['number_of_supported_comparisons']==0
    assert 'index row 9' in str(caught.value)
    with pytest.raises(Exception): adapter.verify_real_fixture(q,i,r)


def test_runner_rejects_prefix(tmp_path):
    q,i,r=real_inputs(); m=manifest_for(q,i); m['largest_index_count']=1001
    with pytest.raises(adapter.VerificationFailure) as caught:
        adapter.run_verification(tmp_path,m,q,i,r)
    assert 'not a prefix' in caught.value.report['diagnostic']
    assert caught.value.report['supported_checks']['real']['status']=='not_run'
