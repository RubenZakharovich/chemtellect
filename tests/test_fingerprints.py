import copy
import pytest
from chemtellect.fingerprints import APPROVED, build_generator, generate_batches, validate_configuration


def vectors(smiles, batch_size=2):
    rows = [dict(source_id=str(i),line_number=i+1,canonical_smiles=s) for i,s in enumerate(smiles)]
    return [v for _,batch,_ in generate_batches(rows,'queries',build_generator(APPROVED),batch_size) for v in batch]


def test_effective_settings_and_determinism():
    g = build_generator(APPROVED)
    assert 'includeRingMembership=1' in g.GetInfoString()
    a = vectors(['CCO','OCC','F[C@H](Cl)Br','F[C@@H](Cl)Br','FC(Cl)Br','CCO'])
    assert all(v.GetNumBits() == 2048 for v in a)
    assert a[0] == a[1] == a[5]
    assert len({v.ToBitString() for v in a[2:5]}) == 3
    assert a == vectors(['CCO','OCC','F[C@H](Cl)Br','F[C@@H](Cl)Br','FC(Cl)Br','CCO'],1)
    bad = copy.deepcopy(APPROVED); bad['generator']['radius'] = 3
    with pytest.raises(ValueError): validate_configuration(bad)


def test_failure_identity():
    with pytest.raises(ValueError,match='queries: 1 line 2: parsing failed'):
        vectors(['C','not smiles','N'])
