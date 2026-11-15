"""Verified packed artifacts; no retrieval or molecular regeneration on reuse."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from time import perf_counter_ns
from uuid import uuid4

import numpy as np
from rdkit import DataStructs, rdBase
from chemtellect.dataset_overview import POLICY, dataset
from chemtellect.fingerprints import validate_configuration, build_generator, generate_batches

STORAGE = {'version': 1, 'dtype': 'uint8', 'bits': 2048, 'bytes_per_row': 256,
           'bitorder': 'little', 'format': 'uncompressed npy', 'bit_indices': '0 through 2047'}


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def validate_split(split):
    dataset(Path('.'), split['source'], split['package'])
    if split['split_schema_version'] != 1 or digest({k:v for k,v in split.items() if k != 'split_id'}) != split['split_id']:
        raise ValueError('Invalid split schema or digest')
    if split['rdkit_version'] != rdBase.rdkitVersion:
        raise ValueError('Split/generation RDKit version mismatch; researcher resolution required')
    if (split['canonicalization'] != POLICY or split['representative_policy'] != 'first raw physical line per canonical structure'
        or split['seed'] != 42 or split['ordering_algorithm'] != 'sha256-json-seed-source-package-id-v1'):
        raise ValueError('Unapproved split policy')
    sizes = split['index_sizes']
    if not sizes or any(type(n) is not int or n <= 0 for n in sizes) or len(set(sizes)) != len(sizes):
        raise ValueError('Invalid index sizes')
    for name in ('query_count', 'eligible_unique_count', 'unused_count'):
        if type(split[name]) is not int or split[name] < (0 if name == 'unused_count' else 1):
            raise ValueError('Invalid counts')
    rows = split['queries'] + split['index_order']
    if (len(split['queries']) != split['query_count'] or len(split['index_order']) != max(sizes)
        or len(rows) + split['unused_count'] != split['eligible_unique_count']):
        raise ValueError('Inconsistent split counts')
    for row in rows:
        if any(not isinstance(row.get(k), str) or not row[k].strip() for k in ('source_id','smiles','canonical_smiles')) or type(row.get('line_number')) is not int or row['line_number'] <= 0:
            raise ValueError('Invalid record fields')
    for key in ('source_id', 'canonical_smiles'):
        if len({r[key] for r in rows}) != len(rows):
            raise ValueError('Duplicate identity or query/index overlap')
    raw = split['raw_sha256']
    if not isinstance(raw, str) or len(raw) != 64 or any(c not in '0123456789abcdef' for c in raw):
        raise ValueError('Invalid inherited raw digest')
    return split


def identity(split, config):
    validate_split(split)
    validate_configuration(config)
    return {'version': 1, 'split_id': split['split_id'], 'configuration': config,
            'rdkit_version': rdBase.rdkitVersion, 'storage': STORAGE}


def pack_fingerprints(vectors):
    bits = np.zeros((len(vectors), 2048), dtype=np.uint8)
    for i, fp in enumerate(vectors):
        if not isinstance(fp, DataStructs.ExplicitBitVect) or fp.GetNumBits() != 2048:
            raise ValueError('Expected 2048-bit ExplicitBitVect')
        DataStructs.ConvertToNumpyArray(fp, bits[i])
    return np.packbits(bits, axis=1, bitorder='little')


def unpack_fingerprints(packed):
    if packed.dtype != np.uint8 or packed.ndim != 2 or packed.shape[1] != 256:
        raise ValueError('Expected uint8 (n, 256) array')
    # One row at a time avoids a full unpacked 100K array.
    # RDKit's binary text uses the same little-bit-order bytes as this store.
    return [DataStructs.CreateFromBinaryText(row.tobytes()) for row in packed]


def mapping(split):
    return {'schema_version': 1, 'source': split['source'], 'package': split['package'],
            **{role: [dict(row_index=i, source_id=r['source_id'], line_number=r['line_number'])
                      for i,r in enumerate(split[key])] for role,key in [('queries','queries'),('index','index_order')]}}


def file_info(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return {'sha256': h.hexdigest(), 'bytes': path.stat().st_size}


def write_json(path, value):
    with path.open('xb') as stream:
        stream.write(canonical_json(value) + b'\n')


def linkage(root, path, split):
    return {'source': split['source'], 'package': split['package'], 'split_id': split['split_id'],
            'split_path': str(path.resolve().relative_to(root.resolve())),
            'split_file_sha256': file_info(path)['sha256'], 'raw_sha256_inherited': split['raw_sha256']}


def verify_artifact(directory, split_path, config, root):
    directory, split_path, root = map(Path, (directory, split_path, root))
    split = validate_split(json.loads(split_path.read_text()))
    ident = identity(split, config)
    artifact_id = digest(ident)
    manifest = json.loads((directory/'manifest.json').read_text())
    expected = {'manifest_schema_version':1, 'status':'complete', 'artifact_id':artifact_id,
                'identity':ident, **linkage(root, split_path, split), 'configuration':config,
                'storage':STORAGE, 'query_count':split['query_count'],
                'largest_index_count':max(split['index_sizes']), 'index_sizes':split['index_sizes']}
    for key,value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f'Manifest mismatch: {key}')
    arrays = {}
    for name in ('queries.npy','index.npy','records.json'):
        if file_info(directory/name) != manifest['files'][name]:
            raise ValueError(f'Checksum/size mismatch: {name}')
    if json.loads((directory/'records.json').read_text()) != mapping(split):
        raise ValueError('Ordered record mapping mismatch')
    for role, count in [('queries',split['query_count']),('index',max(split['index_sizes']))]:
        array = np.load(directory/f'{role}.npy', allow_pickle=False, mmap_mode='r')
        if array.dtype != np.uint8 or array.shape != (count,256) or manifest['dimensions'][role] != [count,256]:
            raise ValueError(f'Invalid array shape/dtype: {role}')
        arrays[role] = array
    return manifest, arrays


def load_artifact(directory, split_path, config, root, index_size):
    manifest, arrays = verify_artifact(directory, split_path, config, root)
    if type(index_size) is not int or index_size not in manifest['index_sizes']:
        raise ValueError('Index size not registered in split')
    return arrays['queries'], arrays['index'][:index_size]


def environment(root):
    def git(*args):
        p = subprocess.run(['git','-C',str(root),*args], capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else None
    dirty = git('status','--porcelain')
    return {'python':platform.python_version(), 'rdkit':rdBase.rdkitVersion, 'numpy':np.__version__,
            'git_revision':git('rev-parse','HEAD'), 'git_dirty':None if dirty is None else bool(dirty)}


def utc():
    return datetime.now(timezone.utc).isoformat()


def prepare_artifact(root, split_path, config, batch_size=1000, progress=None):
    """Generate once or verify reuse. Never repair, overwrite, or break stale locks."""
    started, total = utc(), perf_counter_ns()
    root, split_path = Path(root), Path(split_path)
    if type(batch_size) is not int or not 1 <= batch_size <= 1000:
        raise ValueError('batch_size must be 1..1000; thread count is fixed at one')
    t = perf_counter_ns()
    split = validate_split(json.loads(split_path.read_text()))
    ident = identity(split, config)
    aid = digest(ident)
    link = linkage(root, split_path, split)
    timings = {'verification':perf_counter_ns()-t, 'generator_initialization':None,
               'artifact_writing':None, 'reload_verification':None,
               **{f'{r}_{s}':None for r in ('queries','index') for s in ('parsing','generation','packing')}}
    parent = root/'data/processed'/split['source']/split['package']/'fingerprints'
    parent.mkdir(parents=True, exist_ok=True)
    final = parent/aid
    lock = parent/f'.{aid}.lock'
    try:
        fd = os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f'Publication lock exists: {lock}; inspect owner/staging; never automatically broken') from error
    os.close(fd)
    staging = None
    try:
        operation = 'reused' if final.exists() else 'generated'
        if operation == 'generated':
            staging = Path(tempfile.mkdtemp(prefix=f'.{aid}.staging-', dir=parent))
            t = perf_counter_ns(); generator = build_generator(config)
            timings['generator_initialization'] = perf_counter_ns()-t
            arrays = {}
            for role,key in [('queries','queries'),('index','index_order')]:
                arrays[role] = np.empty((len(split[key]),256), dtype=np.uint8)
                for stage in ('parsing','generation','packing'):
                    timings[f'{role}_{stage}'] = 0
                for offset,vectors,elapsed in generate_batches(split[key],role,generator,batch_size):
                    for stage,ns in elapsed.items(): timings[f'{role}_{stage}'] += ns
                    t = perf_counter_ns(); packed = pack_fingerprints(vectors)
                    arrays[role][offset:offset+len(vectors)] = packed
                    timings[f'{role}_packing'] += perf_counter_ns()-t
                    t = perf_counter_ns()
                    if [v.ToBitString() for v in unpack_fingerprints(packed)] != [v.ToBitString() for v in vectors]:
                        raise ValueError('Bit reconstruction failed')
                    timings['reload_verification'] = (timings['reload_verification'] or 0) + perf_counter_ns()-t
                    if progress: progress(role, offset+len(vectors),len(split[key]))
            t = perf_counter_ns()
            for role,array in arrays.items():
                with (staging/f'{role}.npy').open('xb') as stream: np.save(stream,array,allow_pickle=False)
            write_json(staging/'records.json',mapping(split))
            files = {name:file_info(staging/name) for name in ('queries.npy','index.npy','records.json')}
            timings['artifact_writing'] = perf_counter_ns()-t
            t = perf_counter_ns()
            for role,array in arrays.items():
                if not np.array_equal(array,np.load(staging/f'{role}.npy',allow_pickle=False)):
                    raise ValueError('Reload differs from generated packed bits')
            if json.loads((staging/'records.json').read_text()) != mapping(split): raise ValueError('Mapping differs')
            timings['reload_verification'] += perf_counter_ns()-t
            manifest = {'manifest_schema_version':1,'status':'complete','artifact_id':aid,'identity':ident,
                        **link,'configuration':config,'generator_info':generator.GetInfoString(),
                        'environment':environment(root),'query_count':split['query_count'],
                        'largest_index_count':max(split['index_sizes']),'index_sizes':split['index_sizes'],
                        'storage':STORAGE,'dimensions':{r:list(a.shape) for r,a in arrays.items()},
                        'created_utc':utc(),'execution':{'batch_size':batch_size,'thread_count':1},'files':files}
            t = perf_counter_ns(); write_json(staging/'manifest.json',manifest)
            timings['artifact_writing'] += perf_counter_ns()-t
            t = perf_counter_ns(); verify_artifact(staging,split_path,config,root)
            timings['reload_verification'] += perf_counter_ns()-t
            if final.exists(): raise FileExistsError(f'Publication conflict: {final}')
            staging.rename(final)
        t = perf_counter_ns(); manifest,_ = verify_artifact(final,split_path,config,root)
        timings['reload_verification'] = (timings['reload_verification'] or 0) + perf_counter_ns()-t
        timings['total'] = perf_counter_ns()-total
        report = {'artifact_id':aid,'operation':operation,'start_utc':started,'end_utc':utc(),
                  'environment':environment(root),'timings_ns':timings,
                  'execution':{'batch_size':batch_size,'thread_count':1},
                  'completed_counts':{'queries':split['query_count'],'index':max(split['index_sizes'])}}
        reports = root/'experiments/results/fingerprints'/split['source']/split['package']/aid
        reports.mkdir(parents=True,exist_ok=True)
        report_path = reports/f'run_{uuid4().hex}.json'
        write_json(report_path,report)
        return {'artifact_path':final,'report_path':report_path,'manifest':manifest,'report':report}
    except BaseException as error:
        error.add_note(f'Artifact preparation stopped; inspect retained staging {staging}. No automatic recovery.')
        raise
    finally:
        lock.unlink()
