"""Resumable local raw-to-results orchestration; benchmarking stays in hnsw_benchmark.

The CLI uses a fresh process for every large stage and every size. Immutable stage
markers and run manifests, rather than directory order, determine what is reused.
"""
import argparse
from contextlib import contextmanager
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter_ns
from uuid import uuid4

import psutil
from rdkit import rdBase

from chemtellect import dataset_overview as overview
from chemtellect import fingerprint_store as store
from chemtellect.benchmark_context import memory_snapshot
from chemtellect.fingerprints import load_configuration, validate_nonzero_fingerprints


def read_json(path):
    return json.loads(Path(path).read_text())


def configuration(root, path):
    config = read_json(path)
    overview.dataset(root, config['source'], config['package'])
    if (config['schema_version'] != 1 or config['query_count'] != 100 or config['seed'] != 42
            or config['descriptor_limit'] is not None or not config['index_sizes']
            or config['index_sizes'] != sorted(set(config['index_sizes']))
            or any(type(n) is not int or n < 100 for n in config['index_sizes'])):
        raise ValueError('Unapproved pipeline selection/policy')
    load_configuration(Path(root)/config['fingerprint_configuration'])
    from chemtellect.hnsw_adapter import CONFIGURATION
    if read_json(Path(root)/config['hnsw_configuration']) != CONFIGURATION:
        raise ValueError('HNSW configuration differs from approved settings')
    return config


def resource_observation(root, stage, count=0, calibration=None):
    """Conservative planning envelope, not a measured peak or a promise of fit."""
    snap = memory_snapshot(stage)
    swap = psutil.swap_memory()
    disk = shutil.disk_usage(root)
    # Includes retained split/mapping JSON, RDKit objects, FPSim2/native graph,
    # evidence serialization, and row-wise scratch. Text rows are now streamed.
    estimated = 2*2**30 + count*6500
    if stage in ('audit', 'split', 'fingerprints'):
        estimated = {'audit': 3, 'split': 4, 'fingerprints': 6}[stage]*2**30
    if calibration and count > calibration['index_size']:
        # Calibrate only size-dependent growth after full-parent verification;
        # the baseline already includes loading this parent's full identity data.
        base = next(s['process_rss_bytes'] for s in calibration['snapshots']
                    if s['stage'] == 'verified_and_guarded_inputs')
        observed = max(s['process_rss_bytes'] for s in calibration['snapshots'])
        projected = base + max(0, observed-base)*count/calibration['index_size'] + 2**30
        estimated = max(estimated, int(projected))
    snap.update(swap_total_bytes=swap.total, swap_used_bytes=swap.used,
                disk_free_bytes=disk.free, estimated_additional_ram_bytes=estimated,
                estimated_disk_bytes=2*2**30 + count*4000,
                estimate_basis='2 GiB fixed + 6500 bytes/index row; conservative uncalibrated envelope; '
                'includes Python identities, saved evidence, RDKit/FPSim2/HNSW; excludes unrelated processes',
                reference_payload_bytes=100*count*8,
                avoided_full_text_characters=4095*count,
                thread_environment={k: os.environ.get(k) for k in (
                    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
    if stage in ('audit', 'split', 'fingerprints'):
        snap['estimate_basis'] = f'{estimated/2**30:g} GiB stage envelope for {stage}; includes full-parent Python identities and verification copies; not measured'
    if calibration:
        snap['calibration'] = {'index_size': calibration['index_size'],
                               'maximum_observed_rss_snapshot_bytes': max(s['process_rss_bytes'] for s in calibration['snapshots']),
                               'run_id': calibration['run_id'],
                               'interpretation': 'snapshot-based extrapolation with 1 GiB margin; not a measured peak'}
    snap['safe_to_start'] = (snap['system_ram_available_bytes'] >= estimated + 512*2**20
                             and disk.free >= snap['estimated_disk_bytes'])
    return snap


def verify_audit(root, directory, config):
    directory = Path(directory)
    marker = read_json(directory/'completion.json')
    for key, name in [('audit', 'audit.json'), ('eligible', 'eligible.jsonl')]:
        if marker[key] != store.file_info(directory/name):
            raise ValueError(f'Audit cache checksum mismatch: {name}')
    summary = read_json(directory/'audit.json')
    item = overview.dataset(root, config['source'], config['package'])
    if (summary['source'] != config['source'] or summary['package'] != config['package']
            or summary['sha256'] != store.file_info(item.raw)['sha256']
            or summary['acquisition_manifest'] != read_json(item.manifest)
            or summary['rdkit_version'] != rdBase.rdkitVersion
            or summary['canonicalization'] != overview.POLICY
            or summary['fresh_validation']['success'] is not True
            or summary['fresh_validation']['integrity_rejected_records'] != 0
            or summary['fresh_validation']['sha256'] != summary['sha256']
            or summary['counts'] != {k: summary['fresh_validation'][k] for k in overview.COUNTS}):
        raise ValueError('Audit cache provenance/policy mismatch')
    counts = summary['counts']
    if (counts['total_input_records'] != summary['acquisition_manifest']['actual_extracted_record_count']
            or counts['total_input_records'] != counts['successfully_parsed_records'] + counts['invalid_molecular_records']
            or counts['unique_canonical_structures'] != counts['successfully_parsed_records'] - counts['surplus_duplicate_records']):
        raise ValueError('Audit count reconciliation failed')
    return summary


def audit_stage(root, directory, config):
    directory = Path(directory)
    if (directory/'completion.json').exists():
        return verify_audit(root, directory, config)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f'Incomplete audit retained at {directory}; inspect before retry')
    directory.mkdir(parents=True, exist_ok=True)
    start = perf_counter_ns()
    item = overview.dataset(root, config['source'], config['package'])
    try:
        summary, records = overview.audit(item)
    except Exception as error:
        from dataclasses import asdict
        from chemtellect.data_integrity import validate_jsonl
        diagnostic = {'status': 'failed_audit', 'error': f'{type(error).__name__}: {error}',
                      'integrity': asdict(validate_jsonl(item.raw)),
                      'acquisition_manifest': read_json(item.manifest)}
        store.write_json(directory/'failure.json', diagnostic)
        raise
    store.write_json(directory/'audit.json', summary)
    with (directory/'eligible.jsonl').open('x') as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n')
    store.write_json(directory/'completion.json', {
        'audit': store.file_info(directory/'audit.json'), 'eligible': store.file_info(directory/'eligible.jsonl'),
        'elapsed_ns': perf_counter_ns()-start, 'memory': memory_snapshot('after_audit')})
    return verify_audit(root, directory, config)


def findings_table(root, directory, audit_path, config, summary):
    """Resolve duplicate IDs to physical line references without retaining the raw pool."""
    directory = Path(directory)
    output, marker = directory/'findings.csv', directory/'findings.json'
    link = store.file_info(Path(audit_path)/'completion.json')
    if marker.exists():
        saved = read_json(marker)
        if saved['audit_completion'] != link or saved['file'] != store.file_info(output):
            raise ValueError('Findings resume verification failed')
        return saved
    findings = {i['source_id']: {'category': 'chemical_exclusion', 'reason': i['reason'], 'canonical_smiles': ''}
                for i in summary['fresh_validation']['invalid_records']}
    for canonical, ids in summary['fresh_validation']['duplicate_groups'].items():
        for p, sid in enumerate(ids):
            findings[sid] = {'category': 'duplicate_representative' if p == 0 else 'surplus_duplicate',
                             'reason': 'first physical representative' if p == 0 else f'same canonical structure as {ids[0]}',
                             'canonical_smiles': canonical}
    item = overview.dataset(root, config['source'], config['package'])
    count = 0
    with output.open('x', newline='') as stream, item.raw.open('rb') as raw:
        writer = csv.DictWriter(stream, fieldnames=['line_number', 'source_id', 'category', 'reason', 'canonical_smiles'])
        writer.writeheader()
        for number, line in enumerate(raw, 1):
            sid = json.loads(line)['source_id']
            if sid in findings:
                writer.writerow({'line_number': number, 'source_id': sid, **findings[sid]}); count += 1
    if count != len(findings): raise ValueError('Could not locate every molecular finding')
    saved = {'audit_completion': link, 'file': store.file_info(output), 'rows': count}
    store.write_json(marker, saved)
    return saved


def eligible_records(directory, summary):
    with (Path(directory)/'eligible.jsonl').open() as stream:
        records = [json.loads(line) for line in stream]
    if (len(records) != summary['counts']['unique_canonical_structures']
            or len({r['source_id'] for r in records}) != len(records)
            or len({r['canonical_smiles'] for r in records}) != len(records)
            or [r['line_number'] for r in records] != sorted(r['line_number'] for r in records)):
        raise ValueError('Eligible identity/coverage/order mismatch')
    return records


def descriptors_stage(root, directory, audit_path, config):
    summary = verify_audit(root, audit_path, config)
    output = Path(directory)/'descriptors.csv'
    marker = Path(directory)/'descriptors.json'
    linkage = {'audit': store.file_info(Path(audit_path)/'completion.json'),
               'eligible': store.file_info(Path(audit_path)/'eligible.jsonl'), 'descriptor_limit': None,
               'rdkit_version': rdBase.rdkitVersion}
    if marker.exists():
        saved = read_json(marker)
        if (saved['linkage'] != linkage or saved['file'] != store.file_info(output)
                or saved['rows'] != summary['counts']['unique_canonical_structures']):
            raise ValueError('Descriptor resume verification failed')
        return saved
    start = perf_counter_ns()
    count = 0
    aggregate = {}
    with output.open('x', newline='') as stream, (Path(audit_path)/'eligible.jsonl').open() as source:
        writer = None
        batch = []
        def emit(rows):
            nonlocal writer, count
            values = overview.descriptor_rows(rows)
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(values[0])); writer.writeheader()
            writer.writerows(values)
            for row in values:
                count += 1
                for key, value in row.items():
                    if key == 'source_id': continue
                    a = aggregate.setdefault(key, {'minimum': value, 'maximum': value, 'sum': 0.0})
                    a['minimum'] = min(a['minimum'], value); a['maximum'] = max(a['maximum'], value); a['sum'] += value
        for line in source:
            batch.append(json.loads(line))
            if len(batch) == 256:
                emit(batch); batch = []
        if batch: emit(batch)
    if count != summary['counts']['unique_canonical_structures']:
        raise ValueError('Descriptors do not cover every eligible structure')
    for a in aggregate.values(): a['mean'] = a.pop('sum')/count
    saved = {'linkage': linkage, 'rows': count, 'summary': aggregate, 'file': store.file_info(output),
             'elapsed_ns': perf_counter_ns()-start, 'memory': memory_snapshot('after_descriptors')}
    store.write_json(marker, saved)
    return saved


def split_stage(root, directory, audit_path, config):
    summary = verify_audit(root, audit_path, config)
    records = eligible_records(audit_path, summary)
    split = overview.make_split(summary, records, config['index_sizes'], config['query_count'], config['seed'])
    store.validate_split(split)
    # Artifact identity is independent of report-directory names. Resolve an
    # existing matching artifact's verified split linkage before choosing a path.
    fp_config = load_configuration(Path(root)/config['fingerprint_configuration'])
    aid = store.digest(store.identity(split, fp_config))
    artifact = Path(root)/'data/processed'/config['source']/config['package']/'fingerprints'/aid
    split_path = Path(root)/'data/processed'/config['source']/config['package']/'splits'/f'{split["split_id"]}.json'
    if (artifact/'manifest.json').exists():
        candidate = Path(root)/read_json(artifact/'manifest.json')['split_path']
        candidate.resolve().relative_to(Path(root).resolve())
        manifest, _ = store.verify_artifact(artifact, candidate, fp_config, root)
        if manifest['split_id'] != split['split_id'] or read_json(candidate) != split:
            raise ValueError('Existing artifact belongs to a different deterministic parent')
        split_path = candidate
    else:
        overview.save_json(split_path, split)
    selected = {r['source_id'] for r in split['queries']+split['index_order']}
    unused = [r for r in records if r['source_id'] not in selected]
    unused_path = Path(directory)/'unused_records.json'
    overview.save_json(unused_path, {'split_id': split['split_id'], 'records': unused})
    return {'split_path': str(split_path.relative_to(root)), 'split_id': split['split_id'],
            'query_count': split['query_count'], 'index_sizes': split['index_sizes'],
            'selected_count': len(split['queries'])+len(split['index_order']), 'unused_count': split['unused_count'],
            'file': store.file_info(split_path), 'unused_records_file': store.file_info(unused_path)}


def fingerprints_stage(root, directory, config):
    def progress(role, completed, total):
        if completed == total or completed % 100000 == 0:
            print(json.dumps({'role': role, 'completed': completed, 'total': total,
                              'memory_snapshot': memory_snapshot('fingerprint_generation_progress')}), flush=True)
    split_path = Path(root)/read_json(Path(directory)/'split.json')['split_path']
    result = store.prepare_artifact(root, split_path,
                                   load_configuration(Path(root)/config['fingerprint_configuration']), progress=progress)
    manifest, arrays = store.verify_artifact(result['artifact_path'], split_path,
                                            result['manifest']['configuration'], root)
    split = store.validate_split(read_json(split_path))
    guard = validate_nonzero_fingerprints(arrays['queries'], arrays['index'], store.mapping(split))
    return {'artifact_path': str(result['artifact_path'].relative_to(root)),
            'report_path': str(result['report_path'].relative_to(root)),
            'artifact_id': manifest['artifact_id'], 'operation': result['report']['operation'],
            'nonzero_validation': guard}


def benchmark_stage(root, directory, config, size):
    from chemtellect.hnsw_benchmark import run_comparison, ComparisonFailure
    from chemtellect.hnsw_reporting import write_charts
    completed = [read_json(Path(directory)/f'index_{n}.json') for n in config['index_sizes']
                 if n < size and (Path(directory)/f'index_{n}.json').exists()]
    calibration = completed[-1].get('resource_calibration') if completed else None
    entry_resource = resource_observation(root, 'worker_before_loading', size, calibration)
    resource_path = Path(directory)/f'resource_{size}_{uuid4().hex}.json'
    store.write_json(resource_path, entry_resource)
    if not entry_resource['safe_to_start']:
        raise RuntimeError(f'Resource gate rejected size {size}: {resource_path}; '
                           f'available={entry_resource["system_ram_available_bytes"]}; '
                           f'estimated additional={entry_resource["estimated_additional_ram_bytes"]} plus 512 MiB reserve')
    fp = read_json(Path(directory)/'fingerprints.json')
    try:
        path, report = run_comparison(root, Path(root)/fp['artifact_path'], size,
            expected_selection={k: config[k] for k in ('source', 'package', 'query_count')} | {'index_size': size},
            report_writer=write_charts, progress=lambda snapshot: print(json.dumps(snapshot), flush=True))
    except ComparisonFailure as error:
        store.write_json(Path(directory)/f'failed_index_{size}.json', {
            'path': str(error.report_path.relative_to(root)), 'status': 'failed_validation',
            'error': error.report['error'], 'run_id': error.report['run_id'],
            'manifest': store.file_info(error.report_path/'manifest.json')})
        raise
    # Verification runs in another process in the controller, releasing backend memory.
    return {'path': str(path.relative_to(root)), 'run_id': report['run_id'],
            'manifest': store.file_info(path/'manifest.json'), 'status': 'complete',
            'measured_searches': len(report['benchmark']['measured_results']),
            'warmup_searches': config['query_count']*4*3,
            'resource_calibration': {'index_size': size, 'run_id': report['run_id'],
                                     'snapshots': report['memory_snapshots']},
            'entry_resource_path': str(resource_path.relative_to(root))}


def verified_run(root, saved, config, size, artifact):
    from chemtellect.hnsw_benchmark import load_comparison_run
    path = Path(root)/saved['path']
    if saved['manifest'] != store.file_info(path/'manifest.json'):
        raise ValueError('Resumed run manifest changed')
    report = load_comparison_run(path, root)
    if (not report['passed'] or report['run_id'] != saved['run_id']
            or any(report[k] != config[k] for k in ('source', 'package', 'query_count'))
            or report['index_size'] != size or report['linkage']['artifact_id'] != artifact['artifact_id']
            or report['linkage']['artifact_path'] != artifact['artifact_path']):
        raise ValueError('Resumed run belongs to a different selection/artifact')
    return report


def _worker(root, directory, audit_path, config, stage, size):
    start = perf_counter_ns()
    if stage in ('audit', 'split', 'fingerprints'):
        resource = resource_observation(root, stage)
        store.write_json(Path(directory)/f'resource_{stage}_{uuid4().hex}.json', resource)
        if not resource['safe_to_start']:
            raise RuntimeError(f'Resource gate {stage}: available={resource["system_ram_available_bytes"]}; '
                               f'estimated additional={resource["estimated_additional_ram_bytes"]} plus 512 MiB reserve')
    if stage == 'audit':
        summary = audit_stage(root, audit_path, config)
        result = {'counts': summary['counts'], 'findings': findings_table(root, directory, audit_path, config, summary)}
    elif stage == 'descriptors': result = descriptors_stage(root, directory, audit_path, config)
    elif stage == 'split': result = split_stage(root, directory, audit_path, config)
    elif stage == 'fingerprints': result = fingerprints_stage(root, directory, config)
    elif stage == 'benchmark': result = benchmark_stage(root, directory, config, size)
    elif stage == 'verify':
        result = verified_run(root, read_json(Path(directory)/f'index_{size}.json'), config, size,
                              read_json(Path(directory)/'fingerprints.json'))
        return {'run_id': result['run_id'], 'verified': True}
    else: raise ValueError(stage)
    result['stage_wall_ns'] = perf_counter_ns()-start
    # Existing stage marker may only be reused unchanged; reruns use new experiment directories.
    name = f'index_{size}.json' if stage == 'benchmark' else f'{stage}.json'
    target = Path(directory)/name
    if not target.exists(): store.write_json(target, result)
    elif stage == 'fingerprints':
        store.write_json(Path(directory)/f'fingerprint_verification_{uuid4().hex}.json', result)
    return result


@contextmanager
def pipeline_lock(directory):
    """An interrupted controller requires inspection, never an automatic second graph."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory/'.pipeline.lock'
    owner = json.dumps({'created_utc': store.utc(), 'namespace_pid': os.getpid(),
                        'directory': str(directory.resolve()), 'argv': sys.argv})+'\n'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f'Pipeline controller lock exists: {lock}; inspect the live process and staging evidence; '
                           'never automatically removed or bypassed') from error
    with os.fdopen(fd, 'w') as stream:
        stream.write(owner)
    try:
        yield
    finally:
        if lock.read_text() != owner:
            raise RuntimeError(f'Pipeline lock ownership changed: {lock}')
        lock.unlink()


def run_pipeline(root, config_path, *, directory=None, audit_path=None):
    root = Path(root).resolve()
    config = configuration(root, config_path)
    directory = (Path(directory).resolve() if directory else
                 root/'experiments/results/pipeline'/config['source']/config['package']/f'experiment_{uuid4().hex}')
    directory.relative_to(root)
    with pipeline_lock(directory):
        return _run_pipeline(root, config_path, directory=directory, audit_path=audit_path)


def _run_pipeline(root, config_path, *, directory=None, audit_path=None):
    root = Path(root).resolve(); config_path = Path(config_path).resolve()
    config = configuration(root, config_path)
    directory = (Path(directory).resolve() if directory else
                 root/'experiments/results/pipeline'/config['source']/config['package']/f'experiment_{uuid4().hex}')
    directory.relative_to(root)
    directory.mkdir(parents=True, exist_ok=True)
    settings = {'configuration': config, 'configuration_file': store.file_info(config_path)}
    overview.save_json(directory/'settings.json', settings)
    if audit_path is None and (directory/'state.json').exists():
        audit_path = root/read_json(directory/'state.json')['audit_path']
    audit_path = Path(audit_path).resolve() if audit_path else directory/'acquisition_audit'
    if (directory/'state.json').exists():
        state = read_json(directory/'state.json')
        if state['audit_path'] != str(audit_path.relative_to(root)):
            raise ValueError('Resume audit path differs')
        if state['configuration'] != config:
            raise ValueError('Resume configuration differs')
        if 'last_invocation_wall_ns' in state:
            state.setdefault('previous_invocations', []).append({
                'finished_utc': state.get('finished_utc'), 'status': state['status'],
                'wall_ns_before_reporting': state['last_invocation_wall_ns']})
        if state['blockers']:
            state.setdefault('previous_attempt_blockers', []).append(state['blockers'])
            state['blockers'] = []
        state['status'] = 'running'
    else:
        state = {'schema_version': 1, 'started_utc': store.utc(), 'status': 'running',
                 'configuration': config, 'audit_path': str(audit_path.relative_to(root)),
                 'stages': {}, 'sizes': {}, 'resources': [], 'blockers': []}
    def checkpoint():
        # Mutable controller journal; scientific stages and run directories are immutable.
        temp = directory/'state.tmp'
        temp.write_text(json.dumps(state, indent=2)+'\n'); temp.replace(directory/'state.json')
    def call(stage, size=0):
        smaller = [v['resource_calibration'] for v in state['sizes'].values()
                   if 'resource_calibration' in v and v['resource_calibration']['index_size'] < size]
        calibration = max(smaller, key=lambda c: c['index_size']) if smaller else None
        observation = resource_observation(root, stage, size if stage == 'benchmark' else 0,
                                           calibration if stage == 'benchmark' else None)
        state['resources'].append(observation); checkpoint()
        if not observation['safe_to_start']:
            raise RuntimeError(f'Resource gate {stage}/{size}: available {observation["system_ram_available_bytes"]} bytes; '
                               f'estimated additional {observation["estimated_additional_ram_bytes"]} plus 512 MiB reserve; '
                               f'disk free {observation["disk_free_bytes"]} bytes')
        command = [sys.executable, '-m', 'chemtellect.raw_pipeline', '--root', str(root),
                   '--config', str(config_path), '--directory', str(directory), '--audit-path', str(audit_path),
                   '--worker', stage, '--size', str(size)]
        with (directory/f'{stage}_{size}.log').open('a') as log:
            process = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        if process.returncode:
            failure_path = directory/f'failed_index_{size}.json'
            detail = read_json(failure_path) if failure_path.exists() else None
            raise RuntimeError(f'{stage}/{size} worker exited {process.returncode}; see {stage}_{size}.log; failure={detail}')
        return read_json(directory/(f'index_{size}.json' if stage == 'benchmark' else f'{stage}.json')) if stage != 'verify' else {'verified': True}
    start = perf_counter_ns()
    try:
        print(f'Experiment: {directory}', flush=True)
        state['stages']['audit'] = call('audit'); checkpoint()
        summary = verify_audit(root, audit_path, config)
        required = max(config['index_sizes'])+config['query_count']
        eligible = summary['counts']['unique_canonical_structures']
        state['eligibility'] = {'required': required, 'eligible': eligible,
                                'deficit': max(0, required-eligible), 'passed': eligible >= required}
        print(f'Eligible unique structures: {eligible:,}; required: {required:,}', flush=True)
        # All-eligible diagnostic coverage is useful even if the full split is blocked.
        state['stages']['descriptors'] = call('descriptors'); checkpoint()
        if eligible < required:
            raise RuntimeError(f'Eligibility gate: {eligible} unique structures, need {required}; deficit {required-eligible}. No approved parent created.')
        for stage in ('split', 'fingerprints'):
            print(f'Stage: {stage}', flush=True)
            state['stages'][stage] = call(stage); checkpoint()
        for size in config['index_sizes']:
            print(f'Size: {size:,}', flush=True)
            if (directory/f'index_{size}.json').exists():
                call('verify', size)
            else:
                state['sizes'][str(size)] = call('benchmark', size); checkpoint()
                call('verify', size)
            state['sizes'][str(size)] = read_json(directory/f'index_{size}.json')
            state['sizes'][str(size)]['loader_verified'] = True; checkpoint()
        state['status'] = 'complete'
    except Exception as error:
        state['status'] = 'partial' if state['sizes'] else 'blocked'
        state['blockers'].append(f'{type(error).__name__}: {error}')
        print(state['blockers'][-1], file=sys.stderr, flush=True)
    state['finished_utc'] = store.utc()
    state['last_invocation_wall_ns'] = perf_counter_ns()-start
    checkpoint()
    try:
        render_report(root, directory)
    except Exception as error:
        state['blockers'].append(f'Report verification/export failed: {type(error).__name__}: {error}')
        state['status'] = 'partial' if state['sizes'] else 'blocked'
        checkpoint()
        # Never hide a failed audit or an invalid artifact behind a successful export.
        diagnostic = directory/f'partial_report_{uuid4().hex}.md'
        diagnostic.write_text('# Incomplete raw-to-results workflow\n\nStatus: '+state['status']+
                              '\n\n'+'\n\n'.join(state['blockers'])+
                              '\n\nSee state.json, worker logs and retained stage evidence. No unverified speed/quality publication.\n')
    return directory, state


def load_pipeline(root, directory):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    state = read_json(directory/'state.json')
    config = state['configuration']
    if read_json(directory/'settings.json')['configuration'] != config:
        raise ValueError('Pipeline configuration journal mismatch')
    summary = verify_audit(root, root/state['audit_path'], config)
    if (directory/'findings.json').exists():
        findings_table(root, directory, root/state['audit_path'], config, summary)
    if (directory/'descriptors.json').exists():
        descriptors_stage(root, directory, root/state['audit_path'], config)
    reports = []
    if (directory/'fingerprints.json').exists():
        fp = read_json(directory/'fingerprints.json')
        split_marker = read_json(directory/'split.json')
        split_path = root/split_marker['split_path']
        split_path.resolve().relative_to(root)
        split = store.validate_split(read_json(split_path))
        if (split['raw_sha256'] != summary['sha256'] or split['index_sizes'] != config['index_sizes']
                or split['query_count'] != config['query_count']):
            raise ValueError('Saved split differs from pipeline selection')
        unused_path = directory/'unused_records.json'
        if (split_marker['file'] != store.file_info(split_path)
                or split_marker['unused_records_file'] != store.file_info(unused_path)):
            raise ValueError('Saved split or unused records checksum mismatch')
        unused = read_json(unused_path)
        selected_ids = {r['source_id'] for r in split['queries']+split['index_order']}
        selected_canonical = {r['canonical_smiles'] for r in split['queries']+split['index_order']}
        if (unused['split_id'] != split['split_id'] or len(unused['records']) != split['unused_count']
                or len({r['source_id'] for r in unused['records']}) != split['unused_count']
                or len({r['canonical_smiles'] for r in unused['records']}) != split['unused_count']
                or any(r['source_id'] in selected_ids or r['canonical_smiles'] in selected_canonical
                       for r in unused['records'])):
            raise ValueError('Unused record identity/coverage mismatch')
        store.verify_artifact(root/fp['artifact_path'], split_path,
                              load_configuration(root/config['fingerprint_configuration']), root)
        for size in config['index_sizes']:
            if str(size) in state['sizes']:
                reports.append(verified_run(root, state['sizes'][str(size)], config, size, fp))
        if reports:
            from chemtellect.hnsw_reporting import verify_parent_series
            verify_parent_series(reports)
    if state['status'] == 'complete' and len(reports) != len(config['index_sizes']):
        raise ValueError('Completed pipeline lacks requested runs')
    return state, summary, reports


def render_report(root, directory):
    """Export compact Markdown plus complete CSV tables and readable charts."""
    import pandas as pd
    from chemtellect.hnsw_reporting import tables, write_overview
    root, directory = Path(root).resolve(), Path(directory).resolve()
    state, summary, reports = load_pipeline(root, directory)
    output = directory/f'report_{uuid4().hex}'
    output.mkdir()
    lines = [f'# {state["configuration"]["source"]}/{state["configuration"]["package"]}',
             f'Status: **{state["status"]}**. Requested sizes: {state["configuration"]["index_sizes"]}.',
             f'Raw SHA-256: `{summary["sha256"]}`. RDKit: {summary["rdkit_version"]}.',
             'Canonical sanitized isomeric SMILES; no standardization; first physical representative.',
             '\n## Acquisition and selection',
             *[f'- {k}: {v:,}' for k, v in summary['counts'].items()],
             f'- Eligibility gate: {state.get("eligibility")}',
             f'- Audit and full findings: `{state["audit_path"]}/audit.json` (IDs and lines for chemical exclusions).',
             '- `findings.csv` resolves chemical exclusions and all duplicate group members to source IDs and physical lines; '
             '`eligible.jsonl` retains each first physical representative.',
             f'- Split: {state["stages"].get("split", "NOT CREATED")}',
             f'- Fingerprints: {state["stages"].get("fingerprints", "NOT PREPARED")}',
             '\n## Measured comparisons',
             'Latencies are µs: median, mean and linear p95 across 100 per-query five-pass medians. '
             'One warmup plus five measured passes per k; k=10/25/50/100; thresholds=.3/.5/.7/.9. '
             'Speedup is a ratio of headline medians. Component medians need not add to the total.',
             'No confidence interval was estimated; repetitions share queries and one graph.']
    if len({r['source_files']['nmslib_adapter.py']['sha256'] for r in reports}) > 1:
        lines.append('Preparation implementation note: the first three sizes used NumPy scalar-to-string '
                     'NMSLIB encoding; 1M used an exhaustive byte lookup producing identical 4,095-character '
                     'strings. Conversion costs therefore reflect this disclosed optimization. Native graph '
                     'construction, prepared query values, timed searches and scientific configurations are unchanged.')
    latency_rows, quality_rows, threshold_rows = [], [], []
    for report in reports:
        n = report['index_size']
        saved = state['sizes'][str(n)]
        lines.extend([f'\n### {n:,} index rows', f'Run `{report["run_id"]}`: `{saved["path"]}`.',
                      f'All synthetic and real preflight gates PASS; {saved["measured_searches"]:,} measured '
                      f'and {saved["warmup_searches"]:,} warmup searches. Loader verified.'])
        for k in (10, 25, 50, 100):
            compared = [r for r in report['latency_comparisons'] if r['k'] == k]
            fastest = min(compared, key=lambda r: r['median'])
            hq = report['quality']['by_backend']['hnsw'][str(k)]['top_k']
            lines.append(f'At k={k}, {fastest["backend"]} has the lowest measured headline median '
                         f'({fastest["median"]/1000:.3f} µs, {fastest["rdkit_speedup"]:.3f}× RDKit). '
                         f'HNSW loses {100*(1-hq["mean"]):.6f} percentage points of mean tie-aware recall; '
                         f'its worst query recall is {hq["minimum"]:.8f}.')
        frames = tables(report)
        for name, frame in frames.items():
            frame.to_csv(output/f'index_{n}_{name.lower().replace(" ", "_")}.csv')
        for row in report['latency_comparisons']:
            q = report['quality']['by_backend'][row['backend']][str(row['k'])]
            item = {'index_size': n, 'backend': row['backend'], 'k': row['k'],
                    **{f'{k}_us': row[k]/1000 for k in ('median', 'mean', 'p95')},
                    **{k: row[k] for k in ('rdkit_speedup', 'fpsim2_speedup')},
                    'saved_vs_rdkit_us': row['rdkit_time_saved_ns']/1000,
                    'saved_vs_fpsim2_us': row['fpsim2_time_saved_ns']/1000}
            latency_rows.append(item)
            quality_rows.append({'index_size': n, 'backend': row['backend'], 'k': row['k'], **q['top_k']})
            lines.append(f'- {row["backend"]} k={row["k"]}: median {item["median_us"]:.3f} µs; '
                         f'mean {item["mean_us"]:.3f}; p95 {item["p95_us"]:.3f}; '
                         f'{row["rdkit_speedup"]:.3f}× RDKit, {row["fpsim2_speedup"]:.3f}× FPSim2; '
                         f'saved {item["saved_vs_rdkit_us"]:.3f}/{item["saved_vs_fpsim2_us"]:.3f} µs; '
                         f'tie-aware recall mean {q["top_k"]["mean"]:.8f}, min {q["top_k"]["minimum"]:.8f}.')
        pd.DataFrame(report['quality']['top_k_rows']).to_csv(output/f'index_{n}_recall_distribution.csv', index=False)
        threshold_rows.extend({'index_size': n, **row} for row in report['quality']['threshold_evaluations']['rows'])
        costs = dict(report['timings_ns'])
        costs['outside_search_validation_and_result_assembly'] = report['benchmark']['validation_and_result_assembly_ns']
        pd.DataFrame([{'stage': k, 'seconds': v/1e9} for k, v in costs.items()]).to_csv(output/f'index_{n}_costs.csv', index=False)
        for chart in report['charts']:
            if chart.endswith('.png'):
                relative = os.path.relpath(Path(root)/saved['path']/chart, output)
                lines.append(f'![{chart}]({relative})')
    for name, rows in [('latency', latency_rows), ('quality', quality_rows), ('threshold_per_query', threshold_rows)]:
        if rows: pd.DataFrame(rows).to_csv(output/f'{name}.csv', index=False)
    if len(reports) == 4:
        for name in write_overview(output, reports):
            if name.endswith('.png'): lines.append(f'![{name}]({name})')
    if reports:
        from chemtellect.pipeline_reporting import write_parent_details
        for name in write_parent_details(output, reports):
            if name.endswith('.png'): lines.append(f'![{name}]({name})')
    worker_resources = [read_json(p) for p in directory.glob('resource_*.json')]
    pd.DataFrame(state['resources']+worker_resources).to_csv(output/'resources.csv', index=False)
    stage_costs = [{'stage': k, 'seconds': v.get('elapsed_ns', v.get('stage_wall_ns', 0))/1e9}
                   for k, v in state['stages'].items() if 'elapsed_ns' in v or 'stage_wall_ns' in v]
    audit_completion = read_json(Path(root)/state['audit_path']/'completion.json')
    original_audit_seconds = (audit_completion.get('elapsed_ns', 0)/1e9
                              if 'elapsed_ns' in audit_completion else audit_completion.get('elapsed_seconds'))
    if original_audit_seconds is not None:
        stage_costs.append({'stage': 'original_fresh_acquisition_audit', 'seconds': original_audit_seconds})
    stage_costs += [{'stage': f'index_{n}_worker', 'seconds': saved['stage_wall_ns']/1e9}
                    for n, saved in state['sizes'].items() if 'stage_wall_ns' in saved]
    if 'last_invocation_wall_ns' in state:
        stage_costs.append({'stage': 'controller_invocation_before_reporting', 'seconds': state['last_invocation_wall_ns']/1e9})
    stage_costs.extend({'stage': f'previous_controller_invocation_{i}_before_reporting',
                        'seconds': entry['wall_ns_before_reporting']/1e9}
                       for i, entry in enumerate(state.get('previous_invocations', [])))
    pd.DataFrame(stage_costs).to_csv(output/'workflow_costs.csv', index=False)
    if 'fingerprints' in state['stages']:
        fp_operations = [('original_preparation', state['stages']['fingerprints'])]
        fp_operations += [(p.stem, read_json(p)) for p in directory.glob('fingerprint_verification_*.json')]
        rows = []
        for label, preparation in fp_operations:
            evidence = read_json(Path(root)/preparation['report_path'])
            rows.extend({'operation_id': label, 'operation': evidence['operation'], 'stage': k,
                         'seconds': value/1e9 if value is not None else None}
                        for k, value in evidence['timings_ns'].items())
        pd.DataFrame(rows).to_csv(output/'fingerprint_preparation_costs.csv', index=False)
    lines.extend(['\n## Resource observations and costs',
                  'See resources.csv and per-size memory/cost CSVs. RSS and available RAM are snapshots, not peaks. '
                  'Resource estimates are planning envelopes, not observations. Worker wall times include loader/persistence '
                  'overhead where measured; checksumming/publication and controller/report overhead are not all individually timed.',
                  '\n## Blockers', *(state['blockers'] or ['None.']), '\n## Limitations',
                  'One acquisition, 100 fixed held-out queries, one graph at each size, machine-specific sequential '
                  'in-memory performance and uncontrolled scheduling/cache effects. No measured 100M claim or biological '
                  'validity claim. Quality measures agreement with exact Tanimoto in the approved representation. '
                  'Threshold recall excludes empty ground truth and is limited by k; per-query qualifying counts and '
                  'denominators are retained. Historical acquisitions are separate. Speed and quality have no combined score.'])
    (output/'report.md').write_text('\n\n'.join(lines)+'\n')
    files = {p.name: store.file_info(p) for p in output.iterdir()}
    store.write_json(output/'manifest.json', {'schema_version': 1, 'files': files,
        'source_state': store.file_info(directory/'state.json'),
        'runs': [state['sizes'][str(r['index_size'])] for r in reports]})
    (directory/'latest_report.json').write_text(json.dumps({'path': str(output.relative_to(root)),
        'manifest': store.file_info(output/'manifest.json')}, indent=2)+'\n')
    print(f'Report: {output/"report.md"}', flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--config', type=Path, default=Path('experiments/configs/chemicalspace_new_scale_1m_pipeline.json'))
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--audit-path', type=Path)
    parser.add_argument('--mode', choices=('execute', 'load'), default='load')
    parser.add_argument('--worker', choices=('audit', 'descriptors', 'split', 'fingerprints', 'benchmark', 'verify'), help=argparse.SUPPRESS)
    parser.add_argument('--size', type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.worker:
        _worker(root, args.directory, args.audit_path, configuration(root, args.config), args.worker, args.size)
        return 0
    if args.mode == 'load':
        if args.directory is None: parser.error('--directory is required for load mode')
        render_report(root, args.directory)
        return 0
    directory, state = run_pipeline(root, args.config, directory=args.directory, audit_path=args.audit_path)
    return 0 if state['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
