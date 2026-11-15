"""Tables and charts for one fixed three-backend run; no pooled dataset summaries."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from chemtellect.hnsw_benchmark import BACKENDS


def tables(report):
    provenance = {**{key: report[key] for key in ('source', 'package', 'experiment', 'query_count', 'index_size')},
        **{key: report['linkage'][key] for key in ('artifact_id', 'split_id', 'raw_sha256_inherited')},
        'Git revision': report['environment']['git_revision'], 'validation passed': report['passed']}
    checks = []
    for label, gate in [('Synthetic HNSW', report['synthetic']), ('Synthetic FPSim2', report['fpsim2_synthetic']),
                        ('Real FPSim2', report['preflight']['fpsim2']), ('Real HNSW', report['preflight']['hnsw'])]:
        entries = gate['checks'] if gate else []
        checks.append({'Gate': label, 'Status': ('PASS' if gate['passed'] else 'FAIL') if gate else 'NOT RUN',
                       'Passed responses': sum(c['passed'] for c in entries),
                       'Failed responses': sum(not c['passed'] for c in entries)})
    benchmark = report['benchmark']
    if benchmark:
        for b in BACKENDS:
            entries = [i['check'] for i in benchmark['measured_results'] if i['backend'] == b]
            checks.append({'Gate': f'Measured {b}', 'Status': 'PASS' if report['passed'] else 'INCOMPLETE / FAILED',
                           'Passed responses': sum(c['passed'] for c in entries),
                           'Failed responses': sum(not c['passed'] for c in entries)})
    timings = {**report['timings_ns']}
    if report['hnsw_preparation']:
        timings.update({f'hnsw_{k}': v for k, v in report['hnsw_preparation'].items() if k.endswith('_ns')})
    if report['fpsim2_representation']:
        timings.update({f'fpsim2_{k}': v for k, v in report['fpsim2_representation']['conversion_timings_ns'].items()})
    result = {'Provenance': pd.Series(provenance, name='Value').to_frame(),
        'Configuration': pd.Series(report['configuration'], name='Value').to_frame(),
        'Correctness': pd.DataFrame(checks).set_index('Gate'),
        'Preparation and costs': pd.DataFrame([{'Stage': k.replace('_', ' '), 'Elapsed (ms)': v/1e6}
                                              for k, v in timings.items()]).set_index('Stage'),
        'Memory snapshots (not peaks)': pd.DataFrame([{'Stage': s['stage'], 'UTC': s['sampled_utc'],
            'Total system RAM (GiB)': s['system_ram_total_bytes']/2**30,
            'Available system RAM (GiB)': s['system_ram_available_bytes']/2**30,
            'Process RSS (MiB)': s['process_rss_bytes']/2**20} for s in report['memory_snapshots']]).set_index('Stage')}
    if not report['passed']:
        return result
    rows = []
    recall, thresholds = [], []
    for item in report['latency_comparisons']:
        b, k = item['backend'], item['k']
        stages = report['benchmark']['summaries'][b]['by_k'][str(k)]['across_query_medians_ns']
        names = ('scoring_ns', 'selection_ordering_ns') if b == 'rdkit' else ('native_search_ns', 'normalization_ns')
        rows.append({'Backend': b, 'k': k, 'Scoring / search median (µs)': stages[names[0]]['median']/1000,
            'Selection / normalization median (µs)': stages[names[1]]['median']/1000,
            'Total median (µs)': item['median']/1000, 'Total mean (µs)': item['mean']/1000,
            'p95 of per-query median latency (µs)': item['p95']/1000,
            'Time saved vs RDKit (µs)': item['rdkit_time_saved_ns']/1000,
            'Speedup vs RDKit (×)': item['rdkit_speedup'], 'Speedup vs FPSim2 (×)': item['fpsim2_speedup']})
        quality = report['quality']['by_backend'][b][str(k)]
        recall.append({'Backend': b, 'k': k, **quality['top_k']})
        thresholds.extend({'Backend': b, 'k': k, 'Threshold': float(tau), **values}
                          for tau, values in quality['thresholds'].items())
    result.update({'Sequential latency': pd.DataFrame(rows).set_index(['Backend', 'k']),
        'Tie-aware top-k recall': pd.DataFrame(recall).set_index(['Backend', 'k']),
        'Top-k-limited threshold recall': pd.DataFrame(thresholds).set_index(['Backend', 'k', 'Threshold'])})
    return result


def write_charts(directory, report):
    if not report['passed']:
        return []
    directory = Path(directory)
    frames = tables(report)
    latency = frames['Sequential latency'].reset_index()
    quality = frames['Tie-aware top-k recall'].reset_index()
    title = f"{report['package']} / {report['index_size']:,} index — {report['experiment']}"
    figures = []
    fig, ax = plt.subplots(figsize=(9, 4.6), layout='constrained')
    latency.pivot(index='k', columns='Backend', values='Total median (µs)')[list(BACKENDS)].plot.bar(ax=ax, rot=0)
    ax.set(title=f'Three sequential backends\n{title}', xlabel='Independent top-k request',
           ylabel='Median of per-query median latency (µs)', ylim=(0, None))
    ax.grid(axis='y', alpha=.25)
    figures.append(('three_backend_latency', fig))
    fig, ax = plt.subplots(figsize=(9, 4.4), layout='constrained')
    latency[latency.Backend != 'rdkit'].pivot(index='k', columns='Backend', values='Speedup vs RDKit (×)').plot.bar(ax=ax, rot=0)
    ax.axhline(1, linestyle='--', color='#b33030', label='1×: equal median latency')
    ax.set(title=f'Observed speedup vs fresh RDKit\n{title}', xlabel='Independent top-k request',
           ylabel='RDKit median / backend median (×)', ylim=(0, None))
    ax.grid(axis='y', alpha=.25); ax.legend()
    figures.append(('three_backend_speedup', fig))
    fig, ax = plt.subplots(figsize=(9, 4.6), layout='constrained')
    merged = latency.merge(quality, on=['Backend', 'k'])
    for b in BACKENDS:
        selected = merged[merged.Backend == b]
        ax.plot(selected['Total median (µs)'], selected['mean'], marker='o', label=b)
        for _, row in selected.iterrows():
            ax.annotate(f"k={row['k']}", (row['Total median (µs)'], row['mean']),
                        xytext=(4, -12 if b == 'fpsim2' else 6), textcoords='offset points', fontsize=8)
    ax.set(title=f'First-measured-pass recall vs latency\n{title}',
           xlabel='Median of per-query median latency (µs)', ylabel='Mean tie-aware top-k recall',
           xlim=(0, None), ylim=(0, 1.08))
    ax.grid(alpha=.25); ax.legend()
    figures.append(('recall_vs_latency', fig))
    names = []
    for stem, fig in figures:
        for suffix in ('png', 'svg'):
            name = f'{stem}.{suffix}'
            fig.savefig(directory/name)
            names.append(name)
        plt.close(fig)
    return names


def verify_parent_series(reports):
    """Reject mixed parents/configurations and reordered or unregistered prefixes.

    Schema 1 has identity through its verified split/artifact linkage. Schema 2
    additionally carries ordered query and prefix digests checked by the loader.
    """
    if not reports or not all(r['passed'] for r in reports):
        raise ValueError('Parent series requires successful runs')
    first = reports[0]
    sizes = [r['index_size'] for r in reports]
    if sizes != sorted(set(sizes)):
        raise ValueError('Parent sizes must be strictly increasing')
    for r in reports:
        if (any(r[k] != first[k] for k in ('source', 'package', 'query_count', 'configuration', 'fingerprint_identity'))
                or any(r['linkage'][k] != first['linkage'][k] for k in
                       ('split_id', 'artifact_id', 'raw_sha256_inherited', 'split_file_sha256'))
                or r.get('settings') != first.get('settings')):
            raise ValueError('Cross-parent or configuration mismatch')
    if any('selection_metadata' in r for r in reports):
        if not all('selection_metadata' in r for r in reports):
            raise ValueError('Mixed selection metadata versions')
        largest = reports[-1]['selection_metadata']
        for r in reports:
            m = r['selection_metadata']
            if (m['query_identity_sha256'] != largest['query_identity_sha256']
                    or m['index_sizes'] != largest['index_sizes']
                    or r['index_size'] not in m['index_sizes']
                    or set(m['prefix_identity_sha256']) != {str(n) for n in m['index_sizes'] if n <= r['index_size']}
                    or any(largest['prefix_identity_sha256'].get(k) != v for k, v in m['prefix_identity_sha256'].items())):
                raise ValueError('Query identities or nested prefix ordering mismatch')
    return reports


def write_overview(directory, reports):
    """Presentation-only views of the four completed runs, keeping experiments separate."""
    from matplotlib.lines import Line2D

    expected = [('package_00', 1000), ('scale_100k_00', 1000),
                ('scale_100k_00', 10000), ('scale_100k_00', 100000)]
    historical = [(r['package'], r['index_size']) for r in reports] == expected
    parent = reports[1:] if historical else reports
    verify_parent_series(parent)
    if len(reports) != 4:
        raise ValueError('Overview requires old pilot plus parent, or a verified four-size parent')
    if not all(r['passed'] for r in reports):
        raise ValueError('Overview requires successful runs')
    colors = dict(zip(BACKENDS, ('#386cb0', '#ef8a32', '#389343')))
    markers = {10: 'o', 25: 's', 50: '^', 100: 'D'}
    figures = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for ax, report in zip(axes.flat, reports):
        for b in BACKENDS:
            rows = sorted((r for r in report['latency_comparisons'] if r['backend'] == b), key=lambda r: r['k'])
            for row in rows:
                recall = report['quality']['by_backend'][b][str(row['k'])]['top_k']['mean']
                ax.scatter(row['median']/1000, recall, color=colors[b], marker=markers[row['k']], s=42)
        ax.set(title=f"{report['experiment']} / {report['index_size']:,}",
               xlabel='Median of per-query median latency (µs)', ylabel='Mean tie-aware recall',
               xlim=(0, None), ylim=(0, 1.08))
        ax.grid(alpha=.25)
    handles = [Line2D([], [], color=colors[b], marker='o', linestyle='', label=b) for b in BACKENDS]
    handles += [Line2D([], [], color='#444444', marker=m, linestyle='', label=f'k={k}') for k, m in markers.items()]
    axes[0, 0].legend(handles=handles, loc='lower right', fontsize=8)
    fig.suptitle('First-measured-pass recall and separately timed latency; packages are not pooled')
    figures.append(('recall_latency_overview', fig))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for ax, k in zip(axes, (10, 100)):
        for b in BACKENDS:
            points = [(r['index_size'], next(v['median']/1000 for v in r['latency_comparisons']
                       if v['backend'] == b and v['k'] == k)) for r in parent]
            ax.plot([n for n, _ in points], [v for _, v in points], color=colors[b], marker='o', label=b)
        ax.set(title=f'Nested parent only / k={k}', xscale='log', xlabel='Measured index rows (log scale)',
               ylabel='Median of per-query median latency (µs)', ylim=(0, None), xticks=[r['index_size'] for r in parent])
        ax.set_xticklabels([f"{r['index_size']:,}" for r in parent]); ax.grid(alpha=.25); ax.legend()
    fig.suptitle('Observed parent scaling points; fresh independent graphs, no extrapolation')
    figures.append(('parent_latency_scaling', fig))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), layout='constrained')
    groups = [(reports[:1], 'Independent 1K pilot'), (parent, 'Nested parent; fresh graph at each size')] if historical else [
        (parent, 'Nested parent; fresh graph at each size'), (parent, 'Fresh reference costs')]
    for panel, (ax, (group, title)) in enumerate(zip(axes, groups)):
        ax.bar([f"{r['index_size']:,}" for r in group],
               [(r['timings_ns']['fresh_exhaustive_reference_computation_and_validation'] if not historical and panel == 1
                 else r['hnsw_preparation']['graph_construction_ns'])/1e9 for r in group], color='#389343')
        ax.set(title=title, xlabel='Index rows', ylabel=('Fresh reference (s)' if not historical and panel == 1
                                                     else 'Single HNSW graph construction (s)'), ylim=(0, None))
        ax.grid(axis='y', alpha=.25)
    figures.append(('graph_construction_costs', fig))
    names = []
    for stem, fig in figures:
        for suffix in ('png', 'svg'):
            name = f'{stem}.{suffix}'; fig.savefig(Path(directory)/name); names.append(name)
        plt.close(fig)
    return names
