"""Standalone presentation of verified parent points; never starts a search."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

from chemtellect.hnsw_benchmark import BACKENDS
from chemtellect.hnsw_reporting import verify_parent_series


def write_parent_details(directory, reports):
    verify_parent_series(reports)
    directory = Path(directory)
    colors = dict(zip(BACKENDS, ('#386cb0', '#ef8a32', '#389343')))
    markers = {10: 'o', 25: 's', 50: '^', 100: 'D'}
    sizes = [r['index_size'] for r in reports]
    figures = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for ax, k in zip(axes.flat, markers):
        for backend in BACKENDS:
            values = [next(v['median']/1000 for v in r['latency_comparisons']
                           if v['backend'] == backend and v['k'] == k) for r in reports]
            ax.plot(sizes, values, marker='o', color=colors[backend], label=backend)
        ax.set(title=f'k={k}', xscale='log', yscale='log', xlabel='Verified index rows',
               ylabel='Headline median latency (µs; log scale)', xticks=sizes)
        ax.xaxis.set_major_formatter(ScalarFormatter()); ax.tick_params(axis='x', rotation=20)
        ax.grid(alpha=.25); ax.legend()
    fig.suptitle('Observed nested-parent latency; one fresh graph per size, no extrapolation')
    figures.append(('parent_latency_all_k', fig))

    fig, ax = plt.subplots(figsize=(9, 4.8), layout='constrained')
    lowest = 1.0
    for k, marker in markers.items():
        values = [r['quality']['by_backend']['hnsw'][str(k)]['top_k']['mean'] for r in reports]
        lowest = min(lowest, *values)
        ax.plot(sizes, values, marker=marker, label=f'HNSW k={k}')
    ax.axhline(1, color='#555555', linestyle='--', label='Exact RDKit / FPSim2: 1.0')
    ax.set(xscale='log', xticks=sizes, xlabel='Verified index rows',
           ylabel='Mean tie-aware top-k recall (zoomed axis)',
           ylim=(max(0, lowest-.015), 1.004), title='First-measured-pass recall; zoomed axis')
    ax.xaxis.set_major_formatter(ScalarFormatter()); ax.yaxis.set_major_formatter('{x:.4f}')
    ax.grid(alpha=.25); ax.legend(loc='best')
    figures.append(('parent_recall_zoom', fig))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for ax, k in zip(axes.flat, markers):
        groups = [[v['recall'] for v in r['quality']['top_k_rows']
                   if v['backend'] == 'hnsw' and v['k'] == k] for r in reports]
        ax.boxplot(groups, tick_labels=[f'{n:,}' for n in sizes], showmeans=True,
                   medianprops={'color': '#111111'}, meanprops={'marker': 'D', 'markerfacecolor': '#ef8a32'})
        for position, values in enumerate(groups, 1):
            ax.scatter([position + (q % 11-5)*.012 for q in range(len(values))], values,
                       s=9, alpha=.35, color=colors['hnsw'])
        ax.set(title=f'HNSW k={k}', xlabel='Verified index rows', ylabel='Per-query tie-aware recall', ylim=(0, 1.02))
        ax.grid(axis='y', alpha=.25)
    fig.suptitle('Recall distribution across 100 fixed queries; orange diamond is mean, black line is median')
    figures.append(('parent_recall_distribution', fig))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for ax, report in zip(axes.flat, reports):
        for row in report['latency_comparisons']:
            backend, k = row['backend'], row['k']
            recall = report['quality']['by_backend'][backend][str(k)]['top_k']['mean']
            ax.scatter(row['median']/1000, recall, color=colors[backend], marker=markers[k], s=55)
        ax.set(title=f'{report["index_size"]:,} rows', xscale='log', xlabel='Headline median latency (µs; log scale)',
               ylabel='Mean tie-aware recall (zoomed axis)', ylim=(max(0, lowest-.015), 1.004))
        ax.yaxis.set_major_formatter('{x:.4f}'); ax.grid(alpha=.25)
    for ax in list(axes.flat)[len(reports):]: ax.set_visible(False)
    handles = [Line2D([], [], color=colors[b], marker='o', linestyle='', label=b) for b in BACKENDS]
    handles += [Line2D([], [], color='#444444', marker=m, linestyle='', label=f'k={k}') for k, m in markers.items()]
    fig.legend(handles=handles, loc='outside lower center', ncol=7)
    fig.suptitle('Recall and separately measured latency; marker denotes k, color denotes backend')
    figures.append(('parent_recall_latency_zoom', fig))

    fig, ax = plt.subplots(figsize=(10, 4.8), layout='constrained')
    phases = {
        'Fresh reference (includes mmap writing)': lambda r: r['timings_ns']['fresh_exhaustive_reference_computation_and_validation'],
        'FPSim2 conversion / database / verification': lambda r: r['timings_ns']['fpsim2_conversion_database_loading_and_verification'],
        'HNSW encoding / registration': lambda r: r['hnsw_preparation']['index_conversion_and_original_order_registration_ns'],
        'HNSW graph construction': lambda r: r['hnsw_preparation']['graph_construction_ns'],
        'HNSW serialization': lambda r: r['timings_ns']['hnsw_graph_and_data_saving']}
    for label, value in phases.items():
        ax.plot(sizes, [value(r)/1e9 for r in reports], marker='o', label=label)
    ax.set(xscale='log', yscale='log', xticks=sizes, xlabel='Verified index rows',
           ylabel='Elapsed seconds (log scale)', title='Outside-search preparation costs; components have separate timers')
    ax.xaxis.set_major_formatter(ScalarFormatter()); ax.grid(alpha=.25); ax.legend(fontsize=8)
    figures.append(('parent_preparation_costs', fig))

    fig, ax = plt.subplots(figsize=(11, 5), layout='constrained')
    stages = ['before_loading', 'verified_and_guarded_inputs', 'rdkit_and_fresh_reference_prepared',
              'fpsim2_prepared', 'fresh_hnsw_graph_built', 'after_gate_and_optional_comparative_passes']
    for report in reports:
        snapshots = {s['stage']: s for s in report['memory_snapshots']}
        ax.plot(range(len(stages)), [snapshots[s]['process_rss_bytes']/2**30 for s in stages],
                marker='o', label=f'{report["index_size"]:,} index rows')
    ax.set(xticks=range(len(stages)), xticklabels=['Before load', 'Verified inputs', 'Fresh reference',
           'FPSim2 ready', 'HNSW ready', 'After searches'], ylabel='Process RSS (GiB)',
           title='Synchronous RSS snapshots; these are not peaks', ylim=(0, None))
    ax.grid(alpha=.25); ax.legend()
    figures.append(('parent_resource_snapshots', fig))
    names = []
    for stem, fig in figures:
        for suffix in ('png', 'svg'):
            name = f'{stem}.{suffix}'; fig.savefig(directory/name); names.append(name)
        plt.close(fig)
    return names
