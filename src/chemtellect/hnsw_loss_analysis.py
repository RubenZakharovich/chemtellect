"""Read-only analysis of Notebook 07's original designated HNSW responses."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from uuid import uuid4

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chemtellect import exact_retrieval as exact
from chemtellect import fingerprint_store as store
from chemtellect import hnsw_benchmark as benchmark


OUTCOMES = (
    'no qualifying catalog match',
    'full tie-aware recovery of qualifying top-k',
    'same displayed count but weaker membership',
    'fewer qualifying results displayed',
)
COLORS = ('#a9adb4', '#4c956c', '#e9b44c', '#c44e52')


def miss_groups(scores, positions, k):
    """Return mandatory missing IDs and anonymous missing boundary slots.

    Identity is intentionally unspecified for a boundary deficit: every unreturned
    member of that entire score group could fill a missing slot.
    """
    recall = benchmark.tie_aware_recall(scores, positions, k)
    boundary = recall['boundary']
    returned = set(positions)
    strict = np.flatnonzero(scores > boundary)
    groups = [dict(score=float(scores[p]), missing_slots=1,
                   kind='strict winner', eligible_positions=[int(p)])
              for p in strict if int(p) not in returned]
    deficit = k - recall['strict_winners'] - recall['credited_boundary_hits']
    if deficit:
        groups.append(dict(score=boundary, missing_slots=deficit,
                           kind='boundary tie group', eligible_positions=[
                               int(p) for p in np.flatnonzero(scores == boundary)
                               if int(p) not in returned]))
    groups.sort(key=lambda g: -g['score'])
    assert sum(g['missing_slots'] for g in groups) == round(k * (1-recall['recall']))
    return recall, groups


def slot_metrics(scores, positions, exact_positions, groups, k, cutoff):
    """Inclusive membership uses authoritative float64 scores, never distances."""
    G = int(np.count_nonzero(scores >= np.float64(cutoff)))
    E = min(k, G)
    H = int(np.count_nonzero(scores[list(set(positions))] >= np.float64(cutoff)))
    assert E == int(np.count_nonzero(scores[exact_positions] >= np.float64(cutoff)))
    qualifying = [g for g in groups if g['score'] >= np.float64(cutoff)]
    M = sum(g['missing_slots'] for g in qualifying)
    outcome = OUTCOMES[0 if G == 0 else 3 if H < E else 2 if M else 1]
    assert 0 <= H <= E <= k
    assert E-H <= M <= E
    return dict(G=G, E=E, H=H, qualifying_slots_lost=E-H,
                qualifying_top_k_neighbors_missed=M,
                highest_scoring_qualifying_miss=max((g['score'] for g in qualifying), default=None),
                matches_beyond_output_cap=G-E,
                exact_threshold_recall=E/G if G else None,
                hnsw_threshold_recall=H/G if G else None, outcome=outcome)


def snapshot(paths):
    return {str(p): store.file_info(p) for p in sorted(set(paths)) if p.is_file()}


def load_original(root, experiment):
    root, experiment = Path(root).resolve(), Path(experiment).resolve()
    state = json.loads((experiment/'state.json').read_text())
    point = state['sizes']['1000000']
    run = root/point['path']
    assert run.name == 'run_ea2725c6bba24a8986eb54e24d1c76ba'
    assert store.file_info(run/'manifest.json') == point['manifest']
    # Capture all original experiment files, this run, and Notebook 07. The
    # original loader validates every run payload and raw/fingerprint linkage.
    preserved = snapshot([root/'notebooks/07_dataset_pipeline.ipynb',
                          *experiment.rglob('*'), *run.iterdir()])
    report = benchmark.load_comparison_run(run, root=root)
    assert report['source'] == 'chemicalspace_new' and report['package'] == 'scale_1m_00'
    assert (report['query_count'], report['index_size']) == (100, 1_000_000)
    assert report['settings']['designated_measured_pass_index'] == 0
    source_checks = {}
    for name, expected in report['source_files'].items():
        current = store.file_info(root/'src/chemtellect'/name)
        # The presentation writer can evolve independently of numerical
        # evaluation; retain both hashes and never use it for this analysis.
        if name != 'hnsw_reporting.py':
            assert current == expected, f'Historical evaluation source changed: {name}'
        source_checks[name] = dict(historical=expected, current=current, matches=current == expected)
    fp = root/report['linkage']['artifact_path']
    preserved.update(snapshot(fp.iterdir()))
    split = json.loads((root/report['linkage']['split_path']).read_text())
    records = json.loads((run/'records.json').read_text())
    truth = json.loads((run/'ground_truth_top_k.json').read_text())
    scores = np.load(run/'exact_scores.npy', mmap_mode='r', allow_pickle=False)
    exact.validate_scores(scores, (100, 1_000_000))
    return dict(root=root, experiment=experiment, run=run, report=report,
                split=split, records=records, truth=truth, scores=scores,
                preserved=preserved, source_checks=source_checks)


def analyze(original, primary_k=10, cutoff=.5):
    d, s = original['report'], original['scores']
    rows, missing, retrievals, recall_rows = [], [], {}, []
    for k in d['settings']['candidate_counts']:
        for q, scores in enumerate(s):
            check = d['benchmark']['designated']['hnsw'][str(k)][q]
            hp = [r['index_position'] for r in check['native_results']]
            # Prove each k uses its OWN pass-0 request, including its ordering.
            requests = [r for r in d['benchmark']['measured_results']
                        if r['backend'] == 'hnsw' and r['k'] == k
                        and r['pass_index'] == 0 and r['query_position'] == q]
            assert len(requests) == 1 and requests[0]['check'] == check
            ep = [r['index_position'] for r in original['truth']['by_k'][str(k)][q]['matches']]
            assert ep == exact.exact_top_k(scores, k).tolist()
            recall, groups = miss_groups(scores, hp, k)
            independent = 1 - sum(g['missing_slots'] for g in groups)/k
            assert abs(independent-recall['recall']) < 1e-15
            saved = next(r for r in d['quality']['top_k_rows']
                         if r['backend'] == 'hnsw' and r['k'] == k and r['query_position'] == q)
            assert saved == dict(backend='hnsw', k=k, query_position=q, **recall)
            retrievals[q, k] = dict(hnsw=hp, exact=ep, recall=recall, groups=groups)
            query = original['split']['queries'][q]
            identity = dict(query_position=q, query_source_id=query['source_id'],
                            query_raw_line=query['line_number'],
                            query_canonical_smiles=query['canonical_smiles'], k=k,
                            source_run_id=d['run_id'], source_run_path=str(original['run'].relative_to(original['root'])),
                            pass_index=0, response_key=f'benchmark.designated.hnsw.{k}[{q}]')
            recall_rows.append(dict(**identity, tie_aware_recall=recall['recall'],
                                    genuine_top_k_misses=sum(g['missing_slots'] for g in groups),
                                    highest_genuine_top_k_miss=max((g['score'] for g in groups), default=None)))
            for g in groups:
                candidates = [original['records']['index'][p]['source_id'] for p in g['eligible_positions']]
                missing.append(dict(**identity, **g, eligible_compound_ids=json.dumps(candidates)))
            for t in d['settings']['thresholds']:
                row = dict(**identity, cutoff=t, tie_aware_recall=recall['recall'],
                           **slot_metrics(scores, hp, ep, groups, k, t))
                for backend, numerator in [('rdkit', row['E']), ('hnsw', row['H'])]:
                    old = next(r for r in d['quality']['threshold_evaluations']['rows']
                               if r['backend'] == backend and r['k'] == k
                               and r['query_position'] == q and r['threshold'] == t)
                    assert old['qualifying_count'] == row['G']
                    assert old['designated_qualifying_count'] == numerator
                    assert old['recall'] == (numerator/row['G'] if row['G'] else None)
                rows.append(row)
    metrics, recalls, groups = pd.DataFrame(rows), pd.DataFrame(recall_rows), pd.DataFrame(missing)
    summaries = []
    for (k, t), frame in metrics.groupby(['k', 'cutoff'], sort=True):
        eligible = int((frame.G > 0).sum())
        counts = frame.outcome.value_counts()
        assert sum(int(counts.get(o, 0)) for o in OUTCOMES) == 100
        values = dict(k=int(k), cutoff=float(t), total_queries=100, qualifying_queries=eligible,
                      total_G=int(frame.G.sum()), total_E=int(frame.E.sum()), total_H=int(frame.H.sum()),
                      slots_lost=int(frame.qualifying_slots_lost.sum()),
                      qualifying_neighbors_missed=int(frame.qualifying_top_k_neighbors_missed.sum()),
                      matches_beyond_cap=int(frame.matches_beyond_output_cap.sum()),
                      mean_exact_threshold_recall=None if not eligible else float(frame.exact_threshold_recall.mean()),
                      mean_hnsw_threshold_recall=None if not eligible else float(frame.hnsw_threshold_recall.mean()),
                      highest_qualifying_miss=None if frame.highest_scoring_qualifying_miss.isna().all()
                      else float(frame.highest_scoring_qualifying_miss.max()))
        for i, o in enumerate(OUTCOMES):
            values[f'outcome_{i}_count'] = int(counts.get(o, 0))
            values[f'outcome_{i}_pct_of_qualifying_queries'] = (
                100*values[f'outcome_{i}_count']/eligible if eligible and i else None)
        summaries.append(values)
    summary = pd.DataFrame(summaries)
    primary = metrics[(metrics.k == primary_k) & (metrics.cutoff == cutoff)].copy()
    assert len(primary) == 100
    ten = recalls[recalls.k == 10]
    assert f'{100*ten.tie_aware_recall.mean():.1f}' == '97.4'
    assert f'{100*ten.tie_aware_recall.min():.0f}' == '20'
    baseline = metrics[(metrics.k == 10) & (metrics.cutoff == .5)]
    assert int((baseline.G > 0).sum()) == 87
    assert f'{100*baseline.hnsw_threshold_recall.mean():.2f}' == '83.07'
    assert f'{100*baseline.exact_threshold_recall.mean():.2f}' == '86.81'
    worst = ten.loc[ten.tie_aware_recall == ten.tie_aware_recall.min(), 'query_position'].tolist()
    return dict(metrics=metrics, recalls=recalls, groups=groups, summary=summary,
                primary=primary, retrievals=retrievals, worst=worst)


def neighbor_details(original, analysis, queries, k=10, cutoff=.5):
    rows = []
    for q in queries:
        retrieval = analysis['retrievals'][q, k]
        hp, ep, recall = retrieval['hnsw'], retrieval['exact'], retrieval['recall']
        boundary = recall['boundary']
        tie_deficit = sum(g['missing_slots'] for g in retrieval['groups'] if g['kind'] == 'boundary tie group')
        for engine, positions in [('exact', ep), ('hnsw', hp)]:
            for rank, p in enumerate(positions, 1):
                score = float(original['scores'][q, p])
                if engine == 'exact':
                    status = ('recovered identity' if p in hp else 'genuine strict miss' if score > boundary
                              else f'boundary group: {tie_deficit} missing slots; identity ambiguous' if tie_deficit
                              else 'valid equal-score replacement recovered the slot')
                else:
                    status = ('recovered strict winner' if score > boundary
                              else 'valid boundary-score recovery' if score == boundary
                              else 'weaker replacement below top-k boundary')
                record = original['records']['index'][p]
                rows.append(dict(query_position=q, query_source_id=original['split']['queries'][q]['source_id'],
                                 engine=engine, rank=rank, index_position=p,
                                 compound_id=record['source_id'], raw_line_number=record['line_number'],
                                 canonical_smiles=original['split']['index_order'][p]['canonical_smiles'],
                                 exact_tanimoto=score, meets_cutoff=score >= cutoff,
                                 identity_in_other_engine=p in (hp if engine == 'exact' else ep),
                                 recovery_status=status, boundary_score=boundary,
                                 boundary_missing_slots=tie_deficit,
                                 source_run_id=original['report']['run_id'], k=k, pass_index=0))
    return pd.DataFrame(rows)


def figures(original, analysis, directory):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    plots = []
    worst = analysis['worst']
    fig, axes = plt.subplots(len(worst), 1, figsize=(9, 4.3*len(worst)), squeeze=False, layout='constrained')
    for ax, q in zip(axes[:, 0], worst):
        retrieval = analysis['retrievals'][q, 10]
        for engine, marker in [('exact', 'o'), ('hnsw', 's')]:
            ax.plot(range(1, 11), original['scores'][q, retrieval[engine]], marker=marker, label=engine)
        strongest = retrieval['groups'][0]['score']
        ax.annotate(f'Strongest genuine miss: {strongest:.5f}', xy=(1, strongest),
                    xytext=(3.2, strongest+.035), arrowprops={'arrowstyle': '->'})
        ax.axhline(.5, color='#222222', ls='--', label='Proposed cutoff 0.50')
        ax.set(xlabel='Original response rank (IDs and ties in accompanying table)', ylabel='Exact RDKit Tanimoto',
               title=f'Query row {q}: tie-aware recall@10 = 20%; no results at T ≥ 0.50',
               xticks=range(1, 11), ylim=(.35, .54))
        ax.legend(loc='lower left'); ax.grid(alpha=.2)
    plots.append(('worst_query_scores', fig))
    fig, ax = plt.subplots(figsize=(11, 5), layout='constrained')
    sub = analysis['summary'].query('k == 10').sort_values('cutoff')
    left = np.zeros(len(sub))
    for i, (label, color) in enumerate(zip(OUTCOMES, COLORS)):
        values = sub[f'outcome_{i}_count'].to_numpy()
        ax.barh(range(len(sub)), values, left=left, color=color, label=label, height=.6)
        for y, (value, start) in enumerate(zip(values, left)):
            if value:
                ax.text(start+value/2, y, str(value), ha='center', va='center', fontsize=10)
        left += values
    ax.set(yticks=range(len(sub)), yticklabels=[f'T ≥ {t:.2f}' for t in sub.cutoff],
           xlabel='Number of original queries (same 100 searches in every bar)', xlim=(0, 100),
           title='What scientists would have seen: 1M molecules, original k=10 responses')
    ax.invert_yaxis(); ax.legend(loc='upper center', bbox_to_anchor=(.5, -.18), ncol=2, fontsize=9)
    plots.append(('display_outcomes', fig))
    frame = analysis['recalls'].query('k == 10 and genuine_top_k_misses > 0').sort_values(
        ['highest_genuine_top_k_miss', 'query_position'], ascending=[False, True])
    fig, ax = plt.subplots(figsize=(10, 4.8), layout='constrained')
    ax.scatter(range(len(frame)), frame.highest_genuine_top_k_miss,
               c=['#c44e52' if q in worst else '#386cb0' for q in frame.query_position], s=70, zorder=3)
    for t in (.3, .5, .7):
        ax.axhline(t, ls='--', alpha=.7, color='#555555')
        ax.text(len(frame)-.4, t+.006, f'{t:.2f}', ha='right')
    for x, row in enumerate(frame.itertuples()):
        ax.annotate(f'{row.highest_genuine_top_k_miss:.4f}', (x, row.highest_genuine_top_k_miss),
                    xytext=(0, 9), textcoords='offset points', ha='center', fontsize=8)
    ax.scatter([], [], c='#c44e52', s=70, label='Original minimum-recall query')
    ax.set(xticks=range(len(frame)), xticklabels=[str(q) for q in frame.query_position],
           xlabel='Query row, sorted by highest-scoring genuine top-ten miss',
           ylabel='Exact Tanimoto of strongest genuine miss', ylim=(.27, .74),
           title=f'Missed-neighbor severity: {100-len(frame)}/100 queries had no genuine top-ten miss')
    ax.legend(loc='upper right', bbox_to_anchor=(1, .85))
    plots.append(('miss_severity', fig))
    names = []
    for name, fig in plots:
        for ext in ('png', 'svg'):
            target = directory/f'{name}.{ext}'
            fig.savefig(target, dpi=180)
            names.append(target.name)
        plt.close(fig)
    return names


def markdown_table(frame):
    """No optional tabulate dependency; floats only rounded for presentation."""
    def cell(value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return 'undefined'
        if isinstance(value, float):
            return format(value, '.10g')
        return str(value).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| '+' | '.join(map(str, frame.columns))+' |',
                      '| '+' | '.join(['---']*len(frame.columns))+' |',
                      *['| '+' | '.join(cell(v) for v in row)+' |'
                        for row in frame.itertuples(index=False, name=None)]])


def write_report(original, analysis, directory, primary_k, cutoff, worst_details):
    root, run, d = original['root'], original['run'], original['report']
    primary = analysis['primary']
    affected = primary[primary.qualifying_top_k_neighbors_missed > 0]
    strongest = float(primary.highest_scoring_qualifying_miss.max())
    rel = lambda path: os.path.relpath(path, directory)
    blocks = [
        '# HNSW retrieval loss at the 1M proof-of-concept scale',
        f'The 20% recall@10 query (row 53) lost eight genuine top-ten slots, but its strongest miss was '
        f'only **0.4791666667**: at the proposed **T ≥ {cutoff:.2f}** product cutoff, both engines would '
        f'have displayed **zero compounds**, so neither displayed count nor displayed quality changed. '
        f'Across the original 100 searches at k={primary_k}, **{len(affected)} searches changed**: '
        f'four displayed fewer qualifying compounds, losing seven slots in total; none returned the same '
        f'count with weaker qualifying membership. The strongest consequential miss was **{strongest:.10f}**, '
        'in query row 36, whose exact two qualifying compounds both disappeared from the HNSW display.',
        '## Evidence, scope, and definitions',
        f'Dataset: `chemicalspace_new/scale_1m_00`. Experiment: '
        f'[`{original["experiment"].name}`]({rel(original["experiment"] / "state.json")}). '
        f'Primary source: [`{run.name}`]({rel(run / "manifest.json")}), index size 1,000,000; '
        f'100 original held-out queries, primary k={primary_k}, inclusive display cutoff {cutoff:.2f}. '
        'Each k=10, 25, 50, 100 uses its own `benchmark.designated.hnsw.<k>[query_position]` '
        'response from measured pass 0, checked against the corresponding `measured_results` request. '
        'No HNSW graph was built, loaded, or searched for this analysis.',
        f'The original [`exact_scores.npy`]({rel(run / "exact_scores.npy")}) supplies all 100 × 1M '
        f'RDKit float64 similarities. The original [`ground_truth_top_k.json`]({rel(run / "ground_truth_top_k.json")}), '
        f'[`records.json`]({rel(run / "records.json")}), '
        f'[`parent_split.json`]({rel(root / d["linkage"]["split_path"])}), and '
        f'[`fingerprint manifest`]({rel(root / d["linkage"]["artifact_path"] / "manifest.json")}) '
        'were verified using the project loader, including hashes, packed fingerprints, ordered identities, '
        'raw acquisition linkage, and saved response diagnostics. All required artifacts were present; '
        'no historical question remains unresolved because of missing data. '
        'The source files defining historical evaluation match their recorded SHA-256 hashes.',
        'The current `hnsw_reporting.py` presentation writer differs from its run-time hash. '
        'It is not used here to compute scores or metrics. Both historical and current hashes '
        'are recorded in provenance; all six recorded numerical benchmark/adapter sources match. '
        'The original loader successfully recomputes and verifies the saved quality evaluations.',
        'All rankings, equality ties, and cutoff decisions use full precision. Displayed decimals are '
        'presentation only; CSV floats round-trip to the original binary float64 values. Exact ranking '
        'breaks equal scores by original index position; recovery credits every strict winner and at most '
        'the available slots at the exact k-th score. An equally scoring replacement at that boundary '
        'is valid. A deficit at a tied boundary is a count of missing slots in the entire tied group, '
        'not an assertion that one arbitrarily selected tied compound had to be returned.',
        'One per-query CSV row is one original query × its own k-response × cutoff (1,600 rows). '
        '`G` counts all qualifying catalog compounds; `E=min(k,G)` counts exact qualifying slots; '
        '`H` counts unique returned compounds qualifying by exact score. `E−H` measures display-count loss. '
        '`qualifying_top_k_neighbors_missed` measures tie-aware membership loss and can exceed count loss. '
        '`matches_beyond_output_cap=G−E` is cap loss even for exact retrieval. Threshold recall is H/G '
        '(exact: E/G), undefined for G=0. See [per-query metrics](per_query_metrics.csv) and '
        '[all top-k recalls](top_k_recall.csv). Query/index rows are zero-based; raw line numbers and '
        'response ranks are one-based. Compound identity means source ID, preserving distinct records '
        'even when their fingerprints have equal similarity.',
        '## Reproducing Notebook 07',
        'The unit below is a summary across the 100 original k=10 searches. Threshold-recall means '
        'average only the 87 queries with G>0 at 0.50; they do not divide recovered slots by ten. '
        'Reported percentages are rounded only after averaging.',
    ]
    ten = analysis['recalls'].query('k == 10')
    cross = pd.DataFrame([
        ['Mean tie-aware recall@10', '100 queries', f'{100*ten.tie_aware_recall.mean():.1f}%', '97.4%'],
        ['Minimum tie-aware recall@10', '100 queries', f'{100*ten.tie_aware_recall.min():.0f}%', '20%'],
        ['Qualifying queries at 0.50', '100 queries', str(int((primary.G>0).sum())), '87'],
        ['Mean HNSW threshold recall at 0.50', '87 qualifying queries', f'{100*primary.hnsw_threshold_recall.mean():.2f}%', '83.07%'],
        ['Mean exact threshold recall at 0.50', '87 qualifying queries', f'{100*primary.exact_threshold_recall.mean():.2f}%', '86.81%'],
    ], columns=['Measure', 'Denominator', 'Recomputed', 'Notebook 07'])
    cross.to_csv(directory/'notebook07_crosscheck.csv', index=False)
    blocks += [markdown_table(cross),
               'All summaries reproduce to their displayed precision, with no discrepancy. '
               'The original saved per-query threshold evaluations and top-k recall diagnostics also '
               'match the recomputed values. The exact threshold mean below 100% reflects the output cap; '
               'the additional HNSW shortfall reflects retrieval loss. [Cross-check CSV](notebook07_crosscheck.csv).',
               '## The minimum-recall query']
    for q in analysis['worst']:
        query = original['split']['queries'][q]
        ret = analysis['retrievals'][q, 10]
        ms = [g['score'] for g in ret['groups'] for _ in range(g['missing_slots'])]
        blocks += [f'Every minimum-recall query is included: **row {q}** (query number {q+1}), '
                   f'raw line **{query["line_number"]}**, source compound **`{query["source_id"]}`**. '
                   f'Canonical SMILES: `{query["canonical_smiles"]}`. '
                   f'Independent slot accounting gives {10-len(ms)}/10 = 20% recovery.',
                   'Each row in the following table compares the same response rank in the two original '
                   'ten-result lists. Compound IDs resolve score overlaps; a same-rank entry is not a '
                   'claim of a chemical analogue pairing. HNSW scores here are exact RDKit scores for '
                   'its returned IDs, not 1 minus its rounded native distances. The boundary group is '
                   'shared across ranks and must not be counted as a separate miss per listed ID.']
        detail = worst_details[worst_details.query_position == q]
        left = detail[detail.engine == 'exact'].set_index('rank')
        right = detail[detail.engine == 'hnsw'].set_index('rank')
        table = pd.DataFrame({'Rank': left.index,
            'Exact compound': left.compound_id.to_numpy(), 'Exact T': left.exact_tanimoto.to_numpy(),
            'Exact recovery': left.recovery_status.to_numpy(),
            'HNSW compound': right.compound_id.to_numpy(), 'HNSW exact T': right.exact_tanimoto.to_numpy(),
            'HNSW status': right.recovery_status.to_numpy()})
        blocks += [markdown_table(table), '[Full neighbor details and canonical SMILES](worst_query_neighbor_detail.csv).']
        top = ret['groups'][0]
        ids = [original['records']['index'][p]['source_id'] for p in top['eligible_positions']]
        blocks += [f'**1. Strongest genuine miss:** T={top["score"]!r}, compound '
                   + ', '.join(f'`{i}`' for i in ids)+'.']
        bins = [sum(g['missing_slots'] for g in ret['groups'] if a <= g['score'] < b)
                for a,b in [(0,.3),(.3,.5),(.5,.7),(.7,1.01)]]
        blocks += [f'**2. Miss severity intervals:** T<0.30: **{bins[0]}**; 0.30≤T<0.50: **{bins[1]}**; '
                   f'0.50≤T<0.70: **{bins[2]}**; T≥0.70: **{bins[3]}**. '
                   'The eight individual missing-slot scores, in descending order, are: '
                   + ', '.join(repr(v) for v in sorted(ms, reverse=True))+'.']
        for g in ret['groups']:
            if g['kind'] == 'boundary tie group':
                all_ties = np.flatnonzero(original['scores'][q] == g['score'])
                tie_ids = [original['records']['index'][p]['source_id'] for p in all_ties]
                blocks += [f'At the top-ten boundary T={g["score"]!r}, {len(all_ties)} catalog compounds '
                           f'tie for {10-ret["recall"]["strict_winners"]} exact slots. HNSW recovered '
                           f'{ret["recall"]["credited_boundary_hits"]} slot, leaving **{g["missing_slots"]} '
                           'missing slot**. The full tied group is '+', '.join(f'`{i}`' for i in tie_ids)+
                           '. Only one of the unreturned group members is needed to fill the remaining slot; '
                           'its identity is ambiguous. [Tie-aware miss groups](genuine_miss_groups.csv).']
        blocks += ['**3. Display at 0.50:** exact **0**, HNSW **0** (G=0). There were no stronger '
                   'qualifying neighbors to miss and no displayed replacements. Below the cutoff, '
                   'HNSW kept two valid exact slots and filled the other eight with weaker scores '
                   'ranging from 0.4347826087 to 0.3877551020; the ID table identifies them.',
                   '**4. Display impact:** **neither** count nor quality changed at the proposed 0.50 '
                   'cutoff. Before filtering, both ten-result membership and score quality were worse. '
                   'Threshold recall remains undefined for this query, rather than being scored as 0% or 100%.']
    blocks += ['![Worst-query scores by original response rank](worst_query_scores.png)',
               '[Vector figure](worst_query_scores.svg). Full precision and compound identities are in the neighbor CSV.',
               '## Practical loss across all 100 searches',
               'The following table has one row per outcome at k=10 and T≥0.50. Counts use all 100 '
               'queries. Percentages use only the 87 queries with a qualifying catalog match; no-match '
               'percentage is therefore inapplicable. The categories are exclusive, assigned in this '
               'order: no match; fewer displayed; same count with missing qualifying top-k membership; '
               'otherwise full qualifying recovery. Full recovery does not imply that G fits within ten slots.']
    counts = primary.outcome.value_counts()
    outcome_table = pd.DataFrame([dict(Outcome=o, Count=int(counts.get(o,0)),
        **{'% of 87 qualifying queries': 'inapplicable' if i==0 else f'{100*counts.get(o,0)/87:.2f}%'})
        for i,o in enumerate(OUTCOMES)])
    outcome_table.to_csv(directory/'primary_outcomes.csv', index=False)
    blocks += [markdown_table(outcome_table), '[Outcome CSV](primary_outcomes.csv).',
               f'The qualifying searches had **{int(primary.G.sum()):,}** catalog matches, '
               f'**{int(primary.E.sum())}** exact qualifying slots and **{int(primary.H.sum())}** HNSW '
               f'qualifying slots. **{int(primary.matches_beyond_output_cap.sum()):,}** matches were '
               'beyond the exact ten-result cap. These are sums of query–compound matches; a catalog '
               'compound may qualify for more than one query.',
               'Each row below is one of the four affected searches (denominator: 100 total, 87 '
               'qualifying). `Missed` is qualifying top-ten membership loss, retained here even for '
               'queries that also lose displayed slots. E and H are displayed counts after filtering; '
               'G is the full qualifying catalog count.']
    cols = ['query_position','G','E','H','qualifying_slots_lost','qualifying_top_k_neighbors_missed',
            'highest_scoring_qualifying_miss','matches_beyond_output_cap']
    affected.to_csv(directory/'affected_queries.csv', index=False)
    blocks += [markdown_table(affected[cols]), '[Affected-search metrics](affected_queries.csv).',
               'Each row in the next table is a genuinely missed qualifying compound for those four '
               'searches (seven query–compound pairs). The HNSW entry at the same original rank shows '
               'the rank displacement; it is not a chemical pairing or a claim that HNSW explicitly '
               'chose one structure instead of another. In query 50 the retained 0.50 compound '
               'moves up to rank 1; it must not be mistaken for a newly substituted compound. '
               'The other six same-rank entries shown here fail the 0.50 cutoff.']
    displacement = []
    for q in affected.query_position:
        ret = analysis['retrievals'][q, primary_k]
        for rank, p in enumerate(ret['exact'], 1):
            if p not in ret['hnsw'] and original['scores'][q,p] >= cutoff:
                h = ret['hnsw'][rank-1]
                displacement.append(dict(Query=q, Rank=rank,
                    **{'Missed compound': original['records']['index'][p]['source_id'],
                       'Missed T': float(original['scores'][q,p]),
                       'HNSW same-rank compound': original['records']['index'][h]['source_id'],
                       'HNSW same-rank T': float(original['scores'][q,h])}))
    displacement = pd.DataFrame(displacement)
    displacement.to_csv(directory/'qualifying_miss_displacements.csv', index=False)
    blocks += [markdown_table(displacement),
               '[Displacement CSV](qualifying_miss_displacements.csv); '
               '[complete affected response lists](affected_neighbor_detail.csv). '
               'Query rows 7 and 36 lost all qualifying results (1→0 and 2→0); rows 12 and 50 '
               'retained a qualifying compound but lost 1 and 3 slots respectively (2→1 and 4→1).',
               '## Changing the cutoff',
               'Each row below and each stacked bar represent the same 100 k=10 responses, with a '
               'different inclusive cutoff. The four outcome columns count queries, not compounds; '
               '“qualifying queries” is the denominator for the recovery percentages. The '
               '[full summary CSV](cutoff_k_summary.csv) includes those explicitly labeled percentages, '
               'slot totals, membership losses, cap losses, and both catalog-threshold recall means.']
    sub = analysis['summary'].query('k == 10')
    cut = sub[['cutoff','qualifying_queries','outcome_0_count','outcome_1_count','outcome_2_count','outcome_3_count']].copy()
    cut.columns = ['Cutoff','Qualifying queries','No catalog match','Full qualifying recovery','Same count, weaker','Fewer displayed']
    blocks += [markdown_table(cut),
               'At 0.30, all 100 queries have results: 89% have full qualifying recovery and 11% '
               'return the same count with weaker membership. Both engines have the same 5.7636% '
               'mean catalog-threshold recall at 0.30, yet HNSW missed 26 qualifying top-ten slots '
               'across those 11 queries, replacing them with weaker compounds that still qualify. '
               'Threshold recall alone hides this membership loss. At 0.50, 83/87 (95.40%) qualifying '
               'queries have full recovery and 4/87 (4.60%) display fewer. At 0.70, 10/10 qualifying '
               'queries have full recovery, but 90/100 searches have no qualifying catalog match. '
               'At 0.90, all 100 have no match and threshold recall is undefined. Higher cutoffs '
               'remove the missed-score range and many searchable query results; this is not evidence '
               'of improved HNSW traversal or a reason by itself to raise a product cutoff.',
               '![What scientists would have seen across the four cutoffs](display_outcomes.png)',
               '[Vector figure](display_outcomes.svg). The gray no-match segments stay visible.',
               '## Severity across the original k=10 searches',
               'The figure uses one point per query with at least one genuine top-ten miss, taking '
               'its highest missing score after tie credit. **89/100** queries had no genuine miss; '
               '**11/100** did. The minimum-recall query (row 53, red) was **not** the query with '
               'the strongest miss: row 36 missed T=0.5818181818. No genuine top-ten miss reached '
               '0.70. [Per-query severities and recall](top_k_recall.csv); '
               '[individual misses and ambiguous boundary groups](genuine_miss_groups.csv).',
               '![Highest-scoring genuine top-ten miss by affected query](miss_severity.png)',
               '[Vector figure](miss_severity.svg). Reference lines are 0.30, 0.50, and 0.70.',
               '## Secondary comparisons across k',
               'Each row below is the original pass-0 search set at that k, always 100 queries and '
               '87 qualifying queries at T≥0.50. Count losses and same-count membership losses '
               'are mutually exclusive query counts. Highest miss is the maximum exact score among '
               'genuinely missed qualifying slots, not the average. The run is shared; each request '
               'is identified separately by k, query row, pass index, and response key in the CSVs.']
    secondary = analysis['summary'][analysis['summary'].cutoff == .5][
        ['k','outcome_3_count','outcome_2_count','highest_qualifying_miss']].copy()
    secondary.columns = ['k','Queries losing displayed slots','Same count, weaker membership','Highest qualifying miss']
    secondary.to_csv(directory/'secondary_k_comparison.csv', index=False)
    blocks += [markdown_table(secondary), '[Secondary comparison CSV](secondary_k_comparison.csv). '
               'The affected rows remain 7, 12, 36, and 50 and the seven qualifying misses persist '
               'in each k’s own response. Increasing the output cap alone did not repair those misses. '
               'Exact mean catalog-threshold recall improves from 86.81% at k=10 to 93.31%, 95.25%, '
               'and 96.25% at k=25, 50, and 100; HNSW reaches 83.07%, 89.57%, 91.51%, and 92.51%. '
               'At 0.30, larger k exposes more membership loss: same-count/weaker queries are '
               '11, 20, 37, and 66 and fewer-displayed queries are 0, 1, 2, and 4 respectively. '
               'At 0.70 and 0.90 the outcomes are unchanged across k. '
               'All 16 k/cutoff combinations are in [the complete summary](cutoff_k_summary.csv).',
               '## Provenance and verification',
               f'Source run ID: `{d["run_id"]}`. Fingerprint artifact ID: `{d["linkage"]["artifact_id"]}`. '
               f'Split ID: `{d["linkage"]["split_id"]}`. '
               '[Machine-readable source paths and SHA-256 evidence](provenance.json); '
               '[generated-file manifest and verification](manifest.json). '
               'The checks assert that each outcome partition sums to 100 for all 16 combinations; '
               '0≤H≤E≤k for all 1,600 rows; E equals qualifying records in the exact top-k; '
               'all per-query metrics reproduce saved evaluations; the primary summaries round to '
               'Notebook 07’s values; every local Markdown link and figure resolves; and the original '
               'Notebook 07, original experiment files, primary run, and fingerprint files retain '
               'their initial hashes. Generated results live in a separate analysis directory.',
               '## Implications for the next HNSW tuning experiment',
               'Treat 0.50 as a proposed product display cutoff and 1M as the proof-of-concept scale. '
               'The alarming 20% query is irrelevant to that display, while four searches have a '
               'measurable scientist-facing loss, including two becoming empty and a missed compound '
               'at 0.5818. For a new experiment, keep these fixed queries, exact scores, and tie policy, '
               'and explicitly track lost qualifying slots, same-count membership degradation, '
               'strongest miss, and empty displays alongside recall and latency. Start by varying '
               '`efSearch` above the original 200 on the saved graph, then, if needed, compare fresh '
               'graphs with changed M/efConstruction against the original M=16/efConstruction=200. '
               'Use new run IDs and record each query/k response. Increasing k alone left these '
               'qualifying misses unresolved. Validate across additional held-out queries and graph '
               'builds before choosing a product setting; these fixed-query similarity measurements '
               'do not establish biological utility or performance at 100M molecules.']
    text = '\n\n'.join(blocks)+'\n'
    target = directory/'hnsw_retrieval_loss_report.md'
    target.write_text(text)
    return target


def export(original, analysis, primary_k=10, cutoff=.5):
    assert primary_k == 10 and cutoff == .5, 'This report answers the requested primary configuration.'
    root = original['root']
    directory = root/'experiments/results/hnsw_retrieval_loss'/original['report']['source']/original['report']['package']/original['experiment'].name/f'analysis_{uuid4().hex}'
    directory.mkdir(parents=True, exist_ok=False)
    for key, name in [('metrics','per_query_metrics'), ('recalls','top_k_recall'),
                      ('groups','genuine_miss_groups'), ('summary','cutoff_k_summary')]:
        frame = analysis[key].copy()
        if 'eligible_positions' in frame:
            frame['eligible_positions'] = frame.eligible_positions.map(json.dumps)
        frame.to_csv(directory/f'{name}.csv', index=False)
    details = neighbor_details(original, analysis, analysis['worst'])
    details.to_csv(directory/'worst_query_neighbor_detail.csv', index=False)
    affected = analysis['primary'].query('qualifying_top_k_neighbors_missed > 0').query_position.tolist()
    neighbor_details(original, analysis, affected).to_csv(directory/'affected_neighbor_detail.csv', index=False)
    figures(original, analysis, directory)
    target = write_report(original, analysis, directory, primary_k, cutoff, details)
    provenance = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        experiment_path=str(original['experiment'].relative_to(root)),
        primary_k=primary_k, display_cutoff=cutoff, source_run_id=original['report']['run_id'],
        source_run_path=str(original['run'].relative_to(root)),
        fingerprint_linkage=original['report']['linkage'], pass_index=0,
        response_keys={str(k): f'benchmark.designated.hnsw.{k}[query_position]'
                       for k in original['report']['settings']['candidate_counts']},
        verified_historical_source_files=original['source_checks'],
        source_files_sha256={os.path.relpath(p,root): v for p,v in original['preserved'].items()},
        missing_artifacts=[], unresolved_questions=[],
        score_source='original exact_scores.npy; no reconstruction needed',
        configuration=original['report']['configuration'])
    (directory/'provenance.json').write_bytes(store.canonical_json(provenance)+b'\n')
    # Verify preservation, CSV float round trips, and every local report target.
    assert snapshot(map(Path, original['preserved'])) == original['preserved']
    restored = pd.read_csv(directory/'per_query_metrics.csv', float_precision='round_trip')
    assert len(restored) == 1600
    for column in ('highest_scoring_qualifying_miss','exact_threshold_recall','hnsw_threshold_recall'):
        assert np.array_equal(restored[column], analysis['metrics'][column], equal_nan=True)
    links = re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', target.read_text())
    # The generated manifest is written immediately below.
    assert all((directory/p).is_file() for p in links if p != 'manifest.json')
    marker = dict(schema_version=1, status='complete', source_run_id=original['report']['run_id'],
                  verification=dict(original_files_unchanged=True, outcome_partitions_sum_to_100=True,
                                    all_1600_rows_satisfy_slot_bounds=True, original_quality_reproduced=True,
                                    primary_display_precision_reproduced=True, csv_float64_round_trip=True,
                                    all_report_links_resolve=True),
                  files={p.name:store.file_info(p) for p in sorted(directory.iterdir()) if p.is_file()})
    (directory/'manifest.json').write_bytes(store.canonical_json(marker)+b'\n')
    assert all((directory/p).is_file() for p in links)
    return target
