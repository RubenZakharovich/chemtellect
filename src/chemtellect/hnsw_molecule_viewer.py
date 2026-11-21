"""View an original Chemtellect HNSW response and its highest-scoring misses."""
import base64
import json
from html import escape
from pathlib import Path

import numpy as np


def load_run(root, experiment, index_size):
    """Load once per notebook session. The original loader verifies the run."""
    from chemtellect.hnsw_benchmark import load_comparison_run

    root = Path(root).resolve()
    experiment = root / experiment
    sizes = json.loads((experiment / "state.json").read_text())["sizes"]
    if str(index_size) not in sizes:
        raise ValueError(f"No saved run at size {index_size}. Available sizes: {list(sizes)}")
    run = root / sizes[str(index_size)]["path"]
    report = load_comparison_run(run, root=root)
    split_path = root / report["linkage"]["split_path"]
    split = json.loads(split_path.read_text())
    scores = np.load(run / "exact_scores.npy", mmap_mode="r", allow_pickle=False)
    if scores.shape != (report["query_count"], index_size):
        raise ValueError("Saved scores and selected index have different dimensions.")
    return dict(report=report, scores=scores, queries=split["queries"],
                index=split["index_order"], path=run)


def top_misses(scores, returned, k):
    """Yield up to two genuine missing slots, crediting equivalent boundary hits."""
    boundary = np.partition(scores, len(scores) - k)[-k]
    strict = np.flatnonzero(scores > boundary)
    missing = sorted((int(i) for i in strict if i not in returned),
                     key=lambda i: (-scores[i], i))
    result = [(i, "Missed exact neighbor") for i in missing[:2]]
    boundary_hits = sum(scores[i] == boundary for i in returned)
    deficit = max(0, k - len(strict) - boundary_hits)
    if deficit and len(result) < 2:
        candidates = [int(i) for i in np.flatnonzero(scores == boundary) if i not in returned]
        label = ("Example from tied group" if len(candidates) > deficit
                 else "Missed boundary neighbor")
        result.extend((i, label) for i in candidates[:min(deficit, 2-len(result))])
    return result


def card(record, title, score=None, missed=False):
    from rdkit import Chem
    from rdkit.Chem import Draw, rdDepictor

    smiles = record["canonical_smiles"]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not draw saved structure {record['source_id']}")
    rdDepictor.Compute2DCoords(mol)
    drawing = Draw.MolDraw2DCairo(300, 215)
    drawing.drawOptions().addStereoAnnotation = True
    drawing.DrawMolecule(mol)
    drawing.FinishDrawing()
    encoded = base64.b64encode(drawing.GetDrawingText()).decode("ascii")
    border = "3px solid #bc2828" if missed else "1px solid #b9c8cf"
    value = "" if score is None else f"<br><b>Tanimoto {score:.5f}</b>"
    identity = escape(str(record["source_id"]))
    return (f'<div style="background:white;color:#203444;border:{border};padding:9px;'
            f'box-sizing:border-box;width:285px;max-width:100%;overflow-wrap:anywhere">'
            f'{escape(title)}{value}<br><img width="265" style="max-width:100%" '
            f'alt="2D structure of {identity}" src="data:image/png;base64,{encoded}">'
            f'<br><small>{identity}</small><details><summary>SMILES</summary>'
            f'<code>{escape(smiles)}</code></details></div>')


def show_query(data, query_row=85, k=10, cutoff=0.50, filter_misses=False):
    """Row is zero-based. Cutoffs are inclusive and applied before formatting."""
    from IPython.display import HTML, display

    if not 0 <= query_row < len(data["queries"]):
        raise ValueError(f"Choose a zero-based query row in 0..{len(data['queries'])-1}.")
    if not 0 <= cutoff <= 1:
        raise ValueError("Choose a cutoff between 0 and 1.")
    saved = data["report"]["benchmark"]["designated"]["hnsw"]
    if str(k) not in saved:
        raise ValueError(f"No saved HNSW response for k={k}. Available: {list(saved)}")
    hits = [r["index_position"] for r in saved[str(k)][query_row]["native_results"]]
    scores = data["scores"][query_row]
    if len(hits) != k or len(set(hits)) != k or any(i < 0 or i >= len(scores) for i in hits):
        raise ValueError("Saved response has invalid or duplicate positions.")
    grid = '<div style="display:flex;flex-wrap:wrap;gap:10px">'
    out = [f"<p>{escape(data['path'].name)} · {len(scores):,} indexed molecules · "
           f"query row {query_row} · k={k} · inclusive cutoff {cutoff:g}</p>",
           card(data["queries"][query_row], f"Query row {query_row} (molecule {query_row+1})")]
    shown = [(rank, i) for rank, i in enumerate(hits, 1) if scores[i] >= cutoff]
    out += [f"<h3>HNSW hits · {len(shown)} of {k} meet the cutoff</h3>", grid]
    for rank, i in shown:
        out.append(card(data["index"][i], f"HNSW rank {rank}", scores[i]))
    if not shown:
        out.append("<p>No saved HNSW hit meets this cutoff.</p>")
    out += ["</div><h3>Highest-scoring missed exact neighbors</h3>"]
    misses = top_misses(scores, set(hits), k)
    if filter_misses:
        misses = [(i, label) for i, label in misses if scores[i] >= cutoff]
    out.append(grid)
    for i, label in misses:
        status = "meets cutoff" if scores[i] >= cutoff else "below cutoff"
        out.append(card(data["index"][i], f"{label} · {status}", scores[i], missed=True))
    if not misses:
        out.append("<p>No genuine missed top-k slot in the selected score range.</p>")
    out.append("</div>")
    display(HTML("".join(out)))
