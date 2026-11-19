"""Scientist-facing outcomes and identity ambiguity under the historical tie policy."""
import numpy as np
import pytest

from chemtellect.exact_retrieval import exact_top_k
from chemtellect.hnsw_loss_analysis import OUTCOMES, miss_groups, slot_metrics


def evaluate(values, returned, k, cutoff):
    scores = np.array(values, dtype=np.float64)
    recall, groups = miss_groups(scores, returned, k)
    metrics = slot_metrics(scores, returned, exact_top_k(scores, k), groups, k, cutoff)
    return recall, groups, metrics


def test_equal_score_replacement_is_full_recovery():
    recall, groups, metrics = evaluate([.8, .6, .6, .6, .2], [0, 2], 2, .5)
    assert recall['recall'] == 1
    assert groups == []
    assert (metrics['G'], metrics['E'], metrics['H']) == (4, 2, 2)
    assert metrics['matches_beyond_output_cap'] == 2
    assert metrics['outcome'] == OUTCOMES[1]


def test_missing_boundary_slot_has_ambiguous_identity():
    recall, groups, metrics = evaluate([.8, .6, .6, .6, .2], [0, 4], 2, .5)
    assert recall['recall'] == .5
    assert groups == [dict(score=.6, missing_slots=1, kind='boundary tie group',
                           eligible_positions=[1, 2, 3])]
    assert metrics['qualifying_top_k_neighbors_missed'] == 1
    assert metrics['qualifying_slots_lost'] == 1
    assert metrics['outcome'] == OUTCOMES[3]


def test_same_display_count_can_hide_stronger_missing_membership():
    _, _, metrics = evaluate([.8, .7, .6, .5, .1], [0, 2], 2, .5)
    assert (metrics['E'], metrics['H']) == (2, 2)
    assert metrics['qualifying_slots_lost'] == 0
    assert metrics['qualifying_top_k_neighbors_missed'] == 1
    assert metrics['highest_scoring_qualifying_miss'] == .7
    assert metrics['outcome'] == OUTCOMES[2]


def test_no_match_recall_undefined_even_when_unfiltered_retrieval_misses():
    _, _, metrics = evaluate([.49, .45, .4, .3], [2, 3], 2, .5)
    assert (metrics['G'], metrics['E'], metrics['H']) == (0, 0, 0)
    assert metrics['outcome'] == OUTCOMES[0]
    assert metrics['hnsw_threshold_recall'] is None
    assert metrics['exact_threshold_recall'] is None
    assert metrics['highest_scoring_qualifying_miss'] is None


def test_cutoff_uses_full_precision_and_includes_exact_half():
    below, above = np.nextafter(.5, 0), np.nextafter(.5, 1)
    _, _, metrics = evaluate([above, .5, below, .1], [1, 2], 2, .5)
    assert (metrics['G'], metrics['E'], metrics['H']) == (2, 2, 1)
    assert metrics['highest_scoring_qualifying_miss'] == above


@pytest.mark.parametrize('returned', [[0, 0], [0, 6]])
def test_duplicate_and_invalid_returned_ids_rejected(returned):
    with pytest.raises(ValueError):
        evaluate([.8, .6, .4], returned, 2, .5)
