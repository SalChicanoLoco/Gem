"""
Tests for the self-organizing field and the emergence test.

The property that matters is discrimination: it must find structure that is there
and refuse structure that is not. A detector that always says yes is worthless, so
the negative controls carry as much weight here as the positive cases.
"""

import numpy as np
import pytest

from agents import emergence
from agents.self_organizing import SelfOrganizingField

PROTOTYPES = {0: [0, 0, 3], 1: [3, 0, 0], 2: [0, 3, 0], 3: [3, 3, 3]}
GRAMMAR = {0: [1], 1: [2, 3], 2: [0], 3: [0]}


def _stream(seed, grammar=None, n=300, noise=0.35):
    """A stream of vectors, optionally following a phrase grammar."""
    rng = np.random.default_rng(seed)
    vectors, labels, state = [], [], 0
    for _ in range(n):
        if grammar is None:
            state = int(rng.integers(0, len(PROTOTYPES)))
        labels.append(state)
        vectors.append(np.array(PROTOTYPES[state], dtype=float) + rng.normal(0, noise, 3))
        if grammar is not None:
            state = int(rng.choice(grammar[state]))
    return vectors, labels


def _fitted(seed, grammar=None, n=300):
    vectors, labels = _stream(seed, grammar, n)
    field = SelfOrganizingField(stream_id="test")
    field.observe_many(vectors)
    field.merge_close_prototypes()
    return field, labels


class TestOnlineStateFormation:
    def test_states_form_without_being_told_how_many(self):
        """No preset cluster count, no batch fit, no HDBSCAN."""
        field, _ = _fitted(7, GRAMMAR)
        assert 3 <= len(field.prototypes) <= 8

    def test_each_state_maps_to_one_underlying_type(self):
        from collections import Counter, defaultdict
        field, labels = _fitted(7, GRAMMAR)
        mapping = defaultdict(Counter)
        for got, want in zip(field.sequence, labels):
            mapping[got][want] += 1
        purity = sum(c.most_common(1)[0][1] for c in mapping.values()) / len(labels)
        assert purity > 0.9

    def test_a_single_observation_makes_one_state(self):
        field = SelfOrganizingField()
        field.observe([1.0, 2.0, 3.0])
        assert len(field.prototypes) == 1
        assert field.sequence == ["s0"]

    def test_far_apart_observations_do_not_share_a_state(self):
        field = SelfOrganizingField()
        a = field.observe([0.0, 0.0, 0.0])
        field.observe([0.0, 0.0, 0.0])
        b = field.observe([50.0, 50.0, 50.0])
        assert a != b

    def test_edges_record_observed_transitions(self):
        field, _ = _fitted(7, GRAMMAR)
        assert field.edge_counts
        assert sum(field.edge_counts.values()) == len(field.sequence) - 1

    def test_merging_reduces_states_and_rewrites_history(self):
        """A merged state must not be left dangling in the sequence or edges."""
        field, _ = _fitted(7, GRAMMAR)
        merged = field.merge_close_prototypes(threshold=99.0)
        assert merged
        assert len(field.prototypes) == 1
        assert set(field.sequence) <= set(field.prototypes)
        for source, target in field.edge_counts:
            assert source in field.prototypes and target in field.prototypes


class TestMotifs:
    def test_recurring_subsequences_are_found(self):
        field, _ = _fitted(7, GRAMMAR)
        motifs = field.motifs(level=3)
        assert motifs
        assert all(m.level == 3 for m in motifs)
        assert motifs[0].support >= motifs[-1].support

    def test_a_sequence_shorter_than_the_level_has_no_motifs(self):
        field = SelfOrganizingField()
        field.observe([1.0, 1.0, 1.0])
        assert field.motifs(level=3) == []


class TestEmergenceDiscrimination:
    """The point of the whole exercise."""

    def test_real_grammar_is_detected(self):
        field, _ = _fitted(7, GRAMMAR)
        result = emergence.shuffle_null(field.sequence, shuffles=200)
        assert result["success"]
        assert result["emergence"]["highest_promoted_level"] >= 1
        assert result["statistics"]["transition_mutual_information"]["real"] > \
               result["statistics"]["transition_mutual_information"]["null_mean"]

    def test_clusters_without_ordering_promote_nothing(self):
        """
        The negative control. Same geometry, random order: shuffling preserves
        clusters and destroys only arrangement, so nothing should be promoted.
        An earlier version promoted E2 here at p = 0.022 through multiple
        comparisons.
        """
        field, _ = _fitted(11, grammar=None)
        result = emergence.shuffle_null(field.sequence, shuffles=200)
        assert result["emergence"]["highest_promoted_level"] == 0

    def test_a_level_is_not_promoted_over_an_unsupported_one(self):
        profile = emergence.emergence_profile({
            "transition_mutual_information": {"real": 0.01, "null_mean": 0.01, "z": 0.1, "p": 0.5},
            "kgram_entropy_2": {"real": 1.0, "null_mean": 3.0, "z": 9.0, "p": 0.0001},
        }, max_level=2)
        assert profile["E1"]["status"] == "not-supported"
        assert profile["E2"]["status"] == "unsupported-by-lower-level"
        assert profile["highest_promoted_level"] == 0

    def test_p_is_never_reported_as_zero(self):
        """With n shuffles nothing below 1/(n+1) has been demonstrated."""
        field, _ = _fitted(7, GRAMMAR)
        result = emergence.shuffle_null(field.sequence, shuffles=50)
        # p_floor is rounded to 6dp for display, so compare within that.
        assert result["p_floor"] == pytest.approx(1 / 51, abs=1e-6)
        for stat in result["statistics"].values():
            assert stat["p"] >= result["p_floor"]

    def test_too_short_a_sequence_is_refused(self):
        assert emergence.shuffle_null(["a", "b"])["success"] is False


class TestStatistics:
    def test_a_perfectly_cyclic_sequence_is_fully_predictable(self):
        seq = ["a", "b", "c"] * 20
        assert emergence.next_state_predictability(seq) == pytest.approx(1.0)
        assert emergence.conditional_entropy(seq) == pytest.approx(0.0, abs=1e-9)

    def test_mutual_information_is_zero_for_a_constant_sequence(self):
        assert emergence.transition_mutual_information(["a"] * 20) == pytest.approx(0.0)

    def test_entropy_rises_with_disorder(self):
        ordered = ["a", "b"] * 30
        mixed = list("abbaabababbbaaabbaba") * 3
        assert emergence.kgram_entropy(mixed, 2) > emergence.kgram_entropy(ordered, 2)


class TestSExpression:
    def test_emits_states_edges_and_motifs(self):
        field, _ = _fitted(7, GRAMMAR)
        sexp = field.to_sexp(motif_level=3)
        assert sexp.startswith("(stream test")
        assert "(state (id s" in sexp
        assert "(edge (from s" in sexp
        assert "(motif (id m" in sexp
        assert sexp.count("(") == sexp.count(")"), "unbalanced parentheses"

    def test_statistics_appear_only_when_supplied(self):
        """Absent measurements must not be invented into the canonical form."""
        field, _ = _fitted(7, GRAMMAR)
        assert "null-z" not in field.to_sexp()

        first_edge = next(iter(field.edge_counts))
        with_stats = field.to_sexp(edge_stats={first_edge: 3.91})
        assert "(null-z 3.91)" in with_stats
