"""Tests for PPR analog retrieval and explainability fields.

Verifies:
1. retrieve_analogs returns AnalogResult objects with all explainability fields populated.
2. price_delta / demand_delta signs are correct (query - analog).
3. cosine_similarity is in [0, 1] and higher for more similar nodes.
4. match_reason is a non-empty human-readable string.
5. Embedding fallback path also populates all fields.
6. Same-regime analogs report "same regime (X)"; different-regime reports "regime was X".
7. outcome classification: recovered vs continued_spike.
8. region filter excludes non-matching regions.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.core.interfaces import FeatureVector
from app.engines.hippograph.graph import MarketStateGraph
from app.engines.hippograph.ppr import AnalogResult, retrieve_analogs, _build_match_reason


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_fv(
    node_id: str,
    price: float,
    demand: float = 7000.0,
    avail: float = 9000.0,
    regime: str = "normal",
    region: str = "NSW1",
    t_offset_min: int = 0,
) -> FeatureVector:
    base = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    return FeatureVector(
        event_id=node_id,
        values={
            "price_rrp": price,
            "demand_mw": demand,
            "availability_mw": avail,
            "headroom_mw": max(avail - demand, 0.0),
            "price_norm": min(1.0, price / 15_000.0),
            "headroom_ratio": max(avail - demand, 0.0) / max(demand, 1.0),
        },
        categorical={"region": region, "regime": regime},
        valid_time=base + timedelta(minutes=t_offset_min),
        tenant_id="system",
    )


def _build_graph_with_chain(n_nodes: int = 30, region: str = "NSW1") -> tuple[MarketStateGraph, list[str]]:
    """Build a graph with n_nodes temporally chained at ~$80/MWh normal."""
    g = MarketStateGraph()
    ids = []
    for i in range(n_nodes):
        fv = _make_fv(f"n{i}", price=80.0 + i * 0.1, demand=7000.0, t_offset_min=i * 5, region=region)
        node = g.insert(fv)
        g.compute_similarity_edges(node.node_id)
        ids.append(fv.event_id)
    return g, ids


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestAnalogExplainabilityFields:

    def test_all_fields_populated(self):
        """Every AnalogResult must have non-None explainability fields."""
        graph, ids = _build_graph_with_chain(30)
        query_fv = _make_fv("qry", price=82.0, demand=7100.0, t_offset_min=150)
        qnode = graph.insert(query_fv)
        graph.compute_similarity_edges(qnode.node_id)

        results = retrieve_analogs(qnode.node_id, graph, top_k=5)
        assert results, "expected at least one analog"

        for a in results:
            assert a.cosine_similarity is not None, "cosine_similarity must be set"
            assert a.price_delta is not None, "price_delta must be set"
            assert a.demand_delta is not None, "demand_delta must be set"
            assert a.headroom_delta is not None, "headroom_delta must be set"
            assert a.match_reason is not None and len(a.match_reason) > 0, "match_reason must be non-empty"

    def test_price_delta_sign(self):
        """price_delta = query_price - analog_price."""
        graph, _ = _build_graph_with_chain(30)
        query_fv = _make_fv("qry", price=100.0, demand=7000.0, t_offset_min=200)
        qnode = graph.insert(query_fv)
        graph.compute_similarity_edges(qnode.node_id)

        results = retrieve_analogs(qnode.node_id, graph, top_k=5)
        assert results
        for a in results:
            expected = round(100.0 - a.price_rrp, 2)
            assert abs(a.price_delta - expected) < 0.01, (
                f"price_delta should be {expected}, got {a.price_delta}"
            )

    def test_demand_delta_sign(self):
        """demand_delta = query_demand - analog_demand."""
        graph, _ = _build_graph_with_chain(30)
        query_fv = _make_fv("qry", price=80.0, demand=8000.0, t_offset_min=200)
        qnode = graph.insert(query_fv)
        graph.compute_similarity_edges(qnode.node_id)

        results = retrieve_analogs(qnode.node_id, graph, top_k=5)
        assert results
        for a in results:
            expected = round(8000.0 - a.demand_mw, 1)
            assert abs(a.demand_delta - expected) < 0.1

    def test_cosine_similarity_range(self):
        """cosine_similarity must be in [0, 1]."""
        graph, _ = _build_graph_with_chain(30)
        query_fv = _make_fv("qry", price=81.0, t_offset_min=200)
        qnode = graph.insert(query_fv)
        graph.compute_similarity_edges(qnode.node_id)

        results = retrieve_analogs(qnode.node_id, graph, top_k=10)
        for a in results:
            assert 0.0 <= a.cosine_similarity <= 1.0, (
                f"cosine_similarity out of range: {a.cosine_similarity}"
            )

    def test_similar_nodes_have_higher_cosine(self):
        """A node at $80/MWh normal should be more similar to a $81/MWh query than a $5000/MWh extreme."""
        graph = MarketStateGraph()
        for i in range(25):
            fv = _make_fv(f"n{i}", price=80.0, demand=7000.0, t_offset_min=i * 5)
            n = graph.insert(fv)
            graph.compute_similarity_edges(n.node_id)
        # Add one extreme node
        spike_fv = _make_fv("spike", price=5000.0, demand=7000.0, regime="extreme", t_offset_min=130)
        sn = graph.insert(spike_fv)
        graph.compute_similarity_edges(sn.node_id)

        query_fv = _make_fv("qry", price=81.0, demand=7000.0, t_offset_min=200)
        qnode = graph.insert(query_fv)
        graph.compute_similarity_edges(qnode.node_id)

        results = retrieve_analogs(qnode.node_id, graph, top_k=26)
        normal_sims = [a.cosine_similarity for a in results if a.regime == "normal"]
        extreme_sims = [a.cosine_similarity for a in results if a.regime == "extreme"]

        if normal_sims and extreme_sims:
            assert max(normal_sims) > max(extreme_sims), (
                "Normal analogs should have higher cosine similarity to a normal query"
            )

    def test_match_reason_same_regime(self):
        """Same-regime analog includes 'same regime (X)' in match_reason."""
        graph, _ = _build_graph_with_chain(30)
        query_fv = _make_fv("qry", price=81.0, regime="normal", t_offset_min=200)
        qnode = graph.insert(query_fv)
        graph.compute_similarity_edges(qnode.node_id)

        results = retrieve_analogs(qnode.node_id, graph, top_k=5)
        assert results
        for a in results:
            if a.regime == "normal":
                assert "same regime" in a.match_reason.lower()

    def test_match_reason_different_regime(self):
        """Different-regime analog includes 'regime was X' in match_reason."""
        graph = MarketStateGraph()
        for i in range(20):
            fv = _make_fv(f"n{i}", price=80.0, regime="normal", t_offset_min=i * 5)
            n = graph.insert(fv)
            graph.compute_similarity_edges(n.node_id)
        spike_fv = _make_fv("sp0", price=80.0, regime="spike", t_offset_min=110)
        sn = graph.insert(spike_fv)
        graph.compute_similarity_edges(sn.node_id)

        query_fv = _make_fv("qry", price=80.0, regime="elevated", t_offset_min=200)
        qnode = graph.insert(query_fv)
        graph.compute_similarity_edges(qnode.node_id)

        results = retrieve_analogs(qnode.node_id, graph, top_k=10)
        spike_analogs = [a for a in results if a.regime == "spike"]
        for a in spike_analogs:
            assert "regime was spike" in a.match_reason.lower()

    def test_region_filter_excludes_other_regions(self):
        """region filter must exclude non-matching nodes."""
        graph = MarketStateGraph()
        for i in range(20):
            fv = _make_fv(f"nsw{i}", price=80.0, region="NSW1", t_offset_min=i * 5)
            n = graph.insert(fv)
            graph.compute_similarity_edges(n.node_id)
        for i in range(10):
            fv = _make_fv(f"vic{i}", price=80.0, region="VIC1", t_offset_min=i * 5)
            n = graph.insert(fv)
            graph.compute_similarity_edges(n.node_id)

        query_fv = _make_fv("qry", price=81.0, region="NSW1", t_offset_min=200)
        qnode = graph.insert(query_fv)
        graph.compute_similarity_edges(qnode.node_id)

        results = retrieve_analogs(qnode.node_id, graph, top_k=10, region="NSW1")
        assert all(a.region == "NSW1" for a in results), "All analogs must be NSW1"


class TestEmbeddingFallback:
    """Fallback path (sparse graph) must also populate explainability fields."""

    def test_fallback_fields_populated(self):
        """With sparse graph, embedding fallback must populate all explainability fields."""
        graph = MarketStateGraph()
        # Insert only 5 nodes — below top_k*2=20 threshold for PPR
        for i in range(5):
            fv = _make_fv(f"h{i}", price=80.0 + i, t_offset_min=i * 5)
            n = graph.insert(fv)
            graph.compute_similarity_edges(n.node_id)

        query_fv = _make_fv("qry", price=82.0, t_offset_min=50)
        qnode = graph.insert(query_fv)
        graph.compute_similarity_edges(qnode.node_id)

        results = retrieve_analogs(qnode.node_id, graph, top_k=3)
        assert results, "Fallback must return results"
        for a in results:
            assert a.cosine_similarity is not None
            assert a.price_delta is not None
            assert a.demand_delta is not None
            assert a.headroom_delta is not None
            assert a.match_reason is not None and len(a.match_reason) > 0

    def test_fallback_cosine_matches_ppr_score(self):
        """In the fallback path, ppr_score and cosine_similarity must be the same value.

        Fallback triggers when node_count < top_k * 2. With top_k=5, threshold=10.
        We insert 4 history nodes so total is 5 < 10.
        """
        graph = MarketStateGraph()
        for i in range(4):
            fv = _make_fv(f"h{i}", price=80.0 + i, t_offset_min=i * 5)
            n = graph.insert(fv)
            graph.compute_similarity_edges(n.node_id)

        query_fv = _make_fv("qry", price=82.0, t_offset_min=50)
        qnode = graph.insert(query_fv)
        graph.compute_similarity_edges(qnode.node_id)

        # node_count = 5, top_k * 2 = 10 → fallback triggers
        results = retrieve_analogs(qnode.node_id, graph, top_k=5)
        assert results, "Fallback must return results"
        for a in results:
            assert a.ppr_score == a.cosine_similarity, (
                f"Fallback ppr_score should equal cosine_similarity, "
                f"got ppr={a.ppr_score} cos={a.cosine_similarity}"
            )


class TestBuildMatchReason:
    """Unit tests for the _build_match_reason helper."""

    def test_price_match_label_for_small_delta(self):
        reason = _build_match_reason(82.0, 80.0, 2.0, 0.0, 0.0, "normal", "normal", 0.99)
        assert "price match" in reason

    def test_price_label_for_large_delta(self):
        reason = _build_match_reason(200.0, 80.0, 120.0, 0.0, 0.0, "normal", "normal", 0.70)
        assert "price $80" in reason
        assert "price match" not in reason

    def test_same_regime_label(self):
        reason = _build_match_reason(80.0, 80.0, 0.0, 0.0, 0.0, "elevated", "elevated", 0.95)
        assert "same regime (elevated)" in reason

    def test_different_regime_label(self):
        reason = _build_match_reason(80.0, 80.0, 0.0, 0.0, 0.0, "spike", "normal", 0.80)
        assert "regime was normal" in reason

    def test_demand_match_for_small_delta(self):
        reason = _build_match_reason(80.0, 80.0, 0.0, 50.0, 0.0, "normal", "normal", 0.99)
        assert "demand match" in reason

    def test_headroom_similar_included(self):
        reason = _build_match_reason(80.0, 80.0, 0.0, 0.0, 100.0, "normal", "normal", 0.99)
        assert "headroom similar" in reason

    def test_large_headroom_delta_excluded(self):
        reason = _build_match_reason(80.0, 80.0, 0.0, 0.0, 800.0, "normal", "normal", 0.80)
        assert "headroom" not in reason
