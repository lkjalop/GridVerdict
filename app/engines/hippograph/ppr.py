"""HippoGraph — Personalised PageRank analog retriever.

Given a query node (the current market state), runs PPR to find the most
structurally similar historical nodes in the graph. These are returned as
historical analogs with their outcome (did the market recover within N intervals?).

PPR algorithm:
  r = α × e_q + (1-α) × A^T × r

Where:
  - e_q is the personalisation vector seeded at the query node
  - A is the column-normalised adjacency matrix (temporal + similarity edges)
  - α is the restart probability (0.15 typical)
  - Convergence: ||r_t+1 - r_t||₁ < ε

Implementation: sparse, iterative power method on the in-memory graph.
No external graph library required. O(n × edges × iterations) per query.

Framework boundary: ZERO imports from data/, mcp/, domain/nem/.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.engines.hippograph.embedder import cosine_similarity_vec, embed_node

if TYPE_CHECKING:
    from app.engines.hippograph.graph import GraphNode, MarketStateGraph

logger = logging.getLogger(__name__)

# PPR hyperparameters
_ALPHA = 0.15           # restart probability
_MAX_ITER = 50          # maximum power-method iterations
_CONVERGENCE_EPS = 1e-6 # L1 convergence threshold
_EDGE_WEIGHT_TEMPORAL = 0.4   # weight of temporal edges in adjacency


@dataclass
class AnalogResult:
    """A historical analog identified by PPR."""
    node_id: str
    valid_time: object                  # datetime
    region: str
    regime: str
    price_rrp: float
    demand_mw: float
    ppr_score: float                    # personalised PageRank score
    outcome: str | None = None          # "recovered" | "continued_spike" | None
    outcome_horizon_intervals: int = 6  # how many intervals ahead outcome was measured
    # Feature-distance explainability fields
    cosine_similarity: float | None = None  # cosine sim between query and this node
    price_delta: float | None = None        # query.price_rrp - analog.price_rrp
    demand_delta: float | None = None       # query.demand_mw - analog.demand_mw
    headroom_delta: float | None = None     # query.headroom_mw - analog.headroom_mw
    match_reason: str | None = None         # human-readable explanation


def retrieve_analogs(
    query_node_id: str,
    graph: "MarketStateGraph",
    top_k: int = 10,
    region: str | None = None,
) -> list[AnalogResult]:
    """Run PPR from query_node and return top-k historical analogs.

    Parameters
    ----------
    query_node_id : str
        The node representing the current market state.
    graph : MarketStateGraph
        The in-memory market state graph.
    top_k : int
        Number of analogs to return.
    region : str | None
        If set, restricts analogs to this region.
    """
    query_node = graph.get(query_node_id)
    if query_node is None:
        logger.debug("PPR: query node %s not found", query_node_id)
        return []

    # Fallback to embedding-based similarity if graph has few edges
    if graph.node_count() < top_k * 2:
        return _embedding_fallback(query_node, graph, top_k, region)

    scores = _run_ppr(query_node_id, graph)

    # Pre-compute query features for delta/explainability fields
    query_price = query_node.feature_values.get("price_rrp", 0.0)
    query_demand = query_node.feature_values.get("demand_mw", 0.0)
    query_headroom = query_node.feature_values.get("headroom_mw",
        max(query_node.feature_values.get("availability_mw", 0.0) - query_demand, 0.0))
    query_emb = embed_node(query_node)

    # Filter and rank
    results: list[AnalogResult] = []
    for nid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        if nid == query_node_id:
            continue
        node = graph.get(nid)
        if node is None:
            continue
        if region and node.region != region:
            continue

        analog_price = node.feature_values.get("price_rrp", 0.0)
        analog_demand = node.feature_values.get("demand_mw", 0.0)
        analog_headroom = node.feature_values.get("headroom_mw",
            max(node.feature_values.get("availability_mw", 0.0) - analog_demand, 0.0))

        price_delta = round(query_price - analog_price, 2)
        demand_delta = round(query_demand - analog_demand, 1)
        headroom_delta = round(query_headroom - analog_headroom, 1)
        cos_sim = round(cosine_similarity_vec(query_emb, embed_node(node)), 4)

        outcome = _measure_outcome(node, graph)
        results.append(AnalogResult(
            node_id=nid,
            valid_time=node.valid_time,
            region=node.region,
            regime=node.regime,
            price_rrp=analog_price,
            demand_mw=analog_demand,
            ppr_score=score,
            outcome=outcome,
            cosine_similarity=cos_sim,
            price_delta=price_delta,
            demand_delta=demand_delta,
            headroom_delta=headroom_delta,
            match_reason=_build_match_reason(
                query_price, analog_price, price_delta, demand_delta,
                headroom_delta, query_node.regime, node.regime, cos_sim,
            ),
        ))

        if len(results) >= top_k:
            break

    return results


def _run_ppr(
    query_node_id: str,
    graph: "MarketStateGraph",
) -> dict[str, float]:
    """Iterative personalised PageRank (power method).

    Returns a dict of node_id → score for all reachable nodes.
    """
    # Build local adjacency (avoid full O(n²) scan)
    # Use the query node's similarity edges as the neighbourhood
    query_node = graph.get(query_node_id)
    if query_node is None:
        return {}

    # Gather all relevant nodes: query + all nodes reachable in 2 hops
    node_ids: set[str] = {query_node_id}
    for nid in list(query_node.similarity_edges) + _temporal_neighbours(query_node):
        node_ids.add(nid)
        n = graph.get(nid)
        if n:
            for nn_id in list(n.similarity_edges) + _temporal_neighbours(n):
                node_ids.add(nn_id)

    node_ids_list = list(node_ids)
    n = len(node_ids_list)
    if n < 2:
        return {}

    idx = {nid: i for i, nid in enumerate(node_ids_list)}

    # Build column-normalised adjacency (column = source node)
    adj: list[dict[int, float]] = [{} for _ in range(n)]
    out_weights: list[float] = [0.0] * n

    for nid in node_ids_list:
        node = graph.get(nid)
        if node is None:
            continue
        src = idx[nid]
        # Temporal edges
        for neighbour_id in _temporal_neighbours(node):
            if neighbour_id in idx:
                adj[idx[neighbour_id]][src] = adj[idx[neighbour_id]].get(src, 0.0) + _EDGE_WEIGHT_TEMPORAL
                out_weights[src] += _EDGE_WEIGHT_TEMPORAL
        # Similarity edges
        for neighbour_id, weight in node.similarity_edges.items():
            if neighbour_id in idx:
                adj[idx[neighbour_id]][src] = adj[idx[neighbour_id]].get(src, 0.0) + weight * (1 - _EDGE_WEIGHT_TEMPORAL)
                out_weights[src] += weight * (1 - _EDGE_WEIGHT_TEMPORAL)

    # Personalisation vector — seeded at query node
    query_idx = idx[query_node_id]
    r = [0.0] * n
    r[query_idx] = 1.0

    # Power iteration: r = α × e_q + (1-α) × A_norm × r
    for _ in range(_MAX_ITER):
        r_new = [_ALPHA if i == query_idx else 0.0 for i in range(n)]
        for dest in range(n):
            if adj[dest]:
                total = sum(
                    r[src] * w / max(out_weights[src], 1e-10)
                    for src, w in adj[dest].items()
                )
                r_new[dest] += (1.0 - _ALPHA) * total

        # Normalise
        norm = sum(r_new)
        if norm > 0:
            r_new = [v / norm for v in r_new]

        # Check convergence
        delta = sum(abs(r_new[i] - r[i]) for i in range(n))
        r = r_new
        if delta < _CONVERGENCE_EPS:
            break

    return {node_ids_list[i]: r[i] for i in range(n)}


def _embedding_fallback(
    query_node: "GraphNode",
    graph: "MarketStateGraph",
    top_k: int,
    region: str | None,
) -> list[AnalogResult]:
    """Embedding-based similarity when graph is too sparse for PPR."""
    query_emb = embed_node(query_node)
    query_price = query_node.feature_values.get("price_rrp", 0.0)
    query_demand = query_node.feature_values.get("demand_mw", 0.0)
    query_headroom = query_node.feature_values.get("headroom_mw",
        max(query_node.feature_values.get("availability_mw", 0.0) - query_demand, 0.0))

    region_nodes = graph.get_region_nodes(region or query_node.region)
    scored = []
    for node in region_nodes:
        if node.node_id == query_node.node_id:
            continue
        emb = embed_node(node)
        sim = cosine_similarity_vec(query_emb, emb)
        scored.append((sim, node))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for sim, node in scored[:top_k]:
        analog_price = node.feature_values.get("price_rrp", 0.0)
        analog_demand = node.feature_values.get("demand_mw", 0.0)
        analog_headroom = node.feature_values.get("headroom_mw",
            max(node.feature_values.get("availability_mw", 0.0) - analog_demand, 0.0))

        price_delta = round(query_price - analog_price, 2)
        demand_delta = round(query_demand - analog_demand, 1)
        headroom_delta = round(query_headroom - analog_headroom, 1)
        cos_sim = round(sim, 4)

        outcome = _measure_outcome(node, graph)
        results.append(AnalogResult(
            node_id=node.node_id,
            valid_time=node.valid_time,
            region=node.region,
            regime=node.regime,
            price_rrp=analog_price,
            demand_mw=analog_demand,
            ppr_score=cos_sim,
            outcome=outcome,
            cosine_similarity=cos_sim,
            price_delta=price_delta,
            demand_delta=demand_delta,
            headroom_delta=headroom_delta,
            match_reason=_build_match_reason(
                query_price, analog_price, price_delta, demand_delta,
                headroom_delta, query_node.regime, node.regime, cos_sim,
            ),
        ))

    return results


def _build_match_reason(
    query_price: float,
    analog_price: float,
    price_delta: float,
    demand_delta: float,
    headroom_delta: float,
    query_regime: str,
    analog_regime: str,
    cosine_sim: float,
) -> str:
    """Human-readable explanation of why this analog was selected."""
    parts = []

    if abs(price_delta) < 20:
        parts.append(f"price match (${analog_price:.0f}/MWh, Δ${price_delta:+.0f})")
    else:
        parts.append(f"price ${analog_price:.0f}/MWh (Δ${price_delta:+.0f})")

    if abs(demand_delta) < 300:
        parts.append(f"demand match (Δ{demand_delta:+.0f} MW)")
    else:
        parts.append(f"demand Δ{demand_delta:+.0f} MW")

    if abs(headroom_delta) < 500:
        parts.append(f"headroom similar (Δ{headroom_delta:+.0f} MW)")

    if query_regime == analog_regime:
        parts.append(f"same regime ({analog_regime})")
    else:
        parts.append(f"regime was {analog_regime}")

    return "; ".join(parts)


def _measure_outcome(node: "GraphNode", graph: "MarketStateGraph") -> str | None:
    """Look ahead N intervals from a node to classify the outcome.

    Returns:
    - "recovered": price returned to normal/elevated within N intervals
    - "continued_spike": price remained spike/extreme for N intervals
    - None: insufficient lookahead data
    """
    horizon = 6  # 6 × 5-min = 30 minutes
    current = node
    for _ in range(horizon):
        if current.temporal_next is None:
            return None
        next_node = graph.get(current.temporal_next)
        if next_node is None:
            return None
        current = next_node

    final_regime = current.regime
    if final_regime in ("normal", "elevated"):
        return "recovered"
    return "continued_spike"


def _temporal_neighbours(node: "GraphNode") -> list[str]:
    """Return temporal neighbour IDs (prev + next)."""
    nb = []
    if node.temporal_next:
        nb.append(node.temporal_next)
    if node.temporal_prev:
        nb.append(node.temporal_prev)
    return nb
