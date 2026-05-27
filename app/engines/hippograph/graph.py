"""HippoGraph — market state graph builder.

Converts a FeatureVector into a node in the market state graph.
Each node represents one dispatch interval; edges connect adjacent intervals
and intervals with similar feature vectors (for PPR traversal).

Graph structure:
  - Temporal edges: t → t+1 for consecutive intervals (directed)
  - Similarity edges: bidirectional, weight = cosine_similarity(fv_i, fv_j)
    Only added when similarity > SIMILARITY_THRESHOLD.

Storage: in-memory adjacency list + node dict.
Persistence: Alembic-managed market_events table mirrors graph nodes.
             The graph is rebuilt from DB on cold start.

Framework boundary: ZERO imports from data/, mcp/, domain/nem/.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.core.interfaces import FeatureVector

# Minimum cosine similarity to create a similarity edge
_SIMILARITY_THRESHOLD = 0.85
# Maximum number of neighbours kept per node (prunes weak similarity edges)
_MAX_NEIGHBOURS = 50


@dataclass
class GraphNode:
    """One dispatch interval, stored as a graph node."""
    node_id: str                        # same as FeatureVector.event_id
    valid_time: datetime
    tenant_id: str
    feature_values: dict[str, float]    # numeric features
    categorical: dict[str, str]         # regime, region, etc.

    # Adjacency
    temporal_next: str | None = None    # node_id of t+1
    temporal_prev: str | None = None    # node_id of t-1
    similarity_edges: dict[str, float] = field(default_factory=dict)  # node_id → weight

    @property
    def regime(self) -> str:
        return self.categorical.get("regime", "normal")

    @property
    def region(self) -> str:
        return self.categorical.get("region", "")


class MarketStateGraph:
    """In-memory graph of market state nodes.

    Nodes are keyed by node_id. Edges are stored on the nodes.
    Temporal edges are created automatically on `insert`.
    Similarity edges are computed lazily via `compute_similarity_edges`.
    """

    def __init__(self, max_nodes: int = 8_640) -> None:
        # 8640 = 30 days × 288 intervals/day (5-min dispatch)
        self._max_nodes = max_nodes
        self._nodes: dict[str, GraphNode] = {}
        self._ordered: list[str] = []          # insertion-ordered node_ids
        self._by_region: dict[str, list[str]] = {}  # region → [node_id, ...]

    # ── Insertion ─────────────────────────────────────────────────────

    def insert(self, fv: FeatureVector) -> GraphNode:
        """Add a FeatureVector as a node. Creates temporal edge to previous node."""
        region = fv.categorical.get("region", "")
        node = GraphNode(
            node_id=fv.event_id,
            valid_time=fv.valid_time,
            tenant_id=fv.tenant_id,
            feature_values=dict(fv.values),
            categorical=dict(fv.categorical),
        )

        # Temporal linkage: connect to previous node in same region
        region_nodes = self._by_region.setdefault(region, [])
        if region_nodes:
            prev_id = region_nodes[-1]
            prev_node = self._nodes.get(prev_id)
            if prev_node:
                prev_node.temporal_next = node.node_id
                node.temporal_prev = prev_id

        self._nodes[node.node_id] = node
        self._ordered.append(node.node_id)
        region_nodes.append(node.node_id)

        # Evict oldest nodes if over capacity
        while len(self._nodes) > self._max_nodes:
            self._evict_oldest()

        return node

    def insert_from_dict(self, data: dict[str, Any]) -> GraphNode | None:
        """Reconstruct a node from a DB row dict (market_events table)."""
        try:
            valid_time = data["valid_time"]
            if not isinstance(valid_time, datetime):
                return None
            fv = FeatureVector(
                event_id=data["id"],
                values={
                    "price_rrp": float(data.get("price_rrp", 0.0)),
                    "demand_mw": float(data.get("demand_mw", 0.0)),
                    "availability_mw": float(data.get("availability_mw", 0.0)),
                },
                categorical={
                    "region": data.get("region", ""),
                    "regime": data.get("data", {}).get("regime", "normal"),
                    "source": data.get("source", ""),
                },
                valid_time=valid_time,
                tenant_id=data.get("tenant_id", "system"),
            )
            return self.insert(fv)
        except (KeyError, ValueError, TypeError):
            return None

    # ── Similarity edges ──────────────────────────────────────────────

    def compute_similarity_edges(
        self,
        node_id: str,
        candidates: list[str] | None = None,
    ) -> None:
        """Compute cosine-similarity edges between node and candidate nodes.

        If candidates is None, compares against same-region nodes in a
        recency window (last 2016 intervals = ~7 days).
        """
        node = self._nodes.get(node_id)
        if node is None:
            return

        region = node.region
        if candidates is None:
            region_ids = self._by_region.get(region, [])
            # Last 7-day window, excluding self
            candidates = [nid for nid in region_ids[-2016:] if nid != node_id]

        scores: list[tuple[str, float]] = []
        for cid in candidates:
            cnode = self._nodes.get(cid)
            if cnode is None:
                continue
            sim = _cosine_similarity(node.feature_values, cnode.feature_values)
            if sim >= _SIMILARITY_THRESHOLD:
                scores.append((cid, sim))

        # Keep top-N by similarity weight
        scores.sort(key=lambda x: x[1], reverse=True)
        node.similarity_edges = {nid: w for nid, w in scores[:_MAX_NEIGHBOURS]}

        # Bidirectional: also update the target nodes
        for nid, weight in node.similarity_edges.items():
            cnode = self._nodes.get(nid)
            if cnode and len(cnode.similarity_edges) < _MAX_NEIGHBOURS:
                cnode.similarity_edges[node_id] = weight

    # ── Lookup ────────────────────────────────────────────────────────

    def get(self, node_id: str) -> GraphNode | None:
        return self._nodes.get(node_id)

    def get_region_nodes(self, region: str, limit: int = 288) -> list[GraphNode]:
        """Most recent `limit` nodes for a region."""
        ids = self._by_region.get(region, [])
        return [self._nodes[nid] for nid in ids[-limit:] if nid in self._nodes]

    def node_count(self) -> int:
        return len(self._nodes)

    # ── Internal ──────────────────────────────────────────────────────

    def _evict_oldest(self) -> None:
        if not self._ordered:
            return
        oldest_id = self._ordered.pop(0)
        node = self._nodes.pop(oldest_id, None)
        if node:
            region_list = self._by_region.get(node.region, [])
            if region_list and region_list[0] == oldest_id:
                region_list.pop(0)
            # Clean up temporal links
            if node.temporal_next:
                next_node = self._nodes.get(node.temporal_next)
                if next_node:
                    next_node.temporal_prev = None


# ── Helpers ───────────────────────────────────────────────────────────

def _cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two feature dicts."""
    keys = set(a) & set(b)
    if not keys:
        return 0.0
    dot = sum(a[k] * b[k] for k in keys)
    norm_a = math.sqrt(sum(a[k] ** 2 for k in keys))
    norm_b = math.sqrt(sum(b[k] ** 2 for k in keys))
    denom = norm_a * norm_b
    return dot / denom if denom > 1e-10 else 0.0


# ── Module-level singleton ────────────────────────────────────────────

_graph: MarketStateGraph | None = None


def get_graph() -> MarketStateGraph:
    """Return the module-level MarketStateGraph singleton."""
    global _graph
    if _graph is None:
        _graph = MarketStateGraph()
    return _graph


async def rebuild_from_db(lookback_days: int = 30) -> int:
    """Populate the graph from recent market_events rows at startup.

    Queries AEMO_DISPATCH_PRICE rows for the past `lookback_days` days,
    inserts them into the graph in chronological order, and computes
    similarity edges for the most recent 288 nodes per region.

    Returns the number of nodes inserted.

    Safe to call multiple times — duplicate inserts are de-duped by node_id.
    """
    from datetime import timedelta, timezone
    from sqlalchemy import select, text

    try:
        import app.db.session as _db_mod
        _db_session = _db_mod.db_session
    except Exception:
        return 0  # DB not available (e.g., unit test environment)

    graph = get_graph()
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    inserted = 0
    try:
        _COLS = ("id", "tenant_id", "source", "region", "valid_time",
                 "price_rrp", "demand_mw", "availability_mw", "data")
        async with _db_session() as session:
            result = await session.execute(
                text("""
                    SELECT id, tenant_id, source, region, valid_time,
                           price_rrp, demand_mw, availability_mw, data
                    FROM market_events
                    WHERE source = 'AEMO_DISPATCH_PRICE'
                      AND valid_time >= :cutoff
                    ORDER BY region, valid_time
                """),
                {"cutoff": cutoff},
            )
            raw_rows = result.fetchall()
        import json as _json
        rows = []
        for r in raw_rows:
            d = dict(zip(_COLS, r))
            # SQLite returns JSON columns as strings; parse to dict
            if isinstance(d.get("data"), str):
                try:
                    d["data"] = _json.loads(d["data"])
                except (ValueError, TypeError):
                    d["data"] = {}
            rows.append(d)

        for row in rows:
            row_dict = dict(row)
            # Ensure valid_time is a datetime (SQLite returns strings)
            vt = row_dict.get("valid_time")
            if isinstance(vt, str):
                try:
                    row_dict["valid_time"] = datetime.fromisoformat(vt.replace(" ", "T"))
                    if row_dict["valid_time"].tzinfo is None:
                        from datetime import timezone
                        row_dict["valid_time"] = row_dict["valid_time"].replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
            if row_dict.get("id") not in graph._nodes:
                node = graph.insert_from_dict(row_dict)
                if node:
                    inserted += 1

        # Compute similarity edges for the last day of each region
        for region, node_ids in graph._by_region.items():
            for nid in node_ids[-288:]:
                graph.compute_similarity_edges(nid)

    except Exception as _exc:
        import logging as _logging
        _logging.getLogger(__name__).warning("HippoGraph rebuild error: %s", _exc)

    return inserted
