"""HippoGraph — state vector embedder.

Converts a FeatureVector (or GraphNode) into a fixed-dimension numeric vector
for similarity search and PPR seeding.

Embedding strategy (ordered by availability):
  1. BGE-M3 / sentence-transformers (if installed) — encodes a text description
     of the market state for semantic similarity.
  2. Normalised numeric features — deterministic, no ML dependency.
     Feature vector: [price_norm, demand_norm, headroom_ratio, regime_onehot×4]
     This is the fallback and is always available.

The `embed()` function always returns a numpy-free list[float] so the engine
has zero heavy dependencies in the core path.

Framework boundary: ZERO imports from data/, mcp/, domain/nem/.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.engines.hippograph.graph import GraphNode

# Regime label to one-hot index mapping (deterministic)
_REGIME_INDEX = {"normal": 0, "elevated": 1, "spike": 2, "extreme": 3}

# Normalisation constants — NEM-wide plausible maxima
_PRICE_MAX = 15_000.0   # $/MWh — cap is 17,500 but extremes rare
_DEMAND_MAX = 35_000.0  # MW — NEM peak ~35 GW
_AVAIL_MAX = 55_000.0   # MW — total registered capacity


def embed_node(node: "GraphNode") -> list[float]:
    """Return a fixed-length float embedding for a graph node.

    Returns an 8-dimensional vector:
    [price_norm, demand_norm, avail_norm, headroom_ratio,
     one_hot_normal, one_hot_elevated, one_hot_spike, one_hot_extreme]
    """
    return embed_features(node.feature_values, node.categorical)


def embed_features(
    values: dict[str, float],
    categorical: dict[str, str],
) -> list[float]:
    """Deterministic 8-dim embedding from feature dicts."""
    price = values.get("price_rrp", 0.0)
    demand = values.get("demand_mw", 0.0)
    avail = values.get("availability_mw", 0.0)
    headroom = values.get("headroom_mw", max(avail - demand, 0.0))

    price_norm = _clip(price / _PRICE_MAX)
    demand_norm = _clip(demand / _DEMAND_MAX)
    avail_norm = _clip(avail / _AVAIL_MAX)
    headroom_ratio = _clip(headroom / max(demand, 1.0))

    regime = categorical.get("regime", "normal")
    idx = _REGIME_INDEX.get(regime, 0)
    one_hot = [0.0, 0.0, 0.0, 0.0]
    one_hot[idx] = 1.0

    return [price_norm, demand_norm, avail_norm, headroom_ratio] + one_hot


def cosine_similarity_vec(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    denom = norm_a * norm_b
    return dot / denom if denom > 1e-10 else 0.0


def _clip(v: float) -> float:
    return max(0.0, min(1.0, v))


# ── Optional BGE text encoder ──────────────────────────────────────────

def embed_text(description: str) -> list[float] | None:
    """Encode a text description using BGE-M3 if sentence-transformers is available.

    Returns None if the library is not installed (fallback to embed_features).
    The returned vector is normalised to unit length.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _model = _get_bge_model()
        vec = _model.encode(description, normalize_embeddings=True)
        return vec.tolist()
    except ImportError:
        return None


_bge_model = None


def _get_bge_model():
    global _bge_model
    if _bge_model is None:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _bge_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _bge_model
