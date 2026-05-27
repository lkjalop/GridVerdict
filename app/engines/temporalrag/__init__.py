"""TemporalRAG — bitemporal retrieval-augmented grounding for GridVerdict.

Retrieves evidence from multiple sources scoped to a valid-time window
and enforces a system-time no-leakage fence so historical queries cannot
see data that was recorded after the query point in time.

Public API::

    from app.engines.temporalrag import retrieve, TemporalQuery, RetrievalBundle

    bundle = await retrieve(
        TemporalQuery(
            valid_time_from=dt_from,
            valid_time_to=dt_to,
            system_time_at_query=datetime.now(timezone.utc),
            region="NSW1",
        ),
        session=session,   # optional AsyncSession for DB sources
    )

Framework boundary: DB imports are deferred (inside functions) and guarded
by try/except so the module works in test environments with no DB.
"""
from app.engines.temporalrag.schema import TemporalQuery, TemporalDoc, RetrievalBundle
from app.engines.temporalrag.retriever import retrieve

__all__ = ["retrieve", "TemporalQuery", "TemporalDoc", "RetrievalBundle"]
