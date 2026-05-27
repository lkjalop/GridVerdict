"""TemporalRAG MCP endpoint.

POST /temporalrag/query
  Accepts a TemporalQuery (valid_time_from/to, region, sources, max_docs)
  and returns a ranked RetrievalBundle of bitemporal evidence documents.

  This is a READ-ONLY endpoint — no market actions, no writes to any system.
  Tool outputs are untrusted evidence for the deterministic reasoning layer.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, field_validator

from app.api.auth import TokenPayload
from app.api.deps import get_current_user, get_db
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/temporalrag", tags=["temporalrag"])

_SUPPORTED_SOURCES = {"market_events", "notice", "trace", "analog", "news"}


class TemporalRAGRequest(BaseModel):
    valid_time_from: datetime
    valid_time_to: datetime
    region: str | None = None
    source_types: list[str] | None = None
    max_docs: int = 20

    @field_validator("max_docs")
    @classmethod
    def clamp_max_docs(cls, v: int) -> int:
        return max(1, min(v, 100))

    @field_validator("source_types")
    @classmethod
    def validate_sources(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        unknown = set(v) - _SUPPORTED_SOURCES
        if unknown:
            raise ValueError(f"Unknown source types: {sorted(unknown)}. Valid: {sorted(_SUPPORTED_SOURCES)}")
        return v


class TemporalDocResponse(BaseModel):
    doc_id: str
    source_type: str
    valid_time: datetime
    system_time: datetime
    relevance_score: float
    citation: str
    content: dict[str, Any]


class TemporalRAGResponse(BaseModel):
    total_docs: int
    source_counts: dict[str, int]
    leakage_filtered: int
    elapsed_ms: float
    docs: list[TemporalDocResponse]


@router.post("/query", response_model=TemporalRAGResponse)
async def temporalrag_query(
    body: TemporalRAGRequest,
    user: TokenPayload = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Retrieve bitemporal evidence documents across all configured sources.

    This is read-only and returns untrusted evidence — callers are responsible
    for deterministic verification of all returned content before acting on it.
    """
    from app.engines.temporalrag.retriever import retrieve
    from app.engines.temporalrag.schema import TemporalQuery

    # Normalise timestamps
    vt_from = body.valid_time_from
    vt_to = body.valid_time_to
    if vt_from.tzinfo is None:
        vt_from = vt_from.replace(tzinfo=timezone.utc)
    if vt_to.tzinfo is None:
        vt_to = vt_to.replace(tzinfo=timezone.utc)

    if vt_from >= vt_to:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="valid_time_from must be before valid_time_to",
        )

    region = body.region.upper() if body.region else None
    source_types = body.source_types or list(_SUPPORTED_SOURCES)

    query = TemporalQuery(
        valid_time_from=vt_from,
        valid_time_to=vt_to,
        system_time_at_query=datetime.now(timezone.utc),
        region=region,
        source_types=source_types,
        max_docs=body.max_docs,
    )

    bundle = await retrieve(query, session=db)

    return TemporalRAGResponse(
        total_docs=bundle.total_docs,
        source_counts=bundle.source_counts,
        leakage_filtered=bundle.leakage_filtered,
        elapsed_ms=round(bundle.elapsed_ms, 1),
        docs=[
            TemporalDocResponse(
                doc_id=d.doc_id,
                source_type=d.source_type,
                valid_time=d.valid_time,
                system_time=d.system_time,
                relevance_score=d.relevance_score,
                citation=d.citation,
                content=d.content,
            )
            for d in bundle.docs
        ],
    )
