"""Trace routes — fetch bitemporal decision traces.

GET /traces/{trace_id}         — full trace by ID
GET /queries/{query_id}/trace  — trace for a specific query
GET /traces?valid_from=&valid_to=  — bitemporal range query
"""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import TokenPayload
from app.api.deps import get_current_user, get_db
from app.core.trace import (
    list_traces,
    list_traces_by_valid_time,
    read_trace,
    to_trace_dict,
)

router = APIRouter(tags=["trace"])


@router.get("/traces/{trace_id}")
async def get_trace(
    trace_id: str,
    user: TokenPayload = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    trace = await read_trace(db, trace_id, user.tenant_id)
    if trace is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Trace not found")
    return to_trace_dict(trace)


@router.get("/queries/{query_id}/trace")
async def get_query_trace(
    query_id: str,
    user: TokenPayload = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    traces = await list_traces(db, user.tenant_id, query_id=query_id, limit=1)
    if not traces:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No trace for this query")
    return to_trace_dict(traces[0])


@router.get("/traces")
async def list_traces_range(
    valid_from: datetime | None = Query(None),
    valid_to: datetime | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    user: TokenPayload = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if valid_from and valid_to:
        traces = await list_traces_by_valid_time(db, user.tenant_id, valid_from, valid_to, limit)
    else:
        traces = await list_traces(db, user.tenant_id, limit=limit)
    return [to_trace_dict(t) for t in traces]
