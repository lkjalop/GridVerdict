"""Backtest API routes.

POST /api/backtest/run    — kick off a backtest job (returns job_id)
GET  /api/backtest/{job_id} — poll job status; returns result when done
GET  /api/backtest/regions  — list regions with enough history to backtest
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(tags=["backtest"])

# In-process job store (sufficient for single-instance deployment)
_jobs: dict[str, dict[str, Any]] = {}


# ── Request / response schemas ──────────────────────────────────────────────

class BacktestRequest(BaseModel):
    region: str = Field(..., description="NEM region: NSW1, VIC1, QLD1, SA1, TAS1")
    lookback_days: int = Field(7, ge=1, le=90, description="Days of history to use")
    horizon_intervals: int = Field(6, ge=1, le=36, description="Intervals ahead to forecast")
    include_lnn: bool = Field(True, description="Include LNN model (requires torch + history)")


class BacktestJobResponse(BaseModel):
    job_id: str
    status: str          # pending | running | done | error
    region: str
    submitted_at: str
    completed_at: str | None = None
    result: dict | None = None
    error: str | None = None


# ── Routes ───────────────────────────────────────────────────────────────────

@router.post("/backtest/run", response_model=BacktestJobResponse)
async def start_backtest(req: BacktestRequest) -> BacktestJobResponse:
    """Queue and start a backtest job.  Returns immediately with a job_id to poll."""
    valid_regions = {"NSW1", "VIC1", "QLD1", "SA1", "TAS1"}
    if req.region.upper() not in valid_regions:
        raise HTTPException(status_code=400, detail=f"Unknown region: {req.region}")

    job_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    _jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "region": req.region.upper(),
        "submitted_at": now,
        "completed_at": None,
        "result": None,
        "error": None,
    }

    # Run the backtest in the background so this endpoint returns immediately
    asyncio.create_task(_run_job(job_id, req))
    return BacktestJobResponse(**_jobs[job_id])


@router.get("/backtest/{job_id}", response_model=BacktestJobResponse)
async def get_backtest_job(job_id: str) -> BacktestJobResponse:
    """Poll backtest job status and retrieve results when done."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return BacktestJobResponse(**job)


@router.get("/backtest")
async def list_backtest_jobs(limit: int = 20) -> list[dict]:
    """Return the most recent backtest jobs (newest first)."""
    jobs = sorted(_jobs.values(), key=lambda j: j["submitted_at"], reverse=True)
    return jobs[:limit]


# ── Background task ──────────────────────────────────────────────────────────

async def _run_job(job_id: str, req: BacktestRequest) -> None:
    from app.engines.backtest import run_region_backtest

    job = _jobs[job_id]
    job["status"] = "running"
    try:
        report = await run_region_backtest(
            region=req.region.upper(),
            lookback_days=req.lookback_days,
            horizon_intervals=req.horizon_intervals,
            include_lnn=req.include_lnn,
        )
        job["status"] = "done"
        job["completed_at"] = datetime.now(timezone.utc).isoformat()
        job["result"] = report.to_dict()
        logger.info(
            "Backtest job %s done — %s, %d origins",
            job_id, req.region, report.n_origins,
        )
    except Exception as exc:
        job["status"] = "error"
        job["completed_at"] = datetime.now(timezone.utc).isoformat()
        job["error"] = str(exc)
        logger.warning("Backtest job %s failed: %s", job_id, exc)
