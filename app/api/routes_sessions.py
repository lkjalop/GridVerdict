"""Chat session CRUD routes.

POST   /sessions            — create session
GET    /sessions            — list user's sessions
GET    /sessions/{id}       — get session + queries
DELETE /sessions/{id}       — soft-delete (not implemented, title update only)
PATCH  /sessions/{id}/title — rename
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import TokenPayload
from app.api.deps import CurrentUser, DBSession, get_current_user, get_db
from app.db.models import Query as QueryModel
from app.db.models import Session as SessionModel

router = APIRouter(prefix="/sessions", tags=["sessions"])


class SessionCreate(BaseModel):
    region: str = "NSW1"
    title: str | None = None


class SessionOut(BaseModel):
    id: str
    region: str
    title: str | None
    created_at: datetime
    updated_at: datetime
    query_count: int = 0

    model_config = {"from_attributes": True}


class SessionDetail(SessionOut):
    queries: list[dict] = []


@router.post("", response_model=SessionOut, status_code=status.HTTP_201_CREATED)
async def create_session(
    body: SessionCreate,
    user: TokenPayload = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    session = SessionModel(
        id=str(uuid.uuid4()),
        tenant_id=user.tenant_id,
        user_id=user.sub,
        region=body.region.upper(),
        title=body.title,
    )
    db.add(session)
    await db.flush()
    await db.commit()   # commit before response so the query route sees it immediately
    await db.refresh(session)
    return SessionOut(
        id=session.id,
        region=session.region,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


@router.get("", response_model=list[SessionOut])
async def list_sessions(
    user: TokenPayload = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(SessionModel)
        .where(
            SessionModel.tenant_id == user.tenant_id,
            SessionModel.user_id == user.sub,
            SessionModel.deleted_at.is_(None),
        )
        .order_by(SessionModel.updated_at.desc())
        .limit(100)
    )
    sessions = result.scalars().all()
    return [
        SessionOut(
            id=s.id,
            region=s.region,
            title=s.title,
            created_at=s.created_at,
            updated_at=s.updated_at,
        )
        for s in sessions
    ]


@router.get("/{session_id}", response_model=SessionDetail)
async def get_session(
    session_id: str,
    user: TokenPayload = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    s = await _get_owned_session(session_id, user, db)
    result = await db.execute(
        select(QueryModel)
        .where(QueryModel.session_id == session_id)
        .order_by(QueryModel.created_at.asc())
    )
    queries = result.scalars().all()
    return SessionDetail(
        id=s.id,
        region=s.region,
        title=s.title,
        created_at=s.created_at,
        updated_at=s.updated_at,
        query_count=len(queries),
        queries=[
            {
                "id": q.id,
                "raw_query": q.raw_query,
                "intent": q.intent,
                "verdict": q.verdict,
                "answer": q.answer,
                "created_at": q.created_at.isoformat(),
            }
            for q in queries
        ],
    )


class TitleUpdate(BaseModel):
    title: str


@router.patch("/{session_id}/title", response_model=SessionOut)
async def rename_session(
    session_id: str,
    body: TitleUpdate,
    user: TokenPayload = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    s = await _get_owned_session(session_id, user, db)
    s.title = body.title[:200]
    s.updated_at = datetime.now(timezone.utc)
    await db.flush()
    return SessionOut(
        id=s.id,
        region=s.region,
        title=s.title,
        created_at=s.created_at,
        updated_at=s.updated_at,
    )


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: str,
    user: TokenPayload = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    s = await _get_owned_session(session_id, user, db)
    s.deleted_at = datetime.now(timezone.utc)
    await db.flush()


async def _get_owned_session(
    session_id: str, user: TokenPayload, db: AsyncSession
) -> SessionModel:
    result = await db.execute(
        select(SessionModel).where(
            SessionModel.id == session_id,
            SessionModel.tenant_id == user.tenant_id,
            SessionModel.deleted_at.is_(None),
        )
    )
    s = result.scalar_one_or_none()
    if s is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return s
