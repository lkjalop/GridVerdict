"""Auth routes — register, login, token refresh.

POST /auth/register   — create user + tenant (first user = admin)
POST /auth/token      — issue JWT (form: username + password)
GET  /auth/me         — current user info
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import (
    LOCAL_TENANT_ID,
    TokenPayload,
    TokenResponse,
    create_access_token,
    hash_password,
    verify_password,
)
from app.api.deps import CurrentUser, DBSession, get_current_user, get_db
from app.db.models import Tenant, User

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    email: str
    password: str
    tenant_name: str = "local"


class UserOut(BaseModel):
    id: str
    email: str
    tenant_id: str


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_db)):
    if len(body.password) < 8:
        raise HTTPException(status_code=422, detail="Password must be at least 8 characters")

    # Check email uniqueness
    existing = await db.execute(select(User).where(User.email == body.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email already registered")

    # Get or create tenant
    tenant_result = await db.execute(select(Tenant).where(Tenant.name == body.tenant_name))
    tenant = tenant_result.scalar_one_or_none()
    if tenant is None:
        tenant = Tenant(id=str(uuid.uuid4()), name=body.tenant_name)
        db.add(tenant)
        await db.flush()

    user = User(
        id=str(uuid.uuid4()),
        tenant_id=tenant.id,
        email=body.email,
        hashed_password=hash_password(body.password),
    )
    db.add(user)
    await db.flush()

    return UserOut(id=user.id, email=user.email, tenant_id=user.tenant_id)


@router.post("/token", response_model=TokenResponse)
async def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(User).where(User.email == form.username))
    user = result.scalar_one_or_none()
    if user is None or not verify_password(form.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account inactive")

    return create_access_token(user.id, user.tenant_id, user.email)


@router.get("/me", response_model=UserOut)
async def me(user: TokenPayload = Depends(get_current_user)):
    return UserOut(id=user.sub, email=user.email, tenant_id=user.tenant_id)
