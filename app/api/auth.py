"""JWT authentication — issue tokens, validate tokens.

Dev mode: if GRIDVERDICT_DEV_NO_AUTH=true, every request gets a synthetic
local-tenant token so the API works without a running auth server.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from config.settings import get_settings

_settings = get_settings()
_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

LOCAL_TENANT_ID = "00000000-0000-0000-0000-000000000001"
LOCAL_USER_ID = "00000000-0000-0000-0000-000000000002"


class TokenPayload(BaseModel):
    sub: str          # user_id
    tenant_id: str
    email: str
    exp: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int   # seconds


def hash_password(plain: str) -> str:
    return _pwd_ctx.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    if _settings.gridverdict_dev_no_auth and hashed == "noop":
        return True
    return _pwd_ctx.verify(plain, hashed)


def create_access_token(user_id: str, tenant_id: str, email: str) -> TokenResponse:
    expire = datetime.now(timezone.utc) + timedelta(minutes=_settings.jwt_expire_minutes)
    payload = {
        "sub": user_id,
        "tenant_id": tenant_id,
        "email": email,
        "exp": expire,
    }
    token = jwt.encode(payload, _settings.jwt_secret, algorithm=_settings.jwt_algorithm)
    return TokenResponse(
        access_token=token,
        expires_in=_settings.jwt_expire_minutes * 60,
    )


def decode_token(token: str) -> TokenPayload:
    """Raises JWTError if invalid or expired."""
    data = jwt.decode(token, _settings.jwt_secret, algorithms=[_settings.jwt_algorithm])
    return TokenPayload(
        sub=data["sub"],
        tenant_id=data["tenant_id"],
        email=data["email"],
        exp=datetime.fromtimestamp(data["exp"], tz=timezone.utc),
    )


def dev_token_payload() -> TokenPayload:
    """Synthetic token for GRIDVERDICT_DEV_NO_AUTH=true."""
    return TokenPayload(
        sub=LOCAL_USER_ID,
        tenant_id=LOCAL_TENANT_ID,
        email="dev@local",
        exp=datetime.now(timezone.utc) + timedelta(days=365),
    )
