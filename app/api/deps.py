"""FastAPI dependency injection — current user, DB session, AEMO client."""
from __future__ import annotations

from collections.abc import AsyncGenerator

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import TokenPayload, decode_token, dev_token_payload
from app.data.aemo_live_client import AEMOLiveClient, get_aemo_client
from app.data.cache import MarketCache, get_cache
from app.db.session import get_db
from config.settings import get_settings

_settings = get_settings()
_bearer = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> TokenPayload:
    if _settings.gridverdict_dev_no_auth:
        return dev_token_payload()

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return decode_token(credentials.credentials)
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


CurrentUser = Depends(get_current_user)
DBSession = Depends(get_db)


async def aemo_client() -> AEMOLiveClient:
    return get_aemo_client()


async def market_cache() -> MarketCache:
    return get_cache()


AEMOClient = Depends(aemo_client)
Cache = Depends(market_cache)
