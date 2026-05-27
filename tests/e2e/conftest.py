"""Shared e2e fixtures — live server for Playwright tests."""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

_PROJECT_ROOT = Path(__file__).parent.parent.parent


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def app_url():
    """Start a live uvicorn server on a free port; yield its base URL."""
    port = _free_port()
    env = {
        **os.environ,
        "DATABASE_URL": "sqlite+aiosqlite:///:memory:",
        "GRIDVERDICT_DEV_NO_AUTH": "true",
        "JWT_SECRET": "e2e-test-secret-do-not-use",
        "DECOMPOSER_BACKEND": "rule_based",
    }
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "app.api.main:app",
            "--host", "127.0.0.1",
            "--port", str(port),
            "--log-level", "error",
        ],
        env=env,
        cwd=str(_PROJECT_ROOT),
    )

    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            pytest.fail(f"Server exited early (code {proc.returncode})")
        try:
            httpx.get(f"{base}/api/health", timeout=1.0)
            break
        except Exception:
            time.sleep(0.3)
    else:
        proc.kill()
        pytest.fail("Live server failed to start within 20 seconds")

    yield base

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
