"""Reproducible three-year NEM historical acquisition manifest tooling."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_START = "2023/05/01 00:00:00"
DEFAULT_END = "2026/05/01 00:00:00"
TABLES = {
    "price": "DISPATCHPRICE",
    "region": "DISPATCHREGIONSUM",
    "interconnector": "DISPATCHINTERCONNECTORRES",
    "constraint": "DISPATCHCONSTRAINT",
    "predispatch": "P5MIN_REGIONSOLUTION",
}


def build_manifest(
    start: str,
    end: str,
    cache_dir: str,
    output_dir: str,
    command: list[str],
    acquired_files: list[str] | None = None,
    status: str = "complete",
    error: str | None = None,
) -> dict[str, Any]:
    files = [_file_manifest(Path(p)) for p in (acquired_files or []) if Path(p).exists()]
    return {
        "acquisition_run_id": hashlib.sha256(
            f"{start}|{end}|{','.join(TABLES.values())}".encode()
        ).hexdigest()[:16],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "date_range": {"start": start, "end": end},
        "tables": TABLES,
        "cache_dir": str(Path(cache_dir).expanduser()),
        "output_dir": str(Path(output_dir).expanduser()),
        "acquisition_command": " ".join(command),
        "status": status,
        "error": error,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "files": files,
        "row_counts": {f["path"]: f.get("row_count") for f in files},
        "checksums": {f["path"]: f["sha256"] for f in files},
        "caveat": (
            "Manifest proves acquisition configuration and local file hashes. "
            "Independent reproducibility requires rerunning the same command against public AEMO/NEMWEB data."
        ),
    }


def acquire_with_nemosis(start: str, end: str, cache_dir: str, output_dir: str) -> list[str]:
    """Download public NEM data with NEMOSIS and save CSVs.

    This intentionally imports NEMOSIS lazily so CI can validate the manifest
    path without having the optional dependency installed.
    """
    try:
        from nemosis import dynamic_data_compiler
    except Exception as exc:
        raise RuntimeError("NEMOSIS is not installed. Run: pip install nemosis") from exc

    import pandas as pd  # noqa: F401 - verifies pandas is installed for to_csv

    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    written: list[str] = []
    for key, table in TABLES.items():
        df = dynamic_data_compiler(
            start_time=start,
            end_time=end,
            table_name=table,
            raw_data_location=cache_dir,
        )
        out = Path(output_dir) / f"{key}_{table}_{_safe_dt(start)}_{_safe_dt(end)}.csv"
        df.to_csv(out, index=False)
        written.append(str(out))
    return written


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _file_manifest(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "row_count": _csv_row_count(path) if path.suffix.lower() == ".csv" else None,
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _csv_row_count(path: Path) -> int:
    with path.open("rb") as f:
        line_count = sum(1 for _ in f)
    return max(0, line_count - 1)


def _safe_dt(value: str) -> str:
    return value.replace("/", "").replace(":", "").replace(" ", "_")


def main() -> int:
    parser = argparse.ArgumentParser(description="Acquire reproducible public NEM archive data")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--cache-dir", default=str(Path.home() / "nem_data_cache"))
    parser.add_argument("--output-dir", default="cache/nem_history")
    parser.add_argument("--manifest", default="cache/nem_history/acquisition_manifest.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    command = [sys.executable, "-m", "app.data.historical_acquisition"] + sys.argv[1:]
    files: list[str] = []
    status = "complete"
    error = None
    if not args.dry_run:
        try:
            files = acquire_with_nemosis(args.start, args.end, args.cache_dir, args.output_dir)
        except Exception as exc:
            status = "failed"
            error = str(exc)
    manifest = build_manifest(
        args.start,
        args.end,
        args.cache_dir,
        args.output_dir,
        command,
        files,
        status=status,
        error=error,
    )
    write_manifest(args.manifest, manifest)
    print(json.dumps({"manifest": args.manifest, "files": len(files)}, indent=2))
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
