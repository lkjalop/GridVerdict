from __future__ import annotations

from pathlib import Path

from app.data.historical_acquisition import build_manifest, write_manifest


def test_build_manifest_contains_reproducibility_fields(tmp_path: Path):
    csv = tmp_path / "sample.csv"
    csv.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    manifest = build_manifest(
        start="2023/05/01 00:00:00",
        end="2026/05/01 00:00:00",
        cache_dir=str(tmp_path / "cache"),
        output_dir=str(tmp_path),
        command=["python", "-m", "app.data.historical_acquisition", "--dry-run"],
        acquired_files=[str(csv)],
    )
    assert manifest["date_range"]["start"] == "2023/05/01 00:00:00"
    assert "DISPATCHPRICE" in manifest["tables"].values()
    assert manifest["row_counts"][str(csv)] == 2
    assert len(manifest["checksums"][str(csv)]) == 64


def test_write_manifest_creates_parent(tmp_path: Path):
    out = tmp_path / "nested" / "manifest.json"
    write_manifest(out, {"ok": True})
    assert out.exists()
