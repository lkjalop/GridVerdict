"""Assemble per-region backtest reports into a single checked-in benchmark artifact.

Reads data/backtest_report_{region}.json for each NEM region and produces
data/benchmark_report.json — the reproducibility artifact for the research showcase.

The benchmark_report.json includes:
  - Per-region: CRPS, pinball, calibration, P50/P90 exceedance, skill vs all baselines,
    spike F1, spike-regime breakdown
  - Summary: best model per metric, rank table across regions
  - Provenance: data window, model versions, caveats, coverage status
  - Manifest hash: SHA-256 of the JSON body

Usage:
    python scripts/assemble_benchmark.py
    python scripts/assemble_benchmark.py --regions NSW1 TAS1 SA1
    python scripts/assemble_benchmark.py --window 2023-11-01 2024-02-29
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo))

NEM_REGIONS = ["NSW1", "VIC1", "QLD1", "SA1", "TAS1"]
_DATA_DIR = _repo / "data"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Assemble per-region backtest reports into benchmark artifact")
    p.add_argument("--regions", nargs="+", default=NEM_REGIONS, metavar="REGION")
    p.add_argument("--window", nargs=2, default=["2023-11-01", "2024-02-29"],
                   metavar=("START", "END"),
                   help="Data window (informational only, default: 2023-11-01 2024-02-29)")
    p.add_argument("--out", default=str(_DATA_DIR / "benchmark_report.json"), metavar="PATH")
    return p.parse_args()


def _load_region(region: str) -> dict | None:
    path = _DATA_DIR / f"backtest_report_{region}.json"
    if not path.exists():
        print(f"  WARNING: {path} not found — skipping {region}", file=sys.stderr)
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _best_model(scores: list[dict], metric: str) -> dict | None:
    valid = [s for s in scores if metric in s and s[metric] == s[metric]]  # exclude NaN
    if not valid:
        return None
    return min(valid, key=lambda s: s[metric])


def _rank_table(all_region_data: dict[str, dict]) -> list[dict]:
    """Build a cross-region rank table: for each model, average CRPS across regions."""
    model_crps: dict[str, list[float]] = {}
    model_f1: dict[str, list[float]] = {}
    for region, rd in all_region_data.items():
        for s in rd.get("scores", []):
            m = s["model"]
            model_crps.setdefault(m, []).append(s.get("crps", float("nan")))
            model_f1.setdefault(m, []).append(s.get("spike_f1", float("nan")))

    rows = []
    for model in sorted(model_crps):
        crps_vals = [v for v in model_crps[model] if v == v]
        f1_vals = [v for v in model_f1.get(model, []) if v == v]
        rows.append({
            "model": model,
            "regions_evaluated": len(crps_vals),
            "mean_crps": round(sum(crps_vals) / len(crps_vals), 4) if crps_vals else None,
            "mean_spike_f1": round(sum(f1_vals) / len(f1_vals), 4) if f1_vals else None,
        })
    rows.sort(key=lambda r: r["mean_crps"] or float("inf"))
    return rows


def main() -> None:
    args = _parse_args()

    all_region_data: dict[str, dict] = {}
    for region in args.regions:
        data = _load_region(region)
        if data:
            all_region_data[region] = data

    if not all_region_data:
        print("ERROR: No per-region backtest files found. Run backtest_report.py first.", file=sys.stderr)
        sys.exit(1)

    # Pull provenance from one of the reports
    sample = next(iter(all_region_data.values()))
    generated_at = sample.get("generated_at", datetime.now(timezone.utc).isoformat())

    manifest: dict = {
        "benchmark_version": "1.0",
        "assembled_at": datetime.now(timezone.utc).isoformat(),
        "data_window": {
            "start": args.window[0],
            "end": args.window[1],
        },
        "methodology": {
            "walk_forward_step": "288 intervals (daily origins)",
            "horizon": "6 intervals (30 min)",
            "min_train": "288 intervals (1 day)",
            "spike_threshold": "region-specific (see per-region data)",
        },
        "caveats": [
            (
                "aemo_predispatch feature: historical data uses price[t-6] (30-min lagged price) "
                "as a non-leaking proxy. NOT a real forward-looking AEMO forecast. "
                "skill_vs_aemo reflects skill against this proxy baseline, not real AEMO predispatch. "
                "P5MIN_REGIONSOLUTION backfill not yet implemented."
            ),
            (
                "unit_dispatch_events: DISPATCH_UNIT_SOLUTION not in backfill_tables. "
                "Fuel-type attribution and bidding-behaviour features are unavailable. "
                "Seasonal and demand features are zero-initialised (not from PASA data)."
            ),
            (
                "roll_vol_12 and seasonal_price ARE computed from dispatch price history. "
                "interconnector_room and renewable_frac are zeroed (no PASA/DISPATCH_UNIT data). "
                "These omissions will understate model performance relative to a production system."
            ),
        ],
        "data_sources": {
            "AEMO_DISPATCH_PRICE": "operational",
            "AEMO_PREDISPATCH_30MIN": "scaffolded (30-min lag proxy, not real PD)",
            "driver/constraint": "operational",
            "driver/interconnector": "partial",
            "unit_dispatch_events": "unavailable",
            "generator_units": "operational",
        },
        "per_region": all_region_data,
        "cross_region_rank": _rank_table(all_region_data),
    }

    # Best model summary
    all_scores_flat = [
        {**s, "region": region}
        for region, rd in all_region_data.items()
        for s in rd.get("scores", [])
    ]
    best_crps = _best_model(all_scores_flat, "crps")
    best_f1 = _best_model([{**s, "region": r} for r, rd in all_region_data.items()
                           for s in rd.get("scores", [])], "spike_f1")

    manifest["summary"] = {
        "regions_completed": len(all_region_data),
        "regions_missing": [r for r in args.regions if r not in all_region_data],
        "best_crps_overall": {
            "model": best_crps["model"] if best_crps else None,
            "crps": best_crps["crps"] if best_crps else None,
            "region": best_crps.get("region") if best_crps else None,
        } if best_crps else None,
    }

    # Manifest hash over everything except the hash field
    body = json.dumps(
        {k: v for k, v in manifest.items()},
        sort_keys=True,
        default=str,
    ).encode()
    manifest["manifest_hash"] = hashlib.sha256(body).hexdigest()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print(f"\nBenchmark report assembled: {out_path}")
    print(f"  Regions: {', '.join(all_region_data)}")
    print(f"  Data window: {args.window[0]} to {args.window[1]}")
    print(f"  Manifest hash: {manifest['manifest_hash'][:16]}...")
    print(f"\nCross-region model rank (by mean CRPS):")
    print(f"  {'Model':<22} {'Regions':>8} {'Mean CRPS':>10} {'Mean F1':>9}")
    print(f"  {'-'*54}")
    for row in manifest["cross_region_rank"]:
        crps_str = f"{row['mean_crps']:.3f}" if row["mean_crps"] is not None else "  n/a"
        f1_str = f"{row['mean_spike_f1']:.3f}" if row["mean_spike_f1"] is not None else "  n/a"
        print(f"  {row['model']:<22} {row['regions_evaluated']:>8} {crps_str:>10} {f1_str:>9}")


if __name__ == "__main__":
    main()
