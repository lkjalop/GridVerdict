"""Shared deterministic geography aliases for NEM region mapping."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "geo_aliases.yaml"


@lru_cache
def load_geo_aliases(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else _CONFIG_PATH
    with p.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {
        "regions": data.get("regions", {}),
        "non_nem": data.get("non_nem", {}),
    }


@lru_cache
def alias_to_region() -> dict[str, str]:
    data = load_geo_aliases()
    mapping: dict[str, str] = {}
    for region, info in data["regions"].items():
        mapping[region.lower()] = region
        for alias in info.get("aliases", []):
            mapping[str(alias).lower()] = region
    return mapping


def known_regions() -> list[str]:
    return list(load_geo_aliases()["regions"].keys())


def _contains_phrase(text: str, phrase: str) -> bool:
    # Phrase-aware boundary check. Keeps "sa" from matching "dispatch".
    pattern = r"(?<![a-z0-9])" + re.escape(phrase.lower()) + r"(?![a-z0-9])"
    return re.search(pattern, text.lower()) is not None


def detect_regions(text: str) -> tuple[list[str], list[str]]:
    """Return deterministic NEM region codes and audit notes found in text."""
    found: list[str] = []
    notes: list[str] = []
    lower = text.lower()
    for alias, region in alias_to_region().items():
        if _contains_phrase(lower, alias):
            if region not in found:
                found.append(region)
            if alias != region.lower() and alias not in {"nsw", "vic", "qld", "sa", "tas"}:
                notes.append(f"{alias.title()} mapped to {region}")
    return found, notes


def detect_non_nem(text: str) -> list[str]:
    lower = text.lower()
    notes = []
    for alias, note in load_geo_aliases()["non_nem"].items():
        if _contains_phrase(lower, str(alias)):
            notes.append(str(note))
    return notes


def build_geo_prompt_section() -> str:
    data = load_geo_aliases()
    lines = ["The NEM has exactly 5 regions. Map place names using this table:"]
    for region, info in data["regions"].items():
        aliases = ", ".join(info.get("aliases", []))
        lines.append(f"- {region}: {info.get('label', region)} — includes {aliases}")
    lines.append("")
    lines.append("Regions NOT in the NEM:")
    for alias, note in data["non_nem"].items():
        lines.append(f"- {alias}: {note}")
    return "\n".join(lines)


def correct_region_hallucinations(
    entities: dict[str, list[str]],
    raw_text: str,
) -> tuple[dict[str, list[str]], list[str]]:
    """Add deterministic regions from aliases and flag contradictions.

    The correction is additive: it never deletes an LLM region, but it records
    when a known alias implies a region missing from the LLM output.
    """
    corrected = dict(entities or {})
    current = [r.upper() for r in corrected.get("regions", [])]
    alias_regions, notes = detect_regions(raw_text)
    corrections: list[str] = []
    for region in alias_regions:
        if region not in current:
            current.append(region)
            corrections.append(f"Added {region} from deterministic geography alias")
    if current:
        corrected["regions"] = current
    return corrected, notes + corrections
