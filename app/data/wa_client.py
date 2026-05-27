"""Future Western Australia market data client stub.

GridVerdict v1 is NEM-only. Western Australia is a separate market operated
through AEMO WA, with different data feeds, market notices, and 30-minute
dispatch intervals. A real implementation would need at least:

- STEM / WEM price feed ingestion
- AEMO WA market notice parsing
- WA-specific interval semantics and region codes
- Separate evidence contracts, because NEMWeb DispatchIS parsers do not apply
"""
from __future__ import annotations


class WAMarketClientNotImplemented(RuntimeError):
    pass


class WAMarketClient:
    def __init__(self) -> None:
        raise WAMarketClientNotImplemented(
            "WA/SWIS is outside GridVerdict v1. Enable only after a dedicated WA data path exists."
        )
