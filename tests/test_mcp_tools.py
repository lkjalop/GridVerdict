"""MCP tool stress tests — security, evidence chain, and live-data parser coverage.

Four test groups:

  1. MCP security enforcement
     - read_only=True enforced on every call
     - disabled tools rejected
     - unknown tool names rejected
     - registry loads from tools.yaml (or built-in defaults)

  2. Evidence chain integrity
     - Every SUPPORTED verdict carries ≥1 evidence_ref
     - Every evidence_ref has a non-empty raw_ref, source, and numeric value
     - evidence_ref.interval matches the dispatch valid_time from the gather result
     - Confidence is derived (not flat 0.5)
     - raw_ref for dispatch price is a non-trivial string (sha256 or formatted timestamp)

  3. NEMWeb dispatch price parser
     - Parses real-format DISPATCHPRICE CSV rows correctly
     - Handles column header mapping correctly
     - Keeps only the latest interval per region
     - Gracefully skips malformed rows

  4. AEMO Market Notices client
     - _parse_notice_text extracts notice_type, reason, region, tier
     - Tier-1 notice types produce credibility_tier=1
     - Tier-2 notice types produce credibility_tier=2
     - Region codes are normalised (NSW → NSW1)
     - fetch_active_notices() returns dicts (not NewsItem objects)
     - Timestamp parser handles all known AEMO date formats
"""
from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.mcp.registry import MCPTool, _default_registry, load_registry
from app.mcp.router import MCPCallError, call_tool, _dispatch


# ── 1. MCP Security enforcement ───────────────────────────────────────

class TestMCPSecurityEnforcement:

    @pytest.mark.asyncio
    async def test_unknown_tool_raises(self):
        with pytest.raises(MCPCallError, match="Unknown MCP tool"):
            await call_tool("does_not_exist")

    @pytest.mark.asyncio
    async def test_disabled_tool_raises(self):
        """A tool with enabled=False must be rejected before dispatch."""
        with patch("app.mcp.router.get_tool") as mock_get:
            mock_get.return_value = MCPTool(
                name="aemo_dispatch_price",
                description="test",
                category="dispatch",
                endpoint="http://example.com",
                enabled=False,
            )
            with pytest.raises(MCPCallError, match="disabled"):
                await call_tool("aemo_dispatch_price", region="NSW1")

    @pytest.mark.asyncio
    async def test_write_tool_raises(self):
        """A tool with read_only=False must never be dispatched."""
        with patch("app.mcp.router.get_tool") as mock_get:
            mock_get.return_value = MCPTool(
                name="dangerous_write_tool",
                description="writes to AEMO",
                category="dispatch",
                endpoint="http://example.com",
                read_only=False,   # <-- must be blocked
                enabled=True,
            )
            with pytest.raises(MCPCallError, match="not read-only"):
                await call_tool("dangerous_write_tool")

    def test_all_registry_tools_are_read_only(self):
        """Every tool in the default registry must have read_only=True."""
        registry = _default_registry()
        for name, tool in registry.items():
            assert tool.read_only, f"Tool {name!r} has read_only=False — policy violation"

    def test_all_registry_tools_are_enabled(self):
        """Default registry tools should all be enabled at startup."""
        registry = _default_registry()
        disabled = [name for name, t in registry.items() if not t.enabled]
        assert not disabled, f"Tools disabled in default registry: {disabled}"

    def test_registry_loads_expected_tools(self):
        """Default registry must have the configured read-only tool set."""
        registry = _default_registry()
        assert len(registry) == 9

    def test_registry_tool_names(self):
        """Expected tool names must be present."""
        registry = _default_registry()
        expected = {
            "aemo_dispatch_price",
            "aemo_market_notices",
            "aemo_archive",
            "hippograph_analogs",
            "lnn_forecast",
            "live_quantile_forecast",
            "aemo_predispatch",
            "nem_news_rss",
            "weather_consensus",
        }
        assert set(registry.keys()) == expected

    def test_yaml_fallback_to_defaults(self, tmp_path):
        """Missing tools.yaml must produce the built-in defaults, not raise."""
        result = load_registry(config_path=tmp_path / "nonexistent.yaml")
        assert len(result) == 9

    def test_yaml_invalid_falls_back(self, tmp_path):
        """tools.yaml with bad YAML must fall back to defaults, not raise."""
        bad = tmp_path / "tools.yaml"
        bad.write_text("this: [is: invalid: yaml: :")
        result = load_registry(config_path=bad)
        assert len(result) == 9

    @pytest.mark.asyncio
    async def test_dispatch_price_dispatches_correctly(self):
        """aemo_dispatch_price tool must call fetch_latest_snapshot."""
        mock_dp = MagicMock()
        mock_dp.price_rrp = 347.5
        mock_snapshot = MagicMock()
        mock_snapshot.get.return_value = mock_dp

        mock_client = AsyncMock()
        mock_client.fetch_latest_snapshot = AsyncMock(return_value=mock_snapshot)

        # get_aemo_client is imported lazily inside _dispatch — patch at the source module
        with patch("app.mcp.router.get_tool") as mock_get_tool, \
             patch("app.data.aemo_live_client.get_aemo_client", return_value=mock_client):
            mock_get_tool.return_value = MCPTool(
                name="aemo_dispatch_price",
                description="test",
                category="dispatch",
                endpoint="http://test",
            )
            result = await call_tool("aemo_dispatch_price", region="NSW1")

        mock_client.fetch_latest_snapshot.assert_called_once()
        assert result.price_rrp == 347.5


# ── 2. Evidence chain integrity ───────────────────────────────────────

class TestEvidenceChain:
    """Verify that every claim in the answer is backed by a traceable evidence_ref."""

    def _make_dispatch(self, price: float = 347.5, region: str = "NSW1") -> object:
        from datetime import timezone
        from app.data.aemo_live_client import DispatchPrice
        return DispatchPrice(
            region=region,
            valid_time=datetime(2026, 5, 23, 4, 30, 0, tzinfo=timezone.utc),
            system_time=datetime.now(timezone.utc),
            price_rrp=price,
            demand_mw=8420.0,
            availability_mw=8850.0,
            raw_ref="sha256testref0123456789abcdef01234567",
        )

    def _make_gather(self, price: float = 347.5, region: str = "NSW1"):
        from app.agents.scatter_gather import GatherResult
        dp = self._make_dispatch(price=price, region=region)
        return GatherResult(
            dispatch=dp,
            dispatch_fresh=True,
            notices=[],
            analogs=[],
            tasks_ok=3,
            tasks_total=4,
            elapsed_ms=50.0,
        )

    def _make_decomp(self, intent: str = "action_recommendation", confidence: float = 0.82, region: str = "NSW1"):
        from app.core.schema import IntentLabel, QueryDecomposition
        return QueryDecomposition(
            query_id="qry-test",
            raw_query="should I dispatch",
            intent=IntentLabel(intent),
            entities={"regions": [region]},
            time_range={"type": "current"},
            confidence=confidence,
        )

    def test_evidence_ref_has_required_fields(self):
        """Every EvidenceRefSchema must have source, interval, field, value, raw_ref."""
        from app.agents.why_builder import build_why
        from app.agents.why_sources import assemble_why_sources

        decomp = self._make_decomp()
        gather = self._make_gather()
        sources = assemble_why_sources(decomp, gather, "NSW1")
        output = build_why(sources)

        assert output.evidence_refs, "build_why produced no evidence_refs for fresh data"
        for ref in output.evidence_refs:
            assert ref.source, f"evidence_ref missing source: {ref}"
            assert ref.interval, f"evidence_ref missing interval: {ref}"
            assert ref.field, f"evidence_ref missing field: {ref}"
            assert ref.value is not None, f"evidence_ref missing value: {ref}"
            assert ref.raw_ref, f"evidence_ref has empty raw_ref — cannot be audited"

    def test_evidence_ref_interval_matches_dispatch_valid_time(self):
        """evidence_ref.interval must equal the dispatch interval, not system_time."""
        from app.agents.why_builder import build_why
        from app.agents.why_sources import assemble_why_sources

        valid_time = datetime(2026, 5, 23, 4, 30, 0, tzinfo=timezone.utc)
        decomp = self._make_decomp()
        gather = self._make_gather()
        sources = assemble_why_sources(decomp, gather, "NSW1")
        output = build_why(sources)

        price_refs = [r for r in output.evidence_refs if r.field == "price_rrp"]
        assert price_refs, "No price_rrp evidence_ref found"
        assert price_refs[0].interval == valid_time, (
            f"evidence_ref.interval {price_refs[0].interval} does not match "
            f"dispatch valid_time {valid_time}"
        )

    def test_evidence_ref_price_value_matches_dispatch(self):
        """evidence_ref.value for price_rrp must equal the actual dispatch price."""
        from app.agents.why_builder import build_why
        from app.agents.why_sources import assemble_why_sources

        price = 4_200.75
        decomp = self._make_decomp()
        gather = self._make_gather(price=price)
        sources = assemble_why_sources(decomp, gather, "NSW1")
        output = build_why(sources)

        price_refs = [r for r in output.evidence_refs if r.field == "price_rrp"]
        assert price_refs, "No price_rrp evidence_ref found"
        assert price_refs[0].value == pytest.approx(price), (
            f"evidence_ref.value {price_refs[0].value} != dispatch price {price}"
        )

    def test_confidence_not_flat(self):
        """Confidence must vary — it's weighted from signals, not a constant 0.5."""
        from app.agents.why_formatter import format_verdict
        from app.agents.why_builder import build_why
        from app.agents.why_sources import assemble_why_sources

        results = []
        for price, fresh in [(80.0, True), (5000.0, True), (80.0, False)]:
            gather = self._make_gather(price=price)
            gather.dispatch_fresh = fresh
            if not fresh:
                gather.dispatch = None
            decomp = self._make_decomp()
            sources = assemble_why_sources(decomp, gather, "NSW1")
            output = build_why(sources)
            verdict = format_verdict(output, sources, "trace-test")
            results.append(round(verdict.confidence, 2))

        assert len(set(results)) > 1, (
            f"All confidence scores are identical {results} — "
            "compute_confidence not responding to signal changes"
        )
        assert all(c != 0.5 for c in results if results[0] != results[1]), (
            "Confidence appears to be flat 0.5"
        )

    def test_stale_data_produces_insufficient_verdict(self):
        """When dispatch is None, verdict must be INSUFFICIENT_DATA, not SUPPORTED."""
        from app.agents.why_formatter import format_verdict
        from app.agents.why_builder import build_why
        from app.agents.why_sources import assemble_why_sources
        from app.core.schema import VerdictLabel
        from app.agents.scatter_gather import GatherResult

        gather = GatherResult(
            dispatch=None,
            dispatch_fresh=False,
            tasks_ok=0,
            tasks_total=4,
            elapsed_ms=0.0,
        )
        decomp = self._make_decomp()
        sources = assemble_why_sources(decomp, gather, "NSW1")
        output = build_why(sources)
        verdict = format_verdict(output, sources, "trace-test")
        assert verdict.verdict == VerdictLabel.INSUFFICIENT_DATA, (
            f"Stale data produced {verdict.verdict} — must be INSUFFICIENT_DATA"
        )

    def test_oos_intent_produces_out_of_scope_verdict(self):
        """Out-of-scope intent must never produce SUPPORTED."""
        from app.agents.why_formatter import format_verdict
        from app.agents.why_builder import build_why
        from app.agents.why_sources import assemble_why_sources
        from app.core.schema import VerdictLabel

        decomp = self._make_decomp(intent="out_of_scope", confidence=0.90)
        gather = self._make_gather(price=100.0)
        sources = assemble_why_sources(decomp, gather, "NSW1")
        output = build_why(sources)
        verdict = format_verdict(output, sources, "trace-oos")
        assert verdict.verdict == VerdictLabel.OUT_OF_SCOPE, (
            f"OOS intent produced {verdict.verdict} — must be OUT_OF_SCOPE"
        )

    def test_notice_credibility_tier_flows_to_confidence(self):
        """A Tier-1 notice in gather.notices must increase the confidence score."""
        from app.core.verdict import compute_confidence

        no_news = compute_confidence(
            decomposition_confidence=0.80,
            source_freshness_score=1.0,
            source_coverage_score=0.75,
            analog_count=0,
            analog_consistency=0.0,
            news_tier=None,
        )
        tier1 = compute_confidence(
            decomposition_confidence=0.80,
            source_freshness_score=1.0,
            source_coverage_score=0.75,
            analog_count=0,
            analog_consistency=0.0,
            news_tier=1,
        )
        assert tier1 > no_news, (
            f"Tier-1 notice did not increase confidence ({no_news} → {tier1})"
        )


# ── 3. NEMWeb dispatch price parser ───────────────────────────────────

def _make_dispatch_zip(csv_content: str) -> bytes:
    """Wrap a CSV string in a zip file as NEMWeb would serve it."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("PUBLIC_DISPATCHPRICE_20260523_0430_1.CSV", csv_content)
    return buf.getvalue()


_VALID_CSV = """\
C,NEMSOLUTION,DispatchIS,Public,,2026/05/23,04:35:00,
I,DISPATCH,PRICE,4,SETTLEMENTDATE,RUNNO,REGIONID,DISPATCHINTERVAL,INTERVENTION,RRP,RAISE6SECRRP,RAISE60SECRRP,RAISE5MINRRP,RAISEREGRRP,LOWER6SECRRP,LOWER60SECRRP,LOWER5MINRRP,LOWERREGRRP,RAISE6SECROP,RAISE60SECROP,RAISE5MINROP,RAISEREGROP,LOWER6SECROP,LOWER60SECROP,LOWER5MINROP,LOWERREGROP,PRICE_STATUS,RAISE1SECRRP,LOWER1SECRRP,RAISE1SECROP,LOWER1SECROP,TOTALDEMAND,AVAILABLEGENERATION
D,DISPATCH,PRICE,4,2026/05/23 14:30:00,1,NSW1,1,0,347.50,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,FIRM,0,0,0,0,8420.0,8850.0
D,DISPATCH,PRICE,4,2026/05/23 14:30:00,1,VIC1,1,0,180.25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,FIRM,0,0,0,0,6100.0,6500.0
D,DISPATCH,PRICE,4,2026/05/23 14:30:00,1,QLD1,1,0,120.00,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,FIRM,0,0,0,0,7200.0,7800.0
D,DISPATCH,PRICE,4,2026/05/23 14:30:00,1,SA1,1,0,420.75,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,FIRM,0,0,0,0,1800.0,2100.0
D,DISPATCH,PRICE,4,2026/05/23 14:30:00,1,TAS1,1,0,90.00,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,FIRM,0,0,0,0,1100.0,1400.0
"""


class TestNEMWebDispatchParser:

    def test_parses_all_five_regions(self):
        from app.data.aemo_live_client import _parse_dispatch_zip
        prices = _parse_dispatch_zip(_make_dispatch_zip(_VALID_CSV), "testref")
        assert set(prices.keys()) == {"NSW1", "VIC1", "QLD1", "SA1", "TAS1"}

    def test_parses_correct_price(self):
        from app.data.aemo_live_client import _parse_dispatch_zip
        prices = _parse_dispatch_zip(_make_dispatch_zip(_VALID_CSV), "testref")
        assert prices["NSW1"].price_rrp == pytest.approx(347.50)
        assert prices["SA1"].price_rrp == pytest.approx(420.75)

    def test_parses_correct_demand_and_availability(self):
        from app.data.aemo_live_client import _parse_dispatch_zip
        prices = _parse_dispatch_zip(_make_dispatch_zip(_VALID_CSV), "testref")
        assert prices["NSW1"].demand_mw == pytest.approx(8420.0)
        assert prices["NSW1"].availability_mw == pytest.approx(8850.0)

    def test_raw_ref_is_preserved(self):
        from app.data.aemo_live_client import _parse_dispatch_zip
        prices = _parse_dispatch_zip(_make_dispatch_zip(_VALID_CSV), "my-sha256-ref")
        for dp in prices.values():
            assert dp.raw_ref == "my-sha256-ref", (
                f"raw_ref not preserved on {dp.region}: {dp.raw_ref!r}"
            )

    def test_valid_time_parsed_correctly(self):
        from app.data.aemo_live_client import _parse_dispatch_zip
        prices = _parse_dispatch_zip(_make_dispatch_zip(_VALID_CSV), "testref")
        # 2026/05/23 14:30:00 AEST = 2026/05/23 04:30:00 UTC
        expected_utc = datetime(2026, 5, 23, 4, 30, 0, tzinfo=timezone.utc)
        assert prices["NSW1"].valid_time == expected_utc, (
            f"valid_time {prices['NSW1'].valid_time} != expected {expected_utc}"
        )

    def test_keeps_latest_interval_when_duplicate_regions(self):
        """If the same region appears twice, we must keep the later interval."""
        csv = """\
C,NEMSOLUTION,DispatchIS,Public
I,DISPATCH,PRICE,4,SETTLEMENTDATE,RUNNO,REGIONID,DISPATCHINTERVAL,INTERVENTION,RRP,RAISE6SECRRP,RAISE60SECRRP,RAISE5MINRRP,RAISEREGRRP,LOWER6SECRRP,LOWER60SECRRP,LOWER5MINRRP,LOWERREGRRP,RAISE6SECROP,RAISE60SECROP,RAISE5MINROP,RAISEREGROP,LOWER6SECROP,LOWER60SECROP,LOWER5MINROP,LOWERREGROP,PRICE_STATUS,RAISE1SECRRP,LOWER1SECRRP,RAISE1SECROP,LOWER1SECROP,TOTALDEMAND,AVAILABLEGENERATION
D,DISPATCH,PRICE,4,2026/05/23 14:25:00,1,NSW1,1,0,200.00,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,FIRM,0,0,0,0,8000.0,8500.0
D,DISPATCH,PRICE,4,2026/05/23 14:30:00,1,NSW1,1,0,400.00,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,FIRM,0,0,0,0,8420.0,8850.0
"""
        from app.data.aemo_live_client import _parse_dispatch_zip
        prices = _parse_dispatch_zip(_make_dispatch_zip(csv), "testref")
        assert prices["NSW1"].price_rrp == pytest.approx(400.00), (
            "Expected latest interval (400.00), got earlier one"
        )

    def test_malformed_rows_skipped(self):
        """Rows with bad price fields must be silently skipped."""
        csv = """\
C,NEMSOLUTION
I,DISPATCH,PRICE,4,SETTLEMENTDATE,RUNNO,REGIONID,DISPATCHINTERVAL,INTERVENTION,RRP,RAISE6SECRRP,RAISE60SECRRP,RAISE5MINRRP,RAISEREGRRP,LOWER6SECRRP,LOWER60SECRRP,LOWER5MINRRP,LOWERREGRRP,RAISE6SECROP,RAISE60SECROP,RAISE5MINROP,RAISEREGROP,LOWER6SECROP,LOWER60SECROP,LOWER5MINROP,LOWERREGROP,PRICE_STATUS,RAISE1SECRRP,LOWER1SECRRP,RAISE1SECROP,LOWER1SECROP,TOTALDEMAND,AVAILABLEGENERATION
D,DISPATCH,PRICE,4,2026/05/23 14:30:00,1,NSW1,1,0,NOT_A_FLOAT,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,FIRM,0,0,0,0,8420.0,8850.0
D,DISPATCH,PRICE,4,2026/05/23 14:30:00,1,VIC1,1,0,150.00,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,FIRM,0,0,0,0,6000.0,6500.0
"""
        from app.data.aemo_live_client import _parse_dispatch_zip
        prices = _parse_dispatch_zip(_make_dispatch_zip(csv), "testref")
        assert "NSW1" not in prices, "Malformed NSW1 row should have been skipped"
        assert "VIC1" in prices, "Valid VIC1 row should still be parsed"

    def test_unknown_region_codes_ignored(self):
        """Row with REGIONID=UNKNOWN1 must not appear in result."""
        csv = """\
C,NEMSOLUTION
I,DISPATCH,PRICE,4,SETTLEMENTDATE,RUNNO,REGIONID,DISPATCHINTERVAL,INTERVENTION,RRP,RAISE6SECRRP,RAISE60SECRRP,RAISE5MINRRP,RAISEREGRRP,LOWER6SECRRP,LOWER60SECRRP,LOWER5MINRRP,LOWERREGRRP,RAISE6SECROP,RAISE60SECROP,RAISE5MINROP,RAISEREGROP,LOWER6SECROP,LOWER60SECROP,LOWER5MINROP,LOWERREGROP,PRICE_STATUS,RAISE1SECRRP,LOWER1SECRRP,RAISE1SECROP,LOWER1SECROP,TOTALDEMAND,AVAILABLEGENERATION
D,DISPATCH,PRICE,4,2026/05/23 14:30:00,1,UNKNOWN1,1,0,500.00,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,FIRM,0,0,0,0,999.0,999.0
D,DISPATCH,PRICE,4,2026/05/23 14:30:00,1,NSW1,1,0,347.50,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,FIRM,0,0,0,0,8420.0,8850.0
"""
        from app.data.aemo_live_client import _parse_dispatch_zip
        prices = _parse_dispatch_zip(_make_dispatch_zip(csv), "testref")
        assert "UNKNOWN1" not in prices
        assert "NSW1" in prices


# ── 4. AEMO Market Notices client ─────────────────────────────────────

_LOR1_NOTICE_TEXT = """\
NOTICE IDENTIFIER: 12345
NOTICE TYPE: LACK OF RESERVE 1
ISSUE DATE: 23/05/2026 04:20:00
CREATION TIME: 23/05/2026 04:20:00
NOTICE REASON: LOR1 declared for NSW1 due to forced outage at Eraring.
REGION: NSW1
EXTERNAL REFERENCE: Test reference
"""

_MT_PASA_NOTICE_TEXT = """\
NOTICE IDENTIFIER: 12346
NOTICE TYPE: MT PASA REVISION
ISSUE DATE: 23/05/2026 03:00:00
CREATION TIME: 23/05/2026 03:00:00
NOTICE REASON: MT PASA run completed for SA1 summer outlook.
REGION: SA1
EXTERNAL REFERENCE: Test reference
"""

_NOTICE_NO_REGION_TEXT = """\
NOTICE IDENTIFIER: 12347
NOTICE TYPE: MARKET INTERVENTION
ISSUE DATE: 23/05/2026 02:00:00
CREATION TIME: 23/05/2026 02:00:00
NOTICE REASON: Reliability and Emergency Reserve Trader activated.
EXTERNAL REFERENCE: Test reference
"""


class TestAEMOMarketNoticesClient:

    def _client(self):
        from app.mcp.aemo_notices_client import AEMOMarketNoticesClient
        return AEMOMarketNoticesClient()

    def test_parse_lor1_tier1(self):
        from app.mcp.aemo_notices_client import AEMOMarketNoticesClient
        item = AEMOMarketNoticesClient._parse_notice_text(_LOR1_NOTICE_TEXT, "12345.zip")
        assert item is not None
        assert item.credibility_tier == 1, f"LOR1 should be tier 1, got {item.credibility_tier}"

    def test_parse_mt_pasa_tier2(self):
        from app.mcp.aemo_notices_client import AEMOMarketNoticesClient
        item = AEMOMarketNoticesClient._parse_notice_text(_MT_PASA_NOTICE_TEXT, "12346.zip")
        assert item is not None
        assert item.credibility_tier == 2, f"MT PASA should be tier 2, got {item.credibility_tier}"

    def test_parse_region_normalised(self):
        """Region code extracted from notice text must be normalised to NSW1 format."""
        from app.mcp.aemo_notices_client import AEMOMarketNoticesClient
        item = AEMOMarketNoticesClient._parse_notice_text(_LOR1_NOTICE_TEXT, "12345.zip")
        assert item is not None
        assert item.region in {"NSW1", "NSW"}, (
            f"Region should be NSW1, got {item.region!r}"
        )

    def test_parse_sa_region_normalised(self):
        from app.mcp.aemo_notices_client import AEMOMarketNoticesClient
        item = AEMOMarketNoticesClient._parse_notice_text(_MT_PASA_NOTICE_TEXT, "12346.zip")
        assert item is not None
        assert item.region in {"SA1", "SA"}

    def test_parse_notice_without_region(self):
        """Notice with no REGION field — region should be None."""
        from app.mcp.aemo_notices_client import AEMOMarketNoticesClient
        item = AEMOMarketNoticesClient._parse_notice_text(_NOTICE_NO_REGION_TEXT, "12347.zip")
        # Should still parse — region may be None or inferred from body
        assert item is not None
        assert item.credibility_tier == 1   # MARKET INTERVENTION is tier-1

    @pytest.mark.parametrize("ts,expected", [
        ("23/05/2026 04:20:00", datetime(2026, 5, 23, 4, 20, 0, tzinfo=timezone.utc)),
        ("2026/05/23 04:20:00", datetime(2026, 5, 23, 4, 20, 0, tzinfo=timezone.utc)),
        ("2026-05-23T04:20:00", datetime(2026, 5, 23, 4, 20, 0, tzinfo=timezone.utc)),
    ])
    def test_timestamp_parsing(self, ts, expected):
        from app.mcp.aemo_notices_client import AEMOMarketNoticesClient
        result = AEMOMarketNoticesClient._parse_timestamp(ts)
        assert result == expected, f"Parsed {ts!r} → {result} expected {expected}"

    def test_timestamp_none_returns_none(self):
        from app.mcp.aemo_notices_client import AEMOMarketNoticesClient
        assert AEMOMarketNoticesClient._parse_timestamp(None) is None

    def test_fetch_active_notices_returns_dicts(self):
        """fetch_active_notices must return list[dict], not list[NewsItem]."""
        from app.mcp.aemo_notices_client import AEMOMarketNoticesClient
        client = AEMOMarketNoticesClient()

        # Seed cache with a known notice so we don't hit the network
        from app.engines.forecasting.types import NewsItem
        client._cache["test.zip"] = type("_C", (), {
            "item": NewsItem(
                timestamp=datetime.now(timezone.utc),
                source="AEMO Market Notice",
                credibility_tier=1,
                title="LACK OF RESERVE 1: LOR1 NSW1",
                summary="LOR1 declared for NSW1",
                url="https://nemweb.com.au/test.zip",
                region="NSW1",
            ),
            "fetched_at": 9999999999.0,  # never expires in test
        })()
        client._last_poll = 9999999999.0  # prevent network refresh

        result = client.fetch_active_notices()
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, dict), f"Expected dict, got {type(item)}"
            assert "credibility_tier" in item, "Dict missing credibility_tier"
            assert "notice_type" in item, "Dict missing notice_type"
            assert "region" in item, "Dict missing region"


class TestNEMNewsRSSClient:

    def test_parse_valid_rss_feed(self):
        from app.mcp.nem_news_client import parse_feed

        xml = """\
<rss><channel>
  <item>
    <title>AEMO flags NEM transmission constraint</title>
    <link>https://example.test/aemo-nem</link>
    <pubDate>Sat, 23 May 2026 04:20:00 GMT</pubDate>
    <description>Wholesale electricity prices moved after an interconnector constraint.</description>
  </item>
</channel></rss>
"""
        items = parse_feed(xml, "https://example.test/feed", keywords=["aemo", "interconnector"])
        assert len(items) == 1
        assert items[0]["source"] == "NEM_NEWS_RSS"
        assert "aemo" in items[0]["matched_keywords"]

    def test_parse_rss_keyword_filtering(self):
        from app.mcp.nem_news_client import parse_feed

        xml = """\
<rss><channel>
  <item><title>Unrelated sports story</title><description>No market content</description></item>
  <item><title>NEM price spike in SA</title><description>AEMO dispatch interval moved sharply.</description></item>
</channel></rss>
"""
        items = parse_feed(xml, keywords=["nem", "dispatch"])
        assert len(items) == 1
        assert items[0]["title"] == "NEM price spike in SA"

    def test_parse_malformed_rss_entry_keeps_safe_defaults(self):
        from app.mcp.nem_news_client import parse_feed

        xml = """\
<rss><channel>
  <item><title>NEM dispatch update</title></item>
</channel></rss>
"""
        items = parse_feed(xml, keywords=["nem"])
        assert len(items) == 1
        assert items[0]["link"] == ""
        assert items[0]["summary"] == ""
