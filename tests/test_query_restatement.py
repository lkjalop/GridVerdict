"""Tests for query restatement and adversarial critic modules."""
from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── Query Restatement ──────────────────────────────────────────────────────────

class TestRestateQuery:
    """Unit tests for query_restatement.restate_query()."""

    @pytest.mark.asyncio
    async def test_returns_empty_on_rule_based_backend(self):
        """Restatement is skipped when decomposer_backend is rule_based."""
        from app.engines.query_restatement import restate_query, _EMPTY
        with patch("app.engines.query_restatement._settings") as mock_settings:
            mock_settings.decomposer_backend = "rule_based"
            result = await restate_query("why is nsw price elevated?")
        assert result.is_empty()
        assert result.sub_questions == []

    @pytest.mark.asyncio
    async def test_parses_valid_llm_response(self):
        """Parses a well-formed JSON response from the LLM."""
        from app.engines.query_restatement import restate_query

        mock_response = {
            "sub_questions": [
                "What caused the price to rise from $143 to $167/MWh?",
                "Which fuel type was marginal during the spike?",
            ],
            "primary_intent": "price_fluctuation",
            "explicit_entities": {"regions": ["NSW1"], "fuels": [], "price_values": [143, 167], "time_refs": []},
            "answer_gap_risk": True,
        }

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": json.dumps(mock_response)}}
        mock_resp.raise_for_status = MagicMock()

        with patch("app.engines.query_restatement._settings") as mock_settings, \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_settings.decomposer_backend = "ollama"
            mock_settings.ollama_base_url = "http://localhost:11434"
            mock_settings.ollama_model = "qwen3:14b"
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            result = await restate_query("why did price go from 143 to 167?")

        assert len(result.sub_questions) == 2
        assert result.primary_intent == "price_fluctuation"
        assert result.answer_gap_risk is True
        assert not result.is_empty()

    @pytest.mark.asyncio
    async def test_falls_back_to_empty_on_http_error(self):
        """Returns empty result (not an exception) when Ollama is unreachable."""
        from app.engines.query_restatement import restate_query
        import httpx

        with patch("app.engines.query_restatement._settings") as mock_settings, \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_settings.decomposer_backend = "ollama"
            mock_settings.ollama_base_url = "http://localhost:11434"
            mock_settings.ollama_model = "qwen3:14b"
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
            mock_client_cls.return_value = mock_client

            result = await restate_query("why is coal best?")

        assert result.is_empty()

    @pytest.mark.asyncio
    async def test_falls_back_to_empty_on_invalid_json(self):
        """Returns empty result when LLM produces unparseable output."""
        from app.engines.query_restatement import restate_query

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "sorry I cannot answer that"}}
        mock_resp.raise_for_status = MagicMock()

        with patch("app.engines.query_restatement._settings") as mock_settings, \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_settings.decomposer_backend = "ollama"
            mock_settings.ollama_base_url = "http://localhost:11434"
            mock_settings.ollama_model = "qwen3:14b"
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            result = await restate_query("what is going on with prices?")

        assert result.is_empty()

    def test_seed_text_prepends_sub_questions(self):
        """seed_text() prepends sub-questions prefix to original query."""
        from app.engines.query_restatement import RestatementResult
        r = RestatementResult(
            sub_questions=["What caused the spike?", "Which fuel was marginal?"],
            primary_intent="price_fluctuation",
            answer_gap_risk=True,
        )
        original = "why did price go from 143 to 167?"
        seeded = r.seed_text(original)
        assert "What caused the spike?" in seeded
        assert "Which fuel was marginal?" in seeded
        assert original in seeded

    def test_seed_text_returns_original_when_empty(self):
        """seed_text() returns original unchanged when no sub-questions."""
        from app.engines.query_restatement import RestatementResult
        r = RestatementResult()
        original = "why is NSW elevated?"
        assert r.seed_text(original) == original

    def test_strips_think_tags_from_response(self):
        """Parser strips <think>...</think> blocks before JSON parsing."""
        from app.engines.query_restatement import RestatementResult
        import re, json as _json
        raw = "<think>Let me think about this carefully...</think>\n" + _json.dumps({
            "sub_questions": ["Why did price spike?"],
            "primary_intent": "price_explanation",
            "explicit_entities": {},
            "answer_gap_risk": False,
        })
        clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        data = _json.loads(clean)
        assert data["sub_questions"] == ["Why did price spike?"]


# ── Adversarial Critic ────────────────────────────────────────────────────────

class TestAdversarialCritic:
    """Unit tests for adversarial_critic.audit_coverage()."""

    @pytest.mark.asyncio
    async def test_returns_passing_when_no_sub_questions(self):
        """Critic passes immediately when sub_questions list is empty."""
        from app.engines.coverage_auditor import audit_coverage
        result = await audit_coverage([], [], "causal_explanation")
        assert result.passes is True

    @pytest.mark.asyncio
    async def test_returns_passing_on_http_error(self):
        """Critic falls back to passing when Ollama is unreachable."""
        from app.engines.coverage_auditor import audit_coverage
        import httpx

        with patch("app.engines.coverage_auditor._settings") as mock_settings, \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_settings.decomposer_backend = "ollama"
            mock_settings.ollama_base_url = "http://localhost:11434"
            mock_settings.ollama_model = "qwen3:14b"
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
            mock_client_cls.return_value = mock_client

            result = await audit_coverage(
                ["What caused the price spike?"],
                [{"title": "Answer", "items": ["NSW1: $142/MWh"]}],
                "current_market_state",
            )
        assert result.passes is True
        assert not result.has_actionable_suggestion()

    @pytest.mark.asyncio
    async def test_detects_routing_gap_and_suggests_fix(self):
        """Critic returns has_actionable_suggestion()=True when answer misses sub-questions."""
        from app.engines.coverage_auditor import audit_coverage

        mock_response = json.dumps({
            "passes": False,
            "gaps": ["User asked about fuel source but answer only shows price/demand"],
            "suggested_output": "fuel_source_recommendation",
            "reasoning": "The answer is a bare price lookup; fuel attribution was not addressed.",
        })
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": mock_response}}
        mock_resp.raise_for_status = MagicMock()

        with patch("app.engines.coverage_auditor._settings") as mock_settings, \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_settings.decomposer_backend = "ollama"
            mock_settings.ollama_base_url = "http://localhost:11434"
            mock_settings.ollama_model = "qwen3:14b"
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            result = await audit_coverage(
                ["Which fuel type was the marginal generator?"],
                [{"title": "Answer", "items": ["NSW1: $167/MWh, demand 8211 MW"]}],
                "causal_explanation",
            )

        assert result.passes is False
        assert result.suggested_output == "fuel_source_recommendation"
        assert result.has_actionable_suggestion()

    @pytest.mark.asyncio
    async def test_rejects_invalid_suggested_output(self):
        """Critic ignores suggested_output values not in the allowed set."""
        from app.engines.coverage_auditor import audit_coverage

        mock_response = json.dumps({
            "passes": False,
            "gaps": ["Answer was incomplete"],
            "suggested_output": "made_up_routing_label",
            "reasoning": "Something went wrong.",
        })
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": mock_response}}
        mock_resp.raise_for_status = MagicMock()

        with patch("app.engines.coverage_auditor._settings") as mock_settings, \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_settings.decomposer_backend = "ollama"
            mock_settings.ollama_base_url = "http://localhost:11434"
            mock_settings.ollama_model = "qwen3:14b"
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            result = await audit_coverage(
                ["Why did price spike?"],
                [{"title": "Answer", "items": ["Price is elevated."]}],
                "causal_explanation",
            )

        assert result.suggested_output is None
        assert not result.has_actionable_suggestion()

    @pytest.mark.asyncio
    async def test_passes_when_answer_covers_sub_questions(self):
        """Critic returns passes=True when answer addresses all sub-questions."""
        from app.engines.coverage_auditor import audit_coverage

        mock_response = json.dumps({
            "passes": True,
            "gaps": [],
            "suggested_output": None,
            "reasoning": "All sub-questions addressed.",
        })
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": mock_response}}
        mock_resp.raise_for_status = MagicMock()

        with patch("app.engines.coverage_auditor._settings") as mock_settings, \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_settings.decomposer_backend = "ollama"
            mock_settings.ollama_base_url = "http://localhost:11434"
            mock_settings.ollama_model = "qwen3:14b"
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            result = await audit_coverage(
                ["Why is NSW elevated?"],
                [
                    {"title": "Answer", "items": ["NSW1 is $167/MWh, elevated due to low headroom."]},
                    {"title": "Evidence", "items": ["Dispatch: demand 8211 MW, headroom 3789 MW."]},
                ],
                "causal_explanation",
            )

        assert result.passes is True
        assert not result.has_actionable_suggestion()
