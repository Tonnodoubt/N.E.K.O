from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from plugin.plugins.mahjong_coach import llm_coach
from plugin.plugins.mahjong_coach.llm_coach import (
    build_round_plan_llm,
    format_hand_for_llm,
    merge_heuristic_and_llm,
    parse_llm_plan_response,
)


def test_format_hand_for_llm_groups_by_suit() -> None:
    text = format_hand_for_llm(["1m", "2m", "3p", "4s", "1z", "5z"])

    assert "万子: 1 2" in text
    assert "筒子: 3" in text
    assert "索子: 4" in text
    assert "字牌: 东 白" in text


def test_format_hand_for_llm_marks_red_fives() -> None:
    text = format_hand_for_llm(["0m", "0p", "0s"])

    assert text.count("5(red)") == 3


def test_parse_llm_plan_response_valid_json() -> None:
    plan = parse_llm_plan_response(
        '{"summary":"主线：断幺速度","detail":"保留中张","bias":"attack","targets":["保留：345"],"cautions":["默认跳过副露"],"discard_priority":["孤字"]}'
    )

    assert plan is not None
    assert plan["summary"] == "主线：断幺速度"
    assert plan["bias"] == "attack"
    assert plan["targets"] == ["保留：345"]


def test_parse_llm_plan_response_malformed_json_returns_none() -> None:
    assert parse_llm_plan_response("not json") is None


def test_parse_llm_plan_response_missing_fields_fills_defaults() -> None:
    plan = parse_llm_plan_response('{"summary":"主线：牌效"}')

    assert plan == {
        "summary": "主线：牌效",
        "detail": "",
        "bias": "neutral",
        "targets": [],
        "cautions": [],
        "discard_priority": [],
    }


def test_merge_heuristic_and_llm_prefers_llm() -> None:
    merged = merge_heuristic_and_llm(
        {"summary": "启发式", "detail": "old", "bias": "neutral", "targets": ["旧"], "cautions": ["旧"]},
        {"summary": "LLM", "detail": "new", "bias": "attack", "targets": ["新"], "cautions": ["新"]},
    )

    assert merged["summary"] == "LLM"
    assert merged["bias"] == "attack"
    assert merged["targets"] == ["新"]


def test_merge_heuristic_and_llm_falls_back_on_none() -> None:
    heuristic = {"summary": "启发式", "detail": "old", "bias": "neutral", "targets": ["旧"], "cautions": ["旧"]}

    assert merge_heuristic_and_llm(heuristic, None)["summary"] == "启发式"


@pytest.mark.asyncio
async def test_build_round_plan_llm_calls_with_correct_summary_config(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}

    class FakeConfig:
        def get_model_api_config(self, tier: str) -> dict[str, str]:
            calls["tier"] = tier
            return {"model": "summary-model", "base_url": "https://example.test", "api_key": "key"}

    class FakeLLM:
        async def ainvoke(self, messages):
            calls["messages"] = messages
            return SimpleNamespace(content='{"summary":"AI主线","detail":"细节","bias":"attack","targets":["目标"],"cautions":["风险"],"discard_priority":["孤字"]}')

        async def aclose(self):
            calls["closed"] = True

    def fake_create_chat_llm(**kwargs):
        calls["llm_kwargs"] = kwargs
        return FakeLLM()

    import utils.config_manager as config_manager_module

    monkeypatch.setattr(config_manager_module, "get_config_manager", lambda: FakeConfig())
    monkeypatch.setattr(llm_coach, "create_chat_llm", fake_create_chat_llm)

    plan = await build_round_plan_llm(["1m", "2m", "3m"], timeout=1.0)

    assert plan is not None
    assert calls["tier"] == "summary"
    assert calls["llm_kwargs"]["model"] == "summary-model"
    assert "temperature" not in calls["llm_kwargs"]
    assert calls["closed"] is True
    assert plan["summary"] == "AI主线"


@pytest.mark.asyncio
async def test_build_round_plan_llm_timeout_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeConfig:
        def get_model_api_config(self, _tier: str) -> dict[str, str]:
            return {"model": "summary-model", "base_url": "https://example.test", "api_key": "key"}

    class FakeLLM:
        async def ainvoke(self, _messages):
            await asyncio.sleep(0.2)
            return SimpleNamespace(content='{"summary":"late"}')

        async def aclose(self):
            pass

    import utils.config_manager as config_manager_module

    monkeypatch.setattr(config_manager_module, "get_config_manager", lambda: FakeConfig())
    monkeypatch.setattr(llm_coach, "create_chat_llm", lambda **_kwargs: FakeLLM())

    assert await build_round_plan_llm(["1m", "2m", "3m"], timeout=0.01) is None
