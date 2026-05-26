from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from config.prompts.prompts_mahjong import MAHJONG_COACH_CHECKPOINT_PROMPT, MAHJONG_COACH_OPENING_PROMPT
from utils.file_utils import robust_json_loads
from utils.llm_client import create_chat_llm

from .tile_labels import normalize_tile, tile_rank, tile_suit


SUIT_LABELS = {"m": "万子", "p": "筒子", "s": "索子", "z": "字牌"}
HONOR_LABELS = {"1z": "东", "2z": "南", "3z": "西", "4z": "北", "5z": "白", "6z": "发", "7z": "中"}
PLAN_FIELDS = ("summary", "detail", "bias", "targets", "cautions", "discard_priority")


async def build_round_plan_llm(
    hand_tiles: list[str],
    *,
    previous_plan: str = "",
    turn_number: int | None = None,
    heuristic_plan: dict[str, Any] | None = None,
    timeout: float = 8.0,
    diagnostics: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    """Return a summary-tier strategy plan, or None when the LLM cannot help."""
    _set_diagnostics(diagnostics, "pending")
    try:
        from utils.config_manager import get_config_manager

        api_config = get_config_manager().get_model_api_config("summary")
        model = str(api_config.get("model") or "").strip()
        base_url = str(api_config.get("base_url") or "").strip()
        api_key = str(api_config.get("api_key") or "").strip()
        if not model or not base_url:
            _set_diagnostics(diagnostics, "error", "summary 模型或 base_url 未配置")
            return None

        llm = create_chat_llm(
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout=float(timeout) + 0.5,
            max_retries=0,
        )
        try:
            response = await asyncio.wait_for(
                llm.ainvoke(_build_messages(hand_tiles, previous_plan, turn_number, heuristic_plan)),
                timeout=max(0.1, float(timeout)),
            )
        finally:
            await llm.aclose()
        plan = parse_llm_plan_response(str(getattr(response, "content", "") or response))
        if plan is None:
            _set_diagnostics(diagnostics, "empty", "模型返回内容不是可解析的策略 JSON")
            return None
        _set_diagnostics(diagnostics, "ready")
        return plan
    except asyncio.TimeoutError:
        _set_diagnostics(diagnostics, "timeout", "模型请求超时")
        return None
    except Exception as exc:
        _set_diagnostics(diagnostics, "error", f"模型请求失败：{type(exc).__name__}")
        return None


def format_hand_for_llm(hand_tiles: list[str]) -> str:
    groups: dict[str, list[str]] = {"m": [], "p": [], "s": [], "z": []}
    for tile in hand_tiles:
        raw = str(tile or "").strip()
        normalized = normalize_tile(raw)
        suit = tile_suit(normalized)
        if suit not in groups:
            continue
        if suit == "z":
            groups[suit].append(HONOR_LABELS.get(normalized, normalized))
            continue
        rank = tile_rank(normalized)
        groups[suit].append("5(red)" if raw.startswith("0") else rank)

    parts = []
    for suit in ("m", "p", "s", "z"):
        values = groups[suit]
        parts.append(f"{SUIT_LABELS[suit]}: {' '.join(values) if values else '-'}")
    return " | ".join(parts)


def parse_llm_plan_response(content: str) -> dict[str, Any] | None:
    text = _strip_code_fence(content)
    try:
        parsed = robust_json_loads(text)
    except Exception:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return None
        try:
            parsed = robust_json_loads(match.group(0))
        except Exception:
            return None
    if not isinstance(parsed, dict):
        return None
    return _normalize_plan(parsed)


def merge_heuristic_and_llm(heuristic: dict[str, Any], llm_result: dict[str, Any] | None) -> dict[str, Any]:
    base = _normalize_plan(heuristic)
    if llm_result is None:
        return base
    llm_plan = _normalize_plan(llm_result)
    merged = dict(base)
    for field in PLAN_FIELDS:
        value = llm_plan.get(field)
        if isinstance(value, list):
            if value:
                merged[field] = value
        elif str(value or "").strip():
            merged[field] = value
    return merged


def _build_messages(
    hand_tiles: list[str],
    previous_plan: str,
    turn_number: int | None,
    heuristic_plan: dict[str, Any] | None,
) -> list[dict[str, str]]:
    system = MAHJONG_COACH_CHECKPOINT_PROMPT if previous_plan or turn_number else MAHJONG_COACH_OPENING_PROMPT
    user = {
        "hand": format_hand_for_llm(hand_tiles),
        "turn_number": turn_number or "",
        "previous_plan": previous_plan,
        "heuristic_plan": heuristic_plan or {},
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Analyze this context and return JSON only:\n{json.dumps(user, ensure_ascii=False)}"},
    ]


def _normalize_plan(raw: dict[str, Any]) -> dict[str, Any]:
    summary = _clean_text(raw.get("summary"))
    detail = _clean_text(raw.get("detail"))
    bias = _clean_text(raw.get("bias")).lower()
    if bias not in {"attack", "neutral", "defense"}:
        bias = "neutral"
    return {
        "summary": summary,
        "detail": detail,
        "bias": bias,
        "targets": _string_list(raw.get("targets")),
        "cautions": _string_list(raw.get("cautions")),
        "discard_priority": _string_list(raw.get("discard_priority")),
    }


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_clean_text(item) for item in value if _clean_text(item)]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _strip_code_fence(text: str) -> str:
    value = str(text or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?", "", value, flags=re.IGNORECASE).strip()
        value = re.sub(r"```$", "", value).strip()
    return value


def _set_diagnostics(diagnostics: dict[str, str] | None, status: str, error: str = "") -> None:
    if diagnostics is None:
        return
    diagnostics["status"] = status
    diagnostics["error"] = error
