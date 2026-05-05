from __future__ import annotations

import hashlib
from typing import Any

from ..contracts import DecisionResult
from .events import NarrationEvent
from .view_model import CompanionViewModel

_TEXT_OPTIONS = {
    "danger_action": [
        "这里像是有关键操作，我们先看清楚再点。",
        "我这边看到高优先级按钮了，先别急。",
        "这一手像是有重要机会，我提醒你看一眼。",
    ],
    "action_available": [
        "现在像是轮到你操作了。",
        "底部有可选按钮，我帮你盯到了。",
        "这边有能点的选项，先看一眼比较稳。",
    ],
    "waiting_state": [
        "现在更像是在等待或过渡阶段，我们先安静看着。",
        "这一段不用急，我先陪你盯着画面。",
    ],
    "uncertain_state": [
        "这一帧我还没看太清，再给我一张新的。",
        "这一刻画面信息不够完整，我先继续观察。",
    ],
    "scene_update": [
        "画面有变化了，不过暂时还没到需要提醒的程度。",
        "局面在推进，我先安静陪你看着。",
    ],
    "tile_efficiency_hint": [
        "我看到一个牌效建议，先看这张牌是不是该处理。",
        "这里可以给出一个轻量出牌建议，先别急着点。",
        "这一手有可参考的牌理结果，可以看一下优先处理哪张。",
    ],
}


def _pick_text(key: str, summary: str) -> str:
    options = _TEXT_OPTIONS.get(key) or _TEXT_OPTIONS["scene_update"]
    digest = hashlib.sha1(f"{key}|{summary}".encode("utf-8")).digest()
    index = int.from_bytes(digest[:4], "big") % len(options)
    return options[index]


def _build_event_text(decision: DecisionResult) -> str:
    event_type = decision.decision_type or "scene_update"
    if event_type in {"action_available", "tile_efficiency_hint"}:
        return decision.suggestion or decision.detail or decision.summary or _pick_text(event_type, decision.summary)
    return _pick_text(event_type, decision.summary)


def generate_narration(decision: DecisionResult) -> tuple[NarrationEvent, CompanionViewModel, dict[str, Any]]:
    event_type = decision.decision_type or "scene_update"
    text = _build_event_text(decision)

    mood = "calm"
    suggestion_level = "silent"
    channel = "silent_ui"
    if event_type == "danger_action":
        mood = "alert"
        suggestion_level = "warning"
        channel = "warning"
    elif event_type == "action_available":
        mood = "focused"
        suggestion_level = "nudge"
        channel = "nudge"
    elif event_type == "tile_efficiency_hint":
        mood = "focused"
        suggestion_level = "nudge"
        channel = "nudge"
    elif event_type == "uncertain_state":
        mood = "curious"
        suggestion_level = "silent"
        channel = "silent_ui"

    dedupe_key = "%s|%s|%s|%s" % (
        event_type,
        decision.scene,
        decision.risk_level,
        ",".join(sorted(str(button) for button in decision.buttons)),
    )

    event = NarrationEvent(
        event_type=event_type,
        channel=channel,
        delivery="silent_ui",
        priority=decision.priority,
        summary=decision.summary,
        detail=decision.detail,
        risk_level=decision.risk_level,
        scene=decision.scene,
        buttons=list(decision.buttons),
        text=text,
        speakable=decision.speakable,
        dedupe_key=dedupe_key,
    )
    view_model = CompanionViewModel(
        headline=decision.summary or "有新的局面变化",
        subline=decision.suggestion or (("检测到 %s" % ", ".join(decision.buttons)) if decision.buttons else decision.detail),
        mood=mood,
        suggestion_level=suggestion_level,
        speakable=decision.speakable,
        delivery="silent_ui",
        text=text,
    )
    debug_payload = {
        "decision_result": decision.to_dict(),
        "selected_template": text,
        "event_type": event_type,
        "mood": mood,
        "suggestion_level": suggestion_level,
        "recommended_focus": decision.recommended_focus,
        "review_tags": list(decision.review_tags),
    }
    return event, view_model, debug_payload
