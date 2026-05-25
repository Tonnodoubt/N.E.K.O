from __future__ import annotations

import queue
import threading
from typing import Any


class CoachOverlayController:
    def __init__(self) -> None:
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self.last_error = ""

    def start(self) -> bool:
        if self._thread is not None and self._thread.is_alive():
            return True
        self.last_error = ""
        self._thread = threading.Thread(target=self._run, name="MahjongCoachOverlay", daemon=True)
        self._thread.start()
        return True

    def update(self, text: str) -> None:
        if self._thread is None or not self._thread.is_alive():
            return
        self._queue.put(str(text or "").strip() or "Mahjong Coach")

    def stop(self) -> None:
        if self._thread is None:
            return
        self._queue.put(None)
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        try:
            import tkinter as tk
        except Exception as exc:
            self.last_error = f"tkinter unavailable: {exc}"
            return

        try:
            root = tk.Tk()
            root.title("Mahjong Coach Overlay")
            root.overrideredirect(True)
            root.attributes("-topmost", True)
            root.attributes("-alpha", 0.88)
            frame = tk.Frame(root, bg="#121719", padx=16, pady=12)
            frame.pack(fill="both", expand=True)
            label = tk.Label(
                frame,
                text="Mahjong Coach",
                bg="#121719",
                fg="#f1f4ef",
                font=("Microsoft YaHei UI", 18, "bold"),
                justify="left",
                wraplength=420,
            )
            label.pack(fill="both", expand=True)
            root.update_idletasks()
            screen_width = root.winfo_screenwidth()
            root.geometry(f"460x168+{max(0, screen_width - 490)}+28")

            def pump() -> None:
                while True:
                    try:
                        item = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if item is None:
                        root.destroy()
                        return
                    label.config(text=item)
                root.after(120, pump)

            root.after(120, pump)
            root.mainloop()
        except Exception as exc:
            self.last_error = str(exc)


def overlay_text_from_payload(payload: dict[str, Any]) -> str:
    decision = payload.get("last_decision") if isinstance(payload.get("last_decision"), dict) else payload
    state = payload.get("round_state") if isinstance(payload.get("round_state"), dict) else payload.get("coach_state")
    if not isinstance(state, dict):
        state = {}
    decision_type = str(decision.get("decision_type") or "")
    if decision.get("action_required"):
        return _action_overlay_text(decision_type, decision)
    if decision_type == "round_idle":
        return _format_overlay("等待下一局", "上一局已结束", "新手牌出现后自动重开")
    if not (state.get("current_plan") or state.get("opening_plan")) and decision.get("decision_type") == "observe":
        return _format_overlay("等待手牌", str(decision.get("detail") or "正在找稳定手牌"), "保持牌桌无遮挡")
    title = _title_for_decision(decision_type)
    main = _main_line(state, decision)
    rule = _rule_line(state, decision_type)
    return _format_overlay(title, main, rule, _call_rule_line(state))


def _action_overlay_text(decision_type: str, decision: dict[str, Any]) -> str:
    if decision_type == "win_window":
        return _format_overlay("和牌窗口", "先确认荣和 / 自摸", "不要被其他建议盖掉")
    if decision_type == "riichi_window":
        return _format_overlay("立直选择", "先确认听牌和打点", "不确定就先保持手牌")
    if decision_type == "call_window":
        return _format_overlay("吃碰杠", "默认跳过", "役牌碰/进听/加速主线才开，其余跳过")
    if decision_type == "defense_alert":
        return _format_overlay("防守", str(decision.get("suggestion") or "有人立直，停止贪速度"), "先看现物和安全牌")
    return _format_overlay("操作窗口", str(decision.get("suggestion") or decision.get("detail") or "先处理当前按钮"), "")


def _title_for_decision(decision_type: str) -> str:
    if decision_type == "opening_plan":
        return "开局主线"
    if decision_type == "coach_checkpoint":
        return "三巡复盘"
    if decision_type == "observe":
        return "主线继续"
    return "牌局教练"


def _main_line(state: dict[str, Any], decision: dict[str, Any]) -> str:
    targets = _string_items(state.get("target_shapes"))
    for item in targets:
        if item.startswith("主线"):
            return _clean_line(item)
    plan = str(state.get("current_plan") or state.get("opening_plan") or decision.get("suggestion") or "继续观察牌局")
    return _clean_line(plan)


def _rule_line(state: dict[str, Any], decision_type: str) -> str:
    cautions = _string_items(state.get("caution_points"))
    for item in cautions:
        if item.startswith("路线选择"):
            return _clean_line(item)
    for item in cautions:
        if item.startswith("候选打牌"):
            return _clean_line(item)
    for item in cautions:
        if item.startswith("优先清理"):
            return _clean_line(item)
    for item in cautions:
        if item.startswith("吃碰杠"):
            return _clean_line(item)
    if decision_type == "coach_checkpoint":
        return "按新主线打三巡再看"
    return "三巡后再复盘"


def _call_rule_line(state: dict[str, Any]) -> str:
    cautions = _string_items(state.get("caution_points"))
    for item in cautions:
        if item.startswith("吃碰杠"):
            return "吃碰杠：役牌碰/进听/加速主线才开，其余跳过"
    return "吃碰杠：役牌碰/进听/加速主线才开，其余跳过"


def _string_items(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _clean_line(value: str) -> str:
    text = " ".join(str(value or "").split()).strip()
    keep_separator = text.startswith("路线选择：")
    for prefix in ("主线：", "保留：", "对子：", "路线选择："):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    if not keep_separator:
        for separator in ("；", ";"):
            if separator in text:
                text = text.split(separator, 1)[0]
    return _shorten(text, 34)


def _format_overlay(title: str, main: str, rule: str, extra: str = "") -> str:
    lines = [_shorten(title, 14), _shorten(main, 34), _shorten(rule, 34), _shorten(extra, 34)]
    return "\n".join(line for line in lines if line).strip() or "Mahjong Coach"


def _shorten(value: str, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)].rstrip(" ，,;；") + "…"
