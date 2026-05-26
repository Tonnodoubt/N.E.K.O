from __future__ import annotations

import queue
import threading
from typing import Any


OVERLAY_BLOCK_SEPARATOR = "\n---MAHJONG_COACH_OVERLAY_BLOCK---\n"


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
            transparent = "#010203"
            root.configure(bg=transparent)
            try:
                root.attributes("-transparentcolor", transparent)
            except Exception:
                pass
            container = tk.Frame(root, bg=transparent)
            container.pack(fill="both", expand=True, padx=0, pady=0)

            def build_panel() -> tuple[tk.Frame, tk.Label]:
                panel = tk.Frame(container, bg="#121719", padx=12, pady=10, width=300, height=128)
                panel.pack_propagate(False)
                panel.pack(side="left", fill="both", expand=True, padx=5)
                panel_label = tk.Label(
                    panel,
                    text="Mahjong Coach",
                    bg="#121719",
                    fg="#f1f4ef",
                    font=("Microsoft YaHei UI", 15, "bold"),
                    justify="left",
                    anchor="nw",
                    wraplength=276,
                )
                panel_label.pack(fill="both", expand=True)
                return panel, panel_label

            local_panel, local_label = build_panel()
            ai_panel, ai_label = build_panel()
            root.update_idletasks()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            drag_state = {"offset_x": 0, "offset_y": 0, "x": 0, "y": 0, "manual": False}

            def place_window(width: int, height: int) -> None:
                if drag_state["manual"]:
                    x, y = drag_state["x"], drag_state["y"]
                else:
                    x, y = _overlay_geometry(screen_width, screen_height, width, height)
                drag_state["x"] = x
                drag_state["y"] = y
                root.geometry(f"{width}x{height}+{x}+{y}")

            def start_drag(event: tk.Event) -> None:
                drag_state["offset_x"] = int(event.x_root) - root.winfo_x()
                drag_state["offset_y"] = int(event.y_root) - root.winfo_y()

            def drag_window(event: tk.Event) -> None:
                x = int(event.x_root) - drag_state["offset_x"]
                y = int(event.y_root) - drag_state["offset_y"]
                drag_state["x"] = x
                drag_state["y"] = y
                drag_state["manual"] = True
                root.geometry(f"+{x}+{y}")

            for widget in (container, local_panel, local_label, ai_panel, ai_label):
                widget.bind("<ButtonPress-1>", start_drag)
                widget.bind("<B1-Motion>", drag_window)

            def apply_text(text: str) -> None:
                blocks = _split_overlay_blocks(text)
                if len(blocks) >= 2:
                    if not ai_panel.winfo_ismapped():
                        ai_panel.pack(side="left", fill="both", expand=True, padx=5)
                    local_panel.config(width=300, height=128)
                    ai_panel.config(width=300, height=128)
                    local_label.config(text=blocks[0], font=("Microsoft YaHei UI", 15, "bold"), wraplength=276)
                    ai_label.config(text=blocks[1], font=("Microsoft YaHei UI", 15, "bold"), wraplength=276)
                    width, height = 630, 138
                else:
                    ai_panel.pack_forget()
                    local_panel.config(width=430, height=150)
                    local_label.config(text=blocks[0] if blocks else "Mahjong Coach", font=("Microsoft YaHei UI", 18, "bold"), wraplength=390)
                    width, height = 440, 150
                place_window(width, height)

            apply_text("Mahjong Coach")

            def pump() -> None:
                while True:
                    try:
                        item = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if item is None:
                        root.destroy()
                        return
                    apply_text(item)
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
    has_plan = state.get("local_plan") or state.get("ai_plan") or state.get("current_plan") or state.get("opening_plan")
    if not has_plan and decision.get("decision_type") == "observe":
        return _waiting_hand_overlay(decision)
    local_block = _strategy_overlay_block(
        "本地",
        str(state.get("local_plan") or state.get("current_plan") or state.get("opening_plan") or decision.get("suggestion") or ""),
        _string_items(state.get("local_targets")) or _string_items(state.get("target_shapes")),
        _string_items(state.get("local_cautions")) or _string_items(state.get("caution_points")),
    )
    ai_plan = str(state.get("ai_plan") or "").strip()
    llm_status = str(state.get("llm_status") or "").strip().lower()
    llm_error = str(state.get("llm_error") or "").strip()
    if ai_plan:
        ai_block = _strategy_overlay_block(
            "AI参考" if llm_status == "ready_previous_hand" else "AI",
            ai_plan,
            _string_items(state.get("ai_targets")),
            _string_items(state.get("ai_cautions")),
        )
    elif int(payload.get("llm_pending") or 0) > 0 or llm_status == "pending":
        ai_block = _format_overlay("AI", "思考中")
    elif llm_status in {"timeout", "error", "empty"}:
        ai_block = _format_overlay("AI", "未返回", _brief_error(llm_error or llm_status))
    else:
        ai_block = _format_overlay("AI", "等待阶段更新")
    return _join_overlay_blocks(local_block, ai_block)


def _action_overlay_text(decision_type: str, decision: dict[str, Any]) -> str:
    if decision_type == "win_window":
        return _format_overlay("本地和牌", "先确认荣和 / 自摸", "这个优先级最高")
    if decision_type == "riichi_window":
        suggestion = str(decision.get("suggestion") or "").strip()
        return _format_overlay("本地立直", _riichi_action_line(suggestion), _riichi_reason_line(suggestion))
    if decision_type == "call_window":
        return _format_overlay("本地鸣牌", "默认跳过", "能听牌/明显加速再开")
    if decision_type == "defense_alert":
        return _format_overlay("本地防守", str(decision.get("suggestion") or "有人立直，先防守"), "先看现物和安全牌")
    return _format_overlay("操作窗口", str(decision.get("suggestion") or decision.get("detail") or "先处理当前按钮"), "")


def _strategy_overlay_block(label: str, plan: str, targets: list[str], cautions: list[str]) -> str:
    direction = _direction_text(plan, targets)
    keep = _keep_text(targets, plan)
    discard = _discard_text(cautions, plan)
    return _format_overlay(label, f"方向：{direction}", f"留：{keep}" if keep else "", f"打：{discard}" if discard else "")


def _direction_text(plan: str, targets: list[str]) -> str:
    for item in targets:
        if item.startswith("主线："):
            return _clean_line(item)
    return _clean_line(plan) if plan else "继续观察"


def _keep_text(targets: list[str], plan: str) -> str:
    keep = _first_prefixed_value(targets, "保留：")
    if not keep:
        keep = _extract_after(plan, ("保留",), stop_markers=("，先", "；", "。"))
    return _brief_items(_plain_text(keep), max_items=3)


def _discard_text(cautions: list[str], plan: str) -> str:
    discard = _first_prefixed_value(cautions, "优先清理：")
    if not discard:
        discard = _extract_after(plan, ("先打", "先清", "打："), stop_markers=("，", "；", "。"))
    return _brief_items(_plain_text(discard), max_items=3)


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
        for separator in ("，", "；", ";", "。"):
            if separator in text:
                text = text.split(separator, 1)[0]
    return _plain_text(text)


def _format_overlay(*parts: str) -> str:
    lines = [_plain_text(" ".join(str(value or "").split())) for value in parts]
    return "\n".join(line for line in lines if line).strip() or "Mahjong Coach"


def _join_overlay_blocks(*blocks: str) -> str:
    values = [block.strip() for block in blocks if block.strip()]
    return OVERLAY_BLOCK_SEPARATOR.join(values)


def _split_overlay_blocks(text: str) -> list[str]:
    return [block.strip() for block in str(text or "").split(OVERLAY_BLOCK_SEPARATOR) if block.strip()]


def _overlay_geometry(screen_width: int, screen_height: int, width: int, height: int) -> tuple[int, int]:
    x = max(0, int((screen_width - width) / 2))
    bottom_gap = max(160, int(screen_height * 0.19))
    y = max(28, screen_height - height - bottom_gap)
    return x, y


def _is_llm_decision(decision: dict[str, Any], state: dict[str, Any]) -> bool:
    meta = decision.get("engine_meta") if isinstance(decision.get("engine_meta"), dict) else {}
    source = decision.get("analysis_source") or meta.get("analysis_source") or state.get("plan_source")
    return str(source or "").lower() == "llm"


def _plain_text(value: str) -> str:
    text = str(value or "").strip()
    replacements = {
        "筒子占比很高": "筒子多",
        "万子占比很高": "万子多",
        "索子占比很高": "索子多",
        "保留同色块": "保留同色",
        "同色块": "同色",
        "做搭子": "找顺子",
        "先清": "先打",
        "不硬染": "别强做清一色",
        "吃碰杠": "鸣牌",
        "进听": "听牌",
        "加速主线": "明显加速",
        "役牌碰/听牌/明显加速才开，其余跳过": "默认跳过，能听牌再开",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _brief_items(value: str, *, max_items: int) -> str:
    text = str(value or "").strip(" ，、")
    if not text:
        return ""
    parts = [item.strip() for item in text.replace("，", "、").replace(",", "、").split("、") if item.strip()]
    if len(parts) <= max_items:
        return "、".join(parts) if parts else text
    return f"{'、'.join(parts[:max_items])}等{len(parts)}张"


def _brief_error(value: str) -> str:
    text = _plain_text(str(value or "").strip())
    if len(text) <= 28:
        return text
    return f"{text[:27]}..."


def _riichi_action_line(value: str) -> str:
    text = _plain_text(str(value or "").strip())
    if not text:
        return "先确认听牌和打点"
    for prefix in ("推荐立直", "可以立直", "谨慎立直", "可立直"):
        if text.startswith(prefix):
            discard = _extract_after(text, ("打",), stop_markers=("听", "，", "；", "。"))
            waits = _extract_after(text, ("听",), stop_markers=("，", "；", "。"))
            if discard and waits:
                return f"{prefix}：打{discard}听{_brief_items(waits, max_items=3)}"
            return prefix
    return _clean_line(text)


def _riichi_reason_line(value: str) -> str:
    text = _plain_text(str(value or "").strip())
    for marker in ("好形/枚数够", "待牌尚可", "愚形或枚数少"):
        if marker in text:
            return marker
    if "未拿到稳定手牌" in text:
        return "手牌识别不稳，谨慎确认"
    return "不确定就先别立"


def _first_prefixed_value(values: list[str], prefix: str) -> str:
    for item in values:
        if item.startswith(prefix):
            return item[len(prefix) :].strip()
    return ""


def _extract_after(text: str, keywords: tuple[str, ...], *, stop_markers: tuple[str, ...]) -> str:
    value = str(text or "")
    start = -1
    keyword_len = 0
    for keyword in keywords:
        index = value.find(keyword)
        if index >= 0 and (start < 0 or index < start):
            start = index
            keyword_len = len(keyword)
    if start < 0:
        return ""
    tail = value[start + keyword_len :].strip(" ：:")
    stop_positions = [tail.find(marker) for marker in stop_markers if tail.find(marker) >= 0]
    if stop_positions:
        tail = tail[: min(stop_positions)]
    return tail.strip()


def _waiting_hand_overlay(decision: dict[str, Any]) -> str:
    reason_codes = [str(item) for item in (decision.get("reason_codes") or []) if str(item).strip()]
    hand_reason = ""
    for code in reason_codes:
        if code.startswith("hand_"):
            hand_reason = code[len("hand_"):]
            break
    perception = decision.get("perception") if isinstance(decision.get("perception"), dict) else {}
    hand_perception = perception.get("hand") if isinstance(perception.get("hand"), dict) else {}
    if not hand_reason:
        hand_reason = str(hand_perception.get("reason") or "")
    accepted = sum(1 for item in (hand_perception.get("raw_detections") or []) if item.get("accepted"))
    occupied = sum(1 for item in (hand_perception.get("raw_detections") or []) if item.get("occupied"))

    if hand_reason == "missing_hand_tile_templates":
        return _format_overlay("等待手牌", "截图分辨率无匹配校准", "支持 1920x1080 / 2560x1440")
    if hand_reason in {"image_path_missing", "image_missing"}:
        return _format_overlay("等待手牌", "截图获取失败")
    if accepted > 0:
        return _format_overlay("等待手牌", f"已识别{accepted}张，需要13-14张", "保持牌桌无遮挡")
    if occupied > 0:
        return _format_overlay("等待手牌", f"检测到{occupied}个牌位，识别中", "保持牌桌无遮挡")
    return _format_overlay("等待手牌", "未检测到手牌区域", "确保雀魂窗口可见")
