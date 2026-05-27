from __future__ import annotations

import json
import os
import queue
import threading
from pathlib import Path
from typing import Any


# N.E.K.O design system (light mode)
_BG = "#ffffff"
_CARD = "#f5f5f5"
_BORDER = "#d0d0d0"
_BORDER_FOCUS = "#2a7bc4"
_ACCENT = "#44b7fe"
_BTN_PRIMARY = "#2a7bc4"
_BTN_HOVER = "#3590d9"
_SUCCESS = "#16a34a"
_WARNING = "#d97706"
_DANGER = "#dc2626"
_PURPLE = "#7c3aed"
_TEXT = "#1e1e1e"
_TEXT_MUTED = "#666666"
_FONT = "Segoe UI"
_FONT_SIZE = 18
_HEADER_H = 3

# Badge colors
_BADGE_LOCAL_BG = "#2a7bc4"
_BADGE_LOCAL_FG = "#ffffff"

# Action badge overrides
_ACTION_WIN = ("和牌", _SUCCESS, "#ffffff")
_ACTION_RIICHI = ("立直", _PURPLE, "#ffffff")
_ACTION_CALL = ("鸣牌", _WARNING, "#ffffff")
_ACTION_DEFENSE = ("防守", _DANGER, "#ffffff")
_ACTION_GENERIC = ("操作", _BTN_PRIMARY, "#ffffff")

# Resize bounds
_MIN_WIDTH = 320
_MIN_HEIGHT = 110
_MAX_WIDTH = 1200
_MAX_HEIGHT = 460

# Default size
_DEFAULT_WIDTH = 700
_DEFAULT_HEIGHT = 190

# Prefs file
_PREFS_FILENAME = "overlay_prefs.json"


def _prefs_path() -> Path:
    base = os.environ.get("LOCALAPPDATA", "")
    if base:
        return Path(base) / "N.E.K.O" / "plugins" / "mahjong_coach" / "data" / _PREFS_FILENAME
    return Path(_PREFS_FILENAME)


def _load_prefs() -> dict[str, int]:
    try:
        data = json.loads(_prefs_path().read_text(encoding="utf-8"))
        if isinstance(data, dict):
            w = int(data.get("width", _DEFAULT_WIDTH))
            h = int(data.get("height", _DEFAULT_HEIGHT))
            fs = int(data.get("font_size", _FONT_SIZE))
            return {
                "width": max(_MIN_WIDTH, min(_MAX_WIDTH, w)),
                "height": max(_MIN_HEIGHT, min(_MAX_HEIGHT, h)),
                "font_size": max(16, min(32, fs)),
            }
    except Exception:
        pass
    return {"width": _DEFAULT_WIDTH, "height": _DEFAULT_HEIGHT, "font_size": _FONT_SIZE}


def _save_prefs(width: int, height: int, font_size: int) -> None:
    try:
        path = _prefs_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"width": width, "height": height, "font_size": font_size}), encoding="utf-8")
    except Exception:
        pass


class CoachOverlayController:
    def __init__(
        self,
        on_start: Callable[[str], None] | None = None,
        on_stop: Callable[[], None] | None = None,
    ) -> None:
        self._queue: queue.Queue[Any] = queue.Queue()
        self._thread: threading.Thread | None = None
        self.last_error = ""
        self._on_start = on_start
        self._on_stop = on_stop

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

    def show_config(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            return
        self._queue.put({"cmd": "show_config"})

    def show_strategy(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            return
        self._queue.put({"cmd": "show_strategy"})

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
            root.configure(bg=_BG)

            # Border layer
            shell = tk.Frame(root, bg=_BORDER, padx=1, pady=1)
            shell.pack(fill="both", expand=True)

            inner = tk.Frame(shell, bg=_BG)
            inner.pack(fill="both", expand=True)

            # Header accent line
            header = tk.Canvas(inner, height=_HEADER_H, bg=_BG, highlightthickness=0, bd=0)
            header.pack(fill="x", side="top")

            def _draw_accent(_event: tk.Event | None = None) -> None:
                header.delete("all")
                w = header.winfo_width()
                if w < 2:
                    return
                r1, g1, b1 = 0x44, 0xB7, 0xFE
                r2, g2, b2 = 0x63, 0x66, 0xF1
                steps = min(w, 120)
                for i in range(steps):
                    ratio = i / max(steps - 1, 1)
                    r = int(r1 + (r2 - r1) * ratio)
                    g = int(g1 + (g2 - g1) * ratio)
                    b = int(b1 + (b2 - b1) * ratio)
                    x0 = int(i * w / steps)
                    x1 = int((i + 1) * w / steps)
                    header.create_rectangle(x0, 0, x1, _HEADER_H, fill=f"#{r:02x}{g:02x}{b:02x}", outline="")

            header.bind("<Configure>", _draw_accent)

            # --- close button (top-right) ---
            close_btn = tk.Label(
                inner, text="✕", bg=_BG, fg=_TEXT_MUTED,
                font=(_FONT, 10), cursor="hand2",
            )
            close_btn.place(relx=1.0, x=-8, y=4, anchor="ne")

            def _on_close_click(_event: tk.Event) -> None:
                try:
                    if self._on_stop is not None:
                        self._on_stop()
                except Exception as exc:
                    self.last_error = f"close click failed: {exc}"

            close_btn.bind("<Button-1>", _on_close_click)

            # Content + grip row
            body = tk.Frame(inner, bg=_BG)
            body.pack(fill="both", expand=True)
            close_btn.lift()

            content = tk.Frame(body, bg=_BG, padx=10, pady=6)
            content.pack(side="top", fill="both", expand=True)

            # Resize grip
            grip = tk.Label(
                body, text="⌟", bg=_BG, fg=_BORDER_FOCUS,
                font=(_FONT, 9), anchor="se", padx=4, pady=1,
            )
            grip.pack(side="bottom", fill="x")

            # --- layout state ---
            prefs = _load_prefs()
            layout = {"width": prefs["width"], "height": prefs["height"], "font_size": prefs["font_size"]}
            drag_state = {"offset_x": 0, "offset_y": 0, "x": 0, "y": 0, "manual": False}

            # ---------- Config mode UI ----------
            config_frame = tk.Frame(content, bg=_BG)

            btn_row = tk.Frame(config_frame, bg=_BG)
            btn_row.pack(pady=(16, 8))

            def _make_btn(parent, text, style):
                btn = tk.Label(
                    parent, text=text,
                    bg=_BTN_PRIMARY, fg="#ffffff",
                    font=(_FONT, 13, "bold"),
                    padx=24, pady=10, cursor="hand2",
                )
                btn.pack(side="left", padx=8)

                def _on_enter(_e):
                    btn.config(bg=_BTN_HOVER)

                def _on_leave(_e):
                    btn.config(bg=_BTN_PRIMARY)

                def _on_click(_e):
                    try:
                        if self._on_start is not None:
                            self._on_start(style)
                    except Exception as exc:
                        self.last_error = f"button click failed: {exc}"

                btn.bind("<Enter>", _on_enter)
                btn.bind("<Leave>", _on_leave)
                btn.bind("<Button-1>", _on_click)
                return btn

            _make_btn(btn_row, "立直（门清憋大牌）", "riichi")
            _make_btn(btn_row, "快攻（积极副露）", "fast")

            # ---------- Strategy mode UI ----------
            strategy_frame = tk.Frame(content, bg=_BG)

            def build_panel(parent, badge_text: str, badge_bg: str, badge_fg: str) -> tuple[tk.Frame, tk.Label, tk.Label]:
                panel = tk.Frame(parent, bg=_CARD, padx=10, pady=8)
                panel.pack_propagate(False)

                badge_frame = tk.Frame(panel, bg=_CARD)
                badge_frame.pack(fill="x", pady=(0, 4))

                badge = tk.Label(
                    badge_frame, text=badge_text,
                    bg=badge_bg, fg=badge_fg,
                    font=(_FONT, 9, "bold"),
                    anchor="w", padx=8, pady=1,
                )
                badge.pack(side="left")

                label = tk.Label(
                    panel, text="",
                    bg=_CARD, fg=_TEXT,
                    font=(_FONT, layout["font_size"]),
                    justify="left", anchor="nw",
                    wraplength=240,
                )
                label.pack(fill="both", expand=True)

                return panel, badge, label

            local_panel, local_badge, local_label = build_panel(strategy_frame, "本地", _BADGE_LOCAL_BG, _BADGE_LOCAL_FG)
            local_panel.pack(side="left", fill="both", expand=True)

            def _recompute_panels() -> None:
                w = layout["width"]
                h = layout["height"]
                padding = 30
                panel_h = max(60, h - 40)
                panel_w = max(80, w - padding)
                wrap = max(40, panel_w - 24)
                local_panel.config(width=panel_w, height=panel_h)
                local_label.config(wraplength=wrap)

            def _set_geometry() -> None:
                w, h = layout["width"], layout["height"]
                if drag_state["manual"]:
                    x, y = drag_state["x"], drag_state["y"]
                else:
                    x, y = _overlay_geometry(root.winfo_screenwidth(), root.winfo_screenheight(), w, h)
                    drag_state["x"] = x
                    drag_state["y"] = y
                root.geometry(f"{w}x{h}+{x}+{y}")

            # --- drag to move ---
            def start_drag(event: tk.Event) -> None:
                drag_state["offset_x"] = int(event.x_root) - root.winfo_x()
                drag_state["offset_y"] = int(event.y_root) - root.winfo_y()

            def drag_window(event: tk.Event) -> None:
                drag_state["x"] = int(event.x_root) - drag_state["offset_x"]
                drag_state["y"] = int(event.y_root) - drag_state["offset_y"]
                drag_state["manual"] = True
                root.geometry(f"+{drag_state['x']}+{drag_state['y']}")

            drag_widgets = [shell, inner, header, content, config_frame, strategy_frame, local_panel, local_badge, local_label]
            for widget in drag_widgets:
                widget.bind("<ButtonPress-1>", start_drag)
                widget.bind("<B1-Motion>", drag_window)

            # --- resize grip ---
            resize_state = {"sx": 0, "sy": 0, "sw": 0, "sh": 0}

            def start_resize(event: tk.Event) -> None:
                resize_state["sx"] = event.x_root
                resize_state["sy"] = event.y_root
                resize_state["sw"] = layout["width"]
                resize_state["sh"] = layout["height"]

            def do_resize(event: tk.Event) -> None:
                dw = event.x_root - resize_state["sx"]
                dh = event.y_root - resize_state["sy"]
                layout["width"] = max(_MIN_WIDTH, min(_MAX_WIDTH, resize_state["sw"] + dw))
                layout["height"] = max(_MIN_HEIGHT, min(_MAX_HEIGHT, resize_state["sh"] + dh))
                _recompute_panels()
                _set_geometry()

            def end_resize(_event: tk.Event) -> None:
                _save_prefs(layout["width"], layout["height"], layout["font_size"])

            grip.bind("<ButtonPress-1>", start_resize)
            grip.bind("<B1-Motion>", do_resize)
            grip.bind("<ButtonRelease-1>", end_resize)

            # --- scroll to adjust font size ---
            def scroll_font(event: tk.Event) -> None:
                delta = 1 if int(event.delta) > 0 else -1
                new_fs = max(16, min(32, layout["font_size"] + delta))
                if new_fs != layout["font_size"]:
                    layout["font_size"] = new_fs
                    local_label.config(font=(_FONT, new_fs))
                    _recompute_panels()
                    _save_prefs(layout["width"], layout["height"], layout["font_size"])

            root.bind_all("<MouseWheel>", scroll_font)

            # --- action badge styling ---
            def _style_action_label(badge_label: tk.Label, text: str) -> None:
                for prefix, cfg in [
                    ("本地和牌", _ACTION_WIN),
                    ("本地立直", _ACTION_RIICHI),
                    ("本地鸣牌", _ACTION_CALL),
                    ("本地防守", _ACTION_DEFENSE),
                    ("操作窗口", _ACTION_GENERIC),
                ]:
                    if text.startswith(prefix):
                        label_text, bg, fg = cfg
                        badge_label.config(text=label_text, bg=bg, fg=fg)
                        return
                badge_label.config(text="本地", bg=_BADGE_LOCAL_BG, fg=_BADGE_LOCAL_FG)

            # --- apply text ---
            def apply_text(text: str) -> None:
                value = str(text or "").strip() or "Mahjong Coach"
                first_line = value.split("\n")[0]
                _style_action_label(local_badge, first_line)
                local_label.config(text=value)
                _recompute_panels()
                _set_geometry()

            apply_text("Mahjong Coach")

            # --- mode switching ---
            def show_config_mode() -> None:
                strategy_frame.pack_forget()
                config_frame.pack(fill="both", expand=True)
                _set_geometry()

            def show_strategy_mode() -> None:
                config_frame.pack_forget()
                strategy_frame.pack(fill="both", expand=True)
                _recompute_panels()
                _set_geometry()

            show_config_mode()

            # --- pump ---
            def pump() -> None:
                while True:
                    try:
                        item = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if item is None:
                        _save_prefs(layout["width"], layout["height"], layout["font_size"])
                        root.destroy()
                        return
                    if isinstance(item, dict):
                        cmd = item.get("cmd")
                        if cmd == "show_config":
                            show_config_mode()
                        elif cmd == "show_strategy":
                            show_strategy_mode()
                        continue
                    apply_text(item)
                root.after(120, pump)

            root.after(120, pump)
            root.mainloop()
        except Exception as exc:
            self.last_error = str(exc)


# ---------- text formatting (unchanged logic) ----------

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
    has_plan = state.get("local_direction") or state.get("local_plan") or state.get("current_plan") or state.get("opening_plan")
    if not has_plan and decision.get("decision_type") == "observe":
        return _waiting_hand_overlay(decision)
    local_direction = str(state.get("local_direction") or "").strip()
    local_plan = str(state.get("local_plan") or state.get("current_plan") or state.get("opening_plan") or decision.get("suggestion") or "").strip()
    local_block = _strategy_overlay_block(
        "本地",
        local_direction,
        local_plan,
        _string_items(state.get("local_targets")) or _string_items(state.get("target_shapes")),
        _string_items(state.get("local_cautions")) or _string_items(state.get("caution_points")),
    )
    return local_block


def _action_overlay_text(decision_type: str, decision: dict[str, Any]) -> str:
    if decision_type == "win_window":
        return _format_overlay("本地和牌", "先确认荣和 / 自摸", "这个优先级最高")
    if decision_type == "riichi_window":
        suggestion = str(decision.get("suggestion") or "").strip()
        return _format_overlay("本地立直", _riichi_action_line(suggestion), _riichi_reason_line(suggestion))
    if decision_type == "call_window":
        suggestion = str(decision.get("suggestion") or "").strip()
        return _format_overlay("本地鸣牌", _call_action_line(suggestion), _call_reason_line(suggestion))
    if decision_type == "defense_alert":
        return _format_overlay("本地防守", str(decision.get("suggestion") or "有人立直，先防守"), "先看现物和安全牌")
    return _format_overlay("操作窗口", str(decision.get("suggestion") or decision.get("detail") or "先处理当前按钮"), "")


def _call_text(cautions: list[str]) -> str:
    for item in cautions:
        if item.startswith("鸣牌："):
            return _brief_items(_plain_text(item[3:]), max_items=4)
    return ""


def _strategy_overlay_block(label: str, direction: str, plan: str, targets: list[str], cautions: list[str]) -> str:
    direction = _direction_text(direction, plan, targets)
    yaku = _yaku_text(targets)
    keep = _keep_text(targets, plan)
    discard = _discard_text(cautions, plan)
    efficiency = _efficiency_text(cautions)
    call = _call_text(cautions)
    lines = [f"方向：{direction}"]
    if yaku:
        lines.append(f"役：{yaku}")
    if keep:
        lines.append(f"留：{keep}")
    if discard:
        lines.append(f"打：{discard}")
    if efficiency:
        lines.append(f"牌效：{efficiency}")
    if call:
        lines.append(f"开：{call}")
    return _format_overlay(label, *lines)


def _direction_text(direction: str, plan: str, targets: list[str]) -> str:
    if direction:
        return _plain_text(direction)
    for item in targets:
        if item.startswith("主线："):
            return _clean_line(item)
    return _clean_line(plan) if plan else "继续观察"


def _keep_text(targets: list[str], plan: str) -> str:
    keep = _first_prefixed_value(targets, "保留：")
    if not keep:
        keep = _extract_after(plan, ("保留",), stop_markers=("，先", "；", "。"))
    return _brief_items(_plain_text(keep), max_items=3)


def _yaku_text(targets: list[str]) -> str:
    values: list[str] = []
    for prefix in ("已成役：", "可成役：", "役牌对子："):
        text = _first_prefixed_value(targets, prefix)
        if text:
            values.append(_plain_text(text))
    return _brief_items("、".join(values), max_items=2)


def _discard_text(cautions: list[str], plan: str) -> str:
    discard = _first_prefixed_value(cautions, "优先清理：")
    if not discard:
        for prefix in ("副露收束：", "路线选择：", "下一步：", "副露牌效：", "牌效："):
            value = _first_prefixed_value(cautions, prefix)
            if not value:
                continue
            discard = _extract_after(value, ("主线打", "优先看打", "打"), stop_markers=("后", "；", "，", "。"))
            if discard:
                break
    if not discard:
        discard = _extract_after(plan, ("先打", "先清", "打："), stop_markers=("，", "；", "。"))
    return _brief_items(_plain_text(discard), max_items=3)


def _efficiency_text(cautions: list[str]) -> str:
    for item in cautions:
        text = _plain_text(str(item or "").strip())
        if "牌效" not in text or "当前" not in text:
            continue
        for prefix in ("副露牌效：", "牌效："):
            if text.startswith(prefix):
                text = text[len(prefix) :]
                break
        if text.startswith("估算") and "，" in text:
            text = text.split("，", 1)[1]
        for marker in ("，已扣", "（"):
            if marker in text:
                text = text.split(marker, 1)[0]
        for stop in ("。", "\n"):
            if stop in text:
                text = text.split(stop, 1)[0]
        return text
    return ""


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


def _overlay_geometry(screen_width: int, screen_height: int, width: int, height: int) -> tuple[int, int]:
    x = max(0, int((screen_width - width) / 2))
    bottom_gap = max(160, int(screen_height * 0.19))
    y = max(28, screen_height - height - bottom_gap)
    return x, y


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


def _call_action_line(value: str) -> str:
    text = _plain_text(str(value or "").strip())
    if not text:
        return "默认跳过"
    for sep in ("。", "；", " 当前主线："):
        if sep in text:
            text = text.split(sep, 1)[0]
    return _brief_items(_clean_line(text), max_items=3) or "默认跳过"


def _call_reason_line(value: str) -> str:
    text = _plain_text(str(value or "").strip())
    if not text:
        return "能听牌/明显加速再开"
    if "跳过" in text:
        return "能听牌/明显加速再开"
    return "快攻：能推进就开"


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
        return _format_overlay("等待手牌", f"已识别{accepted}张，继续确认稳定手牌", "副露后少张也会跟踪")
    if occupied > 0:
        return _format_overlay("等待手牌", f"检测到{occupied}个牌位，识别中", "保持牌桌无遮挡")
    return _format_overlay("等待手牌", "未检测到手牌区域", "确保雀魂窗口可见")
