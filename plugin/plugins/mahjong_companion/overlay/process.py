from __future__ import annotations

import queue
from typing import Any

from .view import (
    _DragState,
    _advice_view,
    _position_overlay,
    _primary_font_size,
    _render_screen_markers,
)


def run_overlay_process(status_queue: Any, command_queue: Any) -> None:
    try:
        import tkinter as tk
    except Exception as exc:
        _send_error(command_queue, exc)
        return

    try:
        root = tk.Tk()
        root.title("雀魂陪伴")
        root.configure(bg="#181512")
        root.attributes("-topmost", True)
        root.attributes("-alpha", 0.94)
        root.resizable(False, False)
        try:
            root.attributes("-toolwindow", True)
        except Exception:
            pass

        marker = tk.Toplevel(root)
        marker.title("雀魂陪伴")
        marker.configure(bg="black")
        marker.attributes("-topmost", True)
        marker.resizable(False, False)
        marker.overrideredirect(True)
        try:
            marker.attributes("-toolwindow", True)
        except Exception:
            pass
        try:
            marker.attributes("-transparentcolor", "black")
        except Exception:
            marker.attributes("-alpha", 0.28)
        marker_canvas = tk.Canvas(
            marker,
            bg="black",
            highlightthickness=0,
            bd=0,
            relief="flat",
        )
        marker_canvas.pack(fill="both", expand=True)
        marker.withdraw()

        user_moved = _DragState()
        primary_var = tk.StringVar(value="看牌中")
        reason_var = tk.StringVar(value="选择雀魂窗口并开始后，我会在这里给建议。")

        frame = tk.Frame(root, bg="#181512", padx=12, pady=12)
        frame.pack(fill="both", expand=True)
        primary_label = tk.Label(
            frame,
            textvariable=primary_var,
            bg="#221d18",
            fg="#fff7ea",
            font=("Microsoft YaHei UI", 34, "bold"),
            anchor="center",
            justify="center",
            wraplength=300,
            padx=12,
            pady=8,
        )
        primary_label.pack(fill="x")
        reason_label = tk.Label(
            frame,
            textvariable=reason_var,
            bg="#181512",
            fg="#f0d8b8",
            font=("Microsoft YaHei UI", 10),
            anchor="w",
            justify="left",
            wraplength=300,
            height=2,
        )
        reason_label.pack(fill="x", pady=(10, 0))

        def start_drag(event: Any) -> None:
            user_moved.value = True
            user_moved.dx = int(event.x)
            user_moved.dy = int(event.y)

        def drag(event: Any) -> None:
            x = root.winfo_pointerx() - user_moved.dx
            y = root.winfo_pointery() - user_moved.dy
            root.geometry(f"+{x}+{y}")

        for widget in (root, frame):
            widget.bind("<ButtonPress-1>", start_drag)
            widget.bind("<B1-Motion>", drag)

        def render(status: dict[str, Any]) -> None:
            advice_view = _advice_view(status)
            primary_text = advice_view["primary"]
            primary_var.set(primary_text)
            reason_var.set(advice_view["reason"])
            primary_label.configure(font=("Microsoft YaHei UI", _primary_font_size(primary_text), "bold"))
            _render_screen_markers(marker, marker_canvas, status)
            if not user_moved.value:
                _position_overlay(root, status)

        closed = {"value": False}

        def safe_close() -> None:
            if closed["value"]:
                return
            closed["value"] = True
            try:
                marker.destroy()
            except Exception:
                pass
            try:
                root.destroy()
            except Exception:
                pass

        def poll() -> None:
            if closed["value"]:
                return
            while True:
                try:
                    status = status_queue.get_nowait()
                except queue.Empty:
                    break
                if isinstance(status, dict) and status.get("__control__") == "close":
                    safe_close()
                    return
                if isinstance(status, dict):
                    render(status)
            root.after(100, poll)

        def close_from_window() -> None:
            _put_command(command_queue, "hide_overlay")
            safe_close()

        root.protocol("WM_DELETE_WINDOW", close_from_window)
        root.after(100, poll)
        root.mainloop()
    except Exception as exc:
        _send_error(command_queue, exc)


def _put_command(command_queue: Any, command: str) -> None:
    try:
        command_queue.put_nowait(command)
    except Exception:
        pass


def _send_error(command_queue: Any, exc: Exception) -> None:
    try:
        command_queue.put_nowait({"__control__": "error", "error": str(exc)})
    except Exception:
        pass

