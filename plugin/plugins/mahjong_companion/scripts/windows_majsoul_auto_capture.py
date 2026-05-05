"""
windows_majsoul_auto_capture.py — Windows 专用雀魂友人场自动截图采集脚本。

本脚本只做一件事：在你已经登录雀魂的前提下，按你校准好的点位反复进入友人场、
开局、等待发牌完成、按 Win+PrtSc 截屏、离开房间、回到主页，循环采集手牌截图样本。

明确不做：
    - 不出牌、不点手牌、不做任何对局决策。
    - 不替你决定游戏内任何选项。
    - 仅用于你自己的友人场截图采集。

设计要点：
    1. 单文件、无项目依赖。可以把这一个 .py 拖到任意 Windows 目录运行。
    2. 校准数据和截图输出默认保存在脚本所在目录，可以在 GUI 里改。
    3. 急停热键 Esc / Ctrl+Alt+Q（需要 keyboard 库）+ pyautogui FAILSAFE
       （鼠标移到屏幕左上角触发）+ GUI "停止" 按钮，三道保护。
    4. dry-run 模式只移动鼠标 + 打日志，不真的点击、不真的按 Win+PrtSc。

运行（Windows）::

    pip install pyautogui pygetwindow
    pip install keyboard           # 可选，注册全局 Esc / Ctrl+Alt+Q，可能需要管理员
    python windows_majsoul_auto_capture.py

存储默认值（相对脚本所在目录）::

    ./windows_capture_clicks.json  - 校准数据
    ./captures/                    - 输出目录
    ./captures/manifest.jsonl      - 每张截图一行元数据

GUI 启动后第一件事是校准：
    - 把鼠标依次移到提示的位置
    - 按 Enter 或点 [记录]，脚本记下当前鼠标坐标
    - 全部记录完点 [保存校准]
校准会同时记下当时雀魂窗口的位置；运行时如果窗口位置不同，自动按差值平移点击坐标。

修改设置后请记得点 [保存设置]，否则下次启动恢复默认。
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------
# 必要依赖检查
# ---------------------------------------------------------------------------
try:
    import pyautogui
except ImportError:  # pragma: no cover - environment guard
    print("缺少 pyautogui。请先安装：pip install pyautogui", file=sys.stderr)
    raise

try:
    import pygetwindow as gw
except ImportError:  # pragma: no cover - environment guard
    print("缺少 pygetwindow。请先安装：pip install pygetwindow", file=sys.stderr)
    raise

# 可选：keyboard 用于全局 Esc / Ctrl+Alt+Q 急停。
# 没装也能跑，只是热键不生效，仍可用 GUI 停止按钮 + FAILSAFE。
try:
    import keyboard  # type: ignore[import-not-found]

    _KEYBOARD_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dep
    keyboard = None  # type: ignore[assignment]
    _KEYBOARD_AVAILABLE = False


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
pyautogui.FAILSAFE = True  # 鼠标到屏幕左上角会抛 FailSafeException，是最后一道安全锁
pyautogui.PAUSE = 0.1      # 每次 pyautogui 调用之间的硬性短停

WINDOW_TITLE_PATTERNS: tuple[str, ...] = ("雀魂", "Mahjong Soul", "majsoul")

# 校准点定义：(key, 用户可见描述, 是否可选)
POINT_DEFINITIONS: tuple[tuple[str, str, bool], ...] = (
    ("majsoul_window_anchor", "雀魂窗口左上角（建议用作相对坐标基准）", False),
    ("friend_room_button", "友人场按钮", False),
    ("create_room_button", "创建房间按钮", False),
    ("create_confirm_button", "创建房间-确认按钮", False),
    ("start_button", "开始按钮", False),
    ("leave_room_button", "右上角离开房间按钮", False),
    ("leave_confirm_button", "离开房间-确认弹窗按钮", False),
    ("home_recognition_area", "主页识别区域（可选）", True),
    ("my_turn_recognition_area", "我的回合识别区域（可选）", True),
)

# 一个完整循环里依次点击的点位（不含截图、离开等步骤）
CLICK_SEQUENCE: tuple[str, ...] = (
    "friend_room_button",
    "create_room_button",
    "create_confirm_button",
    "start_button",
)

DEFAULT_STEP_WAITS: dict[str, float] = {
    "after_friend_room": 1.5,
    "after_create_room": 1.0,
    "after_create_confirm": 1.5,
    "after_start": 2.5,
    "before_my_turn_min": 8.0,
    "before_my_turn_max": 15.0,
    "after_screenshot": 0.5,
    "after_leave_room": 1.5,
    "after_leave_confirm": 4.0,
    "between_loops": 1.0,
}

CALIBRATION_FILENAME = "windows_capture_clicks.json"
SETTINGS_FILENAME = "windows_capture_settings.json"
DEFAULT_OUTPUT_DIRNAME = "captures"
MANIFEST_FILENAME = "manifest.jsonl"
CALIBRATION_VERSION = 1


# ---------------------------------------------------------------------------
# 路径辅助
# ---------------------------------------------------------------------------
def script_dir() -> Path:
    """返回脚本所在目录。打包成 exe 时回退到当前工作目录。"""
    try:
        return Path(__file__).resolve().parent
    except NameError:  # pragma: no cover - direct interactive use
        return Path.cwd()


def default_calibration_path() -> Path:
    return script_dir() / CALIBRATION_FILENAME


def default_settings_path() -> Path:
    return script_dir() / SETTINGS_FILENAME


def default_output_dir() -> Path:
    return script_dir() / DEFAULT_OUTPUT_DIRNAME


def windows_screenshots_dir() -> Path:
    """系统 Pictures\\Screenshots 目录，Win+PrtSc 默认存到这。"""
    pictures = os.environ.get("USERPROFILE", "")
    if pictures:
        candidate = Path(pictures) / "Pictures" / "Screenshots"
        if candidate.exists():
            return candidate
    # 退路：用户可能改过位置，先尝试 OneDrive
    onedrive = os.environ.get("OneDrive", "")
    if onedrive:
        candidate = Path(onedrive) / "图片" / "屏幕截图"
        if candidate.exists():
            return candidate
        candidate = Path(onedrive) / "Pictures" / "Screenshots"
        if candidate.exists():
            return candidate
    # 最后退到当前用户主目录下的 Pictures\Screenshots，即使不存在也返回
    return Path.home() / "Pictures" / "Screenshots"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def now_filename_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


# ---------------------------------------------------------------------------
# 配置数据结构
# ---------------------------------------------------------------------------
@dataclass
class CalibrationData:
    version: int = CALIBRATION_VERSION
    calibrated_at: str = ""
    screen_size: tuple[int, int] = (0, 0)
    window_at_calibration: dict[str, Any] = field(default_factory=dict)
    points: dict[str, dict[str, int]] = field(default_factory=dict)

    def has_required_points(self) -> tuple[bool, list[str]]:
        missing: list[str] = []
        for key, _label, optional in POINT_DEFINITIONS:
            if optional:
                continue
            if key not in self.points:
                missing.append(key)
        return (not missing, missing)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "calibrated_at": self.calibrated_at,
            "screen_size": list(self.screen_size),
            "window_at_calibration": dict(self.window_at_calibration),
            "points": {
                name: {"x": int(p["x"]), "y": int(p["y"])} for name, p in self.points.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CalibrationData":
        screen = data.get("screen_size") or [0, 0]
        return cls(
            version=int(data.get("version", CALIBRATION_VERSION)),
            calibrated_at=str(data.get("calibrated_at", "")),
            screen_size=(int(screen[0]) if len(screen) > 0 else 0,
                         int(screen[1]) if len(screen) > 1 else 0),
            window_at_calibration=dict(data.get("window_at_calibration", {})),
            points={
                str(k): {"x": int(v["x"]), "y": int(v["y"])}
                for k, v in (data.get("points") or {}).items()
                if isinstance(v, dict) and "x" in v and "y" in v
            },
        )


@dataclass
class CaptureSettings:
    output_dir: str = ""
    calibration_path: str = ""
    loop_count: int = 20
    dry_run: bool = False
    step_waits: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_STEP_WAITS))
    screenshot_wait_timeout_sec: float = 12.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "calibration_path": self.calibration_path,
            "loop_count": self.loop_count,
            "dry_run": self.dry_run,
            "step_waits": dict(self.step_waits),
            "screenshot_wait_timeout_sec": self.screenshot_wait_timeout_sec,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CaptureSettings":
        waits = dict(DEFAULT_STEP_WAITS)
        for key, value in (data.get("step_waits") or {}).items():
            try:
                waits[key] = float(value)
            except (TypeError, ValueError):
                continue
        return cls(
            output_dir=str(data.get("output_dir", "")),
            calibration_path=str(data.get("calibration_path", "")),
            loop_count=int(data.get("loop_count", 20)),
            dry_run=bool(data.get("dry_run", False)),
            step_waits=waits,
            screenshot_wait_timeout_sec=float(data.get("screenshot_wait_timeout_sec", 12.0)),
        )


# ---------------------------------------------------------------------------
# 配置读写
# ---------------------------------------------------------------------------
def load_calibration(path: Path) -> CalibrationData:
    if not path.exists():
        return CalibrationData()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return CalibrationData()
        return CalibrationData.from_dict(data)
    except (OSError, json.JSONDecodeError):
        return CalibrationData()


def save_calibration(path: Path, calibration: CalibrationData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(calibration.to_dict(), ensure_ascii=False, indent=2)
    path.write_text(payload, encoding="utf-8")


def load_settings(path: Path) -> CaptureSettings:
    if not path.exists():
        return CaptureSettings()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return CaptureSettings()
        return CaptureSettings.from_dict(data)
    except (OSError, json.JSONDecodeError):
        return CaptureSettings()


def save_settings(path: Path, settings: CaptureSettings) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(settings.to_dict(), ensure_ascii=False, indent=2)
    path.write_text(payload, encoding="utf-8")


# ---------------------------------------------------------------------------
# 雀魂窗口管理
# ---------------------------------------------------------------------------
def find_majsoul_window() -> Optional[Any]:
    """根据标题模式找雀魂窗口。返回 pygetwindow 的 Window 对象或 None。"""
    try:
        all_windows = gw.getAllWindows()
    except Exception:
        return None
    for window in all_windows:
        title = (window.title or "").strip()
        if not title:
            continue
        if any(pattern.lower() in title.lower() for pattern in WINDOW_TITLE_PATTERNS):
            return window
    return None


def activate_window(window: Any, log: Callable[[str], None]) -> bool:
    """尝试把窗口提到前台。激活失败不阻塞流程，仅记日志。"""
    if window is None:
        return False
    try:
        if window.isMinimized:
            window.restore()
        try:
            window.activate()
        except Exception as exc:  # noqa: BLE001 - pygetwindow 在某些 Win 状态下会抛
            log(f"[warn] window.activate 失败：{exc}（继续）")
        return True
    except Exception as exc:  # noqa: BLE001
        log(f"[warn] 激活窗口异常：{exc}")
        return False


def window_snapshot(window: Any) -> dict[str, Any]:
    """记录当前窗口几何，校准时和运行时都用这个快照。"""
    if window is None:
        return {}
    try:
        return {
            "title": window.title,
            "left": int(window.left),
            "top": int(window.top),
            "width": int(window.width),
            "height": int(window.height),
        }
    except Exception:
        return {}


def compute_window_offset(
    calibration: CalibrationData, current_window: Optional[Any]
) -> tuple[int, int]:
    """基于校准时窗口位置和当前窗口位置，计算点击坐标的平移量。"""
    calib_geom = calibration.window_at_calibration
    if not calib_geom or current_window is None:
        return (0, 0)
    try:
        dx = int(current_window.left) - int(calib_geom.get("left", 0))
        dy = int(current_window.top) - int(calib_geom.get("top", 0))
        return (dx, dy)
    except Exception:
        return (0, 0)


# ---------------------------------------------------------------------------
# 点击 / 截图 / 等待
# ---------------------------------------------------------------------------
class CaptureAborted(Exception):
    """急停信号，由热键、FAILSAFE、GUI 停止按钮、GUI 关闭触发。"""


def click_point(
    point_name: str,
    calibration: CalibrationData,
    *,
    window: Optional[Any],
    dry_run: bool,
    stop_event: threading.Event,
    log: Callable[[str], None],
) -> None:
    """根据校准坐标 + 窗口当前偏移点击。dry-run 时只移动鼠标 + 打日志。"""
    if stop_event.is_set():
        raise CaptureAborted("stop requested")

    point = calibration.points.get(point_name)
    if point is None:
        raise RuntimeError(f"calibration 缺少点位：{point_name}")

    dx, dy = compute_window_offset(calibration, window)
    target_x = int(point["x"]) + dx
    target_y = int(point["y"]) + dy

    log(f"[click] {point_name} -> ({target_x}, {target_y}) dry_run={dry_run}")

    try:
        pyautogui.moveTo(target_x, target_y, duration=0.25)
        if dry_run:
            return
        if stop_event.is_set():
            raise CaptureAborted("stop requested before click")
        pyautogui.click(target_x, target_y)
    except pyautogui.FailSafeException:
        raise CaptureAborted("FAILSAFE 触发：鼠标到了屏幕左上角")


def press_win_prtsc(*, dry_run: bool, log: Callable[[str], None]) -> None:
    """模拟 Win + PrtSc。Windows 会自动保存到 Pictures\\Screenshots。"""
    log(f"[prtsc] Win + PrtSc dry_run={dry_run}")
    if dry_run:
        return
    try:
        pyautogui.hotkey("win", "printscreen")
    except pyautogui.FailSafeException:
        raise CaptureAborted("FAILSAFE 触发于 Win+PrtSc")


def snapshot_existing_screenshots(folder: Path) -> set[Path]:
    if not folder.exists():
        return set()
    return {p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".png"}


def wait_for_new_screenshot(
    folder: Path,
    before: set[Path],
    *,
    timeout_sec: float,
    stop_event: threading.Event,
    log: Callable[[str], None],
) -> Optional[Path]:
    """等待 Win+PrtSc 把新文件落到截图目录。轮询，最多 timeout_sec 秒。"""
    deadline = time.monotonic() + timeout_sec
    last_seen_count = len(before)
    while time.monotonic() < deadline:
        if stop_event.is_set():
            raise CaptureAborted("stop requested while waiting screenshot")
        if not folder.exists():
            time.sleep(0.3)
            continue
        current = snapshot_existing_screenshots(folder)
        new_files = current - before
        if new_files:
            # 选 mtime 最新的那个
            newest = max(new_files, key=lambda p: p.stat().st_mtime)
            return newest
        if len(current) != last_seen_count:
            last_seen_count = len(current)
            log(f"[debug] 截图目录文件数变化：{last_seen_count}")
        time.sleep(0.3)
    log(f"[warn] {timeout_sec:.1f}s 内没看到新截图，跳过本轮")
    return None


def copy_screenshot_to_output(
    src: Path,
    output_dir: Path,
    loop_index: int,
    log: Callable[[str], None],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_name = f"majsoul_auto_{now_filename_stamp()}_loop{loop_index:03d}.png"
    target = output_dir / target_name
    shutil.copy2(src, target)
    log(f"[save] {src.name} -> {target.name}")
    return target


def append_manifest(
    output_dir: Path,
    record: dict[str, Any],
    log: Callable[[str], None],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_FILENAME
    line = json.dumps(record, ensure_ascii=False)
    with manifest_path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    log(f"[manifest] +1 行：{record.get('saved_path')}")


def sleep_with_stop(seconds: float, stop_event: threading.Event) -> None:
    """可中断的 sleep。stop 触发时立刻抛 CaptureAborted。"""
    if seconds <= 0:
        return
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if stop_event.is_set():
            raise CaptureAborted("stop requested in sleep")
        remain = deadline - time.monotonic()
        time.sleep(min(0.2, max(0.05, remain)))


def wait_for_my_turn(
    *,
    min_sec: float,
    max_sec: float,
    stop_event: threading.Event,
    log: Callable[[str], None],
) -> None:
    """v0 简单实现：在 [min, max] 区间内等一个固定时长。

    后续可以改成检测“我的回合识别区域”里的按钮是否出现：
        from PIL import Image
        ...
        region_metrics = sample_region(my_turn_recognition_area)
        if region_metrics["bright_ratio"] > X: return

    现在不做 OCR，等若干秒就当轮到我。
    """
    target = max(0.0, min_sec)
    if max_sec > min_sec:
        target = (min_sec + max_sec) / 2.0
    log(f"[wait_turn] 等待 {target:.1f}s 假设轮到我（v0 简单实现）")
    sleep_with_stop(target, stop_event)


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------
def ensure_window_or_prompt(
    log: Callable[[str], None],
    stop_event: threading.Event,
    *,
    prompt_callback: Optional[Callable[[], bool]] = None,
) -> Optional[Any]:
    """运行时确保有雀魂窗口；找不到的话让 GUI 弹窗让用户处理。

    prompt_callback 由 GUI 注入，弹窗给用户"我已经把窗口置顶了 [继续 / 取消]"。
    返回 None 说明用户取消或仍找不到窗口。
    """
    window = find_majsoul_window()
    if window is not None:
        return window
    log("[warn] 没找到雀魂窗口，请把雀魂置顶后继续")
    if prompt_callback is None:
        return None
    if stop_event.is_set():
        return None
    user_continued = prompt_callback()
    if not user_continued:
        return None
    return find_majsoul_window()


def run_one_loop(
    loop_index: int,
    *,
    calibration: CalibrationData,
    settings: CaptureSettings,
    stop_event: threading.Event,
    log: Callable[[str], None],
    prompt_callback: Optional[Callable[[], bool]] = None,
) -> bool:
    """跑一个完整循环。返回 True 表示成功采到一张图，False 表示失败但允许继续。

    任何不可恢复的异常会抛出，由 run_capture_loop 捕获并停掉整轮。
    """
    log(f"=== loop {loop_index} 开始 ===")

    window = ensure_window_or_prompt(log, stop_event, prompt_callback=prompt_callback)
    if window is None:
        log("[abort] 窗口缺失，跳过本轮")
        return False

    activate_window(window, log)
    sleep_with_stop(0.3, stop_event)

    # 1-4: 友人场 -> 创建房间 -> 创建确认 -> 开始
    wait_keys = [
        "after_friend_room",
        "after_create_room",
        "after_create_confirm",
        "after_start",
    ]
    for click_name, wait_key in zip(CLICK_SEQUENCE, wait_keys):
        if stop_event.is_set():
            raise CaptureAborted("stop in click sequence")
        click_point(
            click_name,
            calibration,
            window=window,
            dry_run=settings.dry_run,
            stop_event=stop_event,
            log=log,
        )
        sleep_with_stop(settings.step_waits.get(wait_key, 1.0), stop_event)

    # 5: 等到我的回合
    wait_for_my_turn(
        min_sec=settings.step_waits.get("before_my_turn_min", 8.0),
        max_sec=settings.step_waits.get("before_my_turn_max", 15.0),
        stop_event=stop_event,
        log=log,
    )

    # 6: Win+PrtSc
    screenshot_folder = windows_screenshots_dir()
    log(f"[prtsc] 监听目录：{screenshot_folder}")
    before = snapshot_existing_screenshots(screenshot_folder)

    captured_at = now_iso()
    press_win_prtsc(dry_run=settings.dry_run, log=log)
    sleep_with_stop(settings.step_waits.get("after_screenshot", 0.5), stop_event)

    saved_path: Optional[Path] = None
    original_path: Optional[Path] = None
    if not settings.dry_run:
        new_screenshot = wait_for_new_screenshot(
            screenshot_folder,
            before,
            timeout_sec=settings.screenshot_wait_timeout_sec,
            stop_event=stop_event,
            log=log,
        )
        if new_screenshot is not None:
            saved_path = copy_screenshot_to_output(
                new_screenshot, Path(settings.output_dir), loop_index, log,
            )
            original_path = new_screenshot

    # 7-8: 离开房间 -> 离开确认
    click_point(
        "leave_room_button",
        calibration,
        window=window,
        dry_run=settings.dry_run,
        stop_event=stop_event,
        log=log,
    )
    sleep_with_stop(settings.step_waits.get("after_leave_room", 1.5), stop_event)

    click_point(
        "leave_confirm_button",
        calibration,
        window=window,
        dry_run=settings.dry_run,
        stop_event=stop_event,
        log=log,
    )
    sleep_with_stop(settings.step_waits.get("after_leave_confirm", 4.0), stop_event)

    # 9: manifest 留痕
    if saved_path is not None and original_path is not None:
        append_manifest(
            Path(settings.output_dir),
            {
                "loop_index": loop_index,
                "saved_path": str(saved_path),
                "original_path": str(original_path),
                "captured_at": captured_at,
                "window_title": window.title if window else "",
                "step": "my_turn_screenshot",
                "source": "win_prtsc",
                "platform": "windows",
                "dry_run": settings.dry_run,
            },
            log,
        )
        success = True
    elif settings.dry_run:
        log("[dry-run] 跳过 manifest 写入")
        success = True
    else:
        log("[warn] 本轮没采到截图，manifest 不记录")
        success = False

    sleep_with_stop(settings.step_waits.get("between_loops", 1.0), stop_event)
    log(f"=== loop {loop_index} 结束 ok={success} ===")
    return success


def run_capture_loop(
    *,
    calibration: CalibrationData,
    settings: CaptureSettings,
    stop_event: threading.Event,
    log: Callable[[str], None],
    prompt_callback: Optional[Callable[[], bool]] = None,
    on_done: Optional[Callable[[bool, str], None]] = None,
) -> None:
    """整轮跑。被 GUI 用 threading.Thread 启动。"""
    success = False
    error_message = ""
    completed = 0
    failed = 0
    try:
        ok, missing = calibration.has_required_points()
        if not ok:
            raise RuntimeError(f"校准缺少必要点位：{missing}")
        Path(settings.output_dir).mkdir(parents=True, exist_ok=True)
        log(
            f"[start] 开始采集，loop_count={settings.loop_count} "
            f"dry_run={settings.dry_run} output={settings.output_dir}",
        )
        for index in range(1, max(1, settings.loop_count) + 1):
            if stop_event.is_set():
                raise CaptureAborted("stop before loop")
            ok = run_one_loop(
                index,
                calibration=calibration,
                settings=settings,
                stop_event=stop_event,
                log=log,
                prompt_callback=prompt_callback,
            )
            if ok:
                completed += 1
            else:
                failed += 1
        success = True
    except CaptureAborted as exc:
        error_message = f"已停止：{exc}"
        log(f"[stop] {error_message}")
    except pyautogui.FailSafeException:
        error_message = "FAILSAFE 触发，已停止"
        log(f"[stop] {error_message}")
    except Exception as exc:  # noqa: BLE001
        error_message = f"循环异常退出：{exc!r}"
        log(f"[error] {error_message}")
    finally:
        log(f"[summary] 成功 {completed} / 失败 {failed}")
        if on_done is not None:
            on_done(success, error_message)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
class CaptureGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("雀魂自动截图采集（Windows）")
        self.geometry("780x680")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._log_queue: queue.Queue[str] = queue.Queue()
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._calibration_path = default_calibration_path()
        self._settings_path = default_settings_path()
        self._calibration = load_calibration(self._calibration_path)
        self._settings = load_settings(self._settings_path)
        if not self._settings.output_dir:
            self._settings.output_dir = str(default_output_dir())
        if not self._settings.calibration_path:
            self._settings.calibration_path = str(self._calibration_path)
        else:
            self._calibration_path = Path(self._settings.calibration_path)
            self._calibration = load_calibration(self._calibration_path)

        self._build_widgets()
        self._refresh_calibration_status()
        self._poll_log_queue()
        self._setup_global_hotkeys()

    # ------------------------------------------------------------------
    # 控件布局
    # ------------------------------------------------------------------
    def _build_widgets(self) -> None:
        pad = {"padx": 8, "pady": 4}

        # 路径区
        path_frame = ttk.LabelFrame(self, text="存储路径")
        path_frame.pack(fill=tk.X, **pad)

        ttk.Label(path_frame, text="校准 JSON：").grid(row=0, column=0, sticky=tk.W, padx=6, pady=4)
        self._calib_var = tk.StringVar(value=str(self._calibration_path))
        ttk.Entry(path_frame, textvariable=self._calib_var, width=60).grid(
            row=0, column=1, sticky=tk.EW, padx=6, pady=4,
        )
        ttk.Button(path_frame, text="浏览...", command=self._pick_calibration_path).grid(
            row=0, column=2, padx=6, pady=4,
        )

        ttk.Label(path_frame, text="输出目录：").grid(row=1, column=0, sticky=tk.W, padx=6, pady=4)
        self._output_var = tk.StringVar(value=self._settings.output_dir)
        ttk.Entry(path_frame, textvariable=self._output_var, width=60).grid(
            row=1, column=1, sticky=tk.EW, padx=6, pady=4,
        )
        ttk.Button(path_frame, text="浏览...", command=self._pick_output_dir).grid(
            row=1, column=2, padx=6, pady=4,
        )

        path_frame.columnconfigure(1, weight=1)

        # 参数区
        param_frame = ttk.LabelFrame(self, text="运行参数")
        param_frame.pack(fill=tk.X, **pad)

        ttk.Label(param_frame, text="循环次数：").grid(row=0, column=0, sticky=tk.W, padx=6, pady=4)
        self._loop_count_var = tk.IntVar(value=self._settings.loop_count)
        ttk.Spinbox(
            param_frame,
            from_=1,
            to=999,
            textvariable=self._loop_count_var,
            width=8,
        ).grid(row=0, column=1, sticky=tk.W, padx=6, pady=4)

        self._dry_run_var = tk.BooleanVar(value=self._settings.dry_run)
        ttk.Checkbutton(
            param_frame,
            text="Dry-run（只移动鼠标 + 打日志，不真实点击 / 不真实截屏）",
            variable=self._dry_run_var,
        ).grid(row=0, column=2, sticky=tk.W, padx=12, pady=4)

        ttk.Label(param_frame, text="截图等待超时（秒）：").grid(
            row=1, column=0, sticky=tk.W, padx=6, pady=4,
        )
        self._screenshot_timeout_var = tk.DoubleVar(
            value=self._settings.screenshot_wait_timeout_sec,
        )
        ttk.Spinbox(
            param_frame,
            from_=2.0,
            to=60.0,
            increment=0.5,
            textvariable=self._screenshot_timeout_var,
            width=8,
        ).grid(row=1, column=1, sticky=tk.W, padx=6, pady=4)

        # 等待时间细化
        wait_frame = ttk.LabelFrame(self, text="每步等待秒数")
        wait_frame.pack(fill=tk.X, **pad)

        self._wait_vars: dict[str, tk.DoubleVar] = {}
        wait_entries: list[tuple[str, str]] = [
            ("after_friend_room", "点友人场后"),
            ("after_create_room", "点创建房间后"),
            ("after_create_confirm", "点创建确认后"),
            ("after_start", "点开始后"),
            ("before_my_turn_min", "等我的回合 min"),
            ("before_my_turn_max", "等我的回合 max"),
            ("after_screenshot", "截图后"),
            ("after_leave_room", "点离开房间后"),
            ("after_leave_confirm", "点离开确认后"),
            ("between_loops", "循环之间"),
        ]
        for idx, (key, label) in enumerate(wait_entries):
            row = idx // 2
            col_base = (idx % 2) * 2
            ttk.Label(wait_frame, text=f"{label}：").grid(
                row=row, column=col_base, sticky=tk.W, padx=6, pady=2,
            )
            var = tk.DoubleVar(
                value=self._settings.step_waits.get(key, DEFAULT_STEP_WAITS.get(key, 1.0)),
            )
            ttk.Spinbox(
                wait_frame,
                from_=0.0,
                to=60.0,
                increment=0.1,
                textvariable=var,
                width=8,
            ).grid(row=row, column=col_base + 1, sticky=tk.W, padx=6, pady=2)
            self._wait_vars[key] = var

        # 操作区
        ops_frame = ttk.Frame(self)
        ops_frame.pack(fill=tk.X, **pad)

        ttk.Button(ops_frame, text="校准点位", command=self._on_calibrate).pack(
            side=tk.LEFT, padx=4,
        )
        ttk.Button(ops_frame, text="保存设置", command=self._on_save_settings).pack(
            side=tk.LEFT, padx=4,
        )
        self._start_button = ttk.Button(ops_frame, text="开始", command=self._on_start)
        self._start_button.pack(side=tk.LEFT, padx=12)
        self._stop_button = ttk.Button(
            ops_frame, text="停止", command=self._on_stop, state=tk.DISABLED,
        )
        self._stop_button.pack(side=tk.LEFT, padx=4)

        self._status_var = tk.StringVar(value="待命")
        ttk.Label(ops_frame, textvariable=self._status_var, foreground="blue").pack(
            side=tk.RIGHT, padx=8,
        )

        # 校准状态显示
        self._calib_status_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self._calib_status_var, foreground="dark green").pack(
            anchor=tk.W, padx=12,
        )

        # 日志区
        log_frame = ttk.LabelFrame(self, text="状态日志")
        log_frame.pack(fill=tk.BOTH, expand=True, **pad)
        self._log_text = tk.Text(log_frame, height=18, wrap=tk.NONE)
        log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=log_scroll.set)
        self._log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        hint = tk.Label(
            self,
            text=(
                "急停：Esc / Ctrl+Alt+Q（需 keyboard 库）/ 鼠标到屏幕左上角 / 点 [停止] / 关 GUI。"
                + ("" if _KEYBOARD_AVAILABLE else "  [当前未装 keyboard，全局热键不可用]")
            ),
            fg="gray30",
            justify=tk.LEFT,
        )
        hint.pack(anchor=tk.W, padx=12, pady=(0, 8))

    # ------------------------------------------------------------------
    # 全局热键
    # ------------------------------------------------------------------
    def _setup_global_hotkeys(self) -> None:
        if not _KEYBOARD_AVAILABLE:
            self._log("[hotkey] keyboard 未安装，仅 GUI 停止按钮 + FAILSAFE 可用")
            return
        try:
            keyboard.add_hotkey("esc", self._on_stop)
            keyboard.add_hotkey("ctrl+alt+q", self._on_stop)
            self._log("[hotkey] 已注册 Esc 和 Ctrl+Alt+Q 急停")
        except Exception as exc:  # noqa: BLE001
            self._log(f"[hotkey] 注册失败（可能需要管理员权限）：{exc}")

    # ------------------------------------------------------------------
    # 路径选择
    # ------------------------------------------------------------------
    def _pick_calibration_path(self) -> None:
        initial = self._calib_var.get() or str(default_calibration_path())
        path = filedialog.asksaveasfilename(
            title="选择校准 JSON 文件",
            initialfile=Path(initial).name,
            initialdir=str(Path(initial).parent),
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All", "*.*")],
        )
        if path:
            self._calib_var.set(path)
            self._calibration_path = Path(path)
            self._calibration = load_calibration(self._calibration_path)
            self._refresh_calibration_status()

    def _pick_output_dir(self) -> None:
        initial = self._output_var.get() or str(default_output_dir())
        path = filedialog.askdirectory(title="选择输出目录", initialdir=initial)
        if path:
            self._output_var.set(path)

    # ------------------------------------------------------------------
    # 校准
    # ------------------------------------------------------------------
    def _on_calibrate(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            messagebox.showinfo("校准", "请先停止采集再校准。")
            return
        CalibrationDialog(self, self._calibration, on_save=self._save_calibration_from_dialog)

    def _save_calibration_from_dialog(self, calibration: CalibrationData) -> None:
        self._calibration = calibration
        path = Path(self._calib_var.get() or str(default_calibration_path()))
        try:
            save_calibration(path, calibration)
            self._calibration_path = path
            self._refresh_calibration_status()
            self._log(f"[calib] 校准已保存：{path}")
        except OSError as exc:
            messagebox.showerror("保存失败", f"无法写入 {path}\n{exc}")

    def _refresh_calibration_status(self) -> None:
        ok, missing = self._calibration.has_required_points()
        if ok:
            self._calib_status_var.set(
                f"校准 OK（{len(self._calibration.points)} 个点位，"
                f"标定于 {self._calibration.calibrated_at or '未知时间'}）",
            )
        elif self._calibration.points:
            self._calib_status_var.set(f"校准缺少：{', '.join(missing)}")
        else:
            self._calib_status_var.set("校准未完成，请先点 [校准点位]")

    # ------------------------------------------------------------------
    # 设置保存
    # ------------------------------------------------------------------
    def _collect_settings_from_widgets(self) -> CaptureSettings:
        waits = dict(DEFAULT_STEP_WAITS)
        for key, var in self._wait_vars.items():
            try:
                waits[key] = float(var.get())
            except (tk.TclError, ValueError):
                continue
        return CaptureSettings(
            output_dir=self._output_var.get().strip() or str(default_output_dir()),
            calibration_path=self._calib_var.get().strip() or str(default_calibration_path()),
            loop_count=int(self._loop_count_var.get()),
            dry_run=bool(self._dry_run_var.get()),
            step_waits=waits,
            screenshot_wait_timeout_sec=float(self._screenshot_timeout_var.get()),
        )

    def _on_save_settings(self) -> None:
        self._settings = self._collect_settings_from_widgets()
        try:
            save_settings(self._settings_path, self._settings)
            self._log(f"[settings] 已保存到 {self._settings_path}")
        except OSError as exc:
            messagebox.showerror("保存失败", f"无法写入 {self._settings_path}\n{exc}")

    # ------------------------------------------------------------------
    # 开始 / 停止
    # ------------------------------------------------------------------
    def _on_start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._settings = self._collect_settings_from_widgets()
        ok, missing = self._calibration.has_required_points()
        if not ok:
            messagebox.showwarning("校准未完成", f"还缺：{', '.join(missing)}")
            return
        self._stop_event.clear()
        self._start_button.configure(state=tk.DISABLED)
        self._stop_button.configure(state=tk.NORMAL)
        self._status_var.set("运行中")
        self._worker = threading.Thread(
            target=run_capture_loop,
            kwargs={
                "calibration": self._calibration,
                "settings": self._settings,
                "stop_event": self._stop_event,
                "log": self._log,
                "prompt_callback": self._prompt_window_missing,
                "on_done": self._on_worker_done,
            },
            name="majsoul-capture-loop",
            daemon=True,
        )
        self._worker.start()

    def _on_stop(self) -> None:
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        self._log("[stop] 已发出停止信号，等当前点击 / 等待结束")
        self._status_var.set("正在停止…")

    def _on_worker_done(self, success: bool, error_message: str) -> None:
        # 工作线程的回调，需要 marshal 回主线程
        self.after(0, lambda: self._after_worker_done(success, error_message))

    def _after_worker_done(self, success: bool, error_message: str) -> None:
        self._worker = None
        self._start_button.configure(state=tk.NORMAL)
        self._stop_button.configure(state=tk.DISABLED)
        if success:
            self._status_var.set("完成")
        elif error_message:
            self._status_var.set(error_message)
        else:
            self._status_var.set("已停止")

    def _prompt_window_missing(self) -> bool:
        """工作线程里调用，需要 marshal 到 GUI 主线程并阻塞等待回应。"""
        result_holder: list[bool] = []
        done = threading.Event()

        def ask() -> None:
            ok = messagebox.askyesno(
                "找不到雀魂窗口",
                "没找到雀魂窗口。请把雀魂置顶到前台，然后点 [是] 继续，[否] 中止。",
            )
            result_holder.append(bool(ok))
            done.set()

        self.after(0, ask)
        # 最多等 60 秒；超时按取消处理
        if not done.wait(60.0):
            return False
        return bool(result_holder and result_holder[0])

    # ------------------------------------------------------------------
    # 日志
    # ------------------------------------------------------------------
    def _log(self, message: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._log_queue.put(f"[{ts}] {message}")

    def _poll_log_queue(self) -> None:
        try:
            while True:
                line = self._log_queue.get_nowait()
                self._log_text.insert(tk.END, line + "\n")
                self._log_text.see(tk.END)
        except queue.Empty:
            pass
        self.after(100, self._poll_log_queue)

    # ------------------------------------------------------------------
    # 关闭
    # ------------------------------------------------------------------
    def _on_close(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            if not messagebox.askyesno("仍在运行", "采集还在进行，确定关闭吗？"):
                return
            self._stop_event.set()
            self._worker.join(timeout=3.0)
        if _KEYBOARD_AVAILABLE:
            try:
                keyboard.clear_all_hotkeys()
            except Exception:  # noqa: BLE001
                pass
        self.destroy()


# ---------------------------------------------------------------------------
# 校准对话框
# ---------------------------------------------------------------------------
class CalibrationDialog(tk.Toplevel):
    """模态对话框：依次让用户把鼠标放到目标位置，按 Enter 或 [记录] 保存。"""

    def __init__(
        self,
        parent: tk.Tk,
        existing: CalibrationData,
        *,
        on_save: Callable[[CalibrationData], None],
    ) -> None:
        super().__init__(parent)
        self.title("点位校准")
        self.geometry("560x420")
        self.transient(parent)
        self.grab_set()
        self._on_save = on_save
        self._existing = existing
        self._points: dict[str, dict[str, int]] = dict(existing.points)
        self._build_widgets()
        self.bind("<Return>", lambda _e: self._record_current())

    def _build_widgets(self) -> None:
        ttk.Label(
            self,
            text=(
                "依次把鼠标移到下面列表里的点位，然后按 Enter 或点 [记录当前]。\n"
                "可选项可以跳过。完成后点 [保存]。"
            ),
            justify=tk.LEFT,
        ).pack(anchor=tk.W, padx=10, pady=8)

        list_frame = ttk.Frame(self)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10)

        self._tree = ttk.Treeview(list_frame, columns=("desc", "coord"), show="headings", height=10)
        self._tree.heading("desc", text="点位")
        self._tree.heading("coord", text="坐标 (x, y)")
        self._tree.column("desc", width=320)
        self._tree.column("coord", width=160)
        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self._tree.yview)
        self._tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self._refresh_tree()

        bottom = ttk.Frame(self)
        bottom.pack(fill=tk.X, padx=10, pady=8)
        ttk.Button(bottom, text="记录当前 (Enter)", command=self._record_current).pack(
            side=tk.LEFT,
        )
        ttk.Button(bottom, text="跳过此项", command=self._skip_current).pack(
            side=tk.LEFT, padx=6,
        )
        ttk.Button(bottom, text="清除此项", command=self._clear_current).pack(
            side=tk.LEFT, padx=6,
        )
        ttk.Button(bottom, text="保存", command=self._save).pack(side=tk.RIGHT)
        ttk.Button(bottom, text="取消", command=self.destroy).pack(side=tk.RIGHT, padx=6)

        # 默认选中第一个未记录的
        first_unset = self._first_unset_index()
        if first_unset >= 0:
            self._select_index(first_unset)

    def _refresh_tree(self) -> None:
        self._tree.delete(*self._tree.get_children())
        for key, label, optional in POINT_DEFINITIONS:
            coord = self._points.get(key)
            display_label = label + ("（可选）" if optional and "可选" not in label else "")
            display_coord = f"({coord['x']}, {coord['y']})" if coord else "未记录"
            self._tree.insert("", tk.END, iid=key, values=(display_label, display_coord))

    def _first_unset_index(self) -> int:
        for idx, (key, _label, optional) in enumerate(POINT_DEFINITIONS):
            if optional:
                continue
            if key not in self._points:
                return idx
        return 0

    def _select_index(self, index: int) -> None:
        keys = [d[0] for d in POINT_DEFINITIONS]
        if 0 <= index < len(keys):
            self._tree.selection_set(keys[index])
            self._tree.focus(keys[index])
            self._tree.see(keys[index])

    def _selected_key(self) -> Optional[str]:
        sel = self._tree.selection()
        return sel[0] if sel else None

    def _advance_selection(self) -> None:
        keys = [d[0] for d in POINT_DEFINITIONS]
        current = self._selected_key()
        if current is None:
            return
        try:
            idx = keys.index(current)
        except ValueError:
            return
        if idx + 1 < len(keys):
            self._select_index(idx + 1)

    def _record_current(self) -> None:
        key = self._selected_key()
        if key is None:
            messagebox.showinfo("提示", "请先在列表里选择一个点位。")
            return
        x, y = pyautogui.position()
        self._points[key] = {"x": int(x), "y": int(y)}
        self._refresh_tree()
        self._tree.selection_set(key)
        self._advance_selection()

    def _skip_current(self) -> None:
        self._advance_selection()

    def _clear_current(self) -> None:
        key = self._selected_key()
        if key and key in self._points:
            del self._points[key]
            self._refresh_tree()
            self._tree.selection_set(key)

    def _save(self) -> None:
        # 检查必填
        missing = [
            key
            for key, _label, optional in POINT_DEFINITIONS
            if not optional and key not in self._points
        ]
        if missing:
            if not messagebox.askyesno(
                "仍有必填项未记录",
                f"以下点位未记录：{', '.join(missing)}\n仍要保存吗？",
            ):
                return

        window = find_majsoul_window()
        screen_w, screen_h = pyautogui.size()
        calibration = CalibrationData(
            version=CALIBRATION_VERSION,
            calibrated_at=now_iso(),
            screen_size=(int(screen_w), int(screen_h)),
            window_at_calibration=window_snapshot(window),
            points=dict(self._points),
        )
        self._on_save(calibration)
        self.destroy()


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main() -> int:
    if sys.platform != "win32":
        print(
            "本脚本是 Windows 专用。在其他平台下 Win+PrtSc 等行为不一致，已退出。",
            file=sys.stderr,
        )
        return 1
    if not _KEYBOARD_AVAILABLE:
        print(
            "提示：未检测到 keyboard 库，全局 Esc / Ctrl+Alt+Q 不可用。\n"
            "  pip install keyboard 之后重启脚本可启用（注册热键可能需要管理员）。\n"
            "  GUI 的 [停止] 按钮和 pyautogui FAILSAFE（鼠标到屏幕左上角）仍然有效。",
            file=sys.stderr,
        )
    app = CaptureGUI()
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
