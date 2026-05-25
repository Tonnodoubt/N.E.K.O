from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Any

from .models import FramePacket
from .window_binding import WindowBindingResult, bind_window_from_keywords

try:
    import pyautogui
except Exception:
    pyautogui = None

try:
    from PIL import Image, ImageGrab  # type: ignore[import-not-found]
except Exception:
    Image = None
    ImageGrab = None


@dataclass
class CaptureContext:
    file_path: Path
    binding_result: WindowBindingResult


class DefaultCaptureProvider:
    def locate_window(self, keywords: list[str]) -> WindowBindingResult:
        return bind_window_from_keywords(keywords)

    def capture_frame(self, *, samples_dir: Path, binding_result: WindowBindingResult, save_format: str = "png") -> FramePacket:
        samples_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc)
        safe_format = save_format if save_format in {"png", "jpg", "jpeg"} else "png"
        file_path = samples_dir / f"{timestamp.strftime('%Y%m%d-%H%M%S-%f')}-frame.{safe_format}"
        source = self._save_screenshot(CaptureContext(file_path=file_path, binding_result=binding_result))
        return FramePacket(
            timestamp_ms=int(timestamp.timestamp() * 1000),
            image_path=str(file_path),
            window_title=binding_result.window_title or binding_result.app_name,
            width=int(binding_result.width or 0),
            height=int(binding_result.height or 0),
            source=source,
        )

    def _save_screenshot(self, context: CaptureContext) -> str:
        region = self._resolve_capture_region(context.binding_result)
        errors: list[str] = []

        if region is not None and pyautogui is not None:
            try:
                return self._save_with_pyautogui(context.file_path, region)
            except Exception as exc:
                errors.append(f"pyautogui-region: {exc}")

        if region is not None and ImageGrab is not None:
            try:
                return self._save_with_imagegrab(context.file_path, region)
            except Exception as exc:
                errors.append(f"imagegrab-region: {exc}")

        if ImageGrab is not None and context.binding_result.hwnd:
            try:
                return self._save_with_imagegrab_window(context.file_path, int(context.binding_result.hwnd))
            except Exception as exc:
                errors.append(f"imagegrab-window: {exc}")

        if pyautogui is not None:
            try:
                return self._save_with_pyautogui(context.file_path, region)
            except Exception as exc:
                errors.append(f"pyautogui: {exc}")

        if ImageGrab is not None:
            try:
                return self._save_with_imagegrab(context.file_path, region)
            except Exception as exc:
                errors.append(f"imagegrab: {exc}")

        system = platform.system().lower()
        if system == "darwin" and shutil.which("screencapture"):
            try:
                return self._save_with_screencapture(context.file_path, region)
            except Exception as exc:
                errors.append(f"screencapture: {exc}")
        if system == "linux":
            try:
                return self._save_with_linux_tools(context.file_path, region)
            except Exception as exc:
                errors.append(f"linux-tools: {exc}")
        if errors:
            raise RuntimeError("no screenshot backend succeeded: " + "; ".join(errors))
        raise RuntimeError("no screenshot backend available")

    def _resolve_capture_region(self, binding_result: WindowBindingResult) -> tuple[int, int, int, int] | None:
        if not binding_result.bound or not binding_result.has_bounds():
            return None
        assert binding_result.left is not None
        assert binding_result.top is not None
        assert binding_result.width is not None
        assert binding_result.height is not None
        return (int(binding_result.left), int(binding_result.top), int(binding_result.width), int(binding_result.height))

    def _save_with_pyautogui(self, file_path: Path, region: tuple[int, int, int, int] | None) -> str:
        if pyautogui is None:
            raise RuntimeError("pyautogui unavailable")
        if region is not None:
            try:
                image = pyautogui.screenshot(region=region)
                source = "pyautogui-region"
            except Exception:
                image = pyautogui.screenshot()
                source = "pyautogui-fullscreen-fallback"
        else:
            image = pyautogui.screenshot()
            source = "pyautogui"
        self._persist_image(image, file_path)
        return source

    def _save_with_imagegrab(self, file_path: Path, region: tuple[int, int, int, int] | None) -> str:
        if ImageGrab is None:
            raise RuntimeError("ImageGrab unavailable")
        if region is not None:
            left, top, width, height = region
            try:
                image = ImageGrab.grab(bbox=(left, top, left + width, top + height))
                source = "imagegrab-region"
            except Exception:
                image = ImageGrab.grab()
                source = "imagegrab-fullscreen-fallback"
        else:
            image = ImageGrab.grab()
            source = "imagegrab"
        self._persist_image(image, file_path)
        return source

    def _save_with_imagegrab_window(self, file_path: Path, hwnd: int) -> str:
        if ImageGrab is None:
            raise RuntimeError("ImageGrab unavailable")
        image = ImageGrab.grab(window=hwnd)
        width, height = getattr(image, "size", (0, 0))
        if int(width or 0) <= 0 or int(height or 0) <= 0:
            raise RuntimeError("window capture returned empty image")
        self._persist_image(image, file_path)
        return "imagegrab-window"

    def _save_with_screencapture(self, file_path: Path, region: tuple[int, int, int, int] | None) -> str:
        command = ["screencapture", "-x"]
        if region is not None:
            left, top, width, height = region
            try:
                subprocess.run(command + ["-R", f"{left},{top},{width},{height}", str(file_path)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return "screencapture-region"
            except Exception:
                pass
        subprocess.run(command + [str(file_path)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return "screencapture"

    def _save_with_linux_tools(self, file_path: Path, region: tuple[int, int, int, int] | None) -> str:
        if shutil.which("grim"):
            if region is not None:
                left, top, width, height = region
                try:
                    subprocess.run(["grim", "-g", f"{left},{top} {width}x{height}", str(file_path)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return "grim-region"
                except Exception:
                    pass
            subprocess.run(["grim", str(file_path)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return "grim"
        if shutil.which("gnome-screenshot"):
            subprocess.run(["gnome-screenshot", "-f", str(file_path)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return "gnome-screenshot"
        raise RuntimeError("no supported linux screenshot tool found")

    def _persist_image(self, image: Any, file_path: Path) -> None:
        suffix = file_path.suffix.lower()
        if suffix in {".jpg", ".jpeg"} and getattr(image, "mode", "") not in {"RGB", "L"}:
            image = image.convert("RGB")
        if suffix == ".png":
            image.save(file_path, compress_level=1)
        elif suffix in {".jpg", ".jpeg"}:
            image.save(file_path, quality=88)
        else:
            image.save(file_path)


def prune_frames(frames_dir: Path, *, keep: int) -> None:
    if keep <= 0 or not frames_dir.exists():
        return
    frames = sorted(path for path in frames_dir.glob("*-frame.*") if path.is_file())
    extra = len(frames) - keep
    if extra <= 0:
        return
    for path in frames[:extra]:
        try:
            path.unlink()
        except OSError:
            pass
