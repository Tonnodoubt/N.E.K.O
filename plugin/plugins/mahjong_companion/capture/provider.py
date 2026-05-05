from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import platform
import shutil
import subprocess
import time
from typing import Any, Optional, Protocol, Tuple

from ..contracts import FramePacket
from ..window_binding import WindowBindingResult, bind_window_from_keywords

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


class CaptureProvider(Protocol):
    def locate_window(self, keywords: list[str]) -> WindowBindingResult:
        ...

    def capture_frame(self, *, samples_dir: Path, binding_result: WindowBindingResult, save_format: str) -> FramePacket:
        ...


class DefaultCaptureProvider:
    def locate_window(self, keywords: list[str]) -> WindowBindingResult:
        return bind_window_from_keywords(keywords)

    def capture_frame(self, *, samples_dir: Path, binding_result: WindowBindingResult, save_format: str) -> FramePacket:
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

        if context.binding_result.hwnd:
            try:
                return self._save_with_win32_print_window(context.file_path, int(context.binding_result.hwnd))
            except Exception as exc:
                errors.append(f"win32-printwindow: {exc}")

        if ImageGrab is not None and context.binding_result.hwnd:
            try:
                return self._save_with_imagegrab_window(context.file_path, int(context.binding_result.hwnd))
            except Exception as exc:
                errors.append(f"imagegrab-window: {exc}")

        if context.binding_result.hwnd:
            self._activate_window_handle_best_effort(int(context.binding_result.hwnd))

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
            raise RuntimeError("no screenshot backend succeeded: %s" % "; ".join(errors))
        raise RuntimeError("no screenshot backend available")

    def _resolve_capture_region(
        self,
        binding_result: WindowBindingResult,
    ) -> Optional[Tuple[int, int, int, int]]:
        if not binding_result.bound or not binding_result.has_bounds():
            return None
        assert binding_result.left is not None
        assert binding_result.top is not None
        assert binding_result.width is not None
        assert binding_result.height is not None
        return (
            int(binding_result.left),
            int(binding_result.top),
            int(binding_result.width),
            int(binding_result.height),
        )

    def _save_with_pyautogui(
        self,
        file_path: Path,
        region: Optional[Tuple[int, int, int, int]],
    ) -> str:
        source = "pyautogui"
        if region is not None:
            try:
                image = pyautogui.screenshot(region=region)
                source = "pyautogui-region"
            except Exception:
                image = pyautogui.screenshot()
                source = "pyautogui-fullscreen-fallback"
        else:
            image = pyautogui.screenshot()
        self._persist_image(image, file_path)
        return source

    def _save_with_imagegrab(
        self,
        file_path: Path,
        region: Optional[Tuple[int, int, int, int]],
    ) -> str:
        source = "imagegrab"
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
        self._persist_image(image, file_path)
        return source

    def _save_with_imagegrab_window(self, file_path: Path, hwnd: int) -> str:
        image = ImageGrab.grab(window=hwnd)
        width, height = getattr(image, "size", (0, 0))
        if int(width or 0) <= 0 or int(height or 0) <= 0:
            raise RuntimeError("window capture returned an empty image")
        self._persist_image(image, file_path)
        return "imagegrab-window"

    def _save_with_win32_print_window(self, file_path: Path, hwnd: int) -> str:
        if Image is None:
            raise RuntimeError("PIL image module unavailable")

        try:
            import ctypes
            import win32gui  # type: ignore[import-not-found]
            import win32ui  # type: ignore[import-not-found]
        except Exception as exc:
            raise RuntimeError(f"win32 capture backend unavailable: {exc}") from exc

        if not win32gui.IsWindow(hwnd):
            raise RuntimeError("invalid window handle")

        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        width = int(right - left)
        height = int(bottom - top)
        if width <= 0 or height <= 0:
            raise RuntimeError("window bounds are empty")

        hwnd_dc = win32gui.GetWindowDC(hwnd)
        if not hwnd_dc:
            raise RuntimeError("failed to get window device context")

        mfc_dc = None
        save_dc = None
        bitmap = None
        try:
            mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
            save_dc = mfc_dc.CreateCompatibleDC()
            bitmap = win32ui.CreateBitmap()
            bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
            save_dc.SelectObject(bitmap)

            printed = False
            for flag in (0x00000002, 0):
                if ctypes.windll.user32.PrintWindow(int(hwnd), int(save_dc.GetSafeHdc()), flag):
                    printed = True
                    break
            if not printed:
                raise RuntimeError("PrintWindow returned false")

            bitmap_info = bitmap.GetInfo()
            bitmap_bits = bitmap.GetBitmapBits(True)
            image = Image.frombuffer(
                "RGB",
                (int(bitmap_info["bmWidth"]), int(bitmap_info["bmHeight"])),
                bitmap_bits,
                "raw",
                "BGRX",
                0,
                1,
            )
            if self._image_looks_blank(image):
                raise RuntimeError("PrintWindow returned a blank image")
            self._persist_image(image, file_path)
            return "win32-printwindow"
        finally:
            if bitmap is not None:
                try:
                    win32gui.DeleteObject(bitmap.GetHandle())
                except Exception:
                    pass
            if save_dc is not None:
                try:
                    save_dc.DeleteDC()
                except Exception:
                    pass
            if mfc_dc is not None:
                try:
                    mfc_dc.DeleteDC()
                except Exception:
                    pass
            win32gui.ReleaseDC(hwnd, hwnd_dc)

    def _activate_window_handle_best_effort(self, hwnd: int) -> None:
        try:
            import win32con  # type: ignore[import-not-found]
            import win32gui  # type: ignore[import-not-found]
        except Exception:
            return

        try:
            if win32gui.IsIconic(hwnd):
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(hwnd)
            time.sleep(0.08)
        except Exception:
            pass

    def _image_looks_blank(self, image: Any) -> bool:
        try:
            extrema = image.convert("RGB").getextrema()
        except Exception:
            return False
        return all((high - low) <= 2 for low, high in extrema)

    def _save_with_screencapture(
        self,
        file_path: Path,
        region: Optional[Tuple[int, int, int, int]],
    ) -> str:
        command = ["screencapture", "-x"]
        source = "screencapture"
        if region is not None:
            left, top, width, height = region
            try:
                subprocess.run(
                    command + ["-R", f"{left},{top},{width},{height}", str(file_path)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return "screencapture-region"
            except Exception:
                source = "screencapture-fullscreen-fallback"

        subprocess.run(
            command + [str(file_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return source

    def _save_with_linux_tools(
        self,
        file_path: Path,
        region: Optional[Tuple[int, int, int, int]],
    ) -> str:
        if shutil.which("grim"):
            if region is not None:
                left, top, width, height = region
                try:
                    subprocess.run(
                        ["grim", "-g", f"{left},{top} {width}x{height}", str(file_path)],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    return "grim-region"
                except Exception:
                    pass
            subprocess.run(
                ["grim", str(file_path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return "grim-fullscreen-fallback" if region is not None else "grim"

        if shutil.which("gnome-screenshot"):
            subprocess.run(
                ["gnome-screenshot", "-f", str(file_path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return "gnome-screenshot"

        raise RuntimeError("no supported linux screenshot tool found")

    def _persist_image(self, image: Any, file_path: Path) -> None:
        if file_path.suffix.lower() in {".jpg", ".jpeg"} and getattr(image, "mode", "") not in {"RGB", "L"}:
            image = image.convert("RGB")
        suffix = file_path.suffix.lower()
        if suffix == ".png":
            image.save(file_path, compress_level=1)
            return
        if suffix in {".jpg", ".jpeg"}:
            image.save(file_path, quality=88)
            return
        image.save(file_path)
