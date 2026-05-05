from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from plugin.plugins.mahjong_companion.capture import provider as capture_provider_module
from plugin.plugins.mahjong_companion.capture.provider import DefaultCaptureProvider
from plugin.plugins.mahjong_companion.window_binding import WindowBindingResult


def test_capture_provider_prefers_win32_printwindow_capture(tmp_path: Path, monkeypatch) -> None:
    calls: list[int] = []
    provider = DefaultCaptureProvider()

    def _save_with_win32(file_path: Path, hwnd: int) -> str:
        calls.append(hwnd)
        Image.new("RGB", (32, 24), "white").save(file_path)
        return "win32-printwindow"

    monkeypatch.setattr(provider, "_save_with_win32_print_window", _save_with_win32)

    packet = provider.capture_frame(
        samples_dir=tmp_path,
        binding_result=WindowBindingResult(
            bound=True,
            window_title="雀魂麻將",
            hwnd=24680,
            left=0,
            top=0,
            width=2560,
            height=1440,
        ),
        save_format="png",
    )

    assert calls == [24680]
    assert packet.source == "win32-printwindow"
    assert Path(packet.image_path).exists()


def test_capture_provider_falls_back_to_imagegrab_window_capture(tmp_path: Path, monkeypatch) -> None:
    calls: list[int] = []
    provider = DefaultCaptureProvider()

    def _grab(*, window: int):
        calls.append(window)
        return Image.new("RGB", (32, 24), "white")

    monkeypatch.setattr(capture_provider_module, "ImageGrab", SimpleNamespace(grab=_grab))
    monkeypatch.setattr(
        provider,
        "_save_with_win32_print_window",
        lambda _file_path, _hwnd: (_ for _ in ()).throw(RuntimeError("nope")),
    )

    packet = provider.capture_frame(
        samples_dir=tmp_path,
        binding_result=WindowBindingResult(
            bound=True,
            window_title="雀魂麻將",
            hwnd=24680,
            left=0,
            top=0,
            width=2560,
            height=1440,
        ),
        save_format="png",
    )

    assert calls == [24680]
    assert packet.source == "imagegrab-window"
    assert Path(packet.image_path).exists()
