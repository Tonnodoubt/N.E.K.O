from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image


@dataclass
class FrameGateDecision:
    should_process: bool
    signature: str = ""
    distance: int = 0
    reason: str = ""


class FrameChangeGate(Protocol):
    def evaluate(
        self,
        frame_path: Path,
        *,
        enabled: bool = True,
        min_change_distance: int = 3,
        stable_skip_limit: int = 300,
    ) -> FrameGateDecision:
        ...


class DefaultFrameChangeGate:
    def __init__(self) -> None:
        self._last_signature = ""
        self._last_action_signature = ""
        self._stable_skip_count = 0

    def evaluate(
        self,
        frame_path: Path,
        *,
        enabled: bool = True,
        min_change_distance: int = 3,
        stable_skip_limit: int = 300,
    ) -> FrameGateDecision:
        if not enabled:
            return FrameGateDecision(should_process=True, reason="gate_disabled")
        if not frame_path.exists():
            return FrameGateDecision(should_process=True, reason="frame_missing")

        signature, action_signature = self._compute_hashes(frame_path)
        if not self._last_signature:
            self._last_signature = signature
            self._last_action_signature = action_signature
            self._stable_skip_count = 0
            return FrameGateDecision(should_process=True, signature=signature, reason="initial_frame")

        distance = self._hamming_distance(signature, self._last_signature)
        action_distance = self._hamming_distance(action_signature, self._last_action_signature)
        effective_distance = max(distance, action_distance)
        if effective_distance < max(0, int(min_change_distance)) and self._stable_skip_count < max(0, int(stable_skip_limit)):
            self._stable_skip_count += 1
            return FrameGateDecision(
                should_process=False,
                signature=signature,
                distance=effective_distance,
                reason="frame_unchanged",
            )

        self._last_signature = signature
        self._last_action_signature = action_signature
        self._stable_skip_count = 0
        return FrameGateDecision(
            should_process=True,
            signature=signature,
            distance=effective_distance,
            reason="frame_changed" if distance >= action_distance else "action_bar_changed",
        )

    def reset(self) -> None:
        self._last_signature = ""
        self._last_action_signature = ""
        self._stable_skip_count = 0

    def _compute_hashes(self, frame_path: Path) -> tuple[str, str]:
        with Image.open(frame_path) as opened:
            grayscale = opened.convert("L")
            width, height = grayscale.size

            full_frame = grayscale.resize((9, 8))
            pixels = list(full_frame.getdata())

            action_left = max(0, int(width * 0.18))
            action_top = max(0, int(height * 0.76))
            action_right = max(action_left + 9, int(width * 0.82))
            action_bottom = max(action_top + 8, int(height * 0.92))
            action_region = grayscale.crop(
                (action_left, action_top, action_right, action_bottom)
            ).resize((9, 8))
            action_pixels = list(action_region.getdata())

        bits: list[str] = []
        for row in range(8):
            offset = row * 9
            for column in range(8):
                left = pixels[offset + column]
                right = pixels[offset + column + 1]
                bits.append("1" if left > right else "0")
        full_digest = int("".join(bits), 2).to_bytes(8, "big", signed=False)

        action_bits: list[str] = []
        for row in range(8):
            offset = row * 9
            for column in range(8):
                lv = action_pixels[offset + column]
                rv = action_pixels[offset + column + 1]
                action_bits.append("1" if lv > rv else "0")
        action_digest = int("".join(action_bits), 2).to_bytes(8, "big", signed=False)

        return full_digest.hex(), action_digest.hex()

    def _hamming_distance(self, left: str, right: str) -> int:
        left_value = int(left, 16)
        right_value = int(right, 16)
        return (left_value ^ right_value).bit_count()
