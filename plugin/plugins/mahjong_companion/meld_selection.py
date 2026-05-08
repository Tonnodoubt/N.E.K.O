from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from .decision.tile_efficiency import (
    _counter_to_tile_counts,
    _estimate_standard_shanten_with_open_melds,
)
from .decision.utils import meld_group_count as _state_meld_group_count
from .frame_resources import _path_mtime
from .perception.calibration import resolve_calibration_profile
from .perception.hand_layout import TileSlot, build_hand_layout
from .perception.tile_parser import _collect_slot_metrics
from .perception.tile_classifier_dispatch import classify_tile
from .perception.tile_templates import is_probably_occupied_hand_slot
from .session_state import now_iso
from .state_transitions import _image_region_signature
from .tile_labels import normalize_tile as _normalize_tile


class MeldSelectionMixin:
    _meld_selection_pending: bool
    _pending_meld_type: str
    _pending_meld_call_tile: str
    _last_meld_selection_frame_path: str
    _meld_selection_snapshot: dict[str, Any]

    def _maybe_emit_fast_meld_selection_locked(self, frame_path: Path) -> bool:
        if not self._meld_selection_pending:
            return False
        frame_key = str(frame_path)
        if self._last_meld_selection_frame_path == frame_key:
            return False

        try:
            image = Image.open(frame_path).convert("RGB")
        except Exception:
            return False

        width, height = image.size
        layout = build_hand_layout(width, height)
        hand_slots: list[TileSlot] = layout["hand"][:14]
        calibration = resolve_calibration_profile(width, height, calibration_dir=self.plugin.data_path("calibration"))
        template_payload = (
            calibration.hand_tile_templates
            if calibration is not None
            else {}
        )
        if not template_payload:
            return False

        # Collect per-slot brightness and occupancy
        slot_metrics = _collect_slot_metrics(image, hand_slots)
        occupied_slots: list[dict[str, Any]] = []
        for slot, sm in zip(hand_slots, slot_metrics):
            if is_probably_occupied_hand_slot(sm):
                occupied_slots.append({
                    "slot": slot,
                    "mean_luma": float(sm.get("slot_mean_luma", 0) or 0),
                    "bright_ratio": float(sm.get("slot_bright_ratio", 0) or 0),
                })

        if len(occupied_slots) < 4:
            return False

        # Bimodal brightness detection
        lumas = [s["mean_luma"] for s in occupied_slots]
        max_luma = max(lumas)
        min_luma = min(lumas)
        if max_luma <= min_luma or max_luma <= 0:
            return False
        gap_ratio = (max_luma - min_luma) / max_luma
        if gap_ratio < 0.15:
            return False

        threshold = min_luma + (max_luma - min_luma) * 0.5
        highlighted_slots = [s for s in occupied_slots if s["mean_luma"] >= threshold]

        if len(highlighted_slots) < 2 or len(highlighted_slots) > 8:
            return False

        # Identify tiles in highlighted slots
        for entry in highlighted_slots:
            slot = entry["slot"]
            crop = image.crop((slot.box.left, slot.box.top, slot.box.right, slot.box.bottom))
            match = classify_tile(crop, template_payload)
            entry["tile"] = _normalize_tile(str(getattr(match, "tile", "") or "").strip())
            entry["confidence"] = float(getattr(match, "confidence", 0.0) or 0.0)

        highlighted_tiles = [s["tile"] for s in highlighted_slots if s["tile"]]
        if not highlighted_tiles:
            return False

        # Determine meld type from pending state
        meld_type = str(self._pending_meld_type or "").strip()
        if meld_type not in {"chi", "pon", "kan"}:
            return False

        # Get current hand tiles from perception state
        perceived = self._current_perceived_state()
        hand_tiles: list[str] = list(perceived.hand_tiles) if perceived is not None else []
        meld_count = _state_meld_group_count(perceived) if perceived is not None else 0

        options = _evaluate_meld_options(highlighted_tiles, hand_tiles, meld_type, meld_count)
        if not options:
            return False

        best_option = options[0]
        overlays = self._build_meld_selection_overlays(
            best_option,
            highlighted_slots,
            frame_path,
        )
        if not overlays:
            return False

        # Build decision payload for status
        tile_list_label = " ".join(best_option["tiles"])
        decision_payload: dict[str, Any] = {
            "decision_type": "meld_selection",
            "priority": 65,
            "risk_level": "medium",
            "action_required": True,
            "speakable": False,
            "summary": f"当前需要选择{best_option['meld_type']}牌组合。",
            "detail": f"当前推荐组合 {tile_list_label}，牌理上最有利。",
            "suggestion": f"选择 {tile_list_label} 继续推进手牌。",
            "recommended_focus": "meld_selection",
            "scene": "in_match",
            "buttons": [],
            "reason_codes": ["meld_selection.detected", f"meld_selection.{meld_type}"],
            "review_tags": ["meld_selection"],
            "engine_meta": {
                "screen_overlays": overlays,
                "screen_overlay_count": len(overlays),
                "meld_selection": True,
                "meld_type": meld_type,
                "meld_option_tiles": best_option["tiles"],
                "meld_post_shanten": best_option["post_shanten"],
                "meld_highlighted_tiles": highlighted_tiles,
            },
        }
        overlay_updated_at = __import__("time").monotonic()
        self.state.last_decision_at = now_iso()
        self.state.last_decision_ok = True
        self.state.last_decision_type = "meld_selection"
        self.state.last_decision_risk_level = "medium"
        self.state.last_tile_analysis_available = False
        self.state.last_shanten_estimate = None
        self.state.last_ukeire_estimate = None
        self.state.last_decision = decision_payload
        self._last_screen_overlay_update_at = overlay_updated_at
        self._last_meld_selection_frame_path = frame_key
        self._emit_status()
        return True

    def _build_meld_selection_overlays(
        self,
        option: dict[str, Any],
        highlighted_slots: list[dict[str, Any]],
        frame_path: Path,
    ) -> list[dict[str, Any]]:
        tiles = option.get("tiles") if isinstance(option.get("tiles"), list) else []
        tile_set = set(tiles)
        frame_mtime = _path_mtime(frame_path)
        overlays: list[dict[str, Any]] = []
        for entry in highlighted_slots:
            tile = entry.get("tile", "")
            if tile not in tile_set:
                continue
            slot = entry["slot"]
            local_box = {
                "left": slot.box.left,
                "top": slot.box.top,
                "width": slot.box.width,
                "height": slot.box.height,
            }
            screen_box = self._local_box_to_screen_box(local_box)
            if not screen_box:
                continue
            region_signature = _image_region_signature(frame_path, local_box)
            overlays.append({
                "kind": "meld_selection_recommendation",
                "button_type": "meld_selection",
                "label": tile,
                "box": screen_box,
                "local_box": local_box,
                "frame_path": str(frame_path),
                "frame_mtime": frame_mtime,
                "region_signature": region_signature,
                "confidence": entry.get("confidence"),
                "source": "meld_selection_scan",
                "meld_type": option.get("meld_type", ""),
                "tile": tile,
            })
        return overlays

    def _set_meld_selection_pending(self, meld_type: str, call_tile: str = "") -> None:
        self._meld_selection_pending = True
        self._pending_meld_type = meld_type
        self._pending_meld_call_tile = call_tile

    def _clear_meld_selection_pending(self) -> None:
        self._meld_selection_pending = False
        self._pending_meld_type = ""
        self._pending_meld_call_tile = ""
        self._last_meld_selection_frame_path = ""


def _evaluate_meld_options(
    highlighted_tiles: list[str],
    hand_tiles: list[str],
    meld_type: str,
    meld_count: int,
) -> list[dict[str, Any]]:
    if meld_type == "chi":
        return _enumerate_chi_options(highlighted_tiles, hand_tiles, meld_count)
    if meld_type == "pon":
        return _enumerate_pon_options(highlighted_tiles, hand_tiles, meld_count)
    if meld_type == "kan":
        return _enumerate_kan_options(highlighted_tiles, hand_tiles, meld_count)
    return []


def _enumerate_chi_options(
    highlighted_tiles: list[str],
    hand_tiles: list[str],
    meld_count: int,
) -> list[dict[str, Any]]:
    normalized_hand = [_normalize_tile(t) for t in hand_tiles]
    normalized_hand = [t for t in normalized_hand if t]
    hand_counts = Counter(normalized_hand)
    highlighted = list(dict.fromkeys(_normalize_tile(t) for t in highlighted_tiles if _normalize_tile(t)))

    options: list[dict[str, Any]] = []
    # For chi, the highlighted tiles are the ones to select from hand.
    # Mahjong Soul typically highlights 2-4 tiles for chi selection.
    # Valid chi: two hand tiles that form a sequence with the called tile.
    for i, t1 in enumerate(highlighted):
        for t2 in highlighted[i + 1:]:
            if t1 == t2:
                continue
            combo = sorted([t1, t2])
            tiles_key = tuple(combo)
            if any(o["tiles_key"] == tiles_key for o in options):
                continue
            post_shanten = _simulate_meld_shanten(
                hand_counts, combo, meld_count + 1,
            )
            if post_shanten is None:
                continue
            options.append({
                "meld_type": "chi",
                "tiles": list(combo),
                "tiles_key": tiles_key,
                "post_shanten": post_shanten,
            })

    options.sort(key=lambda o: o["post_shanten"])
    return options[:2]


def _enumerate_pon_options(
    highlighted_tiles: list[str],
    hand_tiles: list[str],
    meld_count: int,
) -> list[dict[str, Any]]:
    normalized_hand = [_normalize_tile(t) for t in hand_tiles]
    normalized_hand = [t for t in normalized_hand if t]
    hand_counts = Counter(normalized_hand)
    highlighted = [_normalize_tile(t) for t in highlighted_tiles if _normalize_tile(t)]

    options: list[dict[str, Any]] = []
    # For pon, the highlighted tiles are a pair to meld with the called tile.
    for tile in dict.fromkeys(highlighted):
        if hand_counts.get(tile, 0) < 2:
            continue
        combo = [tile, tile]
        post_shanten = _simulate_meld_shanten(
            hand_counts, combo, meld_count + 1,
        )
        if post_shanten is None:
            continue
        options.append({
            "meld_type": "pon",
            "tiles": combo,
            "tiles_key": (tile, tile),
            "post_shanten": post_shanten,
        })
        break  # Only one pon option for a given tile

    options.sort(key=lambda o: o["post_shanten"])
    return options[:1]


def _enumerate_kan_options(
    highlighted_tiles: list[str],
    hand_tiles: list[str],
    meld_count: int,
) -> list[dict[str, Any]]:
    normalized_hand = [_normalize_tile(t) for t in hand_tiles]
    normalized_hand = [t for t in normalized_hand if t]
    hand_counts = Counter(normalized_hand)
    highlighted = [_normalize_tile(t) for t in highlighted_tiles if _normalize_tile(t)]

    options: list[dict[str, Any]] = []
    for tile in dict.fromkeys(highlighted):
        if hand_counts.get(tile, 0) < 3:
            continue
        combo = [tile, tile, tile]
        post_shanten = _simulate_meld_shanten(
            hand_counts, combo, meld_count + 1,
        )
        if post_shanten is None:
            continue
        options.append({
            "meld_type": "kan",
            "tiles": combo,
            "tiles_key": (tile, tile, tile),
            "post_shanten": post_shanten,
        })
        break

    options.sort(key=lambda o: o["post_shanten"])
    return options[:1]


def _simulate_meld_shanten(
    hand_counts: Counter[str],
    meld_tiles: list[str],
    new_meld_count: int,
) -> int | None:
    after = Counter(hand_counts)
    for tile in meld_tiles:
        if after.get(tile, 0) <= 0:
            return None
        after[tile] -= 1
        if after[tile] <= 0:
            del after[tile]
    if sum(after.values()) < 2:
        return None
    counts_tuple = tuple(_counter_to_tile_counts(after))
    return _estimate_standard_shanten_with_open_melds(counts_tuple, new_meld_count)
