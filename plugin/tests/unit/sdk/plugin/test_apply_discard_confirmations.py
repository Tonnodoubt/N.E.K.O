from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from plugin.plugins.mahjong_companion.scripts.apply_discard_confirmations import (
    apply_discard_confirmations,
    main as apply_discard_confirmations_main,
)
from plugin.plugins.mahjong_companion.scripts.prepare_discard_fixture import prepare_discard_fixture


def test_apply_discard_confirmations_updates_source_and_writes_eval_fixture(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    output_dir = tmp_path / "labeling"
    eval_dir = tmp_path / "eval"
    Image.new("RGB", (1280, 720), color=(30, 70, 120)).save(source)
    prepared = prepare_discard_fixture(image_path=source, case_id="round-1", output_dir=output_dir)

    result = apply_discard_confirmations(
        label_path=Path(prepared["label_path"]),
        confirmations=[{"player": "self", "turn_index": 1, "tile": "5z"}],
        riichi_players=["right_opponent"],
        eval_dir=eval_dir,
        overwrite=True,
    )

    assert result["ok"] is True
    assert result["label_scope"] == "partial"
    assert result["discard_count"] == 1
    source_label = json.loads(Path(result["source_label_path"]).read_text(encoding="utf-8"))
    fixture_label = json.loads(Path(result["fixture_label_path"]).read_text(encoding="utf-8"))
    assert source_label["discard_piles"]["self"][0]["tile"] == "5z"
    assert fixture_label["label_scope"] == "partial"
    assert fixture_label["discard_piles"]["self"][0]["tile"] == "5z"
    assert fixture_label["riichi_players"] == ["right_opponent"]
    assert Path(result["fixture_sheet_path"]).exists()


def test_apply_discard_confirmations_cli_accepts_tile_alias(tmp_path: Path, capsys) -> None:
    source = tmp_path / "source.png"
    output_dir = tmp_path / "labeling"
    eval_dir = tmp_path / "eval"
    Image.new("RGB", (1280, 720), color=(30, 70, 120)).save(source)
    prepared = prepare_discard_fixture(image_path=source, case_id="round-1", output_dir=output_dir)

    exit_code = apply_discard_confirmations_main(
        [
            "--label",
            prepared["label_path"],
            "--confirm",
            "self:1:E",
            "--eval-dir",
            str(eval_dir),
            "--overwrite",
            "--label-scope",
            "full",
            "--pretty",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    label = json.loads(Path(payload["fixture_label_path"]).read_text(encoding="utf-8"))
    assert payload["label_scope"] == "full"
    assert label["label_scope"] == "full"
    assert label["discard_piles"]["self"][0]["tile"] == "1z"
