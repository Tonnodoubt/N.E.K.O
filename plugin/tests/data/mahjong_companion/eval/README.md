# mahjong_companion Eval Fixtures

This directory is the release gate fixture root for v0.3, v0.4, and v0.5:

```bash
.venv/bin/python -m plugin.plugins.mahjong_companion.scripts.evaluate_v03 \
  --eval-dir plugin/tests/data/mahjong_companion/eval \
  --calibration-dir plugin/plugins/mahjong_companion/data/calibration \
  --strict-json

.venv/bin/python -m plugin.plugins.mahjong_companion.scripts.evaluate_v04 \
  --eval-dir plugin/tests/data/mahjong_companion/eval \
  --calibration-dir plugin/plugins/mahjong_companion/data/calibration \
  --template-dir plugin/plugins/mahjong_companion/perception/templates \
  --strict

.venv/bin/python -m plugin.plugins.mahjong_companion.scripts.evaluate_v05 \
  --eval-dir plugin/tests/data/mahjong_companion/eval \
  --calibration-dir plugin/plugins/mahjong_companion/data/calibration \
  --strict

.venv/bin/python -m plugin.plugins.mahjong_companion.scripts.evaluate_v10_release \
  --eval-dir plugin/tests/data/mahjong_companion/eval \
  --calibration-dir plugin/plugins/mahjong_companion/data/calibration \
  --template-dir plugin/plugins/mahjong_companion/perception/templates \
  --pretty
```

Use `--strict-json` while the independent hand screenshot set is still incomplete. Use full `--strict` only for the v0.3 release gate after hand fixtures are present.

## hand_recognition

Store screenshots and labels under resolution folders:

```text
hand_recognition/
  1920x1080/
    frame-001.png
    frame-001.label.json
```

The evaluator uses `hand_tiles` or `expected_hand_tiles` as the expected result. By default it disables sidecar fixture loading while parsing the image, so `.label.json` is only the answer key, not the prediction source.
Pass `--calibration-dir` when evaluating real screenshots so the parser can load the matching trained profile.
Add `--details` when diagnosing a new screenshot batch; the report will include per-image expected/predicted hands, slot mismatches, parser source, confidence, and top confusion pairs.

For local calibration validation without adding screenshots to git, run k-fold holdout against raw labels:

```bash
.venv/bin/python -m plugin.plugins.mahjong_companion.scripts.evaluate_v03 \
  --hand-holdout-dir plugin/plugins/mahjong_companion/data/calibration/raw/manual/屏幕截图\(33\) \
  --holdout-folds 5 \
  --strict-hand \
  --details \
  --pretty
```

The holdout report includes raw accuracy plus coverage-adjusted accuracy. Coverage-adjusted metrics exclude mismatches where the expected tile never appeared in that fold's training labels, so they separate "need more samples" from real template confusion.
Reports also include `red5_normalized_*` metrics. Those keep strict `R5m/R5p/R5s` scoring intact, but additionally score red fives as ordinary `5m/5p/5s` so calibration work can separate red-five exactness from normal tile-rank recognition.

## decision_top1

Each `*.json` case describes a hand state and the expected top discard:

```json
{
  "hand_tiles": ["1m", "2m", "3m"],
  "dora_indicators": [],
  "expected_top1": "9m"
}
```

Current checked-in seed set: 6 JSON cases covering isolated honors, floating terminals, complete-deck ukeire ordering, red-five normalization, and riichi pressure.

## risk_detection

Each `*.json` case describes a state under riichi pressure and the expected confirmed safe tiles:

```json
{
  "hand_tiles": ["1m", "2m", "3m"],
  "riichi_players": ["right_opponent"],
  "expected_genbutsu": true,
  "expected_genbutsu_tiles": ["5z"]
}
```

Current checked-in seed set: 4 JSON cases, including 3 confirmed-genbutsu recall cases and 1 riichi-without-genbutsu control case.

## review_patterns

Each `*.json` case can check structured review summaries, repeated cross-session patterns, or both:

```json
{
  "candidates": [
    {
      "decision_type": "tile_efficiency_hint",
      "priority": 72,
      "risk_level": "medium",
      "recommended_focus": "tile_efficiency",
      "review_tags": ["tile_efficiency"]
    }
  ],
  "review_summaries": [
    {"session_id": "a", "training_points": ["中盘牌效率弃牌优先级"]},
    {"session_id": "b", "training_points": ["中盘牌效率弃牌优先级"]}
  ],
  "expected_repeated_patterns": ["tile_efficiency"]
}
```

Current checked-in seed set: 2 JSON cases covering structured review output and repeated tile-efficiency trend detection.

## button_localization

Each `*.label.json` case points to a frame and expected button bboxes:

```json
{
  "image": {"path": "frame.png"},
  "buttons": [
    {"button_type": "kan", "bbox": [860, 760, 1155, 890]},
    {"button_type": "skip", "bbox": [1130, 760, 1505, 890]}
  ]
}
```

Current checked-in seed set: 6 cases covering all advice-relevant v0.4 templates: `chi`, `pon`, `kan`, `riichi`, `ron`, `tsumo`, and `skip`. The evaluator reports IoU pass rate, precision, and recall.
Use `--strict-templates` to fail the report if any required advice-relevant in-match button template is missing.

When a new screenshot contains a missing button, use `prepare_button_template` to crop the template, update `perception/templates/meta.json`, and optionally create a seed localization case:

```bash
.venv/bin/python -m plugin.plugins.mahjong_companion.scripts.prepare_button_template \
  --image plugin/plugins/mahjong_companion/data/calibration/raw/manual/example.png \
  --button-type chi \
  --bbox 800,735,1080,875 \
  --padding 5 \
  --write-fixture \
  --fixture-case-id chi-skip \
  --fixture-button skip:1100,735,1475,865 \
  --pretty
```

Use `--overwrite` only when intentionally replacing an existing template or fixture.

## discard_recognition

Each `frame.label.json` case points to a frame and expected discard pile items:

```json
{
  "image": {"path": "frame.png"},
  "label_scope": "partial",
  "discard_piles": {
    "right_opponent": [
      {
        "tile": "5z",
        "turn_index": 2,
        "bbox": [1148, 352, 1232, 410],
        "quad": [[1148, 352], [1148, 410], [1232, 410], [1232, 352]],
        "orientation": "right"
      }
    ]
  }
}
```

`label_scope` defaults to `full`, meaning unlabelled slots are treated as known empty slots and extra predictions count as false positives. Use `partial` for sampled labels, such as "only mark visible white dragons"; in that mode the evaluator still scores recall and tile accuracy for labelled slots, but ignores predictions in unlabelled slots unless they are listed in `empty_discard_slots`, `negative_discard_slots`, or `known_empty_discard_slots`.

`prepare_discard_fixture`, `prepare_discard_labeling_batch`, and `apply_discard_confirmations` default new labels to `partial`. Pass `--label-scope full` only when the whole river has been exhaustively labelled.

The v0.5 evaluator reports aggregate metrics plus `by_player` and `by_orientation` breakdowns. Partial-label predictions in unlabelled slots are tracked as `ignored_unlabeled_prediction_count`, so a direction with many ignored predictions still needs full labels before its false-positive quality can be claimed. The report also includes `coverage_warnings` for partial-only fixture sets and for players/orientations that have predictions but no labelled expected slots.

The parser may emit refined four-point `quad` values from visible tile surfaces. For model training or external detector experiments, export the fixture set as JSONL:

```bash
.venv/bin/python -m plugin.plugins.mahjong_companion.scripts.export_discard_recognition_dataset \
  --label-root plugin/tests/data/mahjong_companion/eval/discard_recognition \
  --output-dir plugin/plugins/mahjong_companion/plans/artifacts/discard_recognition_dataset/v0.5-eval \
  --refine-quads \
  --pretty
```

For side/top river alignment review, generate quad overlays and refined slot sheets. The JSON and sheet include both fitted quad candidates and parser `accepted` markers, so review can separate "surface found" from "entered discard_piles":

```bash
.venv/bin/python -m plugin.plugins.mahjong_companion.scripts.prepare_discard_quad_review \
  --input-dir plugin/plugins/mahjong_companion/data/calibration/raw/manual/屏幕截图\(33\)/屏幕截图\(115\) \
  --output-dir plugin/plugins/mahjong_companion/plans/artifacts/discard_quad_review/screenshot-115-folder \
  --include-empty-slots \
  --overwrite \
  --pretty
```

To inspect how accepted side/top crops are cut per screenshot, generate a crop debug index from the quad review artifacts:

```bash
.venv/bin/python -m plugin.plugins.mahjong_companion.scripts.prepare_discard_crop_debug \
  --quad-review-dir plugin/plugins/mahjong_companion/plans/artifacts/discard_quad_review/screenshot-115-folder \
  --output-dir plugin/plugins/mahjong_companion/plans/artifacts/discard_crop_debug/screenshot-115-opponents \
  --pretty
```

The crop debug output writes `discard-crop-debug.md` plus per-case `accepted-quads-on-frame.png`, `left-accepted-crops.png`, `top-accepted-crops.png`, and `right-accepted-crops.png`.

To turn a coverage gap into a focused manual review sheet, generate a gap review from the quad review artifacts:

```bash
.venv/bin/python -m plugin.plugins.mahjong_companion.scripts.prepare_discard_gap_review \
  --quad-review-dir plugin/plugins/mahjong_companion/plans/artifacts/discard_quad_review/screenshot-115-folder \
  --output-dir plugin/plugins/mahjong_companion/plans/artifacts/discard_gap_review/screenshot-115-top \
  --player top_opponent \
  --pretty
```

The gap review writes `discard-gap-candidates.png`, stable candidate IDs such as `top-001`, per-candidate crops under `candidates/`, and a markdown checklist with confirmation commands. Use the IDs when reviewing screenshots manually, then only promote confirmed candidates into partial fixtures.

After manual review, candidate IDs can be applied without retyping `player:turn:tile` specs:

```bash
.venv/bin/python -m plugin.plugins.mahjong_companion.scripts.apply_discard_candidate_confirmations \
  --review plugin/plugins/mahjong_companion/plans/artifacts/discard_gap_review/screenshot-115-opponents/discard-gap-review.json \
  --write-template plugin/plugins/mahjong_companion/plans/artifacts/discard_gap_review/screenshot-115-opponents/candidate-confirmations-template.json \
  --dry-run \
  --pretty
```

Use `--accept top-034`, `--correct top-034=8p`, `--reject top-034`, or pass an edited confirmation JSON via `--confirmations`. Use `--accept-all` only after the whole candidate sheet has been manually reviewed.

Current checked-in seed set: 29 partial-label cases and 71 labeled discard items. The v0.5 closeout gate scores 71/71 labelled discards correctly, including top 39/39, left 7/7, and right 13/13. Because the set still uses partial labels, fresh real-game screenshots remain useful as holdout coverage, but they are not a v0.5 blocker.

## audit_chain

Each `*.json` file contains either one action log entry or an `entries` list. The v0.4 evaluator checks that entries include locator source, target coordinates, risk level, and a non-empty confirmation chain.
