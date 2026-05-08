# Multi-Theme Background Invariance Fixtures

This directory holds screenshots from **different table backgrounds / UI themes**
that the perception pipeline must recognize without retraining or re-calibrating.

The `test_background_invariance.py` regression in
`tests/perception/` walks every theme here, runs the real perception pipeline
against each screenshot, and compares the predicted hand / discards / scene
against the ground-truth sidecar.

If this directory is empty (no themes), the regression test **skips cleanly**
and prints an informational message — so it is safe to merge code first and
populate the fixtures incrementally.

---

## Directory layout

```
fixtures/multi_theme/
├── README.md                   <- you are here
├── _example/                   <- structural reference, no actual image
│   ├── README.md
│   └── theme.json
├── default_blue/               <- one directory per theme
│   ├── theme.json              <- theme metadata (REQUIRED)
│   ├── case_001.png            <- screenshot
│   ├── case_001.tiles.json     <- ground truth sidecar (REQUIRED)
│   ├── case_002.png
│   └── case_002.tiles.json
├── seasonal_sakura/
│   └── ...
└── workshop_woodgrain/
    └── ...
```

Folder name = theme id. Use lowercase ASCII + underscores
(e.g. `default_blue`, `seasonal_sakura_2024`, `workshop_woodgrain_dark`).

Folders whose name starts with `_` (like `_example`) are skipped — reserve
the `_` prefix for documentation/example folders, not real fixtures.

---

## What you need to provide per theme

For each theme directory:

### 1. `theme.json` — required, one per theme

```json
{
  "theme_id": "default_blue",
  "display_name": "默认蓝桌布",
  "is_official": true,
  "screen_width": 1920,
  "screen_height": 1080,
  "notes": "雀魂默认桌布，开发主要校准对象"
}
```

Fields:

| field | required | meaning |
|---|---|---|
| `theme_id` | yes | must equal the folder name |
| `display_name` | yes | human-readable name (CN/EN/JP fine) |
| `is_official` | no | `true` for in-game shipped backgrounds; `false` for community/workshop |
| `screen_width` / `screen_height` | no | source resolution; report-only — pipeline reads it from the PNG |
| `notes` | no | anything worth remembering (season, event, contributor, etc.) |

### 2. `<case_name>.png` + `<case_name>.tiles.json` — at least one pair per theme

The PNG is a raw screenshot from an **in-match scene** (`scene == "in_match"`).
Lobby / result / replay screens don't exercise the recognition path we care
about and should be put in a different fixture directory if needed later.

The `.tiles.json` sidecar uses the **same schema already consumed by**
`perception/tile_parser.py:_load_fixture()`. Minimum viable example:

```json
{
  "hand_tiles": ["1m", "2m", "3m", "5p", "5p", "6p", "7p", "2s", "3s", "4s", "5z", "5z", "6z", "7m"],
  "discard_piles": {
    "self":           [{"tile": "9m"}, {"tile": "1z"}],
    "left_opponent":  [{"tile": "4z"}],
    "top_opponent":   [{"tile": "9p"}, {"tile": "8s"}],
    "right_opponent": []
  },
  "riichi_players": [],
  "dora_indicators": ["3p"]
}
```

Optional fields the regression also looks at if present:
- `scene` — expected scene label (defaults to `"in_match"` if omitted)
- `melds`, `visible_tiles`, `known_genbutsu_tiles`, `analysis_confidence`

The full schema is documented at `_example/README.md` next to this file.

---

## What makes a useful background invariance corpus

Aim for **breadth** before **depth**. 30 themes × 1 screenshot each is more
useful than 3 themes × 10 screenshots — we want to surface theme-specific
failures, not in-theme variance.

Suggested first batch (cover the dimensions that stress the pipeline most):

| dimension | examples |
|---|---|
| color hue | blue (default), green, red, purple, gold, monochrome |
| brightness | dark theme, light theme, high-contrast |
| pattern | solid color, fabric texture, wood grain, marble, illustrated |
| seasonal / event skins | sakura, summer fireworks, halloween, etc. |
| workshop / community | any community-uploaded skin you can find |

For each screenshot, prefer:
- **In-match, your turn (14 tiles)** — exercises both hand recognition AND
  discard pile recognition.
- **Mid-game** — at least 3-4 discards on each opponent side, so the discard
  recall metric is meaningful.
- **No animation in flight** — capture between turns, not while a tile is
  flying out.

---

## How to verify a fixture is valid before committing

```bash
uv run python -m plugin.plugins.mahjong_companion.scripts.audit_background_coupling \
    plugin/plugins/mahjong_companion/tests/fixtures/multi_theme/<theme_id>/<case_name>.png
```

Then run just the regression for that theme:

```bash
uv run pytest plugin/plugins/mahjong_companion/tests/perception/test_background_invariance.py -v
```

The test prints a per-theme accuracy table; new themes will appear there
automatically.
