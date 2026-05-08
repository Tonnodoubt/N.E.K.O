# `_example/` — Reference structure (not a real fixture)

Folders whose names start with `_` are **skipped** by the regression test.
This one exists to document the per-theme layout without contributing to
the metrics.

To add a new theme, copy this folder, rename it, drop in the screenshot and
sidecar, and the regression test will pick it up automatically.

---

## Files in a real theme folder

### `theme.json`

See `theme.json` next to this README for a fully-annotated example.

### `<case_name>.png`

A raw in-match screenshot. PNG preferred (JPG also accepted but PIL will
re-decode either way).

### `<case_name>.tiles.json` — full schema

Only `hand_tiles` is strictly required for the regression to score the case.
Everything else is optional and used to score additional dimensions:

```json
{
  "hand_tiles":          ["1m", "2m", "3m", "...", "7m"],
  "melds":               [["2p", "3p", "4p"]],
  "dora_indicators":     ["3p"],
  "riichi_players":      ["left_opponent"],
  "discard_piles": {
    "self":           [
      {"tile": "9m", "turn_index": 1},
      {"tile": "1z", "turn_index": 2}
    ],
    "left_opponent":  [{"tile": "4z", "turn_index": 1}],
    "top_opponent":   [],
    "right_opponent": []
  },
  "visible_tiles":       ["9m", "1z", "4z", "..."],
  "known_genbutsu_tiles":["4z", "..."],
  "scene":               "in_match",
  "analysis_confidence": 0.86,
  "notes":               "freeform — e.g. 'mid-game, west round, opponent in riichi'"
}
```

The schema mirrors what `perception.tile_parser._load_fixture` already
parses — see `perception/tile_parser.py` for the canonical reader.

### Tile encoding quick reference

| suit | format | example |
|---|---|---|
| 万子 (man) | `Nm` | `1m` ... `9m` |
| 筒子 (pin) | `Np` | `1p` ... `9p` |
| 索子 (sou) | `Ns` | `1s` ... `9s` |
| 字牌 (honors) | `Nz` | `1z`=東, `2z`=南, `3z`=西, `4z`=北, `5z`=白, `6z`=發, `7z`=中 |
| 赤宝牌 | `r5*` or `0*` | `0m` / `r5m` (parser normalizes to `5m`) |
