"""Fixture annotator with auto tile detection + ONNX pre-fill.

Per screenshot:
  1. Auto-detect hand tile positions via brightness scanning
  2. Extract tight crops around each tile face
  3. Run ONNX ViT classifier → pre-fill predictions
  4. Browser UI: show crops + predictions, user corrects, saves .tiles.json

Usage:
    uv run python -m plugin.plugins.mahjong_companion.scripts.annotate_fixtures
"""

from __future__ import annotations

import base64
import io
import json
import threading
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[4]
_FIXTURES_ROOT = _REPO_ROOT / "plugin" / "plugins" / "mahjong_companion" / "tests" / "fixtures" / "multi_theme"
_ARTIFACTS = _REPO_ROOT / "plugin" / "plugins" / "mahjong_companion" / "tests" / "_artifacts"
PORT = 18432
ENLARGE = 3


# ---------------------------------------------------------------------------
# Tile auto-detection
# ---------------------------------------------------------------------------

def _detect_hand_tiles(img: Image.Image) -> list[dict]:
    w, h = img.size
    arr = np.array(img.convert("RGB"))

    band = arr[int(h * 0.78):int(h * 0.95), int(w * 0.12):int(w * 0.75), :].astype(float).mean(axis=(1, 2))
    if band.max() < 140:
        return []
    peak_local = int(np.argmax(band))
    peak_y = int(h * 0.78) + peak_local

    row = arr[peak_y, :, :].astype(float).mean(axis=1)
    bright = row > 155
    trans = np.diff(bright.astype(int))
    starts = np.where(trans == 1)[0] + 1
    ends = np.where(trans == -1)[0] + 1
    if len(bright) > 0 and bright[0]:
        starts = np.concatenate([[0], starts])
    if len(bright) > 0 and bright[-1]:
        ends = np.concatenate([ends, [len(bright)]])
    cols = [(int(s), int(e)) for s, e in zip(starts, ends) if 25 < (e - s) < 120]

    results = []
    for xs, xe in cols[:14]:
        y_lo, y_hi = max(0, peak_y - 50), min(h, peak_y + 50)
        col = arr[y_lo:y_hi, xs:xe, :].astype(float).mean(axis=(1, 2))
        bright_rows = np.where(col > 140)[0]
        if len(bright_rows) == 0:
            yt, yb = peak_y - 30, peak_y + 30
        else:
            yt = y_lo + max(0, int(bright_rows[0]) - 3)
            yb = y_lo + min(y_hi - y_lo, int(bright_rows[-1]) + 3)
        crop = img.crop((xs, yt, xe, yb))
        results.append({"slot": len(results) + 1, "crop": crop, "x": xs, "y": yt})
    return results


def _detect_discard_tiles(img: Image.Image, player: str) -> list[dict]:
    w, h = img.size
    sx, sy = w / 1920, h / 1080
    specs = {
        "self": {"ox": 762, "oy": 542, "tw": 58, "th": 70, "dx": 64, "dy": 70, "cols": 6, "rows": 3},
        "left_opponent": {"ox": 624, "oy": 290, "tw": 84, "th": 58, "dx": 82, "dy": 62, "cols": 3, "rows": 6},
        "top_opponent": {"ox": 802, "oy": 242, "tw": 58, "th": 70, "dx": 64, "dy": -70, "cols": 6, "rows": 3},
        "right_opponent": {"ox": 1148, "oy": 290, "tw": 84, "th": 58, "dx": 82, "dy": 62, "cols": 3, "rows": 6},
    }
    s = specs[player]
    results = []
    idx = 0
    for r in range(s["rows"]):
        for c in range(s["cols"]):
            idx += 1
            left = int((s["ox"] + c * s["dx"]) * sx)
            top = int((s["oy"] + r * s["dy"]) * sy)
            tw = max(1, int(s["tw"] * sx))
            th = max(1, int(s["th"] * sy))
            crop = img.crop((left, top, left + tw, top + th))
            results.append({"turn": idx, "crop": crop})
    return results


# ---------------------------------------------------------------------------
# ONNX classifier
# ---------------------------------------------------------------------------

def _classify(crops: list[Image.Image]) -> list[str]:
    import sys
    sys.path.insert(0, str(_REPO_ROOT))
    from plugin.plugins.mahjong_companion.perception.vit_tile_classifier_onnx import classify_tile_crops_onnx

    if not crops:
        return []
    preds = classify_tile_crops_onnx(crops, top_k=1)
    return [p.tile if p else "" for p in preds]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_b64(img: Image.Image, scale: int = ENLARGE) -> str:
    enlarged = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    enlarged.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _collect() -> list[tuple[str, Path]]:
    out = []
    for td in sorted(_FIXTURES_ROOT.iterdir()):
        if not td.is_dir() or td.name.startswith("_"):
            continue
        tid = json.loads((td / "theme.json").read_text()).get("theme_id", td.name)
        for p in sorted(td.glob("*.png")):
            out.append((tid, p))
    return out


# ---------------------------------------------------------------------------
# Data building
# ---------------------------------------------------------------------------

def _build_data(shots: list[tuple[str, Path]]) -> str:
    entries = []
    for i, (theme_id, png_path) in enumerate(shots):
        print(f"  [{i + 1}/{len(shots)}] {theme_id}/{png_path.name} ...", end="", flush=True)
        img = Image.open(png_path)
        screenshot_b64 = _to_b64(img, scale=1)

        # Hand tiles
        hand_detections = _detect_hand_tiles(img)
        hand_crops = [h["crop"] for h in hand_detections]
        hand_preds = _classify(hand_crops) if hand_crops else []
        hand_data = []
        for j, hd in enumerate(hand_detections):
            hand_data.append({
                "slot": hd["slot"],
                "b64": _to_b64(hd["crop"]),
                "pred": hand_preds[j] if j < len(hand_preds) else "",
            })

        # Discard tiles
        discard_data = {}
        for player in ("self", "left_opponent", "top_opponent", "right_opponent"):
            dd = _detect_discard_tiles(img, player)
            dcrops = [d["crop"] for d in dd]
            dpreds = _classify(dcrops) if dcrops else []
            discard_data[player] = [
                {"turn": d["turn"], "b64": _to_b64(d["crop"]), "pred": dpreds[j] if j < len(dpreds) else ""}
                for j, d in enumerate(dd)
            ]

        # Existing annotation
        sidecar = png_path.with_suffix(".tiles.json")
        existing = None
        if sidecar.exists():
            existing = json.loads(sidecar.read_text())

        entries.append({
            "theme_id": theme_id,
            "image_name": png_path.name,
            "screenshot_b64": screenshot_b64,
            "hand_data": hand_data,
            "discard_data": discard_data,
            "existing": existing,
        })
        hp = " ".join(f"{p:>3s}" for p in hand_preds)
        print(f" {len(hand_data)} tiles: {hp}")

    return json.dumps(entries, ensure_ascii=False)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

def _generate_html(shots: list[tuple[str, Path]]) -> Path:
    print("Extracting crops + running ONNX...")
    js_data = _build_data(shots)

    html = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Mahjong Fixture Annotator</title>
<style>
:root{--bg:#1a1a2e;--surface:#16213e;--accent:#e94560;--text:#eee;--muted:#888;--ok:#2ecc71;--warn:#f39c12}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:-apple-system,"Helvetica Neue",sans-serif;padding:16px}
h1{font-size:18px;margin-bottom:12px}
.nav{display:flex;gap:8px;align-items:center;margin-bottom:12px}
.nav button{background:var(--surface);color:var(--text);border:1px solid #333;padding:5px 12px;border-radius:4px;cursor:pointer}
.nav button:hover{background:#0f3460}
.nav button:disabled{opacity:.4;cursor:default}
.nav .counter{color:var(--muted);font-size:13px}
.section{background:var(--surface);border-radius:6px;padding:12px;margin-bottom:10px}
.section h2{font-size:14px;color:var(--accent);margin-bottom:6px}
.crops{display:flex;flex-wrap:wrap;gap:3px}
.crop-item{display:flex;flex-direction:column;align-items:center}
.crop-item img{border:1px solid #333}
.crop-item .label{font-size:9px;color:var(--muted);margin-top:1px}
.crop-item .label.conf-high{color:var(--ok)}
.crop-item .label.conf-low{color:var(--warn)}
.input-row{display:flex;align-items:center;gap:6px;margin-top:6px}
.input-row label{min-width:90px;font-size:12px;color:var(--muted)}
.input-row input{flex:1;background:#0a0a1a;border:1px solid #333;color:var(--text);padding:5px 8px;border-radius:4px;font-family:monospace;font-size:12px}
.input-row input::placeholder{color:#444}
.save-bar{display:flex;gap:8px;align-items:center;margin-top:10px}
.save-bar button{background:var(--accent);color:white;border:none;padding:7px 16px;border-radius:4px;cursor:pointer;font-size:13px}
.save-bar button:hover{background:#c73650}
.save-bar .status{font-size:11px;color:var(--muted)}
img.screenshot{max-width:100%;border:2px solid #333;border-radius:4px;margin-bottom:10px}
</style>
</head>
<body>
<h1>Mahjong Fixture Annotator</h1>
<p style="color:var(--muted);font-size:11px;margin-bottom:10px">ONNX auto-detected tiles pre-filled. Correct wrong ones, then Save. Green=high conf, orange=low.</p>
<div class="nav">
<button id="prevBtn" onclick="nav(-1)">&#9664; Prev</button>
<span class="counter" id="counter"></span>
<button id="nextBtn" onclick="nav(1)">Next &#9654;</button>
</div>
<div id="content"></div>
<script>
const DATA=""" + js_data + """;
let idx=0;
function nav(d){idx=Math.max(0,Math.min(DATA.length-1,idx+d));render()}
function render(){
  const d=DATA[idx];
  document.getElementById("counter").textContent=(idx+1)+"/"+DATA.length;
  document.getElementById("prevBtn").disabled=idx===0;
  document.getElementById("nextBtn").disabled=idx===DATA.length-1;
  const ex=d.existing||{};
  const useEx=ex.hand_tiles&&ex.hand_tiles.length>0;
  const handVal=useEx?ex.hand_tiles.join(" "):d.hand_data.map(h=>h.pred).filter(Boolean).join(" ");
  const doraVal=(ex.dora_indicators||[]).join(" ");
  let html='<img class="screenshot" src="data:image/png;base64,'+d.screenshot_b64+'" />';
  html+='<div class="section"><h2>Hand Tiles ('+d.hand_data.length+' detected)</h2><div class="crops">';
  for(const c of d.hand_data){
    html+='<div class="crop-item"><img src="data:image/png;base64,'+c.b64+'" /><span>#'+c.slot+' '+(c.pred||'?')+'</span></div>';
  }
  html+='</div><div class="input-row"><label>hand_tiles</label><input id="handInput" value="'+handVal+'" placeholder="e.g. 3m 4m 5m 6p 7p 1z"/></div></div>';
  html+='<div class="section"><h2>Dora</h2><div class="input-row"><label>dora_indicators</label><input id="doraInput" value="'+doraVal+'" placeholder="e.g. 3p"/></div></div>';
  const labels={self:"Self",left_opponent:"Left",top_opponent:"Top",right_opponent:"Right"};
  for(const[player,crops]of Object.entries(d.discard_data)){
    const exT=useEx&&ex.discard_piles&&ex.discard_piles[player]?ex.discard_piles[player].map(t=>t.tile).join(" "):"";
    html+='<div class="section"><h2>Discard: '+(labels[player]||player)+' ('+crops.length+' slots)</h2>';
    html+='<div class="crops" style="max-height:200px;overflow-y:auto">';
    for(const c of crops){
      html+='<div class="crop-item"><img src="data:image/png;base64,'+c.b64+'" /><span>T'+c.turn+'</span></div>';
    }
    html+='</div><div class="input-row"><label>'+player+'</label><input id="discard_'+player+'" value="'+exT+'" placeholder="tiles or empty"/></div></div>';
  }
  html+='<div class="save-bar"><button onclick="save()">Save .tiles.json</button><span class="status" id="saveStatus"></span></div>';
  document.getElementById("content").innerHTML=html;
}
function parse(r){return r?r.trim().split(/\\s+/).filter(Boolean):[]}
function save(){
  const d=DATA[idx];
  const hand=parse(document.getElementById("handInput").value);
  const dora=parse(document.getElementById("doraInput").value);
  const piles={};
  for(const p of["self","left_opponent","top_opponent","right_opponent"]){
    const el=document.getElementById("discard_"+p);
    const tiles=parse(el?el.value:"");
    piles[p]=tiles.map((t,i)=>({tile:t,turn_index:i+1}));
  }
  const payload={theme_id:d.theme_id,source_image:d.image_name,hand_tiles:hand,discard_piles:piles,dora_indicators:dora};
  const blob=new Blob([JSON.stringify(payload,null,2)+"\\n"],{type:"application/json"});
  const a=document.createElement("a");
  a.href=URL.createObjectURL(blob);
  a.download=d.image_name.replace(/\\.png$/,".tiles.json");
  a.click();
  URL.revokeObjectURL(a.href);
  document.getElementById("saveStatus").textContent="Downloaded "+a.download;
}
render();
</script>
</body></html>"""

    out = _ARTIFACTS / "annotate_all.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class _H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(_ARTIFACTS), **kw)
    def log_message(self, *a):
        pass


def main() -> None:
    shots = _collect()
    if not shots:
        print("No screenshots found.")
        return
    print(f"Found {len(shots)} screenshots.")
    html_path = _generate_html(shots)
    size_mb = html_path.stat().st_size / 1024 / 1024
    print(f"\nHTML: {html_path} ({size_mb:.1f} MB)")

    server = HTTPServer(("127.0.0.1", PORT), _H)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    url = f"http://127.0.0.1:{PORT}/{html_path.name}"
    print(f"Serving at {url}")
    webbrowser.open(url)
    print("Ctrl+C to stop.")
    try:
        t.join()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
