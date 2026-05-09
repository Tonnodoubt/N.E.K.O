"""Simplified anchor measurement: 4 clicks per screenshot.

For each screenshot, shows the current grid position (red boxes) for each
player's first discard tile. User clicks where the actual tile is.

Only 4 points: 自家(绿框→点击修正), 左家(蓝), 对家(橙), 右家(红)
"""

from __future__ import annotations

import base64
import io
import json
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_REPO_ROOT = Path(__file__).resolve().parents[4]
FIXTURES = _REPO_ROOT / "plugin/plugins/mahjong_companion/tests/fixtures/multi_theme"
PORT = 18434

PLAYERS = [
    {"id": "self", "label": "自家牌河", "color": "#2ecc71", "desc": "下方自己的牌河"},
    {"id": "left", "label": "左家牌河", "color": "#3498db", "desc": "左侧对手牌河"},
    {"id": "top", "label": "对家牌河", "color": "#e67e22", "desc": "上方对手牌河"},
    {"id": "right", "label": "右家牌河", "color": "#e74c3c", "desc": "右侧对手牌河"},
]

CLICKS = [
    {"id": "self_first", "player": "self", "label": "自家牌河 第1张", "color": "#2ecc71", "desc": "下方自己打出的第一张牌，左上角"},
    {"id": "self_last", "player": "self", "label": "自家牌河 最后1张", "color": "#2ecc71", "desc": "下方自己打出的最后一张牌，左上角"},
    {"id": "left_first", "player": "left", "label": "左家牌河 第1张", "color": "#3498db", "desc": "左侧对手打出的第一张牌，左上角"},
    {"id": "left_last", "player": "left", "label": "左家牌河 最后1张", "color": "#3498db", "desc": "左侧对手打出的最后一张牌，左上角"},
    {"id": "top_first", "player": "top", "label": "对家牌河 第1张", "color": "#e67e22", "desc": "上方对手打出的第一张牌，左上角"},
    {"id": "top_last", "player": "top", "label": "对家牌河 最后1张", "color": "#e67e22", "desc": "上方对手打出的最后一张牌，左上角"},
    {"id": "right_first", "player": "right", "label": "右家牌河 第1张", "color": "#e74c3c", "desc": "右侧对手打出的第一张牌，左上角"},
    {"id": "right_last", "player": "right", "label": "右家牌河 最后1张", "color": "#e74c3c", "desc": "右侧对手打出的最后一张牌，左上角"},
]


def _grid_origin(player_id: str) -> tuple[int, int]:
    """Current hardcoded grid origin for each player's first discard slot."""
    origins = {
        "self": (762, 542),
        "left": (624, 290),
        "top": (802, 242),
        "right": (1148, 290),
    }
    return origins.get(player_id, (0, 0))


def _draw_grid_markers(img: Image.Image) -> str:
    """Draw current grid origin markers on image, return base64."""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 18)
    except OSError:
        font = ImageFont.load_default()

    for p in PLAYERS:
        ox, oy = _grid_origin(p["id"])
        color = p["color"]
        # Draw a bigger box (80x95) at the grid origin for visibility
        draw.rectangle([ox, oy, ox + 80, oy + 95], outline=color, width=3)
        # Crosshair at origin
        draw.line([ox - 12, oy, ox + 12, oy], fill=color, width=2)
        draw.line([ox, oy - 12, ox, oy + 12], fill=color, width=2)
        draw.text((ox + 85, oy + 10), f"网格:{p['label']}", fill=color, font=font)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


def _collect() -> list[dict]:
    shots = []
    for td in sorted(FIXTURES.iterdir()):
        if not td.is_dir() or td.name.startswith("_"):
            continue
        for p in sorted(td.glob("*.png")):
            img = Image.open(p).convert("RGB")
            annotated_b64 = _draw_grid_markers(img.copy())
            shots.append({
                "theme": td.name,
                "file": p.name,
                "b64": annotated_b64,
                "w": img.width,
                "h": img.height,
            })
    return shots


def _build_html(shots: list[dict]) -> str:
    clicks_js = json.dumps(CLICKS, ensure_ascii=False)
    shots_js = json.dumps(shots, ensure_ascii=False)
    grid_js = json.dumps({p["id"]: list(_grid_origin(p["id"])) for p in PLAYERS})

    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Discard Grid Measurement</title>
<style>
:root{{--bg:#1a1a2e;--surface:#16213e;--accent:#e94560;--text:#eee;--muted:#888;--ok:#2ecc71}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:-apple-system,"Helvetica Neue",sans-serif;padding:12px}}
h1{{font-size:15px;margin-bottom:6px}}
.subtitle{{color:var(--muted);font-size:12px;margin-bottom:8px}}
.nav{{display:flex;gap:8px;align-items:center;margin-bottom:8px}}
.nav button{{background:var(--surface);color:var(--text);border:1px solid #333;padding:4px 12px;border-radius:4px;cursor:pointer}}
.nav button:hover{{background:#0f3460}}
.nav button:disabled{{opacity:.4;cursor:default}}
.counter{{color:var(--muted);font-size:13px}}
.prompt{{background:var(--surface);border-radius:6px;padding:8px 12px;margin-bottom:8px}}
.prompt .player{{font-weight:bold;font-size:15px}}
.prompt .desc{{color:var(--muted);font-size:12px;margin-top:2px}}
.canvas-wrap{{position:relative;display:inline-block;cursor:crosshair}}
.canvas-wrap canvas{{max-width:100%;border:1px solid #333;border-radius:4px}}
.legend{{margin-top:8px;font-size:12px;color:var(--muted)}}
.save-bar{{margin-top:10px;display:flex;gap:8px;align-items:center}}
.save-bar button{{background:var(--accent);color:white;border:none;padding:6px 16px;border-radius:4px;cursor:pointer;font-size:13px}}
.save-bar .msg{{font-size:12px;color:var(--muted)}}
</style>
</head>
<body>
<h1>牌河定位测量 — 每张图点 8 下</h1>
<div class="subtitle">彩色框是当前网格位置。每家牌河点第1张和最后1张牌的左上角。</div>
<div class="nav">
<button id="prev" onclick="nav(-1)">&#9664; 上一张</button>
<span class="counter" id="counter"></span>
<button id="next" onclick="nav(1)">下一张 &#9654;</button>
<button onclick="exportJSON()" style="margin-left:16px;background:var(--ok)">导出 JSON</button>
</div>
<div class="prompt" id="prompt"></div>
<div class="canvas-wrap"><canvas id="canvas"></canvas></div>
<div class="legend">← → 键切换截图 | 点错可以重新点覆盖 | 每家点第1张+最后1张共8下</div>

<script>
const P = {clicks_js};
const SHOTS = {shots_js};
const GRID = {grid_js};
let idx = 0;
let data = {{}}; // "theme/file" -> {{player_id: [x,y]}}
let curP = 0;

for(const s of SHOTS) data[s.theme+"/"+s.file] = {{}};

function nav(d) {{
  idx = Math.max(0, Math.min(SHOTS.length-1, idx+d));
  curP = nextUndone();
  render();
}}

function nextUndone() {{
  const k = SHOTS[idx].theme+"/"+SHOTS[idx].file;
  for(let i=0;i<P.length;i++) if(!data[k][P[i].id]) return i;
  return P.length;
}}

function render() {{
  const s = SHOTS[idx], k = s.theme+"/"+s.file;
  document.getElementById("counter").textContent = (idx+1)+"/"+SHOTS.length+" "+s.theme+"/"+s.file;
  document.getElementById("prev").disabled = idx===0;
  document.getElementById("next").disabled = idx===SHOTS.length-1;

  curP = nextUndone();
  const allDone = curP >= P.length;
  const prompt = document.getElementById("prompt");
  if(allDone) {{
    prompt.innerHTML = '<span class="player" style="color:var(--ok)">8 个点都点完了!</span> <span class="desc">→ 键下一张</span>';
  }} else {{
    const p = P[curP];
    prompt.innerHTML = '<span class="player" style="color:'+p.color+'">[' +(curP+1)+'/8] '+p.label+'</span><div class="desc">'+p.desc+'</div>';
  }}

  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");
  const img = new Image();
  img.onload = function() {{
    const scale = Math.min(1, 1200/s.w);
    canvas.width = s.w*scale; canvas.height = s.h*scale;
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    // Draw user clicks
    const d2 = data[k];
    for(let i=0;i<P.length;i++) {{
      const pt = d2[P[i].id];
      if(pt) {{
        const x=pt[0]*scale, y=pt[1]*scale;
        ctx.beginPath(); ctx.arc(x,y,5,0,Math.PI*2);
        ctx.fillStyle=P[i].color; ctx.fill();
        ctx.strokeStyle="#000"; ctx.lineWidth=1; ctx.stroke();
        ctx.fillStyle=P[i].color; ctx.font="bold 11px sans-serif";
        ctx.fillText(P[i].label, x+8, y+4);
      }}
    }}
  }};
  img.src = "data:image/jpeg;base64,"+s.b64;
}}

document.getElementById("canvas").addEventListener("click", function(e) {{
  const rect=this.getBoundingClientRect(), s=SHOTS[idx], k=s.theme+"/"+s.file;
  const scale=Math.min(1,1200/s.w);
  const x=Math.round((e.clientX-rect.left)/scale), y=Math.round((e.clientY-rect.top)/scale);
  if(curP < P.length) {{
    data[k][P[curP].id] = [x,y];
    curP = nextUndone();
    render();
  }}
}});

// Also allow clicking any player to redo it
document.getElementById("canvas").addEventListener("contextmenu", function(e) {{
  e.preventDefault();
  const rect=this.getBoundingClientRect(), s=SHOTS[idx], k=s.theme+"/"+s.file;
  const scale=Math.min(1,1200/s.w);
  const x=Math.round((e.clientX-rect.left)/scale), y=Math.round((e.clientY-rect.top)/scale);
  // Find closest player marker
  let bestDist=Infinity, bestIdx=-1;
  for(let i=0;i<P.length;i++) {{
    const pt=data[k][P[i].id];
    if(pt) {{
      const d=Math.hypot(pt[0]-x, pt[1]-y);
      if(d<bestDist) {{bestDist=d; bestIdx=i;}}
    }}
  }}
  if(bestIdx>=0 && bestDist<40) {{
    data[k][P[bestIdx].id] = [x,y];
    render();
  }}
}});

document.addEventListener("keydown", function(e) {{
  if(e.key==="ArrowLeft") nav(-1);
  if(e.key==="ArrowRight") nav(1);
}});

function exportJSON() {{
  const out=[];
  for(const s of SHOTS) {{
    const k=s.theme+"/"+s.file, d=data[k]||{{}};
    const pts={{}};
    for(const c of P) {{
      const pt=d[c.id];
      pts[c.id] = pt ? {{x:pt[0], y:pt[1]}} : null;
    }}
    out.push({{theme:s.theme, file:s.file, image_size:[s.w,s.h], points:pts}});
  }}
  const jsonStr = JSON.stringify(out, null, 2);

  // Try download
  try {{
    const blob=new Blob([jsonStr],{{type:"application/json"}});
    const a=document.createElement("a");
    a.href=URL.createObjectURL(blob); a.download="discard_offsets.json"; a.click();
    URL.revokeObjectURL(a.href);
  }} catch(e) {{}}

  // Also show in textarea
  let area = document.getElementById("jsonOutput");
  if(!area) {{
    area = document.createElement("textarea");
    area.id = "jsonOutput";
    area.style.cssText = "width:100%;height:300px;margin-top:12px;font-size:11px;font-family:monospace;background:#0a0a1a;color:#eee;border:1px solid #333;padding:8px;border-radius:4px";
    document.body.appendChild(area);
    const btn2 = document.createElement("button");
    btn2.textContent = "复制到剪贴板";
    btn2.onclick = function() {{ area.select(); document.execCommand("copy"); btn2.textContent = "已复制!"; }};
    btn2.style.cssText = "margin-top:8px;padding:6px 16px;background:var(--ok);color:#fff;border:none;border-radius:4px;cursor:pointer";
    document.body.appendChild(btn2);
  }}
  area.value = jsonStr;
}}

render();
</script>
</body></html>"""


class _H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory="/tmp", **kw)
    def log_message(self, *a):
        pass


def main():
    print("Loading screenshots and drawing grid markers...")
    shots = _collect()
    if not shots:
        print("No screenshots found.")
        return

    html = _build_html(shots)
    html_path = Path("/tmp/measure_discard.html")
    html_path.write_text(html, encoding="utf-8")
    size_mb = html_path.stat().st_size / 1024 / 1024
    print(f"HTML ready: {size_mb:.1f} MB")

    server = HTTPServer(("127.0.0.1", PORT), _H)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    url = f"http://127.0.0.1:{PORT}/measure_discard.html"
    print(f"Serving at {url}")
    import webbrowser
    webbrowser.open(url)
    print("每张图点 8 下（每家牌河第1张+最后1张的左上角）。← → 切换。右键可修正已有点。")
    print("Ctrl+C to stop.")
    try:
        t.join()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
