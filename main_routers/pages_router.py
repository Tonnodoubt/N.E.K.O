# -*- coding: utf-8 -*-
"""
Pages Router

Handles HTML page rendering endpoints.
"""

import io
import socket
from typing import Optional
from urllib.parse import quote

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from config import MAIN_SERVER_PORT

from .shared_state import get_config_manager, get_templates

router = APIRouter(tags=["pages"])


def _get_lan_ip() -> Optional[str]:
    """Best-effort LAN IP discovery (non-loopback IPv4)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            if ip and not ip.startswith("127."):
                return ip
    except Exception:
        pass

    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None):
            ip = info[4][0]
            if ip and "." in ip and not ip.startswith("127."):
                return ip
    except Exception:
        pass

    return None


def _get_default_character_name() -> str:
    try:
        config_manager = get_config_manager()
        _, her_name, _, _, _, _, _, _, _, _ = config_manager.get_character_data()
        if isinstance(her_name, str) and her_name.strip():
            return her_name.strip()
    except Exception:
        pass
    return "test"


@router.get("/getipqrcode")
async def get_ip_qrcode(
    request: Request,
    name: str = "",
    host: str = "",
    port: Optional[int] = None,
    format: str = "",
    scheme: str = "",
    path: str = "",
):
    """Return QR code image for RN dev connection config."""
    try:
        import qrcode
    except Exception:
        return JSONResponse(
            {
                "message": "QR 码依赖未安装，请先执行 `uv sync` 或 `pip install qrcode`。",
            },
            status_code=200,
        )

    resolved_host = host.strip() if host else ""
    if not resolved_host:
        resolved_host = _get_lan_ip() or ""

    if not resolved_host or resolved_host.startswith("127."):
        return JSONResponse(
            {
                "message": "无法获取局域网 IP，请确认网络连接或改用手动输入。",
            },
            status_code=200,
        )

    resolved_port = port if isinstance(port, int) and 1 <= port <= 65535 else MAIN_SERVER_PORT
    resolved_name = name.strip() if name else _get_default_character_name()
    encoded_name = quote(resolved_name)

    format_value = (format or "").strip().lower()
    use_deeplink = format_value in {"deeplink", "deep", "app", "rn", "mobile"}
    if use_deeplink:
        safe_scheme = scheme.strip() if scheme else "nekorn"
        safe_path = (path.strip() if path else "main").lstrip("/")
        query = f"host={resolved_host}&port={resolved_port}&name={encoded_name}"
        if safe_path:
            access_url = f"{safe_scheme}://{safe_path}?{query}"
        else:
            access_url = f"{safe_scheme}:///?{query}"
    else:
        access_url = f"{resolved_host}:{resolved_port}?name={encoded_name}"

    try:
        img = qrcode.make(access_url)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        headers = {
            "X-Neko-Access-Url": access_url,
            "Cache-Control": "no-store",
        }
        return Response(content=buf.getvalue(), media_type="image/png", headers=headers)
    except Exception as e:
        return JSONResponse({"message": f"二维码生成失败: {e}"}, status_code=200)


@router.get("/qr", response_class=HTMLResponse)
async def get_qr_page(request: Request):
    """Render a lightweight QR-only page for mobile app connection."""
    params = request.query_params
    name = params.get("name", "")
    host = params.get("host", "")
    port = params.get("port", "")
    format_value = params.get("format", "deeplink")
    scheme = params.get("scheme", "nekorn")
    path = params.get("path", "main")

    query_parts = [
        ("format", format_value),
        ("scheme", scheme),
        ("path", path),
    ]
    if name:
        query_parts.append(("name", name))
    if host:
        query_parts.append(("host", host))
    if port:
        query_parts.append(("port", port))

    query = "&".join(f"{k}={quote(str(v))}" for k, v in query_parts if v)
    img_src = f"/getipqrcode?{query}" if query else "/getipqrcode"

    html = f"""
    <!doctype html>
    <html lang="zh-CN">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>N.E.K.O QR</title>
        <style>
          body {{ margin: 0; font-family: system-ui, sans-serif; background: #0b0b0b; color: #f2f2f2; }}
          .wrap {{ min-height: 100vh; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 16px; padding: 24px; }}
          .card {{ background: #151515; border-radius: 16px; padding: 24px; box-shadow: 0 10px 30px rgba(0,0,0,0.4); text-align: center; }}
          img {{ width: 240px; height: 240px; image-rendering: pixelated; background: #fff; border-radius: 12px; }}
          .title {{ font-size: 18px; font-weight: 600; margin-bottom: 8px; }}
          .hint {{ font-size: 13px; color: #bdbdbd; }}
        </style>
      </head>
      <body>
        <div class="wrap">
          <div class="card">
            <div class="title">N.E.K.O 连接二维码</div>
            <img src="{img_src}" alt="N.E.K.O QR" />
            <div class="hint">用手机 N.E.K.O 扫码，或系统相机识别打开 App</div>
          </div>
        </div>
      </body>
    </html>
    """
    return HTMLResponse(content=html)


@router.get("/", response_class=HTMLResponse)
async def get_default_index(request: Request):
    templates = get_templates()
    return templates.TemplateResponse("templates/index.html", {
        "request": request
    })


def _render_model_manager(request: Request):
    """渲染模型管理器页面的内部实现"""
    templates = get_templates()
    return templates.TemplateResponse("templates/model_manager.html", {
        "request": request
    })


@router.get("/l2d", response_class=HTMLResponse)
async def get_l2d_manager(request: Request):
    """渲染模型管理器页面(兼容旧路由)"""
    return _render_model_manager(request)


@router.get("/model_manager", response_class=HTMLResponse)
async def get_model_manager(request: Request):
    """渲染模型管理器页面"""
    return _render_model_manager(request)


@router.get("/live2d_parameter_editor", response_class=HTMLResponse)
async def live2d_parameter_editor(request: Request):
    """Live2D参数编辑器页面"""
    templates = get_templates()
    return templates.TemplateResponse("templates/live2d_parameter_editor.html", {
        "request": request
    })


@router.get("/live2d_emotion_manager", response_class=HTMLResponse)
async def live2d_emotion_manager(request: Request):
    """Live2D情感映射管理器页面"""
    templates = get_templates()
    return templates.TemplateResponse("templates/live2d_emotion_manager.html", {
        "request": request
    })


@router.get("/vrm_emotion_manager", response_class=HTMLResponse)
async def vrm_emotion_manager(request: Request):
    """VRM情感映射管理器页面"""
    templates = get_templates()
    return templates.TemplateResponse("templates/vrm_emotion_manager.html", {
        "request": request
    })


@router.get('/chara_manager', response_class=HTMLResponse)
async def chara_manager(request: Request):
    """渲染主控制页面"""
    templates = get_templates()
    return templates.TemplateResponse('templates/chara_manager.html', {"request": request})


@router.get('/voice_clone', response_class=HTMLResponse)
async def voice_clone_page(request: Request):
    templates = get_templates()
    return templates.TemplateResponse("templates/voice_clone.html", {"request": request})


@router.get("/api_key", response_class=HTMLResponse)
async def api_key_settings(request: Request):
    """API Key 设置页面"""
    templates = get_templates()
    return templates.TemplateResponse("templates/api_key_settings.html", {
        "request": request
    })


@router.get('/steam_workshop_manager', response_class=HTMLResponse)
async def steam_workshop_manager_page(request: Request, lanlan_name: str = ""):
    templates = get_templates()
    return templates.TemplateResponse("templates/steam_workshop_manager.html", {"request": request, "lanlan_name": lanlan_name})


@router.get('/memory_browser', response_class=HTMLResponse)
async def memory_browser(request: Request):
    templates = get_templates()
    return templates.TemplateResponse('templates/memory_browser.html', {"request": request})


@router.get('/cookies_login', response_class=HTMLResponse)
async def cookies_login_page(request: Request):
    """媒体凭证获取页面"""
    templates = get_templates()
    return templates.TemplateResponse('templates/cookies_login.html', {"request": request})



@router.get("/{lanlan_name}", response_class=HTMLResponse)
async def get_index(request: Request, lanlan_name: str):
    # lanlan_name 将从 URL 中提取，前端会通过 API 获取配置
    templates = get_templates()
    return templates.TemplateResponse("templates/index.html", {
        "request": request
    })
