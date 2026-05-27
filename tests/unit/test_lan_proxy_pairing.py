import json

import lan_proxy


def _patch_pairing_paths(monkeypatch, tmp_path):
    state_dir = tmp_path / ".neko"
    monkeypatch.setattr(lan_proxy, "_device_id_file_path", lambda: state_dir / "device_id")
    monkeypatch.setattr(lan_proxy, "_pairings_file_path", lambda: state_dir / "mobile_pairings.json")
    return state_dir


def test_connection_info_exposes_device_and_pairing_metadata(monkeypatch, tmp_path):
    _patch_pairing_paths(monkeypatch, tmp_path)

    proxy = lan_proxy.LanProxy(
        bind_host="192.168.50.10",
        enable_cloud=False,
        enable_stun=False,
        character="momo",
    )
    proxy.token = "runtime-token"

    info = proxy.get_connection_info()

    assert info["schema"] == "neko.mobile.p2p.v1"
    assert info["service"] == "lan_proxy"
    assert info["mobile_backend"] is True
    assert info["mobile_api_version"] == lan_proxy.MOBILE_API_VERSION
    assert info["client_type"] == "mobile"
    assert info["capabilities"] == ["mobile_pairing"]
    assert info["lan_ip"] == "192.168.50.10"
    assert info["port"] == lan_proxy.PROXY_PORT
    assert info["token"] != "runtime-token"
    assert info["character"] == "momo"
    assert info["device_id"].startswith("neko-")
    assert info["qr_one_time"] is True
    assert info["qr_token_ttl_seconds"] == lan_proxy.QR_TOKEN_TTL_SECONDS
    assert info["qr_expires_at"] > 0
    assert info["pairing_supported"] is True
    assert info["pairing_register_path"] == "/pairing/register"
    assert info["pairing_resolve_path"] == "/pairing/resolve"

    runtime_info = proxy.get_connection_info(token=proxy.token)
    assert runtime_info["token"] == "runtime-token"
    assert runtime_info["qr_one_time"] is False
    assert runtime_info["qr_expires_at"] == 0

    camera_info = proxy.get_connection_info(client_type="nekocam")
    assert camera_info["client_type"] == "nekocam"
    assert camera_info["capabilities"] == [
        "camera_frame",
        "camera_advice",
        "live2d_overlay",
    ]
    assert camera_info["avatar"] == {
        "type": "live2d",
        "character": "momo",
    }


def test_mobile_pairing_persists_and_resolves_fresh_runtime_token(monkeypatch, tmp_path):
    state_dir = _patch_pairing_paths(monkeypatch, tmp_path)

    first_proxy = lan_proxy.LanProxy(
        bind_host="192.168.50.10",
        enable_cloud=False,
        enable_stun=False,
        character="momo",
    )
    first_proxy.token = "token-first"
    pairing = first_proxy.create_mobile_pairing(
        client_name="Tong Phone",
        client_device_id="phone-dev-001",
        user_agent="pytest-phone/1.0",
    )

    stored = json.loads((state_dir / "mobile_pairings.json").read_text(encoding="utf-8"))
    saved_entry = stored["pairings"][pairing["pairing_id"]]
    assert pairing["pairing_secret"]
    assert "pairing_secret" not in saved_entry
    assert saved_entry["secret_sha256"]
    assert saved_entry["client_name"] == "Tong Phone"
    assert saved_entry["client_type"] == "mobile"
    assert saved_entry["capabilities"] == ["mobile_pairing"]
    assert saved_entry["client_device_id"] == "phone-dev-001"

    second_proxy = lan_proxy.LanProxy(
        bind_host="192.168.50.77",
        enable_cloud=False,
        enable_stun=False,
        character="momo",
    )
    second_proxy.token = "token-second"

    resolved = second_proxy.resolve_mobile_pairing(
        pairing_id=pairing["pairing_id"],
        pairing_secret=pairing["pairing_secret"],
        client_name="Tong Phone",
        client_device_id="phone-dev-001",
        user_agent="pytest-phone/2.0",
    )

    assert resolved is not None
    assert resolved["lan_ip"] == "192.168.50.77"
    assert resolved["token"] == "token-second"
    assert resolved["device_id"] == pairing["device_id"]

    refreshed = json.loads((state_dir / "mobile_pairings.json").read_text(encoding="utf-8"))
    refreshed_entry = refreshed["pairings"][pairing["pairing_id"]]
    assert refreshed_entry["last_resolved_at"] > 0
    assert refreshed_entry["last_seen_at"] > 0
    assert refreshed_entry["user_agent"] == "pytest-phone/2.0"


def test_nekocam_pairing_persists_client_type_and_capabilities(monkeypatch, tmp_path):
    state_dir = _patch_pairing_paths(monkeypatch, tmp_path)

    first_proxy = lan_proxy.LanProxy(
        bind_host="192.168.50.10",
        enable_cloud=False,
        enable_stun=False,
        character="momo",
    )
    first_proxy.token = "token-first"
    pairing = first_proxy.create_mobile_pairing(
        client_name="NEKO Camera",
        client_type="nekocam",
        capabilities=["camera_frame", "camera_advice"],
    )

    stored = json.loads((state_dir / "mobile_pairings.json").read_text(encoding="utf-8"))
    saved_entry = stored["pairings"][pairing["pairing_id"]]
    assert pairing["client_type"] == "nekocam"
    assert pairing["capabilities"] == ["camera_frame", "camera_advice"]
    assert saved_entry["client_type"] == "nekocam"
    assert saved_entry["capabilities"] == ["camera_frame", "camera_advice"]

    second_proxy = lan_proxy.LanProxy(
        bind_host="192.168.50.77",
        enable_cloud=False,
        enable_stun=False,
        character="momo",
    )
    second_proxy.token = "token-second"

    resolved = second_proxy.resolve_mobile_pairing(
        pairing_id=pairing["pairing_id"],
        pairing_secret=pairing["pairing_secret"],
    )

    assert resolved is not None
    assert resolved["client_type"] == "nekocam"
    assert resolved["capabilities"] == ["camera_frame", "camera_advice"]
    assert resolved["avatar"]["character"] == "momo"


def test_mobile_pairing_rejects_wrong_secret(monkeypatch, tmp_path):
    _patch_pairing_paths(monkeypatch, tmp_path)

    proxy = lan_proxy.LanProxy(
        bind_host="192.168.50.10",
        enable_cloud=False,
        enable_stun=False,
        character="momo",
    )
    pairing = proxy.create_mobile_pairing(client_name="Tong Phone")

    resolved = proxy.resolve_mobile_pairing(
        pairing_id=pairing["pairing_id"],
        pairing_secret="definitely-not-the-right-secret",
    )

    assert resolved is None


def test_qr_token_is_consumed_once(monkeypatch, tmp_path):
    _patch_pairing_paths(monkeypatch, tmp_path)

    proxy = lan_proxy.LanProxy(
        bind_host="192.168.50.10",
        enable_cloud=False,
        enable_stun=False,
        character="momo",
    )
    token = proxy.get_connection_info()["token"]

    assert proxy._validate_qr_token(token) == (True, "")
    assert proxy._consume_qr_token(token) == (True, "")
    assert proxy._validate_qr_token(token) == (False, "qr_used")

    next_token = proxy.get_connection_info()["token"]
    assert next_token != token
    assert proxy._validate_qr_token(next_token) == (True, "")


def test_qr_token_expires(monkeypatch, tmp_path):
    _patch_pairing_paths(monkeypatch, tmp_path)

    proxy = lan_proxy.LanProxy(
        bind_host="192.168.50.10",
        enable_cloud=False,
        enable_stun=False,
        character="momo",
    )
    token = proxy._issue_qr_token(now=100)

    assert proxy._validate_qr_token(token, now=100) == (True, "")
    assert proxy._validate_qr_token(token, now=100 + lan_proxy.QR_TOKEN_TTL_SECONDS) == (
        False,
        "qr_expired",
    )
