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

    assert info["lan_ip"] == "192.168.50.10"
    assert info["port"] == lan_proxy.PROXY_PORT
    assert info["token"] == "runtime-token"
    assert info["character"] == "momo"
    assert info["device_id"].startswith("neko-")
    assert info["pairing_supported"] is True
    assert info["pairing_register_path"] == "/pairing/register"
    assert info["pairing_resolve_path"] == "/pairing/resolve"


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
    assert refreshed_entry["user_agent"] == "pytest-phone/2.0"


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
