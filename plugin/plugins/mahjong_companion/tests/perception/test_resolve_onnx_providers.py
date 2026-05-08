"""Pin the ONNX provider resolution policy:

  * env var override is honoured verbatim and filtered against
    ``available`` so a stale or typo'd entry doesn't crash the runtime;
  * defaults are platform-specific (CoreML on macOS, DirectML on
    Windows, CUDA elsewhere) — that's the whole point of this patch
    over the previous hardcoded ``CPUExecutionProvider``;
  * ``CPUExecutionProvider`` is always appended last as a guaranteed
    fallback, so an empty / fully-filtered result still yields a usable
    session.
"""

from __future__ import annotations

import pytest

from plugin.plugins.mahjong_companion.perception.vit_tile_classifier_onnx import (
    resolve_onnx_providers,
)


@pytest.mark.unit
def test_macos_default_prefers_coreml() -> None:
    providers = resolve_onnx_providers(
        available=("CoreMLExecutionProvider", "CPUExecutionProvider"),
        platform="darwin",
        env_value="",
    )
    assert providers == ("CoreMLExecutionProvider", "CPUExecutionProvider")


@pytest.mark.unit
def test_windows_default_prefers_directml() -> None:
    providers = resolve_onnx_providers(
        available=("DmlExecutionProvider", "CPUExecutionProvider"),
        platform="win32",
        env_value="",
    )
    assert providers == ("DmlExecutionProvider", "CPUExecutionProvider")


@pytest.mark.unit
def test_linux_default_prefers_cuda() -> None:
    providers = resolve_onnx_providers(
        available=("CUDAExecutionProvider", "CPUExecutionProvider"),
        platform="linux",
        env_value="",
    )
    assert providers == ("CUDAExecutionProvider", "CPUExecutionProvider")


@pytest.mark.unit
def test_env_var_override_wins_over_platform_default() -> None:
    providers = resolve_onnx_providers(
        available=(
            "CUDAExecutionProvider",
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ),
        platform="darwin",
        env_value="CUDAExecutionProvider, CPUExecutionProvider",
    )
    assert providers == ("CUDAExecutionProvider", "CPUExecutionProvider")


@pytest.mark.unit
def test_env_var_filters_unavailable_entries() -> None:
    """Stale env entries (e.g. user pinned CUDA on a CPU-only laptop)
    must not break the session — they should silently drop out."""
    providers = resolve_onnx_providers(
        available=("CPUExecutionProvider",),
        platform="linux",
        env_value="CUDAExecutionProvider",
    )
    assert providers == ("CPUExecutionProvider",)


@pytest.mark.unit
def test_filtered_to_empty_still_falls_back_to_cpu() -> None:
    """If every requested provider is unavailable AND the runtime
    itself reports no providers, we still emit CPUExecutionProvider
    so InferenceSession can be constructed (ORT ships CPU by default)."""
    providers = resolve_onnx_providers(
        available=(),
        platform="linux",
        env_value="CUDAExecutionProvider",
    )
    assert providers == ("CPUExecutionProvider",)


@pytest.mark.unit
def test_cpu_always_appended_last_when_missing() -> None:
    providers = resolve_onnx_providers(
        available=("CoreMLExecutionProvider", "CPUExecutionProvider"),
        platform="darwin",
        env_value="CoreMLExecutionProvider",
    )
    assert providers[-1] == "CPUExecutionProvider"


@pytest.mark.unit
def test_blank_and_whitespace_entries_in_env_dropped() -> None:
    providers = resolve_onnx_providers(
        available=("CPUExecutionProvider",),
        platform="linux",
        env_value=", ,  ,",
    )
    # Empty after split → fall back to platform default (CUDA filtered → CPU).
    assert providers == ("CPUExecutionProvider",)


@pytest.mark.unit
def test_unknown_platform_falls_back_to_cuda_then_cpu() -> None:
    providers = resolve_onnx_providers(
        available=("CUDAExecutionProvider", "CPUExecutionProvider"),
        platform="aix",
        env_value="",
    )
    assert providers == ("CUDAExecutionProvider", "CPUExecutionProvider")
