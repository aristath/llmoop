from __future__ import annotations

import json
from pathlib import Path

import pytest

from nerve.compilation import ModelCompileError
from nerve.compiler_target import (
    CompilerTarget,
    compiler_device_probe_command,
    discover_compiler_target,
)


def device_payload(
    *,
    index: int,
    device_type: str,
    features: list[str],
) -> dict[str, object]:
    return {
        "physical_device_index": index,
        "physical_device_id": f"vulkan:test-{index}",
        "device_name": f"device {index}",
        "device_type": device_type,
        "vendor_id": 1,
        "device_id": index,
        "shader_features": features,
        "subgroup_operations": [],
        "subgroup_compute_supported": True,
        "subgroup_size": 32,
        "max_compute_work_group_invocations": 1024,
        "max_compute_work_group_size_x": 1024,
    }


def test_compiler_target_preserves_dtype_supported_by_any_gpu() -> None:
    target = CompilerTarget.from_json(
        {
            "schema": "nerve.device_capabilities.v1",
            "devices": [
                device_payload(
                    index=0,
                    device_type="discrete_gpu",
                    features=["shader_float16"],
                ),
                device_payload(
                    index=1,
                    device_type="discrete_gpu",
                    features=[
                        "shader_float8",
                        "shader_mixed_float_dot_product_float8_acc_float32",
                        "shader_bfloat16_type",
                    ],
                ),
            ],
        }
    )

    assert target.supports_native_dtype("F8_E4M3")
    assert target.supports_native_dtype("BF16")
    assert target.supports_native_dtype("F16")
    assert target.supports_native_dtype("F32")
    assert not target.supports_native_dtype("Q8_0")


def test_compiler_target_ignores_cpu_vulkan_devices_and_requires_a_gpu() -> None:
    with pytest.raises(ModelCompileError, match="at least one Vulkan GPU"):
        CompilerTarget.from_json(
            {
                "schema": "nerve.device_capabilities.v1",
                "devices": [
                    device_payload(
                        index=0,
                        device_type="cpu",
                        features=[
                            "shader_float8",
                            "shader_mixed_float_dot_product_float8_acc_float32",
                        ],
                    )
                ],
            }
        )


def test_compiler_target_rejects_malformed_device_entries() -> None:
    with pytest.raises(ModelCompileError, match="invalid compiler target device"):
        CompilerTarget.from_json(
            {
                "schema": "nerve.device_capabilities.v1",
                "devices": ["not a device"],
            }
        )


def test_compiler_target_discovery_validates_runtime_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Completed:
        returncode = 0
        stderr = ""
        stdout = json.dumps(
            {
                "schema": "nerve.device_capabilities.v1",
                "devices": [
                    device_payload(
                        index=2,
                        device_type="discrete_gpu",
                        features=[
                            "shader_float8",
                            "shader_mixed_float_dot_product_float8_acc_float32",
                        ],
                    )
                ],
            }
        )

    monkeypatch.setattr(
        "nerve.compiler_target.subprocess.run",
        lambda *args, **kwargs: Completed(),
    )

    target = discover_compiler_target(runtime_bin=Path("/tmp/nerve-runtime"))

    assert target.devices[0].physical_device_index == 2
    assert compiler_device_probe_command(
        runtime_bin=Path("/tmp/nerve-runtime")
    ) == ["/tmp/nerve-runtime", "--inspect-devices", "--json"]


def test_compiler_target_discovery_fails_closed_on_probe_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Completed:
        returncode = 1
        stderr = "device query failed"
        stdout = ""

    monkeypatch.setattr(
        "nerve.compiler_target.subprocess.run",
        lambda *args, **kwargs: Completed(),
    )

    with pytest.raises(ModelCompileError, match="device query failed"):
        discover_compiler_target(runtime_bin=Path("/tmp/nerve-runtime"))
