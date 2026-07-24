from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from nerve.compilation import Json, ModelCompileError


DEVICE_CAPABILITIES_SCHEMA = "nerve.device_capabilities.v1"


@dataclass(frozen=True)
class CompilerTargetDevice:
    physical_device_index: int
    physical_device_id: str
    device_name: str
    device_type: str
    vendor_id: int
    device_id: int
    shader_features: frozenset[str]
    max_compute_work_group_invocations: int
    max_compute_work_group_size_x: int
    cooperative_bfloat16_shapes: tuple[tuple[int, int, int], ...]
    cooperative_float8_e4m3_shapes: tuple[tuple[int, int, int], ...]

    @classmethod
    def from_json(cls, payload: Json) -> CompilerTargetDevice:
        try:
            return cls(
                physical_device_index=int(payload["physical_device_index"]),
                physical_device_id=str(payload["physical_device_id"]),
                device_name=str(payload["device_name"]),
                device_type=str(payload["device_type"]),
                vendor_id=int(payload["vendor_id"]),
                device_id=int(payload["device_id"]),
                shader_features=frozenset(map(str, payload["shader_features"])),
                max_compute_work_group_invocations=int(
                    payload["max_compute_work_group_invocations"]
                ),
                max_compute_work_group_size_x=int(
                    payload["max_compute_work_group_size_x"]
                ),
                cooperative_bfloat16_shapes=cooperative_matrix_shapes(
                    payload, "cooperative_bfloat16_shapes"
                ),
                cooperative_float8_e4m3_shapes=cooperative_matrix_shapes(
                    payload, "cooperative_float8_e4m3_shapes"
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ModelCompileError(
                f"runtime returned an invalid compiler target device: {payload!r}"
            ) from error

    def supports_native_dtype(self, dtype: str) -> bool:
        requirements = native_dtype_shader_features(dtype)
        return requirements is not None and requirements <= self.shader_features

    def to_json(self) -> Json:
        return {
            "physical_device_index": self.physical_device_index,
            "physical_device_id": self.physical_device_id,
            "device_name": self.device_name,
            "device_type": self.device_type,
            "vendor_id": self.vendor_id,
            "device_id": self.device_id,
            "shader_features": sorted(self.shader_features),
            "max_compute_work_group_invocations": (
                self.max_compute_work_group_invocations
            ),
            "max_compute_work_group_size_x": self.max_compute_work_group_size_x,
            "cooperative_bfloat16_shapes": [
                list(shape) for shape in self.cooperative_bfloat16_shapes
            ],
            "cooperative_float8_e4m3_shapes": [
                list(shape) for shape in self.cooperative_float8_e4m3_shapes
            ],
        }


@dataclass(frozen=True)
class CompilerTarget:
    devices: tuple[CompilerTargetDevice, ...]

    @classmethod
    def from_json(cls, payload: Json) -> CompilerTarget:
        if payload.get("schema") != DEVICE_CAPABILITIES_SCHEMA:
            raise ModelCompileError(
                "runtime returned unsupported device-capability schema "
                f"{payload.get('schema')!r}"
            )
        raw_devices = payload.get("devices")
        if not isinstance(raw_devices, list):
            raise ModelCompileError("runtime device-capability report has no device list")
        devices = []
        for raw in raw_devices:
            if not isinstance(raw, dict):
                raise ModelCompileError(
                    f"runtime returned an invalid compiler target device: {raw!r}"
                )
            device = CompilerTargetDevice.from_json(raw)
            if device.device_type in {
                "discrete_gpu",
                "integrated_gpu",
                "virtual_gpu",
            }:
                devices.append(device)
        if not devices:
            raise ModelCompileError(
                "model compilation requires at least one Vulkan GPU target"
            )
        return cls(devices=tuple(devices))

    @classmethod
    def for_features(cls, *feature_sets: Iterable[str]) -> CompilerTarget:
        devices = tuple(
            CompilerTargetDevice(
                physical_device_index=index,
                physical_device_id=f"test-device-{index}",
                device_name=f"test device {index}",
                device_type="discrete_gpu",
                vendor_id=0,
                device_id=index,
                shader_features=frozenset(features),
                max_compute_work_group_invocations=1024,
                max_compute_work_group_size_x=1024,
                cooperative_bfloat16_shapes=(),
                cooperative_float8_e4m3_shapes=(),
            )
            for index, features in enumerate(feature_sets)
        )
        if not devices:
            raise ValueError("a compiler target requires at least one device")
        return cls(devices=devices)

    def supports_native_dtype(self, dtype: str) -> bool:
        return any(device.supports_native_dtype(dtype) for device in self.devices)

    def to_json(self) -> Json:
        return {
            "schema": "nerve.compiler_target.v1",
            "devices": [device.to_json() for device in self.devices],
        }


def native_dtype_shader_features(dtype: str) -> frozenset[str] | None:
    return {
        "F32": frozenset(),
        "F16": frozenset({"shader_float16"}),
        "BF16": frozenset({"shader_bfloat16_type"}),
        "F8_E4M3": frozenset(
            {
                "shader_float8",
                "shader_mixed_float_dot_product_float8_acc_float32",
            }
        ),
    }.get(dtype)


def cooperative_matrix_shapes(
    payload: Json, field: str
) -> tuple[tuple[int, int, int], ...]:
    raw_shapes = payload[field]
    if not isinstance(raw_shapes, list):
        raise ModelCompileError(
            f"runtime compiler target device {field!r} must be a list"
        )
    shapes: list[tuple[int, int, int]] = []
    for raw_shape in raw_shapes:
        if (
            not isinstance(raw_shape, list)
            or len(raw_shape) != 3
            or any(
                not isinstance(dimension, int)
                or isinstance(dimension, bool)
                or dimension <= 0
                for dimension in raw_shape
            )
        ):
            raise ModelCompileError(
                f"runtime compiler target device has invalid {field!r}: "
                f"{raw_shapes!r}"
            )
        shapes.append(tuple(raw_shape))
    if shapes != sorted(set(shapes)):
        raise ModelCompileError(
            f"runtime compiler target device {field!r} must be unique and sorted"
        )
    return tuple(shapes)


def discover_compiler_target(
    *,
    runtime_bin: Path | None = None,
) -> CompilerTarget:
    command = compiler_device_probe_command(runtime_bin=runtime_bin)
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        diagnostic = completed.stderr.strip() or completed.stdout.strip()
        raise ModelCompileError(
            "could not discover GPU compiler capabilities"
            + (f": {diagnostic}" if diagnostic else "")
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ModelCompileError(
            "runtime returned invalid JSON while discovering compiler capabilities"
        ) from error
    if not isinstance(payload, dict):
        raise ModelCompileError(
            "runtime returned a non-object compiler capability report"
        )
    return CompilerTarget.from_json(payload)


def compiler_device_probe_command(*, runtime_bin: Path | None = None) -> list[str]:
    configured = runtime_bin or runtime_bin_from_env()
    if configured is not None:
        return [str(configured), "--inspect-devices", "--json"]

    repo_root = Path(__file__).resolve().parents[1]
    cargo_manifest = repo_root / "runtime-rs" / "Cargo.toml"
    if cargo_manifest.is_file():
        return [
            "cargo",
            "run",
            "--release",
            "--quiet",
            "--manifest-path",
            str(cargo_manifest),
            "--features",
            "vulkan tokenizers",
            "--bin",
            "nerve-runtime",
            "--",
            "--inspect-devices",
            "--json",
        ]

    installed = shutil.which("nerve-runtime")
    if installed:
        return [installed, "--inspect-devices", "--json"]
    raise ModelCompileError(
        "could not find nerve-runtime for GPU compiler-capability discovery"
    )


def runtime_bin_from_env() -> Path | None:
    raw = os.environ.get("NERVE_RUNTIME_BIN")
    return Path(raw).expanduser() if raw else None
