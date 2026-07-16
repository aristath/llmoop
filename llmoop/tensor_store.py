from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    source_file: Path


class SafetensorsTensorStore:
    """Direct tensor access for source checkpoint tensors.

    The store is intentionally simple for now: tensors are loaded by name from
    the safetensors checkpoint and converted to the runtime dtype on first use.
    This removes the need to instantiate the Transformers model just to obtain
    layer weights.
    """

    def __init__(
        self,
        weights_file: Path,
        torch: Any,
        dtype: Any | None = None,
        device: str | None = None,
        tensor_index: Path | None = None,
    ) -> None:
        self.weights_file = weights_file
        self.torch = torch
        self.dtype = dtype if dtype is not None else torch.float32
        self.device = device
        self.tensor_index = tensor_index
        self._cache: dict[str, Any] = {}
        self._infos: dict[str, TensorInfo] | None = None

    @classmethod
    def from_model_dir(
        cls,
        model_dir: Path,
        torch: Any,
        dtype: Any | None = None,
        device: str | None = None,
        tensor_index: Path | None = None,
    ) -> "SafetensorsTensorStore":
        return cls(
            weights_file=model_dir / "model.safetensors",
            torch=torch,
            dtype=dtype,
            device=device,
            tensor_index=tensor_index,
        )

    def get(self, name: str) -> Any:
        if name not in self._cache:
            from safetensors.torch import safe_open

            weights_file = self.infos().get(name, TensorInfo(name, "", (), self.weights_file)).source_file
            with safe_open(weights_file, framework="pt", device=self.device or "cpu") as tensors:
                if name not in tensors.keys():
                    raise KeyError(f"tensor {name!r} not found in {weights_file}")
                tensor = tensors.get_tensor(name)
            self._cache[name] = tensor.to(dtype=self.dtype)
        return self._cache[name]

    def infos(self) -> dict[str, TensorInfo]:
        if self._infos is None:
            if self.tensor_index is not None and self.tensor_index.exists():
                index = json.loads(self.tensor_index.read_text())
                tensor_index_root = self.tensor_index.parent
                self._infos = {
                    name: TensorInfo(
                        name=name,
                        dtype=info["dtype"],
                        shape=tuple(info["shape"]),
                        source_file=resolve_tensor_source_file(
                            tensor_index_root,
                            info["source_file"],
                        ),
                    )
                    for name, info in index["tensors"].items()
                }
            else:
                from safetensors.torch import safe_open

                with safe_open(self.weights_file, framework="pt", device=self.device or "cpu") as tensors:
                    self._infos = {
                        name: TensorInfo(
                            name=name,
                            dtype=str(tensors.get_tensor(name).dtype),
                            shape=tuple(tensors.get_tensor(name).shape),
                            source_file=self.weights_file,
                        )
                        for name in tensors.keys()
                    }
        return self._infos

    def summary(self) -> dict[str, Any]:
        infos = self.infos()
        return {
            "weights_file": str(self.weights_file),
            "tensor_count": len(infos),
            "cached_tensor_count": len(self._cache),
            "dtype": str(self.dtype),
            "device": self.device or "cpu",
        }


def resolve_tensor_source_file(tensor_index_root: Path, source_file: str) -> Path:
    path = Path(source_file)
    return path if path.is_absolute() else tensor_index_root / path
