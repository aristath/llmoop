from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llmoop.pedalboard import Json


@dataclass(frozen=True)
class SourceLayerContractReport:
    ok: bool
    layer_id: str
    operator_type: str
    checks: tuple[str, ...]
    details: Json
    errors: tuple[str, ...] = ()

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ValueError("source layer contract failed:\n" + "\n".join(f"- {error}" for error in self.errors))

    def to_json(self) -> Json:
        return {
            "ok": self.ok,
            "layer_id": self.layer_id,
            "operator_type": self.operator_type,
            "checks": list(self.checks),
            "details": self.details,
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class SourceModelContractReport:
    ok: bool
    model_dir: str
    layer_reports: tuple[SourceLayerContractReport, ...]

    @property
    def errors(self) -> tuple[str, ...]:
        errors: list[str] = []
        for report in self.layer_reports:
            errors.extend(f"{report.layer_id}: {error}" for error in report.errors)
        return tuple(errors)

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ValueError("source model contract failed:\n" + "\n".join(f"- {error}" for error in self.errors))

    def to_json(self) -> Json:
        return {
            "ok": self.ok,
            "model_dir": self.model_dir,
            "layer_count": len(self.layer_reports),
            "layer_reports": [report.to_json() for report in self.layer_reports],
            "errors": list(self.errors),
        }


def check_lfm2_source_layer_contract(
    model_dir: Path,
    pedal_file: Path,
    layer_index: int,
) -> SourceLayerContractReport:
    """Check one transpiled pedal against the real LFM2 source layer.

    This is a contract/oscilloscope check, not a replacement implementation.
    It verifies that the source model can treat a layer as one black-box pedal
    with the declared frame, parameter, and transient-state boundary.
    """

    torch, auto_model, _ = _oracle_imports()
    model = auto_model.from_pretrained(model_dir, dtype=torch.float32)
    model.eval()
    return _check_lfm2_source_layer_contract(model=model, model_dir=model_dir, pedal_file=pedal_file, layer_index=layer_index)


def check_lfm2_source_model_contract(model_dir: Path, pedals_dir: Path) -> SourceModelContractReport:
    torch, auto_model, _ = _oracle_imports()
    model = auto_model.from_pretrained(model_dir, dtype=torch.float32)
    model.eval()
    layer_reports = tuple(
        _check_lfm2_source_layer_contract(
            model=model,
            model_dir=model_dir,
            pedal_file=pedals_dir / f"layer_{layer_index:02d}.json",
            layer_index=layer_index,
        )
        for layer_index in range(model.config.num_hidden_layers)
    )
    return SourceModelContractReport(
        ok=all(report.ok for report in layer_reports),
        model_dir=str(model_dir),
        layer_reports=layer_reports,
    )


def _check_lfm2_source_layer_contract(
    model: Any,
    model_dir: Path,
    pedal_file: Path,
    layer_index: int,
) -> SourceLayerContractReport:
    torch, _, dynamic_cache = _oracle_imports()
    pedal = json.loads(pedal_file.read_text())
    checks: list[str] = []
    errors: list[str] = []

    config = model.config
    layer = model.model.layers[layer_index]
    operator_type = pedal["operator_type"]

    _check(pedal["id"] == f"layer_{layer_index:02d}", checks, errors, "pedal id matches requested layer")
    _check(operator_type == config.layer_types[layer_index], checks, errors, "pedal operator type matches source config")
    _check(hasattr(layer, "operator_norm"), checks, errors, "source layer has operator_norm")
    _check(hasattr(layer, "ffn_norm"), checks, errors, "source layer has ffn_norm")
    _check(hasattr(layer, "feed_forward"), checks, errors, "source layer has feed_forward")

    if operator_type == "conv":
        _check(hasattr(layer, "conv"), checks, errors, "source conv layer has conv operator")
    elif operator_type == "full_attention":
        _check(hasattr(layer, "self_attn"), checks, errors, "source attention layer has self_attn operator")
    else:
        errors.append(f"unsupported source oracle operator type: {operator_type}")

    tensor_shapes = _tensor_shapes(model)
    missing_refs = [name for name in pedal["parameter_block"]["tensor_refs"] if name not in tensor_shapes]
    _check(not missing_refs, checks, errors, "all pedal parameter refs exist in source model")

    expected_frame_shape = tuple(pedal["ports"]["inputs"][0]["shape"])
    token_id = config.bos_token_id if config.bos_token_id is not None else 1
    input_ids = torch.tensor([[token_id]], dtype=torch.long)
    hidden = model.model.embed_tokens(input_ids)
    position_ids = torch.tensor([[0]], dtype=torch.long)
    position_embeddings = model.model.rotary_emb(hidden, position_ids=position_ids)

    _check(tuple(hidden.shape[-1:]) == expected_frame_shape, checks, errors, "source input frame width matches pedal input")

    cache = dynamic_cache(config=config)
    with torch.no_grad():
        output = layer(
            hidden,
            attention_mask=None,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_values=cache,
        )

    expected_output_shape = tuple(pedal["ports"]["outputs"][0]["shape"])
    _check(tuple(output.shape[-1:]) == expected_output_shape, checks, errors, "source output frame width matches pedal output")

    state_details: Json = {}
    if operator_type == "conv":
        state_details = _check_conv_state(cache, layer_index, pedal, checks, errors)
    elif operator_type == "full_attention":
        state_details = _check_attention_state(cache, layer_index, pedal, checks, errors)

    parameter_details = {
        name: list(tensor_shapes[name])
        for name in pedal["parameter_block"]["tensor_refs"]
        if name in tensor_shapes
    }

    details = {
        "model_dir": str(model_dir),
        "pedal_file": str(pedal_file),
        "source_layer_class": type(layer).__name__,
        "input_shape": list(hidden.shape),
        "output_shape": list(output.shape),
        "state": state_details,
        "parameter_shapes": parameter_details,
    }

    return SourceLayerContractReport(
        ok=not errors,
        layer_id=pedal["id"],
        operator_type=operator_type,
        checks=tuple(checks),
        details=details,
        errors=tuple(errors),
    )


def _oracle_imports() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM
        from transformers.cache_utils import DynamicCache
    except ImportError as exc:  # pragma: no cover - exercised by environments without oracle deps
        raise RuntimeError(
            "source oracle checks require torch, transformers, and safetensors. "
            "Use the repo .venv created for oracle work."
        ) from exc
    return torch, AutoModelForCausalLM, DynamicCache


def _check_conv_state(cache: Any, layer_index: int, pedal: Json, checks: list[str], errors: list[str]) -> Json:
    layer_cache = cache.layers[layer_index]
    conv_states = getattr(layer_cache, "conv_states", None)
    if conv_states is None:
        errors.append("source cache did not create conv_states for conv layer")
        return {}

    source_shape = tuple(conv_states.shape)
    if len(source_shape) != 3:
        errors.append(f"expected source conv state [batch, hidden, time], got {source_shape}")
        return {"source_cache_shape": list(source_shape)}

    logical_shape = (source_shape[2], source_shape[1])
    pedal_shape = tuple(pedal["state_ports"][0]["shape"])
    _check(logical_shape == pedal_shape, checks, errors, "source conv cache window matches pedal temporal state")

    return {
        "source_cache_layout": "batch_hidden_time",
        "source_cache_shape": list(source_shape),
        "logical_pedal_layout": "time_hidden",
        "logical_pedal_shape": list(logical_shape),
        "declared_pedal_shape": list(pedal_shape),
    }


def _check_attention_state(cache: Any, layer_index: int, pedal: Json, checks: list[str], errors: list[str]) -> Json:
    layer_cache = cache.layers[layer_index]
    keys = getattr(layer_cache, "keys", None)
    values = getattr(layer_cache, "values", None)
    if keys is None or values is None:
        errors.append("source cache did not create keys/values for attention layer")
        return {}

    key_shape = tuple(keys.shape)
    value_shape = tuple(values.shape)
    if len(key_shape) != 4 or len(value_shape) != 4:
        errors.append(f"expected attention cache [batch, kv_heads, seq, head_dim], got {key_shape} and {value_shape}")
        return {"source_key_shape": list(key_shape), "source_value_shape": list(value_shape)}

    logical_key_shape = (key_shape[1], key_shape[3])
    logical_value_shape = (value_shape[1], value_shape[3])
    state_port = pedal["state_ports"][0]
    declared_key_shape = tuple(state_port["key_shape_per_token"])
    declared_value_shape = tuple(state_port["value_shape_per_token"])

    _check(logical_key_shape == declared_key_shape, checks, errors, "source key shape per token matches pedal KV state")
    _check(logical_value_shape == declared_value_shape, checks, errors, "source value shape per token matches pedal KV state")
    _check(key_shape[2] == 1 and value_shape[2] == 1, checks, errors, "single-token source call creates one KV timestep")

    return {
        "source_cache_layout": "batch_kvheads_seq_headdim",
        "source_key_shape": list(key_shape),
        "source_value_shape": list(value_shape),
        "logical_key_shape_per_token": list(logical_key_shape),
        "logical_value_shape_per_token": list(logical_value_shape),
        "declared_key_shape_per_token": list(declared_key_shape),
        "declared_value_shape_per_token": list(declared_value_shape),
    }


def _tensor_shapes(model: Any) -> dict[str, tuple[int, ...]]:
    return {name: tuple(param.shape) for name, param in model.state_dict().items()}


def _check(condition: bool, checks: list[str], errors: list[str], message: str) -> None:
    if condition:
        checks.append(message)
    else:
        errors.append(message)
