from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llmoop.circuit_ir import load_circuit, validate_circuit
from llmoop.circuit_pedalboard import CircuitPedalboard
from llmoop.pedalboard import Json


@dataclass
class ShortConvCircuitPedal:
    """Executable short-conv stream circuit.

    This is the first reusable numeric circuit executor: one implementation
    runs every LFM2 short-conv layer by swapping only parameter refs and the
    layer-owned transient state index.
    """

    torch: Any
    pedal_id: str
    layer_index: int
    hidden_size: int
    conv_l_cache: int
    norm_eps: float
    circuit_id: str
    weights: dict[str, Any]

    implementation = "executable_lfm2_shortconv_circuit_v1"

    @classmethod
    def from_circuit(cls, model: Any, torch: Any, circuit: Json) -> "ShortConvCircuitPedal":
        _validate_shortconv_circuit(circuit)
        state = model.state_dict()
        refs = circuit["parameters"]["refs"]
        weights = {
            "operator_norm": state[refs["operator_norm"]["tensor"]],
            "ffn_norm": state[refs["ffn_norm"]["tensor"]],
            "conv_in_proj": state[refs["conv_in_projection"]["tensor"]],
            "conv_kernel": state[refs["conv_depthwise_kernel"]["tensor"]],
            "conv_out_proj": state[refs["conv_out_projection"]["tensor"]],
            "ffn_w1": state[refs["ffn_gate"]["tensor"]],
            "ffn_w2": state[refs["ffn_down"]["tensor"]],
            "ffn_w3": state[refs["ffn_up"]["tensor"]],
        }
        return cls(
            torch=torch,
            pedal_id=circuit["source"]["pedal_id"],
            layer_index=int(circuit["source"]["source_layer_index"]),
            hidden_size=int(circuit["boundary"]["inputs"][0]["shape"][0]),
            conv_l_cache=int(circuit["state_ports"][0]["shape"][0]),
            norm_eps=model.config.norm_eps,
            circuit_id=circuit["id"],
            weights=weights,
        )

    @classmethod
    def from_tensor_store(
        cls,
        tensor_store: Any,
        torch: Any,
        circuit: Json,
        config: Json,
    ) -> "ShortConvCircuitPedal":
        _validate_shortconv_circuit(circuit)
        refs = circuit["parameters"]["refs"]
        weights = {
            "operator_norm": tensor_store.get(refs["operator_norm"]["tensor"]),
            "ffn_norm": tensor_store.get(refs["ffn_norm"]["tensor"]),
            "conv_in_proj": tensor_store.get(refs["conv_in_projection"]["tensor"]),
            "conv_kernel": tensor_store.get(refs["conv_depthwise_kernel"]["tensor"]),
            "conv_out_proj": tensor_store.get(refs["conv_out_projection"]["tensor"]),
            "ffn_w1": tensor_store.get(refs["ffn_gate"]["tensor"]),
            "ffn_w2": tensor_store.get(refs["ffn_down"]["tensor"]),
            "ffn_w3": tensor_store.get(refs["ffn_up"]["tensor"]),
        }
        return cls(
            torch=torch,
            pedal_id=circuit["source"]["pedal_id"],
            layer_index=int(circuit["source"]["source_layer_index"]),
            hidden_size=int(circuit["boundary"]["inputs"][0]["shape"][0]),
            conv_l_cache=int(circuit["state_ports"][0]["shape"][0]),
            norm_eps=float(config["norm_eps"]),
            circuit_id=circuit["id"],
            weights=weights,
        )

    @classmethod
    def from_circuit_file(cls, model: Any, torch: Any, circuit_path: Path) -> "ShortConvCircuitPedal":
        circuit = load_circuit(circuit_path)
        return cls.from_circuit(model=model, torch=torch, circuit=circuit)

    def forward(
        self,
        hidden_states: Any,
        past_key_values: Any,
        attention_mask: Any = None,
        **_: Any,
    ) -> Any:
        residual = hidden_states
        normed = self._rms_norm(hidden_states, self.weights["operator_norm"])
        operator_out = self._short_conv(normed, past_key_values=past_key_values, attention_mask=attention_mask)
        hidden_states = residual + operator_out
        hidden_states = hidden_states + self._feed_forward(self._rms_norm(hidden_states, self.weights["ffn_norm"]))
        return hidden_states

    def _short_conv(self, hidden_states: Any, past_key_values: Any, attention_mask: Any = None) -> Any:
        torch = self.torch
        functional = torch.nn.functional
        seqlen = hidden_states.shape[1]

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask[:, -hidden_states.shape[1] :].unsqueeze(-1)

        projected = functional.linear(hidden_states, self.weights["conv_in_proj"]).transpose(-1, -2)
        gate_b, gate_c, projected_x = projected.chunk(3, dim=-2)
        gated_x = gate_b * projected_x
        conv_weight = self.weights["conv_kernel"]

        if past_key_values is not None and past_key_values.has_previous_state(self.layer_index):
            conv_state = past_key_values.update_conv_state(gated_x, self.layer_index)
            conv_out = torch.sum(conv_state.to(gated_x.device) * conv_weight[:, 0, :], dim=-1)
            conv_out = conv_out.unsqueeze(-1)
        else:
            if past_key_values is not None:
                conv_state = functional.pad(gated_x, (self.conv_l_cache - gated_x.shape[-1], 0))
                past_key_values.update_conv_state(conv_state, self.layer_index)
            conv_out = functional.conv1d(
                gated_x,
                conv_weight,
                bias=None,
                padding=self.conv_l_cache - 1,
                groups=self.hidden_size,
            )[..., :seqlen]

        output = gate_c * conv_out
        output = output.transpose(-1, -2).contiguous()
        return functional.linear(output, self.weights["conv_out_proj"])

    def _feed_forward(self, hidden_states: Any) -> Any:
        functional = self.torch.nn.functional
        gate = functional.silu(functional.linear(hidden_states, self.weights["ffn_w1"]))
        up = functional.linear(hidden_states, self.weights["ffn_w3"])
        return functional.linear(gate * up, self.weights["ffn_w2"])

    def _rms_norm(self, hidden_states: Any, weight: Any) -> Any:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(self.torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * self.torch.rsqrt(variance + self.norm_eps)
        return weight * hidden_states.to(input_dtype)


@dataclass
class GQAAttentionCircuitPedal:
    """Executable grouped-query attention stream circuit.

    This keeps KV as stream-owned transient state through the same cache object
    used by the reference stream, while avoiding calls to the source attention
    layer module.
    """

    torch: Any
    pedal_id: str
    layer_index: int
    hidden_size: int
    query_heads: int
    key_value_heads: int
    head_width: int
    query_groups_per_kv_head: int
    norm_eps: float
    circuit_id: str
    weights: dict[str, Any]

    implementation = "executable_lfm2_gqa_attention_circuit_v1"

    @classmethod
    def from_circuit(cls, model: Any, torch: Any, circuit: Json) -> "GQAAttentionCircuitPedal":
        _validate_attention_circuit(circuit)
        state = model.state_dict()
        refs = circuit["parameters"]["refs"]
        heads = _attention_heads(circuit)
        weights = {
            "operator_norm": state[refs["operator_norm"]["tensor"]],
            "ffn_norm": state[refs["ffn_norm"]["tensor"]],
            "q_proj": state[refs["q_projection"]["tensor"]],
            "k_proj": state[refs["k_projection"]["tensor"]],
            "v_proj": state[refs["v_projection"]["tensor"]],
            "out_proj": state[refs["attention_out_projection"]["tensor"]],
            "q_norm": state[refs["q_norm"]["tensor"]],
            "k_norm": state[refs["k_norm"]["tensor"]],
            "ffn_w1": state[refs["ffn_gate"]["tensor"]],
            "ffn_w2": state[refs["ffn_down"]["tensor"]],
            "ffn_w3": state[refs["ffn_up"]["tensor"]],
        }
        return cls(
            torch=torch,
            pedal_id=circuit["source"]["pedal_id"],
            layer_index=int(circuit["source"]["source_layer_index"]),
            hidden_size=int(circuit["boundary"]["inputs"][0]["shape"][0]),
            query_heads=int(heads["query_heads"]),
            key_value_heads=int(heads["key_value_heads"]),
            head_width=int(heads["head_width"]),
            query_groups_per_kv_head=int(heads["query_groups_per_kv_head"]),
            norm_eps=model.config.norm_eps,
            circuit_id=circuit["id"],
            weights=weights,
        )

    @classmethod
    def from_tensor_store(
        cls,
        tensor_store: Any,
        torch: Any,
        circuit: Json,
        config: Json,
    ) -> "GQAAttentionCircuitPedal":
        _validate_attention_circuit(circuit)
        refs = circuit["parameters"]["refs"]
        heads = _attention_heads(circuit)
        weights = {
            "operator_norm": tensor_store.get(refs["operator_norm"]["tensor"]),
            "ffn_norm": tensor_store.get(refs["ffn_norm"]["tensor"]),
            "q_proj": tensor_store.get(refs["q_projection"]["tensor"]),
            "k_proj": tensor_store.get(refs["k_projection"]["tensor"]),
            "v_proj": tensor_store.get(refs["v_projection"]["tensor"]),
            "out_proj": tensor_store.get(refs["attention_out_projection"]["tensor"]),
            "q_norm": tensor_store.get(refs["q_norm"]["tensor"]),
            "k_norm": tensor_store.get(refs["k_norm"]["tensor"]),
            "ffn_w1": tensor_store.get(refs["ffn_gate"]["tensor"]),
            "ffn_w2": tensor_store.get(refs["ffn_down"]["tensor"]),
            "ffn_w3": tensor_store.get(refs["ffn_up"]["tensor"]),
        }
        return cls(
            torch=torch,
            pedal_id=circuit["source"]["pedal_id"],
            layer_index=int(circuit["source"]["source_layer_index"]),
            hidden_size=int(circuit["boundary"]["inputs"][0]["shape"][0]),
            query_heads=int(heads["query_heads"]),
            key_value_heads=int(heads["key_value_heads"]),
            head_width=int(heads["head_width"]),
            query_groups_per_kv_head=int(heads["query_groups_per_kv_head"]),
            norm_eps=float(config["norm_eps"]),
            circuit_id=circuit["id"],
            weights=weights,
        )

    @classmethod
    def from_circuit_file(cls, model: Any, torch: Any, circuit_path: Path) -> "GQAAttentionCircuitPedal":
        circuit = load_circuit(circuit_path)
        return cls.from_circuit(model=model, torch=torch, circuit=circuit)

    def forward(
        self,
        hidden_states: Any,
        past_key_values: Any,
        position_embeddings: tuple[Any, Any],
        attention_mask: Any = None,
        **_: Any,
    ) -> Any:
        residual = hidden_states
        normed = self._rms_norm(hidden_states, self.weights["operator_norm"])
        operator_out = self._attention(
            normed,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = residual + operator_out
        hidden_states = hidden_states + self._feed_forward(self._rms_norm(hidden_states, self.weights["ffn_norm"]))
        return hidden_states

    def _attention(
        self,
        hidden_states: Any,
        past_key_values: Any,
        position_embeddings: tuple[Any, Any],
        attention_mask: Any = None,
    ) -> Any:
        torch = self.torch
        functional = torch.nn.functional
        input_shape = hidden_states.shape[:-1]

        query_states = functional.linear(hidden_states, self.weights["q_proj"])
        query_states = query_states.view(*input_shape, self.query_heads, self.head_width)
        query_states = self._rms_norm(query_states, self.weights["q_norm"]).transpose(1, 2)

        key_states = functional.linear(hidden_states, self.weights["k_proj"])
        key_states = key_states.view(*input_shape, self.key_value_heads, self.head_width)
        key_states = self._rms_norm(key_states, self.weights["k_norm"]).transpose(1, 2)

        value_states = functional.linear(hidden_states, self.weights["v_proj"])
        value_states = value_states.view(*input_shape, self.key_value_heads, self.head_width).transpose(1, 2)

        query_states, key_states = self._apply_rope(query_states, key_states, position_embeddings)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_index)

        key_states = self._repeat_kv(key_states)
        value_states = self._repeat_kv(value_states)
        is_causal = bool(query_states.shape[2] > 1 and attention_mask is None)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=0.0,
            scale=self.head_width**-0.5,
            is_causal=is_causal,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        return functional.linear(attn_output, self.weights["out_proj"])

    def _apply_rope(self, query_states: Any, key_states: Any, position_embeddings: tuple[Any, Any]) -> tuple[Any, Any]:
        cos, sin = position_embeddings
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        query_states = (query_states * cos) + (self._rotate_half(query_states) * sin)
        key_states = (key_states * cos) + (self._rotate_half(key_states) * sin)
        return query_states, key_states

    def _rotate_half(self, tensor: Any) -> Any:
        first = tensor[..., : tensor.shape[-1] // 2]
        second = tensor[..., tensor.shape[-1] // 2 :]
        return self.torch.cat((-second, first), dim=-1)

    def _repeat_kv(self, hidden_states: Any) -> Any:
        if self.query_groups_per_kv_head == 1:
            return hidden_states
        batch, kv_heads, sequence_length, head_width = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch,
            kv_heads,
            self.query_groups_per_kv_head,
            sequence_length,
            head_width,
        )
        return hidden_states.reshape(batch, kv_heads * self.query_groups_per_kv_head, sequence_length, head_width)

    def _feed_forward(self, hidden_states: Any) -> Any:
        functional = self.torch.nn.functional
        gate = functional.silu(functional.linear(hidden_states, self.weights["ffn_w1"]))
        up = functional.linear(hidden_states, self.weights["ffn_w3"])
        return functional.linear(gate * up, self.weights["ffn_w2"])

    def _rms_norm(self, hidden_states: Any, weight: Any) -> Any:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(self.torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * self.torch.rsqrt(variance + self.norm_eps)
        return weight * hidden_states.to(input_dtype)


def install_shortconv_circuit_pedals(executor: Any, circuit_dir: Path) -> tuple[str, ...]:
    board = CircuitPedalboard.from_dir(circuit_dir)
    installed: list[str] = []
    for pedal in board.pedals:
        if pedal.operator_type != "conv":
            continue
        implementation = ShortConvCircuitPedal.from_circuit(
            model=executor.model,
            torch=executor.torch,
            circuit=pedal.circuit,
        )
        executor.install_pedal_implementation(pedal.id, implementation)
        installed.append(pedal.id)
    return tuple(installed)


def install_attention_circuit_pedals(executor: Any, circuit_dir: Path) -> tuple[str, ...]:
    board = CircuitPedalboard.from_dir(circuit_dir)
    installed: list[str] = []
    for pedal in board.pedals:
        if pedal.operator_type != "full_attention":
            continue
        implementation = GQAAttentionCircuitPedal.from_circuit(
            model=executor.model,
            torch=executor.torch,
            circuit=pedal.circuit,
        )
        executor.install_pedal_implementation(pedal.id, implementation)
        installed.append(pedal.id)
    return tuple(installed)


def install_all_circuit_pedals(executor: Any, circuit_dir: Path) -> tuple[str, ...]:
    conv = install_shortconv_circuit_pedals(executor, circuit_dir)
    attention = install_attention_circuit_pedals(executor, circuit_dir)
    return tuple(sorted(conv + attention))


def _validate_shortconv_circuit(circuit: Json) -> None:
    report = validate_circuit(circuit)
    report.raise_for_errors()
    if circuit["source"]["source_operator_type"] != "conv":
        raise ValueError(f"{circuit['id']} is not a conv circuit")
    expected_params = {
        "operator_norm",
        "ffn_norm",
        "ffn_gate",
        "ffn_down",
        "ffn_up",
        "conv_in_projection",
        "conv_depthwise_kernel",
        "conv_out_projection",
    }
    actual_params = set(circuit["parameters"]["refs"])
    if actual_params != expected_params:
        raise ValueError(f"{circuit['id']} has unexpected params: {sorted(actual_params)}")
    if len(circuit["state_ports"]) != 1 or circuit["state_ports"][0]["id"] != "temporal_memory":
        raise ValueError(f"{circuit['id']} must declare one temporal_memory state port")


def _validate_attention_circuit(circuit: Json) -> None:
    report = validate_circuit(circuit)
    report.raise_for_errors()
    if circuit["source"]["source_operator_type"] != "full_attention":
        raise ValueError(f"{circuit['id']} is not a full_attention circuit")
    expected_params = {
        "operator_norm",
        "ffn_norm",
        "ffn_gate",
        "ffn_down",
        "ffn_up",
        "q_projection",
        "k_projection",
        "v_projection",
        "attention_out_projection",
        "q_norm",
        "k_norm",
    }
    actual_params = set(circuit["parameters"]["refs"])
    if actual_params != expected_params:
        raise ValueError(f"{circuit['id']} has unexpected params: {sorted(actual_params)}")
    if len(circuit["state_ports"]) != 1 or circuit["state_ports"][0]["id"] != "kv_memory":
        raise ValueError(f"{circuit['id']} must declare one kv_memory state port")


def _attention_heads(circuit: Json) -> Json:
    for node in circuit["nodes"]:
        attrs = node.get("attrs", {})
        if {"query_heads", "key_value_heads", "head_width", "query_groups_per_kv_head"} <= set(attrs):
            return attrs
    state = circuit["state_ports"][0]
    key_value_heads, head_width = state["key_shape_per_token"]
    query_heads = circuit["boundary"]["inputs"][0]["shape"][0] // head_width
    return {
        "query_heads": query_heads,
        "key_value_heads": key_value_heads,
        "head_width": head_width,
        "query_groups_per_kv_head": query_heads // key_value_heads,
    }
