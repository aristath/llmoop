from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ExactLayer00ConvPedal:
    """Exact direct lowering of LFM2 layer_00 into a pedal implementation.

    This is not yet a synthesized behavioral replacement. It reproduces the
    source layer's math directly from tensors while avoiding a call to the
    source `model.layers[0]` module. Its purpose is to establish the first
    non-source-module implementation boundary that can later be optimized,
    fused, or replaced.
    """

    torch: Any
    layer_index: int
    hidden_size: int
    conv_l_cache: int
    norm_eps: float
    weights: dict[str, Any]

    pedal_id = "layer_00"
    implementation = "exact_lowering_lfm2_conv_layer_v1"

    @classmethod
    def from_model(cls, model: Any, torch: Any, layer_index: int = 0) -> "ExactLayer00ConvPedal":
        if layer_index != 0:
            raise ValueError("ExactLayer00ConvPedal currently only supports layer_00")
        prefix = f"model.layers.{layer_index}"
        state = model.state_dict()
        weights = {
            "operator_norm": state[f"{prefix}.operator_norm.weight"],
            "ffn_norm": state[f"{prefix}.ffn_norm.weight"],
            "conv_in_proj": state[f"{prefix}.conv.in_proj.weight"],
            "conv_kernel": state[f"{prefix}.conv.conv.weight"],
            "conv_out_proj": state[f"{prefix}.conv.out_proj.weight"],
            "ffn_w1": state[f"{prefix}.feed_forward.w1.weight"],
            "ffn_w2": state[f"{prefix}.feed_forward.w2.weight"],
            "ffn_w3": state[f"{prefix}.feed_forward.w3.weight"],
        }
        return cls(
            torch=torch,
            layer_index=layer_index,
            hidden_size=model.config.hidden_size,
            conv_l_cache=model.config.conv_L_cache,
            norm_eps=model.config.norm_eps,
            weights=weights,
        )

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
