import math
from contextlib import contextmanager
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Linear):
    """Linear layer with an independently switchable low-rank residual."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        bias: bool = True,
        enabled: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}.")

        self.rank = rank
        self.scaling = alpha / rank
        self.lora_enabled = enabled
        self.lora_dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
        enabled: bool,
    ) -> "LoRALinear":
        adapter = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            bias=linear.bias is not None,
            enabled=enabled,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        # Keep pretrained parameter names unchanged for source-checkpoint loading.
        adapter.weight = linear.weight
        adapter.bias = linear.bias
        return adapter

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = F.linear(inputs, self.weight, self.bias)
        if not self.lora_enabled:
            return outputs

        low_rank_features = F.linear(self.lora_dropout(inputs), self.lora_A)
        return outputs + F.linear(low_rank_features, self.lora_B) * self.scaling


class LoRAMultiheadAttention(nn.MultiheadAttention):
    """Multi-head attention with checkpoint-compatible Q/V LoRA deltas."""

    def __init__(self, *args, rank: int, alpha: float, enabled: bool, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}.")
        self.scaling = alpha / rank
        self.lora_enabled = enabled
        self.lora_A_q = nn.Parameter(torch.empty(rank, self.embed_dim))
        self.lora_B_q = nn.Parameter(torch.empty(self.embed_dim, rank))
        self.lora_A_v = nn.Parameter(torch.empty(rank, self.embed_dim))
        self.lora_B_v = nn.Parameter(torch.empty(self.embed_dim, rank))
        nn.init.kaiming_uniform_(self.lora_A_q, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_v, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_q)
        nn.init.zeros_(self.lora_B_v)

    @classmethod
    def from_attention(
        cls,
        attention: nn.MultiheadAttention,
        rank: int,
        alpha: float,
        enabled: bool,
    ) -> "LoRAMultiheadAttention":
        adapter = cls(
            embed_dim=attention.embed_dim,
            num_heads=attention.num_heads,
            dropout=attention.dropout,
            bias=attention.in_proj_bias is not None,
            add_bias_kv=attention.bias_k is not None,
            add_zero_attn=attention.add_zero_attn,
            kdim=attention.kdim,
            vdim=attention.vdim,
            batch_first=attention.batch_first,
            device=attention.in_proj_weight.device,
            dtype=attention.in_proj_weight.dtype,
            rank=rank,
            alpha=alpha,
            enabled=enabled,
        )
        adapter.in_proj_weight = attention.in_proj_weight
        adapter.in_proj_bias = attention.in_proj_bias
        adapter.bias_k = attention.bias_k
        adapter.bias_v = attention.bias_v
        adapter.out_proj.weight = attention.out_proj.weight
        adapter.out_proj.bias = attention.out_proj.bias
        return adapter

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
        is_causal=False,
    ):
        in_proj_weight = self.in_proj_weight
        if self.lora_enabled:
            zero_delta = torch.zeros_like(in_proj_weight[: self.embed_dim])
            in_proj_weight = in_proj_weight + self.scaling * torch.cat(
                [
                    self.lora_B_q @ self.lora_A_q,
                    zero_delta,
                    self.lora_B_v @ self.lora_A_v,
                ],
                dim=0,
            )

        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            query, key, value = (tensor.transpose(0, 1) for tensor in (query, key, value))

        output, weights = F.multi_head_attention_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            in_proj_weight,
            self.in_proj_bias,
            self.bias_k,
            self.bias_v,
            self.add_zero_attn,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )
        if self.batch_first and is_batched:
            output = output.transpose(0, 1)
        return output, weights


def inject_lora_into_mlp(
    layers: nn.Sequential,
    rank: int,
    alpha: float,
    dropout: float,
    enabled: bool,
) -> None:
    """Replace an MLP's linear layers while preserving its base state-dict keys."""
    for index, layer in enumerate(layers):
        if type(layer) is nn.Linear:
            layers[index] = LoRALinear.from_linear(
                layer,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                enabled=enabled,
            )


def iter_lora_modules(module: nn.Module) -> Iterable[nn.Module]:
    return (
        submodule
        for submodule in module.modules()
        if isinstance(submodule, (LoRALinear, LoRAMultiheadAttention))
    )


def set_lora_enabled(module: nn.Module, enabled: bool) -> None:
    for adapter in iter_lora_modules(module):
        adapter.lora_enabled = enabled


def set_lora_trainable(module: nn.Module, trainable: bool) -> None:
    for adapter in iter_lora_modules(module):
        for name, parameter in adapter.named_parameters(recurse=False):
            if name.startswith("lora_A") or name.startswith("lora_B"):
                parameter.requires_grad_(trainable)


@contextmanager
def lora_activation(module: nn.Module, enabled: bool):
    """Temporarily select the source or adapted path without changing trainability."""
    adapters = list(iter_lora_modules(module))
    previous_states = [adapter.lora_enabled for adapter in adapters]
    for adapter in adapters:
        adapter.lora_enabled = enabled
    try:
        yield
    finally:
        for adapter, previous_state in zip(adapters, previous_states):
            adapter.lora_enabled = previous_state
