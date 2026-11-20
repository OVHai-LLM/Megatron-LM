# Copyright (c) 2026, OVHai-LLM.
"""Core attention of dragon's SelfDiffAttention with FlashAttention-4 (CuTe-DSL), for Blackwell.

A drop-in for TEDotProductAttention in dragon's layer specs (``--dragon-core-attention fa4``), with
the same constructor and forward arguments. Why not TE's own FA4 backend: dragon uses tanh logit
softcapping (``--softcap-attn``), and TE (2.20) disables both its cuDNN and its FA4 backends when
softcap is set ("TE does not pass it softcap"). On Hopper that left FA3, which has softcap; on
Blackwell nothing fast remains. FA4 implements softcap itself, forward and backward.

Scope: what dragon's training uses. Packed sequences (thd, cu_seqlens) or a plain batch (sbhd),
causal or not, GQA, sliding window, no dropout, no bias, no context parallelism, vanilla softmax.
Anything else raises rather than computing something different from TE.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import is_layer_window_attention

try:
    from flash_attn.cute import flash_attn_func, flash_attn_varlen_func

    HAVE_FA4 = True
except ImportError:
    flash_attn_func = flash_attn_varlen_func = None
    HAVE_FA4 = False

_CAUSAL = (AttnMaskType.causal, AttnMaskType.padding_causal)
_NON_CAUSAL = (AttnMaskType.no_mask, AttnMaskType.padding)


class FA4DotProductAttention(MegatronModule):
    """softmax(softcap * tanh(scale * q k^T / softcap) + mask) v, per attention head, with FA4."""

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        num_attention_heads: Optional[int] = None,
        num_query_groups: Optional[int] = None,
        k_channels: Optional[int] = None,
        v_channels: Optional[int] = None,
        softcap: Optional[float] = None,
        cp_comm_type: str = "p2p",
        pg_collection: ProcessGroupCollection = None,
    ):
        super().__init__(config=config)
        if not HAVE_FA4:
            raise ImportError("--dragon-core-attention fa4 needs flash-attn-4 (pip install 'flash-attn-4[cu13]')")
        if config.context_parallel_size > 1:
            raise NotImplementedError("FA4DotProductAttention: no context parallelism")
        dropout = config.attention_dropout if attention_dropout is None else attention_dropout
        if dropout != 0.0:
            raise NotImplementedError(f"FA4DotProductAttention: no attention dropout (got {dropout})")
        if getattr(config, "softmax_type", "vanilla") != "vanilla":
            raise NotImplementedError(f"FA4DotProductAttention: softmax_type {config.softmax_type}")
        self.attn_mask_type = attn_mask_type
        head_dim = k_channels if k_channels is not None else config.kv_channels
        # TE's default scale (softmax_scale=None) is 1/sqrt(kv_channels), as FA4's
        self.softmax_scale = softmax_scale if softmax_scale is not None else head_dim ** -0.5
        self.softcap = float(softcap if softcap is not None else getattr(config, "softcap_attn", 0.0))
        # TE takes config.window_size at construction for the windowed layers, and a forward
        # window_size overrides it
        self.window_size = (
            config.window_size
            if is_layer_window_attention(config.window_size, config.window_attn_skip_freq, layer_number)
            else None
        )

    @staticmethod
    def _fa4_window(window_size, max_seqlen):
        """TE's (left, right), -1 = unbounded, as FA4's (left, right), None = unbounded.

        A left window covering the longest sequence is no window: the same attention, and FA4's
        causal kernel instead of its local one (the SFT's sliding-window schedule is at SEQ_LEN)."""
        if window_size is None:
            return None, None
        left, right = window_size
        left = None if left is None or left < 0 or left >= max_seqlen - 1 else int(left)
        right = None if right is None or right < 0 else int(right)
        return left, right

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor],
        attn_mask_type: Optional[AttnMaskType] = None,
        attention_bias: Optional[Tensor] = None,
        window_size: Optional[Tuple[int, int]] = None,
        concat_heads: bool = True,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tensor:
        if attention_bias is not None:
            raise NotImplementedError("FA4DotProductAttention: no attention bias")
        mask_type = self.attn_mask_type if attn_mask_type is None else attn_mask_type
        if mask_type in _CAUSAL:
            causal = True
        elif mask_type in _NON_CAUSAL:
            causal = False
        else:
            raise NotImplementedError(f"FA4DotProductAttention: mask type {mask_type.name}")
        if window_size is None:
            window_size = self.window_size
        common = dict(softmax_scale=self.softmax_scale, causal=causal, softcap=self.softcap)

        if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
            # (t, h, d): the documents of a pack, cu_seqlens their boundaries
            p = packed_seq_params
            if getattr(p, "cu_seqlens_q_padded", None) is not None or getattr(p, "cu_seqlens_kv_padded", None) is not None:
                raise NotImplementedError("FA4DotProductAttention: padded cu_seqlens")
            max_q, max_kv = int(p.max_seqlen_q), int(p.max_seqlen_kv)
            out = flash_attn_varlen_func(
                query, key, value,
                cu_seqlens_q=p.cu_seqlens_q, cu_seqlens_k=p.cu_seqlens_kv,
                max_seqlen_q=max_q, max_seqlen_k=max_kv,
                window_size=self._fa4_window(window_size, max(max_q, max_kv)), **common,
            )
            out = out[0] if isinstance(out, tuple) else out          # (t, h, d)
            return out.reshape(out.size(0), -1) if concat_heads else out

        # (s, b, h, d) -> FA4's (b, s, h, d)
        if attention_mask is not None and mask_type not in (AttnMaskType.causal, AttnMaskType.no_mask):
            raise NotImplementedError("FA4DotProductAttention: explicit attention_mask")
        q, k, v = (x.transpose(0, 1) for x in (query, key, value))
        out = flash_attn_func(
            q, k, v, window_size=self._fa4_window(window_size, max(q.size(1), k.size(1))), **common
        )
        out = (out[0] if isinstance(out, tuple) else out).transpose(0, 1)   # (s, b, h, d)
        return out.reshape(out.size(0), out.size(1), -1) if concat_heads else out
