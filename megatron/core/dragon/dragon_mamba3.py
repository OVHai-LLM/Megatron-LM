# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Tri Dao, Albert Gu.

# Some of this code was adopted from https://github.com/state-spaces/mamba/
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import math
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange

from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.transformer.module import MegatronModule
from megatron.core.dragon.dragon_config import DragonConfig
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.utils import (
    make_sharded_tensors_for_checkpoint,
    sharded_state_dict_default,
)
from megatron.core.utils import deprecate_inference_params, log_single_rank, make_tp_sharded_tensor_for_checkpoint, nvtx_range_pop, nvtx_range_push

# Public state-spaces/mamba Mamba-3 MIMO kernel, used WITH cu_seqlens so the SSM
# state resets at artificial_seq_len window boundaries AND document boundaries
# (packed-sequence isolation). It takes RAW angles (computes the angle cumsum
# internally), so we no longer call angle_dt externally.
#
# PERF: the public wrappers ship with T.dynamic("B/S/H/G/NS") which compiles a
# shape-generic kernel ~10-20x slower on GH200. We patched mamba3_mimo_{fwd,bwd}
# {,_varlen}.py to pass CONCRETE shapes -> @tilelang.jit specializes+caches one
# kernel per shape (verified: varlen B=1 S=32768 NS=2 == 9ms, == private speed).
HAVE_FAST_MAMBA_SSM = False
try:
    from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo import mamba3_mimo
    HAVE_FAST_MAMBA_SSM = True
except ImportError as exc:
    HAVE_FAST_MAMBA_SSM = False
    raise exc

# Silence the per-call TileLang kernel-cache WARNING (fires on every fwd/bwd,
# floods logs at 1024 ranks; it is a cache HIT, not a recompile).
try:
    import tilelang as _tilelang
    _tilelang.set_log_level("ERROR")
except Exception:
    import logging as _logging
    _logging.getLogger("tilelang").setLevel(_logging.ERROR)


_M3_PATH_REPORTED = False


def _check_mamba3_sm100_path(q, k, v, mimo_out, z, D, angles, chunk_size, rotary_dim_divisor, dtype, cu_seqlens):
    """Report (once) whether mamba3_mimo will take the OVHai sm100 tcgen05 kernels or fall back to TileLang.

    The decision is mamba_ssm's own (sm100.common.config_ok, same arguments as mamba3_mimo passes it). With
    DRAGON_REQUIRE_FAST_KERNELS=1 a fallback raises instead of silently running ~several x slower."""
    global _M3_PATH_REPORTED
    require = os.environ.get("DRAGON_REQUIRE_FAST_KERNELS", "0") == "1"
    if _M3_PATH_REPORTED and not require:
        return
    try:
        from mamba_ssm.ops.cuda.mamba3.sm100 import common as _sm100
        cu = None if cu_seqlens is None else cu_seqlens.to(torch.int32)
        ok = bool(_sm100.config_ok(q, k, v, v.shape[2], mimo_out, z, D, angles, chunk_size, rotary_dim_divisor,
                                   dtype, False, cu, False))
        why = "" if ok else (
            f"enabled={_sm100.enabled()} cu_seqlens={'set' if cu is not None else 'None'} q={tuple(q.shape)} "
            f"v={tuple(v.shape)} angles={tuple(angles.shape)} rotary_dim_divisor={rotary_dim_divisor} "
            f"dtypes q/k/v/z={q.dtype}/{k.dtype}/{v.dtype}/{z.dtype}")
    except Exception as e:  # pragma: no cover
        ok, why = False, f"mamba_ssm sm100 module unavailable: {e}"
    if not _M3_PATH_REPORTED:
        _M3_PATH_REPORTED = True
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f"[dragon] Mamba-3 MIMO kernels: {'OVHai sm100 tcgen05' if ok else 'TileLang fallback (' + why + ')'}",
                  flush=True)
    if require and not ok:
        raise RuntimeError(f"DRAGON_REQUIRE_FAST_KERNELS=1 but Mamba-3 would use the TileLang fallback: {why}")


def _canon_shape_strides(shape):
    strides, acc = [], 1
    for s in reversed(shape):
        strides.append(acc)
        acc *= s
    return tuple(reversed(strides))


class _CanonView(torch.autograd.Function):
    """as_strided to the canonical strides of an already contiguous tensor, with an identity backward
    (autograd's AsStridedBackward would zero-fill and scatter the whole storage)."""

    @staticmethod
    def forward(ctx, t):
        return t.as_strided(t.shape, _canon_shape_strides(t.shape))

    @staticmethod
    def backward(ctx, grad):
        return grad


def _canon_strides(t):
    """Force canonical C-contiguous strides.

    torch's .contiguous() leaves a size-1 leading dim with a collapsed stride
    (a contiguous (1, L, H, P) tensor can report stride[0]==H*P instead of L*H*P).
    The TileLang MIMO kernel validates strides strictly and rejects this at micro
    batch size 1 (the cu_seqlens / B=1 case). as_strided fixes it without a copy.
    """
    t = t.contiguous()
    canon = _canon_shape_strides(t.shape)
    if tuple(t.stride()) == canon:
        return t
    if not t.requires_grad:
        return t.as_strided(t.shape, canon)
    return _CanonView.apply(t)


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _split_zx_fwd_kernel(IN, Z, X, DT, A, TRAP, S, H, P: tl.constexpr, BT: tl.constexpr, HB: tl.constexpr):
        t = tl.program_id(0) * BT + tl.arange(0, BT)[:, None, None]
        h = tl.program_id(1) * HB + tl.arange(0, HB)[None, :, None]
        j = tl.arange(0, P)[None, None, :]
        m = (t < S) & (h < H)
        W = 2 * P + 3
        src = IN + (t.to(tl.int64) * H + h) * W
        dst = (t.to(tl.int64) * H + h) * P + j
        tl.store(Z + dst, tl.load(src + j, mask=m), mask=m)
        tl.store(X + dst, tl.load(src + P + j, mask=m), mask=m)
        t2 = tl.reshape(t, (BT, 1))
        h2 = tl.reshape(h, (1, HB))
        m2 = (t2 < S) & (h2 < H)
        src2 = IN + (t2.to(tl.int64) * H + h2) * W + 2 * P
        tl.store(DT + t2 * H + h2, tl.load(src2, mask=m2), mask=m2)
        tl.store(A + t2 * H + h2, tl.load(src2 + 1, mask=m2), mask=m2)
        tl.store(TRAP + h2.to(tl.int64) * S + t2, tl.load(src2 + 2, mask=m2), mask=m2)

    @triton.jit
    def _split_zx_bwd_kernel(G, DZ, DX, DDT, DA, DTRAP, S, H, P: tl.constexpr, BT: tl.constexpr, HB: tl.constexpr):
        t = tl.program_id(0) * BT + tl.arange(0, BT)[:, None, None]
        h = tl.program_id(1) * HB + tl.arange(0, HB)[None, :, None]
        j = tl.arange(0, P)[None, None, :]
        m = (t < S) & (h < H)
        W = 2 * P + 3
        dst = G + (t.to(tl.int64) * H + h) * W
        src = (t.to(tl.int64) * H + h) * P + j
        tl.store(dst + j, tl.load(DZ + src, mask=m), mask=m)
        tl.store(dst + P + j, tl.load(DX + src, mask=m), mask=m)
        t2 = tl.reshape(t, (BT, 1))
        h2 = tl.reshape(h, (1, HB))
        m2 = (t2 < S) & (h2 < H)
        dst2 = G + (t2.to(tl.int64) * H + h2) * W + 2 * P
        tl.store(dst2, tl.load(DDT + t2 * H + h2, mask=m2), mask=m2)
        tl.store(dst2 + 1, tl.load(DA + t2 * H + h2, mask=m2), mask=m2)
        tl.store(dst2 + 2, tl.load(DTRAP + h2.to(tl.int64) * S + t2, mask=m2), mask=m2)

    _HAVE_TRITON_SPLIT = True
except ImportError:  # pragma: no cover
    _HAVE_TRITON_SPLIT = False

_SPLIT_BT, _SPLIT_HB = 4, 16


class _SplitZXDtATrap(torch.autograd.Function):
    """Splits the per-head in_proj output (L, B=1, H, 2P + 3) into contiguous z, x (1, L, H, P), dt, A (1, L, H)
    and trap (1, H, L) in one pass. Backward assembles the input gradient in one pass instead of five zero-filled
    slice / select gradients summed by autograd (same values: those sums only add zeros)."""

    @staticmethod
    def forward(ctx, per_head, P):
        ctx.P = P
        ctx.shape = per_head.shape
        L, B, H, W = per_head.shape
        assert W == 2 * P + 3
        dev, dt_ = per_head.device, per_head.dtype
        z = torch.empty((B, L, H, P), dtype=dt_, device=dev)
        x = torch.empty_like(z)
        if _HAVE_TRITON_SPLIT and B == 1 and per_head.is_contiguous() and per_head.is_cuda and (P & (P - 1)) == 0:
            dt = torch.empty((B, L, H), dtype=dt_, device=dev)
            A = torch.empty_like(dt)
            trap = torch.empty((B, H, L), dtype=dt_, device=dev)
            grid = (triton.cdiv(L, _SPLIT_BT), triton.cdiv(H, _SPLIT_HB))
            _split_zx_fwd_kernel[grid](per_head, z, x, dt, A, trap, L, H, P=P, BT=_SPLIT_BT, HB=_SPLIT_HB,
                                       num_warps=4)
        else:
            z.copy_(per_head[..., :P].transpose(0, 1))
            x.copy_(per_head[..., P:2 * P].transpose(0, 1))
            dt = per_head[..., 2 * P].transpose(0, 1).contiguous()
            A = per_head[..., 2 * P + 1].transpose(0, 1).contiguous()
            trap = per_head[..., 2 * P + 2].permute(1, 2, 0).contiguous()
        return z, x, dt, A, trap

    @staticmethod
    def backward(ctx, dz, dx, ddt, dA, dtrap):
        P = ctx.P
        L, B, H, W = ctx.shape
        ref = next(t for t in (dz, dx, ddt, dA, dtrap) if t is not None)
        dev, dty = ref.device, ref.dtype
        g = torch.empty((L, B, H, W), dtype=dty, device=dev)
        if _HAVE_TRITON_SPLIT and B == 1 and dev.type == "cuda" and (P & (P - 1)) == 0:
            fill = lambda t, shape: (torch.zeros(shape, dtype=dty, device=dev) if t is None  # noqa: E731
                                     else t.to(dty).contiguous())
            grid = (triton.cdiv(L, _SPLIT_BT), triton.cdiv(H, _SPLIT_HB))
            _split_zx_bwd_kernel[grid](g, fill(dz, (B, L, H, P)), fill(dx, (B, L, H, P)), fill(ddt, (B, L, H)),
                                       fill(dA, (B, L, H)), fill(dtrap, (B, H, L)), L, H, P=P, BT=_SPLIT_BT,
                                       HB=_SPLIT_HB, num_warps=4)
            return g, None
        for t, sl, perm in ((dz, slice(0, P), (1, 0, 2, 3)), (dx, slice(P, 2 * P), (1, 0, 2, 3)),
                            (ddt, 2 * P, (1, 0, 2)), (dA, 2 * P + 1, (1, 0, 2)), (dtrap, 2 * P + 2, (2, 0, 1))):
            dst = g[..., sl]
            if t is None:
                dst.zero_()
            else:
                dst.copy_(t.permute(*perm))
        return g, None


class _CanonGradStrides(torch.autograd.Function):
    """Identity fwd; canonicalize the gradient's strides in bwd.

    The varlen MIMO backward validates DOUT (grad-of-output) strides just as
    strictly as the forward validates inputs. At B=1 the downstream rearrange
    hands back a grad whose size-1 leading dim has a collapsed stride that
    .contiguous() does not fix; this restores canonical strides for the kernel.
    """

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        return _canon_strides(grad)


logger = logging.getLogger(__name__)

@dataclass
class Mamba3Submodules:
    """
    Contains the module specs for the input and output linear layers.
    """

    in_proj: Union[ModuleSpec, type] = None
    b_norm: Union[ModuleSpec, type] = None
    c_norm: Union[ModuleSpec, type] = None
    rope_proj: Union[ModuleSpec, type] = None
    output_norm: Union[ModuleSpec, type] = None
    dyn_proj: Union[ModuleSpec, type] = None

class FastMamba3(MegatronModule):
    def __init__(
        self,
        config: DragonConfig,
        submodules: Mamba3Submodules,
        layer_number: int,
        input_scalar: float = 1.,
        pg_collection: ProcessGroupCollection = None,
    ):
        if not HAVE_FAST_MAMBA_SSM:
            raise ImportError(
                "MambaSSM is not installed. Please install it with `pip install mamba-ssm`."
            )

        super().__init__(config)
        self.config = config
        self.layer_number = layer_number

        self.d_model = config.hidden_size
        self.d_inner = int(2 * self.d_model)
        self.rope_fraction = self.config.mamba_rope_fraction
        if self.rope_fraction == 0.5:
            self.rotary_dim_divisor = 4
        elif self.rope_fraction == 1.0:
            self.rotary_dim_divisor = 2
        else:
            raise ValueError(f"rope fraction of {self.rope_fraction} is currently not supported.")

        self.A_floor = 1e-4
        self.chunk_size = 64 // self.config.mamba_mimo_dim
        assert pg_collection is not None, "pg_collection must be provided for Mamba3"
        self.pg_collection = pg_collection

        self.use_mem_eff_path = self.config.use_mamba_mem_eff_path
        if not self.use_mem_eff_path:
            log_single_rank(
                logger,
                logging.WARNING,
                (
                    "We are not currently using or functionally testing use_mem_eff_path==False "
                    "for training. It may not work as expected."
                ),
            )

        self.d_state = self.config.mamba_state_dim
        self.headdim = self.config.mamba_head_dim
        self.ngroups = self.config.mamba_num_groups
        self.mimo_dim = self.config.mamba_mimo_dim
        self.mimo_proj_block_order = self.config.mamba_mimo_proj_block_order

        assert self.ngroups == 1
        assert self.d_state is not None and self.d_state > 0
        assert self.headdim is not None and self.headdim > 0
        assert self.ngroups is not None and self.ngroups > 0

        if self.config.mamba_num_heads is not None:
            self.nheads = self.config.mamba_num_heads
            assert self.nheads > 0
            self.d_inner = self.nheads * self.headdim
        else:
            assert self.d_inner % self.headdim == 0, "d_inner must be evenly divisible by headdim"
            self.nheads = self.d_inner // self.headdim
        self.dr_out_dim = self.d_inner // self.mimo_proj_block_order

        self.split_tensor_size = int(self.d_state * self.rope_fraction)
        if self.split_tensor_size % 2 != 0:
            self.split_tensor_size -= 1
        self.num_rope_angles = self.split_tensor_size // 2

        if self.config.fp8:
            assert (2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads) % 16 == 0, (
                "For FP8, the innermost dimension of the Mamba layer "
                "input projection output tensor must be a multiple of 16."
            )

        tp_size = self.pg_collection.tp.size()

        # Ensure that each TP rank gets at least one head:
        assert self.nheads % tp_size == 0, "nheads must be evenly divisble by tp_size"
        self.nheads_per_group = self.nheads // self.ngroups
        self.nheads_local_tp = self.nheads // tp_size

        # Note that we do not need to confirm that `d_inner % tp_size == 0` because
        # `d_inner % headdim == 0`, `nheads = d_inner // headdim`, and `nheads % tp_size == 0`
        self.d_inner_per_group = self.d_inner // self.ngroups
        self.d_inner_local_tp = self.d_inner // tp_size
        self.dr_out_dim_local_tp = self.dr_out_dim // tp_size

        # Ensure that each TP rank gets at least one group:
        #assert self.ngroups % tp_size == 0, "ngroups must be evenly divisible by tp_size"
        self.ngroups_local_tp = self.ngroups // tp_size

        # Ensure that each group has a positive integer number of heads:
        assert self.nheads % self.ngroups == 0, "nheads must be evenly divisible by ngroups"

        if not self.config.use_geodesic_norm:
            args = {
                "alpha_fwd": input_scalar,
                "alpha_bwd": input_scalar,
                "return_layernorm_output": True,
            }
        else:
            args = {
                "alpha_fwd": input_scalar,
                "alpha_bwd": input_scalar,
            }

        self.in_proj = build_module(
            submodules.in_proj,
            self.d_model,
            self.d_inner * 2 + 3 * self.nheads,  # z x dt A trap angle
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="in_proj",
            tp_group=self.pg_collection.tp,
            **args,
        )

        args = {
            "alpha_fwd": input_scalar,
            "alpha_bwd": input_scalar,
        }
        self.in_proj_dyn = build_module(
            submodules.dyn_proj,
            self.config.hidden_size,
            2 * self.ngroups * self.d_state * self.mimo_dim + self.num_rope_angles, # B C angle
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode='duplicated',
            is_expert=False,
            tp_comm_buffer_name='in_proj_dyn',
            **args,
        )
        w = self.in_proj_dyn.weight
        b = getattr(self.in_proj_dyn, "bias", None)
        if self.config.sequence_parallel:
            # doesnt see the same data => sum grads
            setattr(w, 'tp_sync', True)
            if b is not None:
                setattr(b, 'tp_sync', True)
        else:
            # see the same data => avg grads
            setattr(w, 'average_gradients_across_tp_domain', True)
            if b is not None:
                setattr(b, 'average_gradients_across_tp_domain', True)

        self.B_bias = nn.Parameter(torch.ones((self.nheads_local_tp, self.mimo_dim, self.d_state), dtype=torch.float32), requires_grad=True)
        self.C_bias = nn.Parameter(torch.ones((self.nheads_local_tp, self.mimo_dim, self.d_state), dtype=torch.float32), requires_grad=True)
        setattr(self.B_bias, "tensor_model_parallel", True)
        setattr(self.C_bias, "tensor_model_parallel", True)

        self.B_norm = build_module(
            submodules.b_norm,
            hidden_size=self.d_state,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )
        self.C_norm = build_module(
            submodules.c_norm,
            hidden_size=self.d_state,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )
        setattr(self.B_norm.weight, "tp_sync", True)
        setattr(self.C_norm.weight, "tp_sync", True)

        # Initialize up/down MIMO projection (for x and z)
        in_proj_mimo_x_init_weights = torch.ones(self.nheads_local_tp, self.mimo_dim, self.headdim, dtype=torch.float32)/self.mimo_dim
        in_proj_mimo_z_init_weights = torch.ones(self.nheads_local_tp, self.mimo_dim, self.headdim, dtype=torch.float32)
        out_proj_mimo_init_weights = torch.ones(self.nheads_local_tp, self.mimo_dim, self.headdim, dtype=torch.float32)/self.mimo_dim
        self.in_proj_mimo_x = nn.Parameter(in_proj_mimo_x_init_weights, requires_grad=True)
        self.in_proj_mimo_z = nn.Parameter(in_proj_mimo_z_init_weights, requires_grad=True)
        self.out_proj_mimo = nn.Parameter(out_proj_mimo_init_weights, requires_grad=True)
        setattr(self.in_proj_mimo_x, "tensor_model_parallel", True)
        setattr(self.in_proj_mimo_z, "tensor_model_parallel", True)
        setattr(self.out_proj_mimo, "tensor_model_parallel", True)

        with get_cuda_rng_tracker().fork():
            dt_min = 0.001
            dt_max = 0.1
            dt_init_floor = 1e-4
            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            dt = torch.exp(
                torch.rand(
                    self.nheads_local_tp,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
                * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_bias = nn.Parameter(inv_dt)
            # Our initialization would set all Linear.bias to zero,
            # need to mark this one as _no_reinit
            self.dt_bias._no_reinit = True
            # Just to be explicit. Without this we already don't
            # put wd on dt_bias because of the check
            # name.endswith("bias") in param_grouping.py
            self.dt_bias._no_weight_decay = True
            setattr(self.dt_bias, "tensor_model_parallel", True)

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads_local_tp, device=torch.cuda.current_device())) # Keep in fp32
        self.D._no_weight_decay = True # useless flag
        setattr(self.D, "tensor_model_parallel", True)

        self.output_norm = build_module(
            submodules.output_norm,
            hidden_size=self.d_inner,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )
        w = getattr(self.output_norm, "weight", None)
        if w is not None:
            w.tp_sync = True

        self.window_size = 0

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        key_value_states: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        window_size: Optional[Tuple[int, int]] = None, # not used, for compatibility
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ):
        """
        hidden_states: (nL, B, D) / (L B D)
        Returns: same shape as hidden_states
        """

        #self._maintain_float32_params()
        inference_context = deprecate_inference_params(inference_context, inference_params)
        in_inference_mode = inference_context is not None and not self.training
        assert not in_inference_mode

        # Input projection
        nvtx_range_push(suffix="M3_input_proj")
        out, _ = self.in_proj(hidden_states)

        if self.config.use_geodesic_norm:
            zxdtAtrap = out
            normed_hidden_states = hidden_states
        else:
            zxdtAtrap, normed_hidden_states = out
        
        BCangle, _ = self.in_proj_dyn(normed_hidden_states)

        # Window/document resets are handled by cu_seqlens passed to the kernel
        # (built below), NOT by folding windows into the batch dim. The kernel runs
        # at B=1 with the full sequence; the static-shape patch keeps it fast.
        per_head = zxdtAtrap.view(*zxdtAtrap.shape[:-1], self.nheads_local_tp, 2*self.headdim+3)
        # z, x: (B, L, H, p) contiguous; dt, A: (B, L, H); trap: (B, H, L)
        z, x, dt, A, trap = _SplitZXDtATrap.apply(per_head, self.headdim)
        dt = dt.to(torch.float32)

        off = 0
        B     = BCangle[..., off : off + self.ngroups*self.mimo_dim*self.d_state]; off += self.ngroups*self.mimo_dim*self.d_state
        C     = BCangle[..., off : off + self.ngroups*self.mimo_dim*self.d_state]; off += self.ngroups*self.mimo_dim*self.d_state
        angle = BCangle[..., off :]
        B = rearrange(B, "l b (G r n) -> l b r G n", G=self.ngroups, r=self.mimo_dim)
        C = rearrange(C, "l b (G r n) -> l b r G n", G=self.ngroups, r=self.mimo_dim)
        nvtx_range_pop(suffix="M3_input_proj")

        _A = -F.softplus(A.to(torch.float32)) # (B, L, N)
        _A = torch.clamp(_A, max=-self.A_floor)
        dt = F.softplus(dt + self.dt_bias) # (B, L, N)
        ADT = _A * dt

        nvtx_range_push(suffix="M3_mimo_BC_norm")
        B = self.B_norm(B)
        C = self.C_norm(C)

        if self.config.sequence_parallel:
            B = gather_from_sequence_parallel_region(B, group=self.pg_collection.tp)
            C = gather_from_sequence_parallel_region(C, group=self.pg_collection.tp)
            angle = gather_from_sequence_parallel_region(angle, group=self.pg_collection.tp)

        B = _canon_strides(rearrange(B, "l b r G n -> b l r G n"))
        C = _canon_strides(rearrange(C, "l b r G n -> b l r G n"))
        nvtx_range_pop(suffix="M3_mimo_BC_norm")

        # Public kernel applies the angle cumsum internally (varlen-aware), so we
        # pass RAW angles (no external angle_dt).
        # fp32 before the head broadcast: one materialized copy instead of two (bf16 expand, then fp32 cast);
        # the backward sums the head gradients in fp32 before the bf16 cast
        angle = angle.transpose(0, 1).to(torch.float32) # (B, L, S)
        angle = angle.unsqueeze(-2).expand(-1, -1, self.nheads_local_tp, -1).contiguous() # (B, L, H, S)

        ADT = rearrange(ADT, "b l n -> b n l")
        dt = rearrange(dt, "b l n -> b n l")

        # cu_seqlens: reset the SSM state at artificial_seq_len window boundaries
        # AND document boundaries (packed-sequence isolation). This replaces the
        # old reshape-into-batch "complete SLW".
        b_size, L = x.shape[0], x.shape[1]
        cu_seqlens = None
        if packed_seq_params is not None and getattr(packed_seq_params, "cu_seqlens_q", None) is not None:
            cu_seqlens = packed_seq_params.cu_seqlens_q.to(torch.int32)
        if self.config.artificial_seq_len > 0:
            window_boundaries = torch.arange(
                0, L + 1, self.config.artificial_seq_len,
                device=x.device, dtype=torch.int32,
            )
            cu_seqlens = window_boundaries if cu_seqlens is None else \
                torch.unique(torch.cat([window_boundaries, cu_seqlens]), sorted=True)
        assert cu_seqlens is None or b_size == 1, (
            "Mamba-3 cu_seqlens path requires micro batch size 1 at the kernel "
            f"(got batch={b_size}); pack sequences instead of batching."
        )

        _check_mamba3_sm100_path(C, B, x, self.out_proj_mimo, z, self.D, angle, self.chunk_size,
                                 self.rotary_dim_divisor, x.dtype, cu_seqlens)
        y = mamba3_mimo(
            Q=_canon_strides(C),
            K=_canon_strides(B),
            V=_canon_strides(x),
            ADT=_canon_strides(ADT.to(torch.float32)),
            DT=_canon_strides(dt.to(torch.float32)),
            Trap=_canon_strides(trap),
            Q_bias=self.C_bias.to(torch.float32),
            K_bias=self.B_bias.to(torch.float32),
            MIMO_V=self.in_proj_mimo_x.to(torch.float32),
            MIMO_Z=self.in_proj_mimo_z.to(torch.float32),
            MIMO_Out=self.out_proj_mimo.to(torch.float32),
            Angles=_canon_strides(angle.to(torch.float32)),
            D=self.D.to(torch.float32).contiguous(),
            Z=_canon_strides(z),
            chunk_size=self.chunk_size,
            rotary_dim_divisor=self.rotary_dim_divisor,
            dtype=x.dtype,
            return_state=False,
            cu_seqlens=cu_seqlens,
        )
        if isinstance(y, tuple):
            y, new_state = y
        # Canonicalize grad strides for the varlen backward (B=1 collapse).
        y = _CanonGradStrides.apply(y)
        nvtx_range_pop(suffix="M3_mimo_chunk_scan")

        y = rearrange(y, "b l h p -> l b (h p)")

        y = self.output_norm(y)

        return y, normed_hidden_states

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Provide a sharded state dictionary for distributed checkpointing."""
        sharded_state_dict = {}
        # Parameters
        self._save_to_state_dict(sharded_state_dict, "", keep_vars=True)
        
        axis_map = {
            "dt_bias": 0,
            "D": 0,
            "B_bias": 0,
            "C_bias": 0,
            "in_proj_mimo_x": 0,
            "in_proj_mimo_z": 0,
            "out_proj_mimo": 0,
        }

        sharded_state_dict = make_sharded_tensors_for_checkpoint(
            sharded_state_dict,
            prefix,
            tensor_parallel_layers_axis_map=axis_map, # parameters sharded across TP
            sharded_offsets=sharded_offsets,
        )
        # Submodules
        for name, module in self.named_children():
            module_sharded_sd = sharded_state_dict_default(
                module, f"{prefix}{name}.", sharded_offsets, metadata
            )

            sharded_state_dict.update(module_sharded_sd)

        return sharded_state_dict
    
    def _load_from_state_dict(self, *args, **kwargs):
        """Load the state dict of the router."""
        #self._maintain_float32_params() # switch to float32 before loading
        return super()._load_from_state_dict(*args, **kwargs)

    def _save_to_state_dict(self, *args, **kwargs):
        """Save the state dict of the router."""
        #self._maintain_float32_params() # switch to float32 before saving
        return super()._save_to_state_dict(*args, **kwargs)
