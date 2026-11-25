"""Output layer + cross entropy without materializing the logits (QuACK chunked linear cross entropy).

The stock path all-gathers the sequence-parallel hidden states and computes vocab-parallel logits
[S, V/TP] on every TP rank: at S=256k that is one 18.6 GiB bf16 tensor (plus its gradient), which does
not fit next to the model on an 80 GB H100.

Here the parallelism of the head is flipped from vocab-parallel to sequence-parallel:
  * the [V/TP, h] weight shards are all-gathered into the full [V, h] weight (467 MB for 152k x 1536),
  * each TP rank computes the cross entropy of ITS OWN sequence-parallel tokens (S/TP of them) over the
    full vocabulary with QuACK's chunked kernel: logits exist only one [chunk, V] block at a time, and
    the per-token dx and the fp32 dW are produced in the same pass,
  * backward scales dx by the incoming per-token gradient (exact), finishes dW in fp32, reduce-scatters
    it back to the [V/TP, h] shards over TP and accumulates it into main_grad (gradient-accumulation
    fusion, like TE / Megatron linear layers),
  * the per-token losses are all-gathered over TP (no gradient reduction) so the caller gets the same
    [b, s] loss tensor as compute_language_model_loss.

dW is exact when the per-token loss gradient is the same for every non-masked token, which is what
Megatron's loss (sum(loss * mask) / num_tokens, times a global scale) produces: tokens with
loss_mask == 0 are given ignore_index in the forward so they contribute nothing. Set
DRAGON_QUACK_LCE_CHECK=1 to verify that assumption at every backward (host sync).
"""

import os
from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor

from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

try:
    from quack.cross_entropy import cross_entropy_fwd_out  # noqa: F401  (import check)
    from quack.gemm_interface import gemm, gemm_add_inplace
    from quack.linear_cross_entropy import chunked_linear_cross_entropy_fwd

    HAVE_QUACK = True
except ImportError:  # pragma: no cover
    HAVE_QUACK = False

IGNORE_INDEX = -100


def _all_gather_dim0(t: Tensor, group) -> Tensor:
    world = dist.get_world_size(group)
    out = torch.empty((world * t.shape[0], *t.shape[1:]), dtype=t.dtype, device=t.device)
    dist.all_gather_into_tensor(out, t.contiguous(), group=group)
    return out


def _reduce_scatter_dim0(t: Tensor, group) -> Tensor:
    world = dist.get_world_size(group)
    out = torch.empty((t.shape[0] // world, *t.shape[1:]), dtype=t.dtype, device=t.device)
    dist.reduce_scatter_tensor(out, t.contiguous(), op=dist.ReduceOp.SUM, group=group)
    return out


class _SeqParallelChunkedLCE(torch.autograd.Function):
    """Per-token CE of the local sequence-parallel tokens against the TP-gathered full output weight."""

    @staticmethod
    def forward(ctx, x: Tensor, weight_shard: Tensor, target: Tensor, tp_group, chunk_size: int,
                grad_accum_fusion: bool):
        # (grad mode is off inside forward; needs_input_grad says what backward will want)
        need_dx, need_dw = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        w_full = _all_gather_dim0(weight_shard.to(x.dtype), tp_group)  # [V, h]
        loss, dx, dw, last_dlogits, last_x = chunked_linear_cross_entropy_fwd(
            x, w_full, target, chunk_size, IGNORE_INDEX, tuned=True, need_dx=need_dx, need_dw=need_dw)
        del w_full
        ctx.save_for_backward(dx, dw, last_dlogits, last_x, target, weight_shard)
        ctx.tp_group = tp_group
        ctx.grad_accum_fusion = grad_accum_fusion
        return loss

    @staticmethod
    def backward(ctx, g: Tensor):
        dx, dw, last_dlogits, last_x, target, weight = ctx.saved_tensors
        valid = target != IGNORE_INDEX
        g = g.float()
        # The (uniform) gradient of a non-masked token's loss; 0 when this rank has none.
        c = (g * valid).sum() / valid.sum().clamp_min(1)
        if os.environ.get("DRAGON_QUACK_LCE_CHECK", "0") == "1":
            dev = (g - c).abs().masked_fill(~valid, 0).max().item()
            assert dev <= 1e-6 * max(c.abs().item(), 1e-30) + 1e-30, (
                f"quack LCE: non-uniform loss gradient over unmasked tokens (max dev {dev}, c {c.item()})")

        grad_x = dx.mul_(g.unsqueeze(-1).to(dx.dtype)) if dx is not None else None
        if last_dlogits is None:  # the weight did not need a gradient
            return grad_x, None, None, None, None, None

        # dW over the full vocabulary, fp32: earlier chunks are already in `dw`, add the last one.
        if dw is None:
            dw = gemm(last_dlogits.T, last_x, out_dtype=torch.float32)
        else:
            gemm_add_inplace(last_dlogits.T, last_x, dw)
        dw.mul_(c)
        dw_shard = _reduce_scatter_dim0(dw, ctx.tp_group)  # [V/TP, h] fp32
        del dw

        if ctx.grad_accum_fusion and hasattr(weight, "main_grad"):
            weight.main_grad.add_(dw_shard)
            if hasattr(weight, "grad_added_to_main_grad"):
                # Same contract as Megatron's linear layers: a dummy wgrad keeps the DDP hooks on the
                # main backward thread; the real gradient is already in main_grad.
                grad_w = torch.empty(weight.main_grad.shape, dtype=weight.dtype, device=weight.device,
                                     requires_grad=False)
                weight.grad_added_to_main_grad = True
            else:
                grad_w = None
        else:
            grad_w = dw_shard.to(weight.dtype)
        return grad_x, grad_w, None, None, None, None


def sequence_parallel_chunked_lce(
    hidden_states: Tensor,
    weight_shard: Tensor,
    labels: Tensor,
    loss_mask: Optional[Tensor],
    tp_group,
    chunk_size: int = 4096,
    grad_accum_fusion: bool = True,
) -> Tensor:
    """Per-token LM loss [b, s] (fp32) from sequence-parallel hidden states, never forming the logits.

    Args:
        hidden_states: [s/TP, b, h] local sequence-parallel shard (the input of the output layer).
        weight_shard: [V/TP, h] this rank's output-layer weight (vocab-parallel).
        labels: [b, s] full-sequence targets.
        loss_mask: [b, s] or None; tokens with 0 are excluded (ignore_index) from loss and gradients.
    """
    assert HAVE_QUACK, "quack-kernels is required for --quack-chunked-lce"
    tp = dist.get_world_size(tp_group)
    rank = dist.get_rank(tp_group)
    s_local, b, h = hidden_states.shape
    assert labels.shape[1] == s_local * tp, (labels.shape, hidden_states.shape, tp)
    # [b, s] -> [s, b] -> this rank's contiguous sequence-parallel slice, token order (s, b).
    tgt = labels.transpose(0, 1)[rank * s_local:(rank + 1) * s_local]
    if loss_mask is not None:
        m = loss_mask.transpose(0, 1)[rank * s_local:(rank + 1) * s_local]
        tgt = tgt.masked_fill(m == 0, IGNORE_INDEX)
    x = hidden_states.reshape(s_local * b, h)
    loss = _SeqParallelChunkedLCE.apply(
        x, weight_shard, tgt.reshape(-1).long().contiguous(), tp_group, chunk_size, grad_accum_fusion)
    loss = loss.view(s_local, b)
    # Every TP rank gets the full [s, b] losses; backward only keeps this rank's slice (no reduction:
    # the loss is replicated over TP, so each rank's slice gradient is already the right one).
    loss = gather_from_sequence_parallel_region(loss, tensor_parallel_output_grad=False, group=tp_group)
    return loss.transpose(0, 1).contiguous()  # [b, s]
