# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pretrain and SFT GPT."""

import os
import torch

from functools import partial
from typing import List, Optional, Tuple
from megatron.core import mpu, parallel_state
from megatron.training import inprocess_restart
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.enums import ModelType
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.models.dragon import DragonModel
from megatron.core.dragon.dragon_config import DragonConfig
from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_hybrid_data_context_parallel_groups,
)
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.utils import (
    StragglerDetector,
    flatten_batch_for_packed_sequences,
    get_attr_wrapped_model,
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
)
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.core.transformer.multi_token_prediction import get_mtp_ranks
from megatron.core.transformer.multi_token_prediction import (
    mtp_on_this_rank as mtp_on_this_rank_func,
)
from megatron.training import get_args, get_timers, pretrain, print_rank_0
from megatron.training.argument_utils import (
    pretrain_cfg_container_from_args,
    resolve_tokenizer_vocab_size,
)
from megatron.training.arguments import core_transformer_config_from_args, parse_and_validate_args
from megatron.training.global_vars import (
    get_run_config,
    initialize_runtime_services,
    set_run_config,
)
from megatron.training.utils import get_blend_and_blend_per_split, is_first_or_last_pipeline_stage
from megatron.training.datasets.sft_dataset import SFTDataset
from model_provider import model_provider
from gpt_builders import dragon_builder

import transformers

try:
    from megatron.post_training.arguments import add_modelopt_args
    from megatron.post_training.loss_func import loss_func as loss_func_modelopt

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

stimer = StragglerDetector()
g_scheduler = {'scheduler': None}

import traceback, pickle
import torch
old_wrap = torch.distributed.checkpoint.utils._wrap_exception                                                                                                                        
def patched_wrap(exc):
    traceback.print_exception(exc)  # Print the REAL error                                                                                                                           
    exc.__traceback__ = None        # Strip traceback so pickle works                                                                                                                
    return old_wrap(exc)                                                                                                                                                             
torch.distributed.checkpoint.utils._wrap_exception = patched_wrap                                                                                                                    


# Keys a Dragon micro-batch carries (the dataloader may add others, e.g. "dataset_id").
DRAGON_BATCH_KEYS = [
    "tokens", "labels", "loss_mask", "attention_mask", "position_ids", "cu_seqlens", "max_seqlen",
]

_DRAGON_CONFIG = {}


def _dragon_config(args):
    """DragonConfig from args, built once (it is only read here for the MTP / PP layout)."""
    if id(args) not in _DRAGON_CONFIG:
        _DRAGON_CONFIG.clear()
        _DRAGON_CONFIG[id(args)] = core_transformer_config_from_args(args, config_class=DragonConfig)
    return _DRAGON_CONFIG[id(args)]


def _has_cu_seqlens(args):
    """Packed / intra-document-masked batches: cu_seqlens + max_seqlen come from the dataloader."""
    return bool(args.create_cu_seqlens_in_dataloader or args.sft)


def _mtp_on_this_rank(config, vp_stage):
    return mtp_on_this_rank_func(
        layout=config.pipeline_model_parallel_layout,
        mtp_num_layers=config.mtp_num_layers,
        ignore_virtual=False,
        vp_stage=vp_stage,
    )


def get_batch(data_iterator, vp_stage: Optional[int] = None):
    """Generate a batch: (tokens, labels, loss_mask, attention_mask, position_ids, cu_seqlens, max_seqlen)."""
    args = get_args()
    config = _dragon_config(args)
    has_cu_seqlens = _has_cu_seqlens(args)
    mtp_on_this_rank = _mtp_on_this_rank(config, vp_stage)

    # Intermediate pipeline stages need nothing, except the THD metadata (cu_seqlens, max_seqlen).
    if not is_first_or_last_pipeline_stage(vp_stage) and not mtp_on_this_rank and not has_cu_seqlens:
        return None, None, None, None, None, None, None

    tp_rank = mpu.get_tensor_model_parallel_rank()
    batch = {}
    if tp_rank == 0:
        data = next(data_iterator)
        for key in DRAGON_BATCH_KEYS:
            batch[key] = (
                data[key].cuda(non_blocking=True) if key in data and data[key] is not None else None
            )
        # The receivers allocate int32 (mbs, n) cu_seqlens and int32 (mbs,) max_seqlen: send
        # exactly that, whatever the dataset produced (a dtype/shape mismatch makes the NCCL
        # broadcast sizes differ between ranks -> hang; JZ port fix, same contract as before).
        if has_cu_seqlens:
            mbs = args.micro_batch_size
            assert batch['cu_seqlens'] is not None, "the dataset did not produce cu_seqlens"
            batch['cu_seqlens'] = batch['cu_seqlens'].to(torch.int32).reshape(mbs, -1).contiguous()
            if batch['max_seqlen'] is None:
                lens = batch['cu_seqlens'][:, 1:] - batch['cu_seqlens'][:, :-1]
                batch['max_seqlen'] = lens.max(dim=1).values
            batch['max_seqlen'] = batch['max_seqlen'].to(torch.int32).reshape(-1)[:mbs].contiguous()

    batch = get_batch_on_this_tp_rank(
        batch,
        has_cu_seqlens=has_cu_seqlens,
        is_hybrid_cp=False,
        create_attention_mask_in_dataloader=args.create_attention_mask_in_dataloader,
        broadcast_src_rank=mpu.get_tensor_model_parallel_src_rank(),
        broadcast_group=mpu.get_tensor_model_parallel_group(),
        cp_size=args.context_parallel_size,
        tp_rank=tp_rank,
        micro_batch_size=args.micro_batch_size,
        seq_length=args.seq_length,
        mtp_on_this_rank=mtp_on_this_rank,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        is_pipeline_first_stage=mpu.is_pipeline_first_stage(),
        is_pipeline_last_stage=mpu.is_pipeline_last_stage(),
    )
    # (mbs, n) cu_seqlens -> one (1, N) pack; tokens etc. -> (1, mbs * seq)
    batch = flatten_batch_for_packed_sequences(batch)

    if args.context_parallel_size > 1:
        batch = get_batch_on_this_cp_rank(
            batch,
            is_hybrid_cp=False,
            cp_group=get_context_parallel_group(),
            hybrid_cp_group_func=get_hybrid_data_context_parallel_groups,
        )

    return tuple(batch.get(key) for key in DRAGON_BATCH_KEYS)


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10


def loss_func(
    loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: Optional[DragonModel] = None
):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
        model (DragonModel, optional): The model (can be wrapped)

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()
    if has_nvidia_modelopt and getattr(args, 'modelopt_enabled', False):  # [ModelOpt]
        return loss_func_modelopt(loss_mask, output_tensor, model=model)

    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=False,
        )

    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    reporting_loss = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])

    return (loss, num_tokens, {'lm loss': reporting_loss})


def debug_batch(tokens, position_ids, cu_seqlens, max_seqlen):
    tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    # debug to check tokens
    seq_tokens = tokens[0]
    seq_pos_ids = position_ids[0]

    # 2. Find indices where position_id resets to 0
    # .nonzero() gives us the indices where the condition is True
    reset_indices = (seq_pos_ids == 0).nonzero(as_tuple=False).flatten().tolist()

    print_rank_0(f"Found document starts at indices: {reset_indices}")

    # 3. Iterate through these resets and print context
    context_window = 10  # How many tokens to show before/after

    for idx in reset_indices:
        # Define the slice bounds
        start = max(0, idx - context_window)
        end = min(len(seq_tokens), idx + context_window)
        
        # Slice the tensors
        # 'pre_tokens' are the end of the previous document
        # 'post_tokens' are the start of the new document (starting at idx)
        pre_tokens = seq_tokens[start:idx]
        post_tokens = seq_tokens[idx:end]
        
        # Decode to string
        # We use replace('\n', '\\n') so you can see newlines in the logs clearly
        pre_text = tok.decode(pre_tokens).replace('\n', '\\n')
        post_text = tok.decode(post_tokens).replace('\n', '\\n')
        
        print_rank_0(f"--- Reset at Index {idx} ---")
        print_rank_0(f"   End of Doc A: \"...{pre_text}\"")
        print_rank_0(f"   Start of Doc B: \"{post_text}...\"")
        print_rank_0("-" * 30)
    
    # 1. Prepare Data
    # cu_seqlens describes the entire flattened batch. 
    # We flatten position_ids to match the dimension of cu_seqlens.
    flat_pos_ids = position_ids.view(-1) 
    cu_seqlens_cpu = cu_seqlens.cpu().view(-1)

    # 2. Derive expected boundaries from Position IDs (The "Ground Truth")
    # We look for where pos_id is 0. 
    # Note: This finds the START of every document.
    pos_reset_indices = (flat_pos_ids == 0).nonzero(as_tuple=False).flatten().cpu()

    # 3. Derive boundaries from cu_seqlens (The "Metadata")
    # cu_seqlens is [0, len1, len1+len2, ...]. 
    # So the starts are everything except the very last element.
    cu_start_indices = cu_seqlens_cpu[:-1]

    # 4. Compare specific Start Indices
    print_rank_0(f"\n--- Checking cu_seqlens Consistency ---")
    print_rank_0(f"Num documents (pos_ids): {len(pos_reset_indices)}")
    print_rank_0(f"Num documents (cu_seq):  {len(cu_start_indices)}")

    # Check if they match exactly
    if torch.equal(pos_reset_indices, cu_start_indices):
        print_rank_0("✅ cu_seqlens boundaries match position_ids exactly.")
    else:
        print_rank_0("❌ MISMATCH DETECTED!")
        print_rank_0(f"   Pos ID Resets (first 10): {pos_reset_indices[:10].tolist()}")
        print_rank_0(f"   Cu Seq Starts (first 10): {cu_start_indices[:10].tolist()}")
        
        # Identify the first mismatch index
        min_len = min(len(pos_reset_indices), len(cu_start_indices))
        diff = (pos_reset_indices[:min_len] != cu_start_indices[:min_len]).nonzero(as_tuple=False)
        if diff.numel() > 0:
            first_diff = diff[0].item()
            print_rank_0(f"   First divergence at doc index {first_diff}:")
            print_rank_0(f"   -> Pos ID says start is at: {pos_reset_indices[first_diff].item()}")
            print_rank_0(f"   -> cu_seqlens says start is at: {cu_start_indices[first_diff].item()}")

    # 5. Check Max Seqlen
    # Calculate individual lengths: cu_seqlens[i+1] - cu_seqlens[i]
    doc_lengths = cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]
    calculated_max = doc_lengths.max().item()
    provided_max = max_seqlen if isinstance(max_seqlen, int) else max_seqlen.item()

    print_rank_0(f"\n--- Checking max_seqlen Consistency ---")
    print_rank_0(f"Calculated Max (from cu_seqlens): {calculated_max}")
    print_rank_0(f"Provided Max (variable):          {provided_max}")

    if calculated_max == provided_max:
        print_rank_0("✅ max_seqlen is correct.")
    else:
        print_rank_0(f"❌ max_seqlen is WRONG. Diff: {provided_max - calculated_max}")

    print_rank_0("-" * 30)



_NAN_HOOKS_INSTALLED = False


def _install_nan_hooks(model):
    """DRAGON_NAN_HOOKS=1: print (every rank) the first modules whose forward output is non-finite, the
    batch's cu_seqlens summary, and non-finite parameters. Debug aid for the Jean Zay port."""
    global _NAN_HOOKS_INSTALLED
    if _NAN_HOOKS_INSTALLED or os.environ.get("DRAGON_NAN_HOOKS", "0") != "1":
        return
    _NAN_HOOKS_INSTALLED = True
    rank = torch.distributed.get_rank()
    reported = {"n": 0}
    for name, p in model.named_parameters():
        if not torch.isfinite(p.detach()).all():
            print(f"[nan-hook r{rank}] non-finite PARAM {name}", flush=True)

    def first(t):
        if isinstance(t, torch.Tensor):
            return t
        if isinstance(t, (tuple, list)):
            for x in t:
                y = first(x)
                if y is not None:
                    return y
        return None

    def mk(name):
        def hook(mod, inp, out):
            if reported["n"] >= 5:
                return
            o = first(out)
            if o is not None and o.is_floating_point() and not torch.isfinite(o).all():
                i = first(inp)
                ifin = None if i is None or not i.is_floating_point() else bool(torch.isfinite(i).all())
                reported["n"] += 1
                print(f"[nan-hook r{rank}] non-finite OUTPUT of {name} ({type(mod).__name__}) "
                      f"shape={tuple(o.shape)} nonfinite={(~torch.isfinite(o)).sum().item()} input_finite={ifin}", flush=True)
        return hook

    for name, mod in model.named_modules():
        mod.register_forward_hook(mk(name))


def forward_step(data_iterator, model: DragonModel, return_schedule_plan: bool = False):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (DragonModel): The Dragon Model
        return_schedule_plan (bool): Whether to return the schedule plan instead of the output tensor
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        vp_stage = get_attr_wrapped_model(model, "vp_stage")
        tokens, labels, loss_mask, attention_mask, position_ids, cu_seqlens, max_seqlen = get_batch(data_iterator, vp_stage)
        #print(f"PP rank : {parallel_state.get_pipeline_model_parallel_rank()} VPP rank : {parallel_state.get_virtual_pipeline_model_parallel_rank()} got batch with tokens shape {tokens.shape if tokens is not None else 'None'} and position_ids shape {position_ids.shape if position_ids is not None else 'None'} and cu_seqlens shape {cu_seqlens.shape if cu_seqlens is not None else 'None'} and max_seqlen {max_seqlen if max_seqlen is not None else 'None'}")
    timers('batch-generator').stop()

    _install_nan_hooks(model)
    if os.environ.get("DRAGON_NAN_HOOKS", "0") == "1" and cu_seqlens is not None:
        cs = cu_seqlens.view(-1)
        print(f"[nan-hook r{torch.distributed.get_rank()}] tokens={tuple(tokens.shape)} n_cu={cs.numel()} "
              f"cu_head={cs[:4].tolist()} cu_tail={cs[-3:].tolist()} max_seqlen={max_seqlen}", flush=True)

    #debug_batch(tokens, position_ids, cu_seqlens, max_seqlen)

    # check init mean & std
    """with torch.no_grad():
        for name, p in model.named_parameters():
            if p is None or p.numel() == 0:
                continue
            t = p.detach().float()
            mean = t.mean().item()
            std  = t.std(unbiased=False).item()
            print_rank_0(f"{name:60s} shape={tuple(p.shape)} mean={mean:+.4e} std={std:.4e}")"""

    packed_seq_params = None
    if cu_seqlens is not None:
        assert model.config.intra_doc_masking
        cu_seqlens = cu_seqlens.reshape(-1).to(torch.int32)
        max_seqlen = int(max_seqlen.max().item())
        packed_seq_params = PackedSeqParams(
            qkv_format='thd',
            position_ids=position_ids.squeeze(0) if position_ids is not None else None,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
        )

    with stimer:
        if return_schedule_plan:
            assert args.overlap_moe_expert_parallel_comm, \
                "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
            schedule_plan = model.build_schedule_plan(
                tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask, window_size=(g_scheduler["scheduler"].get_wsize(), 0), packed_seq_params=packed_seq_params,
            )
            return schedule_plan, partial(loss_func, loss_mask, model=model)
        else:
            output_tensor = model(
                tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask, window_size=(g_scheduler["scheduler"].get_wsize(), 0), packed_seq_params=packed_seq_params,
            )

    # [ModelOpt]: model is needed to access ModelOpt distillation losses
    return output_tensor, partial(loss_func, loss_mask, model=model)


def is_dataset_built_on_rank(vp_stage=None):
    args = get_args()
    if parallel_state.get_tensor_model_parallel_rank() != 0:
        return False
    if _has_cu_seqlens(args):
        # Every pipeline stage needs the THD metadata, so every TP-src rank reads the data.
        return True
    return is_first_or_last_pipeline_stage(vp_stage) or _mtp_on_this_rank(_dragon_config(args), vp_stage)


def core_gpt_dataset_config_from_args(args):
    cfg = get_run_config()
    tokenizer = build_tokenizer(args)

    # Sometimes --data-path is too long, instead we parse it from a file.
    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    if args.create_cu_seqlens_in_dataloader:
        assert not args.eod_mask_loss, "we don't want to mask the loss of the first sequence!"

    return GPTDatasetConfig(
        random_seed=cfg.rng.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        multiple_validation_sets=args.multiple_validation_sets,
        full_validation=args.full_validation,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        # Dragon intra-document masking: (unpadded, int32) cu_seqlens from the dataloader
        create_cu_seqlens=args.create_cu_seqlens_in_dataloader,
        object_storage_cache_path=args.object_storage_cache_path,
        mid_level_dataset_surplus=args.mid_level_dataset_surplus,
        allow_ambiguous_pad_tokens=args.allow_ambiguous_pad_tokens,
        fast_cache_load=args.dataloader_fast_cache_load,
        defer_npy_index_mmap=args.dataloader_defer_npy_index_mmap,
        context_parallel_size=args.context_parallel_size,
        data_parallel_size=args.data_parallel_size,
        sequence_parallel_size=args.tensor_model_parallel_size * args.sequence_parallel,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if args.sft:
        dataset_type = SFTDataset
    else:
        if args.mock_data:
            dataset_type = MockGPTDataset
        else:
            dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type, train_val_test_num_samples, partial(is_dataset_built_on_rank, vp_stage=vp_stage), config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


def get_embedding_ranks(pp_ranks: List[int]):
    """Get the embedding ranks."""
    embedding_ranks = [pp_ranks[0]]
    if len(pp_ranks) > 1:
        args = get_args()
        if not args.untie_embeddings_and_output_weights:
            embedding_ranks.append(pp_ranks[-1])
        config = core_transformer_config_from_args(args)
        mtp_ranks = get_mtp_ranks(pp_ranks, config)
        embedding_ranks.extend(mtp_ranks)
    embedding_ranks = list(set(embedding_ranks))
    embedding_ranks = sorted(embedding_ranks)
    return embedding_ranks


if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    # Optionally enable inprocess restart on pretrain
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    args = parse_and_validate_args(
        extra_args_provider=add_modelopt_args if has_nvidia_modelopt else None,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )
    # Dragon is built by a model provider (dragon_builder), not by a ModelConfig: model_cfg=None.
    full_config = pretrain_cfg_container_from_args(args, model_cfg=None)
    set_run_config(full_config)
    initialize_runtime_services(args, training=True)
    resolve_tokenizer_vocab_size(full_config, args.padded_vocab_size)
    pretrain(
        full_config,
        train_valid_test_datasets_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        model_provider=partial(model_provider, dragon_builder),
        store=store,
        get_embedding_ranks=get_embedding_ranks,
        g_scheduler=g_scheduler,
    )
