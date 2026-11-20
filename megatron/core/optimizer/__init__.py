# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import copy
import logging
import math
import warnings
from collections import defaultdict
from dataclasses import astuple
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.optim import SGD as CPUSGD
from torch.optim import AdamW as CPUAdam

try:
    from transformer_engine.pytorch.optimizers import FusedAdam as Adam
    from transformer_engine.pytorch.optimizers import FusedSGD as SGD

    USING_PYTORCH_OPTIMIZER = False
except ImportError:
    try:
        from apex.optimizers import FusedAdam as Adam
        from apex.optimizers import FusedSGD as SGD

        USING_PYTORCH_OPTIMIZER = False
    except ImportError:
        warnings.warn(
            f'Transformer Engine and Apex are not installed. Falling back to Torch optimizers.'
        )

        # Apex's FusedAdam is a drop-in replacement for torch's AdamW.
        # pylint: disable-next=line-too-long.
        # See https://github.com/NVIDIA/apex/blob/7b73b12361068a10b0f44844534613f252a5ea75/apex/optimizers/fused_adam.py#L16.
        from torch.optim import SGD
        from torch.optim import AdamW as Adam

        USING_PYTORCH_OPTIMIZER = True

try:
    # Dragon: AdEMAMix from the OVHai TransformerEngine fork.
    from transformer_engine.pytorch.optimizers import FusedAdemamix as Ademamix
except ImportError:
    Ademamix = None

try:
    import transformer_engine as te

    # Dragon's uscaling / CompleteDP param groups walk TE linear modules (TE fork: row_parallel).
    _TE_LINEAR_TYPES = (te.pytorch.Linear, te.pytorch.LayerNormLinear, te.pytorch.GroupedLinear)
except ImportError:
    te = None
    _TE_LINEAR_TYPES = ()

try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    _eo_ver = tuple(int(x) for x in _pkg_version('emerging-optimizers').split('.')[:2])
except (ImportError, PackageNotFoundError):
    _eo_ver = (0, 0)

HAVE_EMERGING_OPTIMIZERS = _eo_ver >= (0, 2)

if HAVE_EMERGING_OPTIMIZERS:
    from emerging_optimizers.scalar_optimizers import Lion

from megatron.core import parallel_state
from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer
from megatron.core.optimizer_param_scheduler import (
    ParamGroupOverride,
    combine_param_group_overrides,
    param_group_override_to_tuple,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.fsdp_dtensor_checkpoint import get_global_unique_param_name

from ..distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallelV2
from ..distributed.param_and_grad_buffer import _ParamAndGradBuffer
from ..transformer.module import MegatronModule
from ..utils import get_model_config, get_pg_rank, get_pg_size, is_te_min_version, log_single_rank
from .distrib_optimizer import DistributedOptimizer
from .emerging_optimizers import (
    _EMERGING_OPTIMIZERS,
    HAVE_EMERGING_OPTIMIZERS,
    _create_emerging_optimizer,
    _get_qkv_split_shapes,
)
from .fully_sharded_optimizer import FullyShardedOptimizer
from .grad_scaler import ConstantGradScaler, DynamicGradScaler
from .layer_wise_optimizer import LayerWiseDistributedOptimizer, is_managed_by_layer_wise_optimizer
from .optimizer import (
    ChainedOptimizer,
    Float16OptimizerWithFloat16Params,
    FP32Optimizer,
    MegatronOptimizer,
    param_group_identifier_keys,
)

# Subclass aliases kept for backward compatibility; all are OptimizerConfig.
from .optimizer_config import (
    AdamOptimizerConfig,
    OptimizerConfig,
    ParamKey,
    ParamPredicate,
    ParamWithNamePredicate,
    SGDOptimizerConfig,
)

logger = logging.getLogger(__name__)


def get_standard_config_overrides(config: OptimizerConfig) -> Dict[ParamKey, ParamGroupOverride]:
    """Get standard config overrides for the optimizer, handling decoupled LR and common wd skips.

    Args:
        config (OptimizerConfig): optimizer configuration object.

    Returns:
        Dict[ParamKey, ParamGroupOverride]: standard config overrides.
    """
    config_overrides: Optional[Dict[ParamKey, ParamGroupOverride]] = {}
    # Select the model-family convention for zero weight decay on vector-like parameters:
    # the classic rule skips all 1-D parameters and biases, while the Qwen3-Next rule keeps
    # weight decay on Q/K layernorm parameters. Wide-residual retention controllers intentionally
    # follow the run's ordinary weight-decay policy rather than this generic vector exemption.
    if config.apply_wd_to_qk_layernorm:
        shape_1_not_qkln_param = ParamWithNamePredicate(
            name="s1_not_qkln",
            fn=lambda param, name: (len(param.shape) == 1 or name.endswith(".bias"))
            and not ("q_layernorm." in name or "k_layernorm." in name)
            and not getattr(param, "is_wide_residual_retention_parameter", False),
        )
        param_wd_mult_key = ParamKey(with_name_predicate=shape_1_not_qkln_param)
    else:
        param_length_1_match = ParamPredicate(
            name="param_len_1_except_wide_residual_retention",
            fn=lambda param: len(param.shape) == 1
            and not getattr(param, "is_wide_residual_retention_parameter", False),
        )
        param_wd_mult_key = ParamKey(name="*.bias", predicate=param_length_1_match)

    config_overrides[param_wd_mult_key] = ParamGroupOverride(wd_mult=0.0)

    if config.decoupled_lr is not None:
        decoupled_lr_config: ParamGroupOverride = {"max_lr": config.decoupled_lr}
        decoupled_param_key = ParamKey(attr="is_embedding_or_output_parameter")
        if config.decoupled_min_lr is not None:
            decoupled_lr_config["min_lr"] = config.decoupled_min_lr
        config_overrides[decoupled_param_key] = decoupled_lr_config

    return config_overrides


def get_mup_config_overrides(
    config: OptimizerConfig, mup_width_mult: float, optimizer_type: str = 'adam'
) -> Dict[ParamKey, ParamGroupOverride]:
    """Get MuP config overrides for per-layer LR and Adam epsilon scaling.

    In MuP, optimizer learning rates are adjusted by parameter class to ensure
    stable update scales across model widths and enable hyperparameter transfer.

    MuP optimizer scaling rules (as implemented here):
    - Adam/AdamW:
      - hidden (matrix-like) lr = base_lr / width_mult
      - hidden (matrix-like) eps = base_eps / width_mult
      - vector-like params keep base lr and eps
    - SGD:
      - vector-like lr = base_lr * width_mult
      - hidden (matrix-like) lr keeps base_lr in the current uniform-width setup
      - no eps override is applied
    - Non-Adam optimizers:
      - hidden (matrix-like) lr = base_lr / width_mult
      - no eps override is applied.
      - for Muon optimizers, matrix-like params managed by Muon itself are
        excluded from these Adam-style MuP overrides.

    With decoupled_lr enabled, embedding/output params continue using decoupled LR
    and MuP will not override those explicit decoupled values.

    Args:
        config (OptimizerConfig): optimizer configuration object.
        mup_width_mult (float): Width multiplier (hidden_size / base_hidden_size).
        optimizer_type (str): Optimizer type string from config.optimizer.

    Returns:
        Dict[ParamKey, ParamGroupOverride]: MuP optimizer overrides.
    """
    optimizer_type_lower = optimizer_type.lower()
    is_sgd_optimizer = optimizer_type_lower == 'sgd'
    is_adam_optimizer = 'adam' in optimizer_type_lower
    is_muon_optimizer = 'muon' in optimizer_type_lower

    decoupled_lr_enabled = config.decoupled_lr is not None
    if decoupled_lr_enabled:
        message = (
            "Both decoupled_lr and MuP LR scaling are enabled. decoupled_lr sets an "
            "absolute LR for embedding+output params, and MuP LR scaling will not "
            "override those parameters."
        )
        if is_adam_optimizer:
            message += " MuP Adam epsilon scaling remains applied to hidden matrix-like parameters."
        log_single_rank(logger, logging.WARNING, message)

    if is_muon_optimizer:
        muon_scale_mode = getattr(config, 'muon_scale_mode', 'spectral')
        if muon_scale_mode == 'spectral':
            log_single_rank(
                logger,
                logging.WARNING,
                "Both MuP and muon_scale_mode=spectral are enabled. "
                "Muon-managed matrix parameters will continue using spectral Muon scaling. "
                "Set --muon-scale-mode unit_rms_norm to use unit_rms_norm scaling for "
                "Muon-managed matrices with MuP.",
            )

    if mup_width_mult == 1.0:
        # No scaling needed when width_mult is 1
        return {}

    hidden_lr_mult = 1.0 / mup_width_mult
    base_lr = config.lr
    base_min_lr = config.min_lr

    # Hidden matrix-like layers get scaled LR/eps; vector-like params keep base values.
    # Prefer the explicit parameter attribute set by LanguageModule. Fall back to
    # a conservative name check for older or non-language modules.
    def is_embedding_parameter(param: torch.nn.Parameter, param_name: str) -> bool:
        if getattr(param, 'shared_embedding', False):
            return True
        if hasattr(param, 'is_embedding_parameter'):
            return bool(param.is_embedding_parameter)
        return 'embedding' in param_name.lower()

    def is_vector_like_parameter(param: torch.nn.Parameter, param_name: str) -> bool:
        if is_embedding_parameter(param, param_name):
            return True
        if param.dim() <= 1:
            return True
        return False

    def is_muon_managed_matrix_parameter(param: torch.nn.Parameter, _: str) -> bool:
        if not is_muon_optimizer:
            return False
        return is_managed_by_layer_wise_optimizer(param)

    def should_scale_lr_with_mup(param: torch.nn.Parameter, param_name: str) -> bool:
        if decoupled_lr_enabled and getattr(param, 'is_embedding_or_output_parameter', False):
            return False
        if is_muon_managed_matrix_parameter(param, param_name):
            return False
        return not is_vector_like_parameter(param, param_name)

    def should_scale_vector_like_lr_with_mup(param: torch.nn.Parameter, param_name: str) -> bool:
        if decoupled_lr_enabled and getattr(param, 'is_embedding_or_output_parameter', False):
            return False
        return is_vector_like_parameter(param, param_name)

    def should_scale_eps_with_mup(param: torch.nn.Parameter, param_name: str) -> bool:
        if is_vector_like_parameter(param, param_name):
            return False
        if is_muon_managed_matrix_parameter(param, param_name):
            return False
        # MuP Appendix B.3: eps scales with fan_in when non-negligible.
        # This implementation follows the common denominator form: sqrt(v) + eps.
        return True

    mup_overrides: Dict[ParamKey, ParamGroupOverride] = {}

    if is_sgd_optimizer:
        vector_like_lr_mult = mup_width_mult
        vector_like_lr_override: ParamGroupOverride = {}
        if base_lr is not None:
            vector_like_lr_override["max_lr"] = base_lr * vector_like_lr_mult
        if base_min_lr is not None:
            vector_like_lr_override["min_lr"] = base_min_lr * vector_like_lr_mult

        if vector_like_lr_override:
            vector_like_predicate = ParamWithNamePredicate(
                name="mup_sgd_vector_like_excluding_embedding_output",
                fn=should_scale_vector_like_lr_with_mup,
            )
            mup_overrides[ParamKey(with_name_predicate=vector_like_predicate)] = (
                vector_like_lr_override
            )

        return mup_overrides

    lr_override: ParamGroupOverride = {}
    if base_lr is not None:
        lr_override["max_lr"] = base_lr * hidden_lr_mult
    if base_min_lr is not None:
        lr_override["min_lr"] = base_min_lr * hidden_lr_mult

    eps_override: ParamGroupOverride = {}
    if is_adam_optimizer and config.adam_eps is not None:
        eps_override["eps"] = config.adam_eps * hidden_lr_mult

    if decoupled_lr_enabled:
        if lr_override:
            hidden_predicate = ParamWithNamePredicate(
                name="mup_hidden_only_excluding_embedding_output", fn=should_scale_lr_with_mup
            )
            mup_overrides[ParamKey(with_name_predicate=hidden_predicate)] = lr_override

        if eps_override:
            hidden_output_predicate = ParamWithNamePredicate(
                name="mup_hidden_only_for_adam_eps", fn=should_scale_eps_with_mup
            )
            mup_overrides[ParamKey(with_name_predicate=hidden_output_predicate)] = eps_override
    else:
        combined_override: ParamGroupOverride = {}
        combined_override.update(lr_override)
        combined_override.update(eps_override)
        if combined_override:
            hidden_output_predicate = ParamWithNamePredicate(
                name="mup_hidden_and_output", fn=should_scale_eps_with_mup
            )
            mup_overrides[ParamKey(with_name_predicate=hidden_output_predicate)] = combined_override

    return mup_overrides


def _get_param_groups_uscaling(
    model_chunks: List[MegatronModule],
    base_lr: float,
    lr_mult_emb: float,
    lr_mult_scalar: float,
    lr_mult_head: float,
    min_lr: Optional[float] = None,
    process_group: Optional[torch.distributed.ProcessGroup] = None,
) -> List[Dict]:
    """Dragon u-muP ("uscaling") param groups: per-tensor lr_mult from fan-in, truly decoupled WD."""

    # Map (wd_mult, lr_mult, is_expert_parallel, is_decoupled_lr) to params.
    params_map = {}
    for model_chunk in model_chunks:
        seen = set()
        for name, mod in model_chunk.named_modules():
            if isinstance(mod, _TE_LINEAR_TYPES):
                assert hasattr(mod, "row_parallel")
                size_mult = 1
                if mod.row_parallel:
                    if not isinstance(mod, te.pytorch.GroupedLinear):
                        size_mult = parallel_state.get_tensor_model_parallel_world_size()
                    else:
                        size_mult = parallel_state.get_expert_tensor_parallel_world_size()

                if isinstance(mod, te.pytorch.GroupedLinear):
                    is_grouped = True
                    num_weights = mod.num_gemms
                else:
                    is_grouped = False
                    num_weights = 1

                for w_idx in range(num_weights if num_weights > 0 else 1):
                    weight_attr = "weight" if not is_grouped else f"weight{w_idx}"
                    weight = getattr(mod, weight_attr)

                    if not weight.requires_grad:
                        continue           

                    is_expert_parallel = not (getattr(mod, 'allreduce', True) and getattr(weight, 'allreduce', True))
                    is_decoupled_lr = False
                    is_scalar = getattr(mod, "is_scalar_weight", False)

                    # compute wd_mult, lr_mult
                    if "output_layer" in name:
                        _lr_mult = lr_mult_head
                        _wd_mult = 0.
                    elif is_scalar:
                        _lr_mult = lr_mult_scalar
                        _wd_mult = 0.
                    else: # hidden matrix weight
                        fan_in = size_mult * weight.shape[1]
                        _lr_mult = 1 / math.sqrt(fan_in)
                        _wd_mult = 1.
                    _wd_mult = _wd_mult / (base_lr * _lr_mult) # truly decoupled weight decay

                    if parallel_state.get_data_parallel_rank() == 0 and parallel_state.get_tensor_model_parallel_rank() == 0:
                        print(f"param {name}.{weight_attr} | shape {weight.shape} (multiplied: {size_mult}) | lr_mult={_lr_mult:.3e} | wd_mult={_wd_mult:.3e} | is_expert_parallel={is_expert_parallel}")

                    key = (_wd_mult, _lr_mult, is_expert_parallel, is_decoupled_lr)
                    if key not in params_map:
                        params_map[key] = []
                    params_map[key].append(weight)
                    seen.add(weight)

                    bias_attr = "bias" if num_weights == 1 else f"bias{w_idx}"
                    bias = getattr(mod, bias_attr, None)
                    if bias is not None and bias.requires_grad:
                        is_expert_parallel = not getattr(bias, 'allreduce', True)
                        is_decoupled_lr = False
                        _lr_mult = lr_mult_scalar
                        _wd_mult = 0.

                        _wd_mult = _wd_mult / (base_lr * _lr_mult) # truly decoupled weight decay

                        key = (_wd_mult, _lr_mult, is_expert_parallel, is_decoupled_lr)
                        if key not in params_map:
                            params_map[key] = []
                        params_map[key].append(bias)
                        seen.add(bias)

                        #if parallel_state.get_data_parallel_rank() == 0:
                        #    print(f"param {name}.{bias_attr} | shape {mod.bias.shape} | lr_mult={_lr_mult:.3e} | wd_mult={_wd_mult:.3e}")

        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue
            if param in seen:
                continue

            is_expert_parallel = not getattr(param, 'allreduce', True)
            is_decoupled_lr = False

            # compute wd_mult, lr_mult
            if "embedding" in name:
                _lr_mult = lr_mult_emb
                _wd_mult = 0.
            elif "router.weight" in name:
                fan_in = size_mult * param.shape[1]
                _lr_mult = 1 / math.sqrt(fan_in)
                _wd_mult = 1.
            else:
                _lr_mult = lr_mult_scalar
                _wd_mult = 0.
            
            if getattr(param, 'requires_weight_decay', False):
                _wd_mult = 1.0

            _wd_mult = _wd_mult / (base_lr * _lr_mult) # truly decoupled weight decay

            key = (_wd_mult, _lr_mult, is_expert_parallel, is_decoupled_lr)
            if key not in params_map:
                params_map[key] = []
            params_map[key].append(param)

            if parallel_state.get_data_parallel_rank() == 0 and parallel_state.get_tensor_model_parallel_rank() == 0:
                print(f"param {name} | shape {param.shape} | lr_mult={_lr_mult:.3e} | wd_mult={_wd_mult:.3e} | is_expert_parallel={is_expert_parallel}")
    
    # Distributed checkpoint requires all ranks to have the same param groups,
    # so we need to align the param groups across ranks, otherwise we may have
    # runtime error when loading the checkpoint or numerical error when resuming training.
    params_key = list(params_map.keys())
    gathered_params_key = [None] * torch.distributed.get_world_size(group=process_group)
    torch.distributed.all_gather_object(gathered_params_key, params_key, group=process_group)
    for keys in gathered_params_key:
        for key in keys:
            if key not in params_key:
                params_key.append(key)

    param_groups = []
    for key in params_key:
        wd_mult, _lr_mult, is_expert_parallel, is_decoupled_lr = key
        params = params_map[key] if key in params_map else []
        param_group = {
            'params': params,
            'wd_mult': wd_mult,
            'lr_mult': _lr_mult,
            'is_expert_parallel': is_expert_parallel,
            'is_decoupled_lr': is_decoupled_lr,
            # upstream keys: the scheduler reads max_lr / min_lr per group; lr = sched * lr_mult
            'default_config': True,
            'max_lr': base_lr,
            'min_lr': min_lr,
        }
        param_groups.append(param_group)

    return param_groups


def _get_param_groups_completedp(
    model_chunks: List[MegatronModule],
    base_lr_hidden: float,
    base_lr_emb: float,
    base_lr_scalar: float,
    base_lr_head: float,
    base_lr_expert: float,
    base_wd_hidden: float,
    base_eps: float,
    alpha_completedp: float,
    rho_adjusted: float,
    width_adjusted: float,
    depth_adjusted: float,
    min_lr: float = None,
    process_group: Optional[torch.distributed.ProcessGroup] = None,
) -> List[Dict]:
    """Dragon CompleteDP param groups: per-tensor LR / WD / Adam-eps multipliers from the width,
    depth and horizon ratios to the base run (lr_mult = lr / base_lr_hidden, wd_mult = wd /
    base_wd_hidden; the scheduler multiplies its LR by lr_mult)."""

    MAX_NAME  = 90   # tweak
    MAX_SHAPE = 14   # tweak

    def _clip(s: str, w: int) -> str:
        return s if len(s) <= w else (s[:w-3] + "...")
    def _shape_str(shape) -> str:
        return "x".join(map(str, shape))

    # Map (wd_mult, lr_mult, eps, is_expert_parallel, is_decoupled_lr) to params.
    params_map = {}
    for model_chunk in model_chunks:
        seen = set()
        for name, mod in model_chunk.named_modules():
            if isinstance(mod, _TE_LINEAR_TYPES):
                assert hasattr(mod, "row_parallel")

                if isinstance(mod, te.pytorch.GroupedLinear):
                    is_grouped = True
                    num_weights = mod.num_gemms
                else:
                    is_grouped = False
                    num_weights = 1

                for w_idx in range(num_weights if num_weights > 0 else 1):
                    weight_attr = "weight" if not is_grouped else f"weight{w_idx}"
                    weight = getattr(mod, weight_attr)

                    if not weight.requires_grad:
                        continue           

                    is_expert_parallel = not (getattr(mod, 'allreduce', True) and getattr(weight, 'allreduce', True))
                    is_decoupled_lr = False

                    # compute wd_mult, lr_mult
                    if "output_layer" in name:
                        base_lr = base_lr_head
                        scale_lr = (width_adjusted ** (-1)) * rho_adjusted
                        lr = base_lr * scale_lr

                        base_wd = base_wd_hidden # TODO: wd or not ?
                        scale_wd = width_adjusted * rho_adjusted
                        wd = base_wd * scale_wd

                        base_eps = base_eps
                        scale_eps = 1/rho_adjusted
                        eps = base_eps * scale_eps
                    else: # hidden matrix weight
                        base_lr = base_lr_hidden
                        if "experts.linear_fc" in name and "shared" not in name:
                            base_lr = base_lr_expert
                        scale_lr = (width_adjusted ** (-1)) * (depth_adjusted ** (alpha_completedp-1)) * rho_adjusted
                        lr = base_lr * scale_lr

                        base_wd = base_wd_hidden
                        scale_wd = (width_adjusted) * rho_adjusted
                        wd = base_wd * scale_wd

                        base_eps = base_eps
                        scale_eps = ((width_adjusted) ** (-1)) * (depth_adjusted ** (-alpha_completedp)) * 1/rho_adjusted
                        eps = base_eps * scale_eps

                    if parallel_state.get_data_parallel_rank() == 0 and parallel_state.get_tensor_model_parallel_rank() == 0:
                        if not "mlp.experts.linear" in name:
                            print(
                                f"param {_clip(name, MAX_NAME):<{MAX_NAME}}"
                                f" | shape {_shape_str(weight.shape):>{MAX_SHAPE}}"
                                f" | lr={lr:9.3e} | wd={wd:9.3e} | eps={eps:9.3e}"
                                f" | is_expert_parallel={is_expert_parallel}"
                            )

                    key = (wd/base_wd_hidden, lr/base_lr_hidden, eps, is_expert_parallel, is_decoupled_lr)
                    if key not in params_map:
                        params_map[key] = []
                    params_map[key].append(weight)
                    seen.add(weight)

                    bias_attr = "bias" if num_weights == 1 else f"bias{w_idx}"
                    bias = getattr(mod, bias_attr, None)
                    if bias is not None and bias.requires_grad:
                        is_expert_parallel = not getattr(bias, 'allreduce', True)
                        is_decoupled_lr = False

                        base_lr = base_lr_scalar
                        scale_lr = (depth_adjusted ** (alpha_completedp-1)) * rho_adjusted
                        lr = base_lr * scale_lr

                        wd = 0.

                        base_eps = base_eps
                        if "output_layer" in name:
                            scale_eps = 1/rho_adjusted
                        else:
                            scale_eps = ((width_adjusted) ** (-1)) * (depth_adjusted ** (-alpha_completedp)) * 1/rho_adjusted
                        eps = base_eps * scale_eps

                        key = (wd/base_wd_hidden, lr/base_lr_hidden, eps, is_expert_parallel, is_decoupled_lr)
                        if key not in params_map:
                            params_map[key] = []
                        params_map[key].append(bias)
                        seen.add(bias)

                        if parallel_state.get_data_parallel_rank() == 0 and parallel_state.get_tensor_model_parallel_rank() == 0:
                            print(
                                f"param {_clip(name, MAX_NAME):<{MAX_NAME}}"
                                f" | shape {_shape_str(bias.shape):>{MAX_SHAPE}}"
                                f" | lr={lr:9.3e} | wd={wd:9.3e} | eps={eps:9.3e}"
                                f" | is_expert_parallel={is_expert_parallel}"
                            )

        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue
            if param in seen:
                continue

            is_expert_parallel = not getattr(param, 'allreduce', True)
            is_decoupled_lr = False

            # compute wd_mult, lr_mult
            if "embedding" in name:
                base_lr = base_lr_emb
                scale_lr = rho_adjusted
                lr = base_lr * scale_lr

                base_wd = 0.
                scale_wd = rho_adjusted
                wd = base_wd * scale_wd

                base_eps = base_eps
                scale_eps = ((width_adjusted) ** (-1)) * 1/rho_adjusted
                eps = base_eps * scale_eps
            elif "final_layernorm" in name:
                base_lr = base_lr_scalar
                scale_lr = rho_adjusted
                lr = base_lr * scale_lr

                base_wd = 0.
                scale_wd = rho_adjusted
                wd = base_wd * scale_wd

                base_eps = base_eps
                scale_eps = 1/rho_adjusted
                eps = base_eps * scale_eps
            elif "router.weight" in name: # TODO: for the router, same base lr as hidden? same WD? it has a finite second dimension
                base_lr = base_lr_hidden
                scale_lr = (width_adjusted ** (-1)) * (depth_adjusted ** (alpha_completedp-1)) * rho_adjusted
                lr = base_lr * scale_lr

                base_wd = base_wd_hidden
                scale_wd = (width_adjusted) * rho_adjusted
                wd = base_wd * scale_wd

                base_eps = base_eps
                scale_eps = ((width_adjusted) ** (-1)) * (depth_adjusted ** (-alpha_completedp)) * 1/rho_adjusted
                eps = base_eps * scale_eps
            else:
                base_lr = base_lr_scalar
                scale_lr = (depth_adjusted ** (alpha_completedp-1)) * rho_adjusted
                lr = base_lr * scale_lr

                base_wd = 0.
                scale_wd = rho_adjusted
                wd = base_wd * scale_wd

                base_eps = base_eps
                if not("q_layernorm" in name or "k_layernorm" in name):
                    scale_eps = (width_adjusted ** (-1)) * (depth_adjusted ** (-alpha_completedp)) * 1/rho_adjusted
                else:
                    scale_eps = (depth_adjusted ** (-alpha_completedp)) * 1/rho_adjusted
                eps = base_eps * scale_eps

            assert not getattr(param, 'requires_weight_decay', False)

            key = (wd/base_wd_hidden, lr/base_lr_hidden, eps, is_expert_parallel, is_decoupled_lr)
            if key not in params_map:
                params_map[key] = []
            params_map[key].append(param)

            if parallel_state.get_data_parallel_rank() == 0 and parallel_state.get_tensor_model_parallel_rank() == 0:
                print(
                    f"param {_clip(name, MAX_NAME):<{MAX_NAME}}"
                    f" | shape {_shape_str(param.shape):>{MAX_SHAPE}}"
                    f" | lr={lr:9.3e} | wd={wd:9.3e} | eps={eps:9.3e}"
                    f" | is_expert_parallel={is_expert_parallel}"
                )

    # Distributed checkpoint requires all ranks to have the same param groups,
    # so we need to align the param groups across ranks, otherwise we may have
    # runtime error when loading the checkpoint or numerical error when resuming training.
    params_key = list(params_map.keys())
    gathered_params_key = [None] * torch.distributed.get_world_size(group=process_group)
    torch.distributed.all_gather_object(gathered_params_key, params_key, group=process_group)
    for keys in gathered_params_key:
        for key in keys:
            if key not in params_key:
                params_key.append(key)

    param_groups = []
    for key in params_key:
        wd_mult, _lr_mult, eps, is_expert_parallel, is_decoupled_lr = key
        params = params_map[key] if key in params_map else []
        param_group = {
            'params': params,
            'wd_mult': wd_mult,
            'lr_mult': _lr_mult,
            'eps': eps,
            'is_expert_parallel': is_expert_parallel,
            'is_decoupled_lr': is_decoupled_lr,
            # upstream keys: the scheduler reads max_lr / min_lr per group; lr = sched * lr_mult
            'default_config': True,
            'max_lr': base_lr_hidden,
        }
        if min_lr is not None:
            param_group['min_lr'] = min_lr
        param_groups.append(param_group)

    return param_groups


def _get_dragon_legacy_config_overrides(
    config: OptimizerConfig,
    config_overrides: Optional[Dict[ParamKey, ParamGroupOverride]],
    no_weight_decay_cond: Optional[Callable],
    scale_lr_cond: Optional[Callable],
    lr_mult: float,
    default_skip_embedding_weight_decay: bool,
) -> Optional[Dict[ParamKey, ParamGroupOverride]]:
    """Translate the pre-0.16 (no_weight_decay_cond, scale_lr_cond, lr_mult,
    default_skip_embedding_weight_decay) arguments Dragon's training.py still passes into upstream
    ParamKey config overrides. Returns config_overrides unchanged when none of them is set."""
    if (
        no_weight_decay_cond is None
        and scale_lr_cond is None
        and not default_skip_embedding_weight_decay
    ):
        return config_overrides
    overrides = dict(config_overrides) if config_overrides is not None else None
    if no_weight_decay_cond is not None:
        # The condition replaces the default "bias / 1-D" no-WD rule (pre-0.16 semantics).
        overrides = {
            k: v
            for k, v in (overrides or get_standard_config_overrides(config)).items()
            if set(v.keys()) != {'wd_mult'}
        }
        overrides[
            ParamKey(
                with_name_predicate=ParamWithNamePredicate(
                    name="dragon_no_weight_decay_cond",
                    fn=lambda param, name: bool(no_weight_decay_cond(name, param)),
                )
            )
        ] = ParamGroupOverride(wd_mult=0.0)
    elif default_skip_embedding_weight_decay:
        overrides = overrides if overrides is not None else get_standard_config_overrides(config)
        overrides[
            ParamKey(
                with_name_predicate=ParamWithNamePredicate(
                    name="dragon_skip_embedding_wd", fn=lambda param, name: "embedding" in name
                )
            )
        ] = ParamGroupOverride(wd_mult=0.0)
    if scale_lr_cond is not None and lr_mult != 1.0:
        overrides = overrides if overrides is not None else get_standard_config_overrides(config)
        scaled: ParamGroupOverride = {"max_lr": config.lr * lr_mult}
        if config.min_lr is not None:
            scaled["min_lr"] = config.min_lr * lr_mult
        overrides[
            ParamKey(
                with_name_predicate=ParamWithNamePredicate(
                    name="dragon_scale_lr_cond",
                    fn=lambda param, name: bool(scale_lr_cond(name, param)),
                )
            )
        ] = scaled
    return overrides


def _get_param_groups(
    model_chunks: List[MegatronModule],
    config: OptimizerConfig,
    config_overrides: Optional[Dict[ParamKey, ParamGroupOverride]],
    process_group: Optional[torch.distributed.ProcessGroup] = None,
) -> List[Dict]:
    """Create parameter groups for optimizer.

    Creates parameter groups from provided optimizer config object.

    NOTE There can be more than one match between a ParamKey and a parameter.
        What we do is merge all of the matching ParamKey overrides into a single ParamGroupOverride
        for that parameter and use that as the key for that parameter. Any parameters that get
        the same set of merged overrides will be mapped into the same parameter group.

    Args:
        model_chunks (List[MegatronModule]): model chunks to create parameter
            groups for.
        config (OptimizerConfig): optimizer configuration object.
        config_overrides (Optional[Dict[ParamKey, ParamGroupOverride]): optimizer overrides,
            specified on a per-layer basis. NOTE: if you want to skip applying weight decay on bias
            and length 1 parameters, and also do not want to do any other overrides, set this to an
            empty dictionary rather than the default value of None.
        process_group (Optional[torch.distributed.ProcessGroup]): group whose ranks must construct
            aligned parameter groups. ``None`` preserves the WORLD-group behavior.
    Returns:
        List of parameter groups.
    """

    # Map (pg_overrides, is_expert_parallel) to params.
    params_map = {}

    for model_chunk in model_chunks:
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue

            uses_default_config = False
            # Get optimizer config overrides for this parameter.
            param_overrides_list: list[ParamGroupOverride] = []
            if config_overrides is not None:
                for param_key, param_override in config_overrides.items():
                    if param_key.matches(param, name):
                        param_overrides_list.append(param_override)

            if param_overrides_list:
                param_override: ParamGroupOverride | None = combine_param_group_overrides(
                    param_overrides_list
                )
            else:
                param_override = None

            is_expert_parallel = not getattr(param, 'allreduce', True)

            # Create config_tuple that is hash-able, and has a consistent ordering of the keys.
            param_override_tuple: tuple[tuple[str, Any], ...] | None = (
                param_group_override_to_tuple(param_override)
            )
            key = (param_override_tuple, is_expert_parallel)
            if key not in params_map:
                params_map[key] = []
            params_map[key].append(param)

    # Distributed checkpoint requires all ranks to have the same param groups,
    # so we need to align the param groups across ranks, otherwise we may have
    # runtime error when loading the checkpoint or numerical error when resuming training.
    params_key = list(params_map.keys())
    gathered_params_key = [None] * torch.distributed.get_world_size(group=process_group)
    torch.distributed.all_gather_object(gathered_params_key, params_key, group=process_group)
    for keys in gathered_params_key:
        for key in keys:
            if key not in params_key:
                params_key.append(key)
    # Need to pick one of the param_override_tuples to use for the param group.
    param_groups = []
    # Sort keys, None first.
    for key in sorted(params_key, key=lambda x: (x[0] is not None, x[0])):
        param_override_tuple, is_expert_parallel = key
        params = params_map[key] if key in params_map else []
        if param_override_tuple is None:
            param_override: ParamGroupOverride = {}
        else:
            param_override: ParamGroupOverride = {k: v for (k, v) in param_override_tuple}

        # False if param_group_override is None or empty tuple or if we do not modify the
        #  LR schedule.
        #  NOTE: "default_config" is used for logging the learning rate in training.py.
        #   so set to True if we do not modify the learning rate.
        #  if param_group['default_config']:
        #    learning_rate = param_group['lr']
        uses_default_lr_schedule: bool = (not bool(param_override_tuple)) or not any(
            ["lr" in k for k in param_override]
        )

        # TODO: Remove "backwards compatible" fields below eventually.
        default_config: ParamGroupOverride = {
            'wd_mult': 1.0,
            'lr_mult': 1.0,
            'is_decoupled_lr': False,
            # The following two fields may be important to keep even when we remove the
            #   above "backwards compatible" fields.
            "max_lr": config.lr,  # user may override this in param_override
            "min_lr": config.min_lr,  # user may override this in param_override
        }
        assert (
            "params" not in param_override
        ), "'params' should not be in param_override, this is a protected key"
        param_group = {
            'params': params,
            'is_expert_parallel': is_expert_parallel,
            'default_config': uses_default_lr_schedule,
            **default_config,
            **param_override,  # keep **param_override last so that users can override other fields.
        }
        param_groups.append(param_group)

    return param_groups


def _get_param_groups_and_buffers(
    model_chunks: List[MegatronModule],
    model_chunk_offset: int,
    config: OptimizerConfig,
    config_overrides: Optional[Dict[ParamKey, ParamGroupOverride]],
    filter_fn: Callable,
    buffer_name: str,
    process_group: Optional[torch.distributed.ProcessGroup] = None,
    uscaling: bool = False,
    completedp: bool = False,
    alpha_completedp: Optional[float] = None,
    rho_adjusted: Optional[float] = None,
    width_adjusted: Optional[float] = None,
    depth_adjusted: Optional[float] = None,
) -> Tuple[List[Dict], Dict[int, List[_ParamAndGradBuffer]]]:
    """Returns parameter groups and buffer for optimizer.

    Args:
        model_chunks (List[MegatronModule]): model chunks to create parameter
            groups for.
        model_chunk_offset (int): offset of model_chunks in global model_chunks list.
        config (OptimizerConfig): optimizer configuration object.
        config_overrides (Optional[Dict[ParamKey, ParamGroupOverride]): optimizer/scheduler
            overrides, specified on the basis of ParamKey matches with each parameter.
        lr (float): learning rate.
        min_lr (float): minimum learning rate.
        filter_fn (callable): filtering function for param_groups.
        buffer_name (str): name of buffer.
        process_group (Optional[torch.distributed.ProcessGroup]): group used to align parameter
            groups across ranks. ``None`` preserves the WORLD-group behavior.

    Returns:
        List of parameter groups and dictionary of model chunk IDs to buffers.
    """
    if uscaling:
        assert config.lr_mult_emb is not None
        assert config.lr_mult_scalar is not None
        assert config.lr_mult_head is not None
        param_groups = _get_param_groups_uscaling(
            model_chunks,
            base_lr=config.lr,
            lr_mult_emb=config.lr_mult_emb,
            lr_mult_scalar=config.lr_mult_scalar,
            lr_mult_head=config.lr_mult_head,
            min_lr=config.min_lr,
            process_group=process_group,
        )
    elif completedp:
        assert config.lr_emb is not None
        assert config.lr_scalar is not None
        assert config.lr_head is not None
        param_groups = _get_param_groups_completedp(
            model_chunks,
            base_lr_hidden=config.lr,
            base_lr_emb=config.lr_emb,
            base_lr_scalar=config.lr_scalar,
            base_lr_head=config.lr_head,
            base_lr_expert=config.lr_expert if config.lr_expert is not None else config.lr,
            base_wd_hidden=config.weight_decay,
            base_eps=config.adam_eps,
            alpha_completedp=alpha_completedp,
            rho_adjusted=rho_adjusted,
            width_adjusted=width_adjusted,
            depth_adjusted=depth_adjusted,
            min_lr=config.min_lr,
            process_group=process_group,
        )
    else:
        param_groups = _get_param_groups(model_chunks, config, config_overrides, process_group)
    param_groups = list(filter(filter_fn, param_groups))
    buffers = {}
    for model_chunk_idx, model_chunk in enumerate(model_chunks):
        if hasattr(model_chunk, buffer_name):
            buffers[model_chunk_idx + model_chunk_offset] = getattr(model_chunk, buffer_name)

    return param_groups, buffers


def _get_megatron_optimizer_based_on_param_groups(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    param_groups: List,
    per_model_buffers: Optional[Dict[int, List[_ParamAndGradBuffer]]] = None,
    model_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    data_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    data_parallel_group_gloo: Optional[torch.distributed.ProcessGroup] = None,
    data_parallel_group_idx: Optional[int] = None,
    intra_dist_opt_group: Optional[torch.distributed.ProcessGroup] = None,
    distributed_optimizer_instance_id: Optional[int] = 0,
    pg_collection: Optional[ProcessGroupCollection] = None,
    skip_megatron_wrapping: bool = False,
) -> Union[MegatronOptimizer, Tuple[Optional[torch.optim.Optimizer], Optional[Callable]]]:
    """Get Megatron optimizer based on parameter groups.

    Args:
        config (OptimizerConfig): optimizer configuration object.
        model_chunks (list): list of model chunks.
        param_groups (list): list of parameter groups.
        per_model_buffers (dict, optional): buffers for distributed optimizer. Defaults to None.
        data_parallel_group (torch.distributed.ProcessGroup, optional): data-parallel group for
            distributed optimizer. Defaults to None.
        data_parallel_group_gloo (torch.distributed.ProcessGroup, optional): gloo data-parallel
            group for distributed optimizer. Defaults to None.
        data_parallel_group_idx (int, optional): data-parallel group index for distributed
            optimizer. Defaults to None.
        distributed_optimizer_instance_id (int, optional): Distributed optimizer instance. Defaults
            0.
        skip_megatron_wrapping (bool): if True, return a
            ``(optimizer, init_state_fn)`` tuple of the raw PyTorch optimizer
            without any Megatron wrapping. Useful when the caller
            (e.g. LayerWiseDistributedOptimizer) performs its own wrapping.

    Returns:
        Instance of MegatronOptimizer, or ``(optimizer, init_state_fn)`` when
        *skip_megatron_wrapping=True*.
    """
    # All param_groups passed here must belong to the same optimizer type (adam / sgd).
    # Callers are responsible for splitting by optimizer type before calling this function.

    if skip_megatron_wrapping and config.use_precision_aware_optimizer:
        raise ValueError(
            "skip_megatron_wrapping=True is incompatible with use_precision_aware_optimizer."
        )
    if skip_megatron_wrapping and config.optimizer_cpu_offload:
        raise ValueError("skip_megatron_wrapping=True is incompatible with optimizer_cpu_offload.")

    # When freezing sub-models we may have no trainable parameters on a rank and
    # hence an empty param_groups. However, we still need to create an optimizer
    # for the purposes of grad stats reductions.
    if param_groups:
        if config.optimizer_cpu_offload:
            assert config.optimizer != 'ademamix', "AdEMAMix does not support optimizer CPU offload"
            if torch.__version__ < '2.3.0':
                warnings.warn(
                    "CPU offload is recommended for PyTorch >= 2.3.0, "
                    "untested versions below this may have convergence issues."
                )
            assert (
                config.decoupled_weight_decay
            ), "CPU offloading only supported with decoupled_weight_decay enabled (AdamW mode)."
            gpu_optimizer_cls = Adam if config.optimizer == 'adam' else SGD
            cpu_optimizer_cls = CPUAdam if config.optimizer == 'adam' else CPUSGD
            if config.use_torch_optimizer_for_cpu_offload:
                gpu_optimizer_cls = cpu_optimizer_cls
            if config.optimizer == 'adam':
                gpu_optimizer_cls = Adam
                cpu_optimizer_cls = CPUAdam
                optimizer_defaults = dict(
                    lr=config.lr,
                    weight_decay=config.weight_decay,
                    betas=(config.adam_beta1, config.adam_beta2),
                    eps=config.adam_eps,
                    bias_correction=True,
                    fused=True,  # this flag is used to improve the performance of the cpu optimizer
                )
            else:
                gpu_optimizer_cls = SGD
                cpu_optimizer_cls = CPUSGD
                optimizer_defaults = dict(
                    lr=config.lr, weight_decay=config.weight_decay, momentum=config.sgd_momentum
                )
            optimizer = HybridDeviceOptimizer(
                param_groups,
                offload_fraction=config.optimizer_offload_fraction,
                cpu_optimizer_cls=cpu_optimizer_cls,
                gpu_optimizer_cls=gpu_optimizer_cls,
                overlap_cpu_optimizer_d2h_h2d=config.overlap_cpu_optimizer_d2h_h2d,
                pin_cpu_grads=config.pin_cpu_grads,
                pin_cpu_params=config.pin_cpu_params,
                param_update_in_fp32=True,
                **optimizer_defaults,
            )
            init_state_fn = None
        elif config.optimizer == 'adam':
            kwargs = {
                "params": param_groups,
                "lr": config.lr,
                "weight_decay": config.weight_decay,
                "betas": (config.adam_beta1, config.adam_beta2),
                "eps": config.adam_eps,
                "capturable": config.optimizer_cuda_graph,
            }

            # set Adam class and weight decay mode depending
            # on source of optimizer (Torch or TE/Apex)
            if USING_PYTORCH_OPTIMIZER:
                adam_cls = torch.optim.AdamW if config.decoupled_weight_decay else torch.optim.Adam
            else:
                kwargs["adam_w_mode"] = config.decoupled_weight_decay
                adam_cls = Adam

            if config.use_precision_aware_optimizer:
                kwargs.update(
                    {
                        "exp_avg_dtype": config.exp_avg_dtype,
                        "exp_avg_sq_dtype": config.exp_avg_sq_dtype,
                    }
                )
                # Master weight is managed by MCore when main_params_dtype is fp32. This is
                # because we want to use fp8 primary weight with precision aware optimizer.
                # Otherwise, master weight will be managed by TransformerEngine.
                # Delayed scaling is an exception because casting as well as the computation
                # of the scaling factor can be conducted in the adam kernel.
                if config.use_precision_aware_optimizer_no_fp8_or_ds_fp8 and not isinstance(
                    model_chunks[0], FullyShardedDataParallelV2
                ):
                    kwargs.update(
                        {
                            "master_weights": True,
                            "use_decoupled_grad": True,
                            "master_weight_dtype": config.main_params_dtype,
                        }
                    )

                if is_te_min_version("2.1.0.dev0"):
                    kwargs.update({"store_param_remainders": config.store_param_remainders})

            optimizer = adam_cls(**kwargs)

            def init_state_fn(opt, config=None):
                for group in opt.param_groups:
                    for p in group['params']:
                        if len(opt.state[p]) == 0:
                            if config is None or not config.use_precision_aware_optimizer:
                                opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                                opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)
                            else:
                                opt.initialize_state(p)

        elif config.optimizer == 'ademamix':
            # Dragon: TE fork's FusedAdemamix (betas = (beta1, beta2, beta3), alpha; the
            # scheduler warms up beta3 / alpha every step).
            if Ademamix is None:
                raise ImportError(
                    "--optimizer ademamix needs the OVHai TransformerEngine fork (FusedAdemamix)."
                )
            kwargs = {
                "params": param_groups,
                "lr": config.lr,
                "weight_decay": config.weight_decay,
                "betas": (config.adam_beta1, config.adam_beta2, config.ademamix_beta3),
                "alpha": config.ademamix_alpha,
                "eps": config.adam_eps,
            }

            if config.use_precision_aware_optimizer:
                kwargs.update(
                    {
                        "exp_avg_dtype": config.exp_avg_dtype,
                        "exp_avg_sq_dtype": config.exp_avg_sq_dtype,
                    }
                )
                if config.use_precision_aware_optimizer_no_fp8_or_ds_fp8 and not isinstance(
                    model_chunks[0], FullyShardedDataParallelV2
                ):
                    kwargs.update(
                        {
                            "master_weights": True,
                            "use_decoupled_grad": True,
                            "master_weight_dtype": config.main_params_dtype,
                        }
                    )

                if is_te_min_version("2.1.0.dev0"):
                    kwargs.update({"store_param_remainders": config.store_param_remainders})

            optimizer = Ademamix(**kwargs)

            def init_state_fn(opt, config=None):
                for group in opt.param_groups:
                    for p in group['params']:
                        if len(opt.state[p]) == 0:
                            if config is None or not config.use_precision_aware_optimizer:
                                opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                                opt.state[p]['exp_avg_slow'] = torch.zeros_like(p.data)
                                opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)
                            else:
                                opt.initialize_state(p)

        elif config.optimizer == 'lion':
            if not HAVE_EMERGING_OPTIMIZERS:
                raise ImportError(
                    "Lion optimizer requires emerging_optimizers >= 0.2. "
                    "Please install or upgrade it to use --optimizer lion."
                )
            optimizer = Lion(  # pylint: disable=possibly-used-before-assignment
                param_groups,
                lr=config.lr,
                betas=(config.lion_beta1, config.lion_beta2),
                weight_decay=config.weight_decay,
            )

            def init_state_fn(opt, config=None):
                for group in opt.param_groups:
                    for p in group['params']:
                        if len(opt.state[p]) == 0:
                            opt.state[p]['exp_avg'] = torch.zeros_like(p.data)

        elif config.optimizer == 'sgd':
            optimizer = SGD(
                param_groups,
                lr=config.lr,
                weight_decay=config.weight_decay,
                momentum=config.sgd_momentum,
            )
            init_state_fn = None
        else:
            raise Exception('{} optimizer is not supported.'.format(config.optimizer))
    else:
        optimizer = None
        init_state_fn = None

    if skip_megatron_wrapping:
        return optimizer, init_state_fn

    # Grad scaler:
    #    if loss-scale is provided, instantiate the constant scaler.
    #    if we are using fp16 and loss-scale is not present, use a
    #       dynamic scaler.
    #    otherwise we are running in bf16 with no loss-scale so
    #       leave it as None.
    grad_scaler = None
    if config.fp16 or config.bf16 or config.use_distributed_optimizer:
        if config.loss_scale:
            grad_scaler = ConstantGradScaler(config.loss_scale)
        elif config.fp16:
            grad_scaler = DynamicGradScaler(
                initial_scale=config.initial_loss_scale,
                min_scale=config.min_loss_scale,
                growth_factor=2.0,
                backoff_factor=0.5,
                growth_interval=config.loss_scale_window,
                hysteresis=config.hysteresis,
            )

    if config.use_distributed_optimizer:
        optimizer = DistributedOptimizer(
            optimizer,
            config,
            grad_scaler,
            init_state_fn,
            model_chunks=model_chunks,
            per_model_buffers=per_model_buffers,
            data_parallel_group=data_parallel_group,
            data_parallel_group_gloo=data_parallel_group_gloo,
            data_parallel_group_idx=data_parallel_group_idx,
            distributed_optimizer_instance_id=distributed_optimizer_instance_id,
        )
        # This is needed for case where num_distributed_optimizer_instances > 1. In this case,
        # weight gradients are all-reduced across optimizer instances, so each instance has
        # the duplicated weight gradients, need to reduce gradient stats inside each instance.
        setattr(optimizer, 'grad_stats_parallel_group', intra_dist_opt_group)
    elif isinstance(model_chunks[0], FullyShardedDataParallelV2):
        optimizer = FullyShardedOptimizer(
            optimizer, config, grad_scaler, init_state_fn, model_chunks=model_chunks
        )
        # get_grad_norm divides each rank's contribution by that gradient's replication
        # factor, so summing it over every rank is correct by construction.
        setattr(optimizer, 'grad_stats_parallel_group', torch.distributed.group.WORLD)
    elif config.fp16 or config.bf16:
        optimizer = Float16OptimizerWithFloat16Params(optimizer, config, grad_scaler, init_state_fn)
        setattr(optimizer, 'grad_stats_parallel_group', model_parallel_group)
    else:
        # FP32 optimizer.
        assert grad_scaler is None
        optimizer = FP32Optimizer(optimizer, config, init_state_fn)
        setattr(optimizer, 'grad_stats_parallel_group', model_parallel_group)

    if pg_collection is None or not hasattr(pg_collection, 'tp'):
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    tp_group = pg_collection.tp
    expert_tp_group = getattr(pg_collection, 'expt_tp', tp_group)
    # TODO(M4): plumb TP groups through optimizer constructors so these setattrs disappear.
    setattr(optimizer, 'tp_group', tp_group)
    setattr(optimizer, 'expert_tp_group', expert_tp_group)
    # The GTP axes this optimizer's params are sharded over. Taken from the collection rather
    # than MPU because a MIMO module owns its axes through its own grid and leaves the MPU
    # globals unset, where the duplicate filter would read rank 0 everywhere.
    setattr(optimizer, 'gtp_group', getattr(pg_collection, 'gtp_remat', None))
    setattr(optimizer, 'expert_gtp_group', getattr(pg_collection, 'expt_gtp_remat', None))

    return optimizer


def check_config_overrides_consistency(
    config: OptimizerConfig, config_overrides: Optional[Dict[ParamKey, ParamGroupOverride]]
):
    """Check if the config overrides are consistent with the config."""

    # TODO: Remove `optimizer` from this eventually (e.g., if we use Muon for some layers and
    # Adam for other layers). This would need some more refactoring to work though (param_groups
    # filtered by optimizer passed into _get_megatron_optimizer_based_on_param_groups).
    if config_overrides is not None:
        fields_to_check_for_consistency = [
            'overlap_param_gather_with_optimizer_step',
            'optimizer',
            'optimizer_cpu_offload',
        ]
        for field_name in fields_to_check_for_consistency:
            base_field = getattr(config, field_name, None)
            all_config_overrides = list(config_overrides.values())
            for config_override in all_config_overrides:
                if field_name in config_override:
                    field = config_override[field_name]
                    if field != base_field:
                        raise ValueError(
                            f"Field {field_name} should not be overriden in a config override."
                        )
    return True


def _get_megatron_emerging_optimizer(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    config_overrides: Optional[Dict[ParamKey, Any]] = None,
    pg_collection: Optional[ProcessGroupCollection] = None,
    param_group_process_group: Optional[torch.distributed.ProcessGroup] = None,
) -> MegatronOptimizer:
    """Build an emerging optimizer (e.g. Muon) for the given model chunks.

    Parameter separation (e.g., linear weights -> Muon, rest -> Adam) is expressed as a
    config_override, the same mechanism used for weight-decay and learning-rate overrides.
    Adam/SGD groups are delegated to _get_megatron_optimizer_based_on_param_groups so they
    go through the exact same code path as the standard optimizer factory.

    When ``config.use_layer_wise_distributed_optimizer`` is True, the underlying optimizers
    are wrapped with :class:`LayerWiseDistributedOptimizer`.
    """
    eopt_name = config.optimizer
    use_layer_wise = config.use_layer_wise_distributed_optimizer

    # Handle legacy "dist_*" optimizer names (e.g. "dist_muon" → "muon" + layer-wise).
    if eopt_name.startswith('dist_'):
        bare_name = eopt_name[len('dist_') :]
        warnings.warn(
            f"optimizer='{eopt_name}' is deprecated. "
            f"Use optimizer='{bare_name}' with use_layer_wise_distributed_optimizer=True.",
            DeprecationWarning,
            stacklevel=3,
        )
        eopt_name = bare_name
        use_layer_wise = True

    if isinstance(model_chunks[0], FullyShardedDataParallelV2):
        raise NotImplementedError("MFSDP v2 with emerging optimizers is not currently validated.")

    if not HAVE_EMERGING_OPTIMIZERS:
        raise ImportError(
            f"emerging-optimizers package is required for optimizer='{eopt_name}'. "
            "Install it with: pip install emerging-optimizers"
        )
    assert not (use_layer_wise and config.overlap_param_gather_with_optimizer_step), (
        "overlap_param_gather_with_optimizer_step is not supported with "
        "use_layer_wise_distributed_optimizer: the emerging-optimizer path does not "
        "split model_chunks into (first, rest) groups, so the per-chunk param-gather "
        "dispatch never fires. Disable one of the two flags."
    )
    if eopt_name not in _EMERGING_OPTIMIZERS:
        raise ValueError(f"Unsupported emerging optimizer: {eopt_name}")
    if config.fp16:
        raise ValueError('emerging optimizer with fp16 is not supported.')

    if pg_collection is None:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    log_single_rank(logger, logging.INFO, f'Setting up emerging optimizer with config {config}')

    # Tag parameters with optimizer-specific attributes (expert_tp, is_qkv).
    for model_chunk in model_chunks:
        qkv_split_shapes = None
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue
            if 'experts' in name and 'shared' not in name:
                param.expert_tp = True
            # TODO(deyuf): support MLA
            if 'linear_qkv.weight' in name and len(param.shape) == 2:
                if qkv_split_shapes is None:
                    qkv_split_shapes = _get_qkv_split_shapes(model_chunk.config)
                # MUST be pre-GTP-sharding rows, not param.shape[0] (this rank's shard):
                # a shard-local test flips as the GTP degree changes, giving the SAME
                # weight two different Muon update rules depending on parallel layout.
                rows_before_gtp_sharding = (
                    param._unsharded_shape[0]
                    if getattr(param, 'is_gtp_weight_remat', False)
                    else param.shape[0]
                )
                if rows_before_gtp_sharding % sum(qkv_split_shapes) == 0:
                    param.is_qkv = True
                    param.qkv_split_shapes = qkv_split_shapes
                else:
                    log_single_rank(
                        logger,
                        logging.DEBUG,
                        f"Emerging optimizer QKV split skipped for {name}: "
                        f"shape={tuple(param.shape)}, split_shapes={qkv_split_shapes}",
                    )

    # Apply optimizer-specific default param overrides (e.g. muon: non-linear -> adam).
    # For Muon-family optimizers, the scalar optimizer that handles non-linear/embedding
    # params is configurable via ``config.muon_scalar_optimizer`` (e.g., 'adam' or 'lion');
    # deep-copy the registry defaults before rewriting so we never mutate shared state.
    default_param_overrides = copy.deepcopy(_EMERGING_OPTIMIZERS[eopt_name].default_param_overrides)
    if eopt_name in ('muon', 'adaptive_muon'):
        for override in default_param_overrides.values():
            if override.get('optimizer') in ('adam', 'lion'):
                override['optimizer'] = config.muon_scalar_optimizer
    config_overrides.update(default_param_overrides)

    # Build param groups and bucket by (optimizer_name, is_expert_parallel).
    # Layer-wise distributed optimizer handles expert params internally so we skip that split.
    all_param_groups = _get_param_groups(
        model_chunks, config, config_overrides, param_group_process_group
    )
    grouped_param_groups = defaultdict(list)
    for group in all_param_groups:
        opt_name = group.get('optimizer', eopt_name)
        is_expert = group['is_expert_parallel'] and not use_layer_wise
        grouped_param_groups[(opt_name, is_expert)].append(group)

    # Set up DistOpt process groups + filtered buffers once, only if we'll
    # construct a DistributedOptimizer for non-Muon groups in layer-wise mode.
    # The DistOpt-vs-LayerWise buffer split only happens when DDP was wrapped
    # with ``use_distributed_optimizer=True`` (i.e. the layout-based path); in
    # legacy ping-pong mode all params share one unpadded DDP buffer that
    # DistOpt cannot manage, so we keep non-Muon params inside LayerWise.
    ddp_uses_distributed_optimizer = (
        bool(getattr(model_chunks[0], 'ddp_config', None))
        and model_chunks[0].ddp_config.use_distributed_optimizer
    )
    distopt_process_groups = None
    distopt_per_model_buffers = None
    use_separate_distributed_optimizer = ddp_uses_distributed_optimizer and use_layer_wise
    if use_separate_distributed_optimizer:
        ddp_config = model_chunks[0].ddp_config
        assert ddp_config.num_distributed_optimizer_instances == 1, (
            "Layer-wise + DistributedOptimizer split path does not yet support "
            "num_distributed_optimizer_instances > 1: distributed_optimizer_instance_id "
            "is hardcoded to 0 in this path. Disable use_layer_wise_param_layout to "
            "fall back to the legacy LayerWise ping-pong path."
        )
    if use_separate_distributed_optimizer and any(
        # A separate DistributedOptimizer with byte-level sharding handles any group
        # whose optimizer is not the primary emerging optimizer (stored in ``eopt_name``,
        # e.g., Muon). This includes scalar optimizers like Adam or Lion.
        not (opt_name == eopt_name and opt_name in _EMERGING_OPTIMIZERS)
        for (opt_name, _), groups in grouped_param_groups.items()
        if groups
    ):
        # ``setup_process_groups_for_optimizer`` rejects Gloo groups whenever
        # an explicit ``pg_collection`` is supplied, so the only legal value
        # here is False.
        distopt_process_groups = ProcessGroupCollection.setup_process_groups_for_optimizer(
            pg_collection, model_chunks, use_gloo_process_groups=False
        )
        # DistOpt should only manage non-LayerWise buffers (those holding
        # embeddings, biases, layernorm, etc.). Filter out the LayerWise
        # shard-aligned buffers that the LayerWiseDistributedOptimizer owns.
        distopt_per_model_buffers = {}
        for model_chunk_idx, model_chunk in enumerate(model_chunks):
            if not hasattr(model_chunk, 'buffers'):
                continue
            non_layer_wise_buffers = [
                buffer
                for buffer in model_chunk.buffers
                if buffer.params
                and not getattr(buffer.params[0], 'is_managed_by_layer_wise_optimizer', False)
            ]
            if non_layer_wise_buffers:
                distopt_per_model_buffers[model_chunk_idx] = non_layer_wise_buffers

    # Build an optimizer for each (optimizer_name, is_expert) bucket and combine.
    # In layer-wise mode, emerging-optimizer (Muon) groups feed into LayerWise,
    # while non-emerging (Adam) groups are managed by a separate DistributedOptimizer
    # — that is, the LayerWise optimizer only owns Muon-managed matrix parameters,
    # and the rest go through DistOpt's standard byte-level shard machinery.
    results = []
    layer_wise_base_results = []  # (raw_optimizer, init_state_fn) feeding LayerWise.
    for (opt_name, is_expert), groups in grouped_param_groups.items():
        if not groups:
            continue

        model_parallel_group = pg_collection.tp_ep_pp if is_expert else pg_collection.mp

        # Only the primary emerging optimizer (stored in ``eopt_name``, e.g., Muon) is
        # constructed via ``_create_emerging_optimizer``. Scalar optimizers that also appear
        # in ``_EMERGING_OPTIMIZERS`` (e.g., Lion) fall through to the standard fallback path.
        if opt_name == eopt_name and opt_name in _EMERGING_OPTIMIZERS:
            optimizer, init_state_fn = _create_emerging_optimizer(
                config, groups, eopt_name, model_chunks, pg_collection
            )
            if use_layer_wise:
                layer_wise_base_results.append((optimizer, init_state_fn))
                continue
            if config.bf16:
                optimizer = Float16OptimizerWithFloat16Params(
                    optimizer, config, None, init_state_fn
                )
            else:
                optimizer = FP32Optimizer(optimizer, config, init_state_fn)
            setattr(optimizer, 'grad_stats_parallel_group', model_parallel_group)
            tp_group = pg_collection.tp
            expert_tp_group = getattr(pg_collection, 'expt_tp', tp_group)
            setattr(optimizer, 'tp_group', tp_group)
            setattr(optimizer, 'expert_tp_group', expert_tp_group)
            setattr(optimizer, 'gtp_group', getattr(pg_collection, 'gtp_remat', None))
            setattr(optimizer, 'expert_gtp_group', getattr(pg_collection, 'expt_gtp_remat', None))
            results.append(optimizer)
            continue
        else:
            fallback_config = copy.copy(config)
            fallback_config.optimizer = opt_name
            if use_separate_distributed_optimizer:
                # Route non-emerging params (adam/lion) through a real DistributedOptimizer
                # (byte-level sharding) instead of stuffing them inside LayerWise.
                for group in groups:
                    assert not group['is_expert_parallel'], (
                        "Non-emerging expert-parallel param groups are not yet "
                        "supported on the layer-wise + DistributedOptimizer "
                        "path: they need a separate DistOpt instance with the "
                        "expert-DP process group, which is not wired up yet. "
                        "Disable use_layer_wise_param_layout to fall back to "
                        "the legacy LayerWise ping-pong path for MoE models."
                    )
                fallback_config.use_distributed_optimizer = True
                result = _get_megatron_optimizer_based_on_param_groups(
                    config=fallback_config,
                    model_chunks=model_chunks,
                    param_groups=groups,
                    per_model_buffers=distopt_per_model_buffers,
                    model_parallel_group=distopt_process_groups['mp_group'],
                    data_parallel_group=distopt_process_groups['intra_dp_cp_group'],
                    data_parallel_group_gloo=distopt_process_groups['intra_dp_cp_group_gloo'],
                    data_parallel_group_idx=get_pg_rank(distopt_process_groups['mp_group']),
                    intra_dist_opt_group=distopt_process_groups['intra_dist_opt_group'],
                    distributed_optimizer_instance_id=0,
                    pg_collection=pg_collection,
                    skip_megatron_wrapping=False,
                )
                # TODO(deyuf): ChainedOptimizer currently asserts all sub-optimizers
                # share the same config. Reset to the top-level config so the
                # assertion holds when DistOpt+LayerWise are chained.
                if hasattr(result, 'config'):
                    result.config = config
                results.append(result)
            else:
                # Legacy ping-pong layer-wise path (use_layer_wise=True) or the
                # non-layer-wise standard chain: keep ``use_distributed_optimizer``
                # off; in layer-wise mode the raw torch optimizer (returned as a
                # ``(optimizer, init_state_fn)`` tuple via ``skip_megatron_wrapping``)
                # feeds into ``LayerWiseDistributedOptimizer``.
                fallback_config.use_distributed_optimizer = False
                result = _get_megatron_optimizer_based_on_param_groups(
                    config=fallback_config,
                    model_chunks=model_chunks,
                    param_groups=groups,
                    model_parallel_group=model_parallel_group,
                    pg_collection=pg_collection,
                    skip_megatron_wrapping=use_layer_wise,
                )
                if use_layer_wise:
                    layer_wise_base_results.append(result)
                else:
                    if hasattr(result, 'config'):
                        result.config = config
                    results.append(result)

    if use_layer_wise:
        log_single_rank(
            logger, logging.INFO, f'Using LayerWiseDistributedOptimizer for {eopt_name}'
        )
        base_optimizers, init_fns = (), ()
        if layer_wise_base_results:
            base_optimizers, init_fns = zip(*layer_wise_base_results)
        layer_wise_optimizer = LayerWiseDistributedOptimizer(
            list(base_optimizers),
            config,
            pg_collection,
            init_state_fn_list=list(init_fns),
            model_chunks=model_chunks,
        )
        # LayerWise owns Muon-managed params; DistOpt instances in ``results``
        # own the rest. Chain them so the training loop sees one optimizer.
        if results:
            return ChainedOptimizer([layer_wise_optimizer] + results)
        return layer_wise_optimizer

    return ChainedOptimizer(results)


def _clear_high_precision_initializers(model_chunks: List[MegatronModule]) -> None:
    """Release saved initializers after all local optimizer master weights are constructed.

    Sharded optimizers only consume initializers for locally owned parameters. Sweep the
    model as well so non-owned and frozen parameters do not retain their CPU copies.
    """
    for model_chunk in model_chunks:
        for param in model_chunk.parameters():
            getter_fn = getattr(param, 'get_high_precision_init_val', None)
            clearer_fn = getattr(param, 'clear_high_precision_init_val', None)
            if getter_fn is not None and clearer_fn is not None and getter_fn() is not None:
                clearer_fn()


def get_megatron_optimizer(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    uscaling: bool = False,
    completedp: bool = False,
    alpha_completedp: Optional[float] = None,
    rho_adjusted: Optional[float] = None,
    width_adjusted: Optional[float] = None,
    depth_adjusted: Optional[float] = None,
    no_weight_decay_cond: Optional[Callable] = None,
    scale_lr_cond: Optional[Callable] = None,
    lr_mult: float = 1.0,
    *,
    config_overrides: Optional[Dict[ParamKey, ParamGroupOverride]] = None,
    use_gloo_process_groups: bool = True,
    pg_collection: Optional[ProcessGroupCollection] = None,
    dump_param_to_param_group_map: Optional[str] = None,
    param_group_process_group: Optional[torch.distributed.ProcessGroup] = None,
    default_skip_embedding_weight_decay: bool = False,
) -> MegatronOptimizer:
    """Retrieve the Megatron optimizer for model chunks.

    Handles both standard optimizers (Adam, SGD) and emerging optimizers (e.g. Muon).
    We use separate optimizers for expert parameters and non-expert parameters.
    For emerging optimizers with ``config.use_layer_wise_distributed_optimizer=True``,
    the optimizer is automatically wrapped with :class:`LayerWiseDistributedOptimizer`.

    Args:
        config (OptimizerConfig): optimizer configuration object.
        model_chunks (List[MegatronModule]): model chunks to get optimizer for.
        config_overrides (Optional[Dict[ParamKey, OptimizerConfig]]): optional dictionary of
            optimizer configuration objects to override default optimizer behavior for different
            subsets of parameters (identified by ParamKey).
        use_gloo_process_groups (bool): if false, disable use of Gloo process groups
            in underlying Megatron optimizers.
        pg_collection: Optional unified process group for distributed training.
        dump_param_to_param_group_map (Optional[str]): path to dump parameter to param group map.
        param_group_process_group (Optional[torch.distributed.ProcessGroup]): group whose ranks
            must construct aligned optimizer parameter groups. ``None`` preserves WORLD alignment.

    Dragon arguments (positional, in the order Dragon's training.py passes them):
        uscaling / completedp: build Dragon's u-muP or CompleteDP param groups instead of the
            config_overrides ones (alpha_completedp, rho_adjusted, width_adjusted and
            depth_adjusted parametrize CompleteDP).
        no_weight_decay_cond, scale_lr_cond, lr_mult, default_skip_embedding_weight_decay:
            pre-0.16 API, translated into config_overrides.

    Returns:
        Instance of MegatronOptimizer.
    """
    dragon_groups_kwargs = dict(
        uscaling=uscaling,
        completedp=completedp,
        alpha_completedp=alpha_completedp,
        rho_adjusted=rho_adjusted,
        width_adjusted=width_adjusted,
        depth_adjusted=depth_adjusted,
    )

    # A MimoModel routes to the heterogeneous per-module optimizer builder.
    from megatron.core.models.mimo.model.base import MimoModel

    if isinstance(model_chunks[0], MimoModel):
        from megatron.core.models.mimo.optimizer import get_mimo_optimizer

        assert (
            len(model_chunks) == 1
        ), "MimoModel does not support virtual pipeline parallelism (multiple model chunks)"
        return get_mimo_optimizer(model_chunks[0], config)

    # None → apply standard defaults. To extend defaults with custom overrides,
    # start from get_standard_config_overrides(config) and merge yours in.
    config_overrides = _get_dragon_legacy_config_overrides(
        config,
        config_overrides,
        no_weight_decay_cond,
        scale_lr_cond,
        lr_mult,
        default_skip_embedding_weight_decay,
    )
    if config_overrides is None:
        config_overrides = get_standard_config_overrides(config)

    check_config_overrides_consistency(config, config_overrides)

    is_mfsdp_v2 = isinstance(model_chunks[0], FullyShardedDataParallelV2)
    # TODO: the standard and emerging optimizer paths handle pg_collection differently;
    # unify them so both use a single pg_collection-based flow.
    if config.optimizer not in ('adam', 'sgd', 'ademamix'):
        assert not (uscaling or completedp), "Dragon uscaling / CompleteDP need adam or ademamix"
        optimizer = _get_megatron_emerging_optimizer(
            config=config,
            model_chunks=model_chunks,
            config_overrides=config_overrides,
            pg_collection=pg_collection,
            param_group_process_group=param_group_process_group,
        )
        _clear_high_precision_initializers(model_chunks)
        return optimizer

    log_single_rank(logger, logging.INFO, f'Setting up optimizer with config {config}')

    if is_mfsdp_v2:
        if config.use_distributed_optimizer:
            raise ValueError("MFSDP v2 currently requires use_distributed_optimizer=False.")
    elif config.use_precision_aware_optimizer and not config.use_distributed_optimizer:
        raise ValueError(
            "--use-precision-aware-optimizer only supported with distributed optimizer"
        )

    # Separate out first model chunk if overlapping param AG with optimizer step.
    if config.overlap_param_gather_with_optimizer_step:
        all_dense_model_chunks = [[model_chunks[0]], model_chunks[1:]]
        overlap_param_gather_with_optimizer_step_flags = [True, False]
    else:
        all_dense_model_chunks = [model_chunks]
        overlap_param_gather_with_optimizer_step_flags = [False]

    # Setup process groups using helper method
    process_groups_dict = ProcessGroupCollection.setup_process_groups_for_optimizer(
        pg_collection, model_chunks, use_gloo_process_groups
    )

    dp_cp_group = process_groups_dict['dp_cp_group']
    intra_dp_cp_group = process_groups_dict['intra_dp_cp_group']
    intra_expt_dp_group = process_groups_dict['intra_expt_dp_group']
    mp_group = process_groups_dict['mp_group']
    expt_tp_pp_group = process_groups_dict['expt_tp_pp_group']
    expt_tp_pp_with_egtp_remat_group = process_groups_dict['expt_tp_pp_with_egtp_remat_group']
    intra_dp_cp_group_gloo = process_groups_dict['intra_dp_cp_group_gloo']
    intra_expt_dp_group_gloo = process_groups_dict['intra_expt_dp_group_gloo']
    intra_dist_opt_group = process_groups_dict['intra_dist_opt_group']

    # ``mp_group`` spans TP×GTP_remat×PP (GTP_remat-merged).
    model_parallel_rank = get_pg_rank(mp_group)

    if get_pg_size(dp_cp_group) > get_pg_size(intra_dp_cp_group):
        inter_dist_opt_group = process_groups_dict['inter_dist_opt_group']
        distributed_optimizer_instance_id = get_pg_rank(inter_dist_opt_group)
    else:
        distributed_optimizer_instance_id = 0

    optimizers = []
    model_chunk_offset = 0
    ddp_config = model_chunks[0].ddp_config  # Use the first model chunk's DDP config
    if ddp_config.use_megatron_fsdp:
        # For no_shard, gradients are replicated across DP ranks after all-reduce, so grad stats
        # should only be reduced over TP/PP (model_parallel_group) to avoid inflating the norm.
        if ddp_config.data_parallel_sharding_strategy == 'no_shard':
            effective_intra_dist_opt_group = mp_group
        elif ddp_config.outer_dp_sharding_strategy != 'no_shard':
            # Hybrid FSDP that shards the optimizer state over DP-Outer leaves each optimizer
            # instance with a distinct slice of the gradient instead of a replica of it, so the
            # stats have to be reduced over every instance to cover the whole gradient.
            effective_intra_dist_opt_group = dp_cp_group
        else:
            effective_intra_dist_opt_group = intra_dist_opt_group
        for model_chunk, overlap_param_gather_with_optimizer_step in zip(
            all_dense_model_chunks, overlap_param_gather_with_optimizer_step_flags
        ):
            if is_mfsdp_v2:
                param_groups = _get_param_groups(
                    model_chunk, config, config_overrides, param_group_process_group
                )
                if not is_te_min_version("2.18.0"):
                    # TE FusedAdam can skip pending updates when a group ends in an empty tensor:
                    # https://github.com/NVIDIA/TransformerEngine/issues/3207.
                    # Empty local shards have no optimizer state or data to update, so omit them.
                    for param_group in param_groups:
                        param_group['params'] = [
                            parameter
                            for parameter in param_group['params']
                            if parameter.to_local().numel() > 0
                        ]
                    param_groups = [
                        param_group for param_group in param_groups if param_group['params']
                    ]
                # MFSDP v2 owns its sharded parameter and gradient storage, so
                # FullyShardedOptimizer does not need DDP param-and-grad buffers.
                buffers = None
            else:
                param_groups, buffers = _get_param_groups_and_buffers(
                    model_chunk,
                    model_chunk_offset=model_chunk_offset,
                    config=config,
                    config_overrides=config_overrides,
                    filter_fn=lambda g: True,
                    buffer_name='buffers',
                    process_group=param_group_process_group,
                    **dragon_groups_kwargs,
                )

            optimizer_part = _get_megatron_optimizer_based_on_param_groups(
                config=config,
                model_chunks=model_chunk,
                param_groups=param_groups,
                per_model_buffers=buffers,
                model_parallel_group=mp_group,
                data_parallel_group=dp_cp_group,
                data_parallel_group_gloo=intra_dp_cp_group_gloo,
                data_parallel_group_idx=model_parallel_rank,
                intra_dist_opt_group=effective_intra_dist_opt_group,
                distributed_optimizer_instance_id=distributed_optimizer_instance_id,
                pg_collection=pg_collection,
            )
            if (
                not USING_PYTORCH_OPTIMIZER
                and config.use_precision_aware_optimizer
                and not is_mfsdp_v2
                and getattr(optimizer_part.optimizer, "master_weights", None) is not None
            ):
                # NOTE(@cspades): FusedAdam is provided Megatron-FSDP's main weights as
                # non-quantized DTensor(s). Megatron-FSDP should NEVER use FusedAdam's
                # main weights, complete waste of memory as the optimizer step is still
                # applied to the Megatron-FSDP main weight and extended to FusedAdam
                # main weights. Override this here.
                setattr(optimizer_part.optimizer, "master_weights", False)
                # Megatron-FSDP always uses a decoupled gradient when using FusedAdam.
                setattr(optimizer_part.optimizer, "use_decoupled_grad", True)

            optimizers.append(optimizer_part)
            model_chunk_offset += 1

        _clear_high_precision_initializers(model_chunks)
        if len(optimizers) == 1:
            return optimizers[0]

        return ChainedOptimizer(optimizers)

    if dump_param_to_param_group_map is not None:
        param_to_param_group = {}
        param_group_id = 0
    for dense_model_chunks, overlap_param_gather_with_optimizer_step in zip(
        all_dense_model_chunks, overlap_param_gather_with_optimizer_step_flags
    ):
        param_groups, buffers = _get_param_groups_and_buffers(
            dense_model_chunks,
            model_chunk_offset=model_chunk_offset,
            config=config,
            config_overrides=config_overrides,
            filter_fn=lambda g: not g['is_expert_parallel'],
            buffer_name='buffers',
            process_group=param_group_process_group,
            **dragon_groups_kwargs,
        )
        for model_chunk in dense_model_chunks:
            model_chunk.overlap_param_gather_with_optimizer_step = (
                overlap_param_gather_with_optimizer_step
            )
        if dump_param_to_param_group_map is not None:
            for param_group in param_groups:
                for param in param_group["params"]:
                    param_name = get_global_unique_param_name(model_chunks, param)
                    param_to_param_group[param_name] = param_group_id
                param_group_id += 1

        # Pass Gloo process groups into optimizer only if needed.
        optimizers.append(
            _get_megatron_optimizer_based_on_param_groups(
                config=config,
                model_chunks=dense_model_chunks,
                param_groups=param_groups,
                per_model_buffers=buffers,
                model_parallel_group=mp_group,
                data_parallel_group=intra_dp_cp_group,
                data_parallel_group_gloo=intra_dp_cp_group_gloo,
                data_parallel_group_idx=model_parallel_rank,
                intra_dist_opt_group=intra_dist_opt_group,
                distributed_optimizer_instance_id=distributed_optimizer_instance_id,
                pg_collection=pg_collection,
            )
        )
        model_chunk_offset += 1

    moe_param_groups, moe_buffers = _get_param_groups_and_buffers(
        model_chunks,
        model_chunk_offset=0,
        config=config,
        config_overrides=config_overrides,
        filter_fn=lambda g: g['is_expert_parallel'],
        buffer_name='expert_parallel_buffers',
        process_group=param_group_process_group,
        **dragon_groups_kwargs,
    )
    if dump_param_to_param_group_map is not None:
        for param_group in moe_param_groups:
            for param in param_group["params"]:
                param_name = get_global_unique_param_name(model_chunks, param)
                param_to_param_group[param_name] = param_group_id
            param_group_id += 1
    if len(moe_param_groups) > 0:
        # Expert analog of dense ``model_parallel_rank``; use the EGTP_remat-merged group so each
        # EGTP_remat peer gets a distinct distopt ShardedObject key (else DCP "duplicate" error).
        expt_model_parallel_rank = get_pg_rank(expt_tp_pp_with_egtp_remat_group)
        if use_gloo_process_groups:
            expt_data_parallel_group_gloo = intra_expt_dp_group_gloo
        else:
            expt_data_parallel_group_gloo = None
        optimizers.append(
            _get_megatron_optimizer_based_on_param_groups(
                config=config,
                model_chunks=model_chunks,
                param_groups=moe_param_groups,
                per_model_buffers=moe_buffers,
                model_parallel_group=expt_tp_pp_with_egtp_remat_group,
                data_parallel_group=intra_expt_dp_group,
                data_parallel_group_gloo=expt_data_parallel_group_gloo,
                data_parallel_group_idx=expt_model_parallel_rank,
                intra_dist_opt_group=intra_dist_opt_group,
                distributed_optimizer_instance_id=distributed_optimizer_instance_id,
                pg_collection=pg_collection,
            )
        )

    if dump_param_to_param_group_map is not None:
        torch.distributed.checkpoint.save(
            state_dict=param_to_param_group, checkpoint_id=dump_param_to_param_group_map
        )

    _clear_high_precision_initializers(model_chunks)
    return ChainedOptimizer(optimizers)
