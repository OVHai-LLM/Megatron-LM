# Copyright (c) 2026, OVHai-LLM.
"""Dragon's fine-grained activation offloading calls, on top of Megatron-Core >= 0.20's
FineGrainedActivationOffloadingInterface (the free functions Dragon v1.5 used were removed)."""

from megatron.core import parallel_state
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as _off_interface,
)
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    fine_grained_offloading_group_offload,
    fine_grained_offloading_group_start,
)

__all__ = [
    "fine_grained_offloading_group_commit",
    "fine_grained_offloading_group_start",
    "fine_grained_offloading_init_chunk_handler",
    "fine_grained_offloading_set_last_layer",
    "get_fine_grained_offloading_context",
]


def get_fine_grained_offloading_context(flag):
    """The offload context manager when `flag`, else a null context."""
    return _off_interface.get_context(flag)


def fine_grained_offloading_group_commit(*tensors, name, forced_released_tensors=None):
    """End of an offload group: returns `tensors` (the first one carries the commit op)."""
    if len(tensors) == 0:
        return tensors
    first = fine_grained_offloading_group_offload(
        tensors[0], name=name, forced_released_tensors=forced_released_tensors
    )
    return (first,) + tuple(tensors[1:])


def fine_grained_offloading_set_last_layer(is_last_layer):
    """No-op: Megatron-Core now pre-reloads the last layer of a chunk itself."""
    del is_last_layer


def fine_grained_offloading_init_chunk_handler(vp_size, vp_stage, min_offloaded_tensor_size, config=None):
    """Start a model chunk (microbatch) for the offload manager."""
    kwargs = {}
    if config is not None:
        kwargs = dict(
            delta_offload_bytes_across_pp_ranks=getattr(
                config, "delta_offload_bytes_across_pp_ranks", 0
            ),
            activation_offload_fraction=getattr(config, "activation_offload_fraction", 1.0),
            max_inflight_offloads=getattr(
                config, "fine_grained_offloading_max_inflight_offloads", None
            ),
        )
    _off_interface.init_chunk_handler(
        pp_rank=parallel_state.get_pipeline_model_parallel_rank(),
        vp_size=vp_size,
        vp_stage=vp_stage,
        min_offloaded_tensor_size=min_offloaded_tensor_size,
        **kwargs,
    )
