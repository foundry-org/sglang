"""Communication state that the first forward otherwise creates lazily, built
once before graph capture so that no capture records its creation."""

from __future__ import annotations

import logging

import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.utils import log_info_on_rank0

logger = logging.getLogger(__name__)

# DIAGNOSTIC (SGLANG_DIAG_ELASTIC_SCALE_TIMELINE): process-wide counts of the work a graph capture does.
DIAG_CAPTURE_COUNTS = {
    "kernel_warmup": 0,
    "capture_warmup_forward": 0,
    "native_capture": 0,
}


def prepare_capture_state(
    *,
    model: nn.Module,
    dtype: torch.dtype,
    is_draft_worker: bool,
    is_ep_scale_joiner: bool,
) -> None:
    """Build the MoE dispatchers' buffers / expert maps and the logits all-gather
    state now; natively the first warmup forward builds the same objects."""
    # Lazy imports keep the runner package free of a layers import cycle.
    from sglang.srt.layers.logits_processor import prebuild_logits_gatherers
    from sglang.srt.layers.moe.token_dispatcher.base import prebuild_dispatchers

    # A joiner's EP buffers are collectives with ranks it meets only at the
    # elastic rendezvous; they keep their lazy (or post-rendezvous) creation.
    # DIAGNOSTIC joiner_init_load: the joiner builds its buffers now and connects at the scale step.
    joiner_prebuild = envs.SGLANG_DIAG_ELASTIC_JOINER_INIT_LOAD.get()
    num_dispatchers = (
        0 if is_ep_scale_joiner and not joiner_prebuild else prebuild_dispatchers(model)
    )
    # A draft warms up after the target's graphs are captured, past the point
    # where persisted graphs need driver-placed symmetric memory; it stays lazy.
    num_gatherers = (
        0 if is_draft_worker else prebuild_logits_gatherers(model, dtype=dtype)
    )
    if num_dispatchers or num_gatherers:
        log_info_on_rank0(
            logger,
            f"Prebuilt capture state: {num_dispatchers} MoE dispatchers, "
            f"{num_gatherers} logits gatherers.",
        )
