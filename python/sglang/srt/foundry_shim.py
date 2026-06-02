# SPDX-License-Identifier: Apache-2.0
"""Small activation shim for the Foundry SGLang integration."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def apply_server_args(server_args) -> None:
    cfg_path = getattr(server_args, "foundry_graph_extension_config_path", None)
    if not cfg_path:
        return

    # Keep phase-1 SAVE/LOAD deterministic while preserving full-graph
    # torch.compile semantics. In SGLang, torch.compile is triggered from the
    # full CudaGraphRunner capture path for batch sizes in compile_bs, so
    # overriding enable_torch_compile here would silently change the graph.
    server_args.disable_piecewise_cuda_graph = True
    server_args.enable_profile_cuda_graph = False
    server_args.disable_flashinfer_autotune = True

    from foundry.integration.sglang.hooks import install_hooks

    install_hooks(server_args)
    logger.info("Foundry SGLang integration activated from %s", cfg_path)
