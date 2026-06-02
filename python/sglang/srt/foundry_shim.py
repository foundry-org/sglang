# SPDX-License-Identifier: Apache-2.0
"""Small activation shim for the Foundry SGLang integration.

Config constraints live in arg_groups.cuda_graph_hook.handle_graph_extension
(declared through the resolution pipeline); this shim only installs the
per-process runtime hooks. Process entries (scheduler, DP controller) call it
on the pickled resolved record before publish(); the launcher reaches it
through the pipeline handler.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def apply_server_args(server_args) -> None:
    cfg_path = server_args.foundry_graph_extension_config_path
    if not cfg_path:
        return

    from foundry.integration.sglang.hooks import install_hooks

    install_hooks(server_args)
    logger.info("Foundry SGLang integration activated from %s", cfg_path)
