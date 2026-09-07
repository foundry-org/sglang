"""Start the worker processes before the launcher imports its HTTP/tokenizer
stack (experimental, ``SGLANG_PRESPAWN_WORKERS=1``).

A cold ``sglang serve`` spends ~7 s importing before ``Engine._launch_subprocesses``
can spawn the first worker, and the workers cannot start their own init
(config, tokenizer, weights, graphs) until then. ``maybe_prespawn()`` runs the
same preparation ``_launch_subprocesses`` does (logger, resolution, env, plugins,
validation, publish, ports) using only light modules, spawns the schedulers (or
the DP controller) with lazy entry points, and parks the result here;
``_launch_subprocesses`` adopts it via ``take()`` and skips the setup it already
did. Combined with ``SGLANG_EARLY_FORKSERVER=1`` the workers start ~1 s after
the CLI is entered.
"""

import dataclasses
import logging
import os
from typing import Any, List, Optional

from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.server_args import PortArgs, ServerArgs

logger = logging.getLogger(__name__)

_PRESPAWNED: Optional["Prespawned"] = None


@dataclasses.dataclass
class Prespawned:
    server_args: ServerArgs
    port_args: PortArgs
    result: Any  # worker_launch.SchedulerInitResult
    procs: List


def enabled() -> bool:
    return os.environ.get("SGLANG_PRESPAWN_WORKERS") == "1"


def _eligible(server_args: ServerArgs) -> bool:
    cfg = resolving_view(server_args)
    # Paths where _launch_subprocesses does extra work *before* spawning that
    # the workers depend on; leave those to the normal sequence.
    if getattr(cfg, "weight_cache_mode", "off") == "daemon":
        return False
    if getattr(
        cfg, "remote_instance_weight_loader_start_seed_via_transfer_engine", False
    ):
        return False
    if getattr(cfg, "use_ray", False):
        return False
    return True


def maybe_prespawn(server_args: ServerArgs) -> None:
    """Called from `run_server` before importing the HTTP server module."""
    global _PRESPAWNED
    if not enabled() or _PRESPAWNED is not None:
        return
    from sglang.srt.entrypoints.worker_launch import (
        _set_envs_and_config,
        launch_scheduler_processes,
    )
    from sglang.srt.managers import process_entry
    from sglang.srt.plugins import load_plugins
    from sglang.srt.runtime_context import publish
    from sglang.srt.utils import configure_logger

    configure_logger(server_args)
    server_args.resolve_once()
    if not _eligible(server_args):
        logger.info(
            "[prespawn] configuration not eligible; using the normal launch path"
        )
        return
    _set_envs_and_config(server_args)
    load_plugins()
    server_args.check_server_args()
    parsers = resolving_view(server_args)
    if parsers.reasoning_parser == "auto" or parsers.tool_call_parser == "auto":
        from sglang.srt.parser.template_detection import resolve_auto_parsers

        resolve_auto_parsers(server_args)
    publish(server_args, role="tokenizer")
    port_args = PortArgs.init_new(server_args)
    logger.info(f"server_args={server_args.resolved_dict()}")
    result, procs = launch_scheduler_processes(
        server_args,
        port_args,
        process_entry.run_scheduler_process,
        process_entry.run_data_parallel_controller_process,
    )
    logger.info(
        "[prespawn] started %d worker process(es) before the server imports",
        len(procs or []),
    )
    _PRESPAWNED = Prespawned(server_args, port_args, result, procs)


def take(server_args: ServerArgs) -> Optional[Prespawned]:
    """Hand the pre-spawned workers to `_launch_subprocesses` (once)."""
    global _PRESPAWNED
    pre = _PRESPAWNED
    if pre is None or pre.server_args is not server_args:
        return None
    _PRESPAWNED = None
    return pre


def abandon(pre: Prespawned) -> None:
    """The caller cannot use the pre-spawned workers (custom entry point): stop them."""
    logger.warning("[prespawn] pre-spawned workers not adopted; terminating them")
    for p in pre.procs or []:
        try:
            p.terminate()
        except Exception:
            pass
