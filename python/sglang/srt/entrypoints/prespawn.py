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
    """Called from `run_server` before importing the HTTP server module.

    Deliberately does *not* resolve/validate/publish the config: resolution
    imports the model stack (~3-5 s in the launcher). The workers resolve on
    their own (`publish()` in the child does), and `_launch_subprocesses`
    performs the launcher-side resolution, validation and publish when it adopts
    the pre-spawned processes.
    """
    global _PRESPAWNED
    if not enabled() or _PRESPAWNED is not None:
        return
    import time

    from sglang.srt.entrypoints.worker_launch import (
        _ParallelView,
        _set_envs_and_config,
        launch_scheduler_processes,
    )
    from sglang.srt.managers import process_entry
    from sglang.srt.utils import configure_logger

    t0 = time.perf_counter()
    configure_logger(server_args)
    if not _eligible(server_args):
        logger.info(
            "[prespawn] configuration not eligible; using the normal launch path"
        )
        return
    t1 = time.perf_counter()
    _set_envs_and_config(server_args)
    _maybe_setup_foundry_env(server_args)
    t2 = time.perf_counter()
    port_args = PortArgs.init_new(server_args)
    t3 = time.perf_counter()
    result, procs = launch_scheduler_processes(
        server_args,
        port_args,
        process_entry.run_scheduler_process,
        process_entry.run_data_parallel_controller_process,
        parallel=_ParallelView(resolving_view(server_args)),
    )
    t4 = time.perf_counter()
    logger.info(
        "[prespawn] started %d worker process(es) before the server imports "
        "(logger %.2fs, envs %.2fs, ports %.2fs, spawn %.2fs; %.1fs since interpreter start)",
        len(procs or []),
        t1 - t0,
        t2 - t1,
        t3 - t2,
        t4 - t3,
        _since_process_start(),
    )
    _PRESPAWNED = Prespawned(server_args, port_args, result, procs)


def _maybe_setup_foundry_env(server_args: ServerArgs) -> None:
    """Foundry (CUDA graph save/load) injects its hook library via LD_PRELOAD
    from a patch on Engine._launch_scheduler_processes, installed during config
    resolution -- both of which run after this pre-spawn. Do the env part here
    so spawned workers inherit the hook; the workers install the Python-side
    hooks themselves (scheduler entry -> foundry_shim.apply_server_args)."""
    cfg_path = getattr(server_args, "foundry_graph_extension_config_path", None)
    if not cfg_path:
        return
    from foundry.integration.sglang import runtime as rt
    from foundry.integration.sglang.config import load_graph_extension_config

    load_graph_extension_config(cfg_path)
    rt.setup_ld_preload_env()
    logger.info(
        "[prespawn] foundry env prepared (LD_PRELOAD hook, mode=%s)",
        os.environ.get("FOUNDRY_MODE"),
    )


def _since_process_start() -> float:
    """Seconds since this process was created (Linux /proc)."""
    try:

        with open("/proc/self/stat") as f:
            start_ticks = int(f.read().split(")")[-1].split()[19])
        with open("/proc/uptime") as f:
            uptime = float(f.read().split()[0])
        return uptime - start_ticks / os.sysconf("SC_CLK_TCK")
    except Exception:
        return -1.0


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
