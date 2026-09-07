"""Worker process launch helpers, kept free of the model/tokenizer import graph.

`sglang.srt.entrypoints.engine` re-exports everything here (it used to define
it), so importers of the engine module see no change. Keeping these in a light
module lets `prespawn.maybe_prespawn()` start the scheduler / DP-controller
processes right after argument parsing, before the launcher imports its HTTP
and tokenizer stack (~7 s), so the workers' own init overlaps that import.
"""

import dataclasses
import gc
import logging
import multiprocessing as mp
import os
import random
import signal
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from sglang.srt.arg_groups.overrides import (
    attention_backends_of,
    resolved_view,
    resolving_view,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_exec, get_parallel
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    assert_pkg_version,
    get_bool_env_var,
    is_cuda,
    is_mnnvl_fabric_device,
    kill_process_tree,
    maybe_reindex_device_id,
    numa_utils,
    set_prometheus_multiproc_dir,
    set_ulimit,
)
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()

# Same value as sglang.srt.managers.data_parallel_controller.SCHEDULER_PIDS_ARG
# (asserted there); duplicated so this module does not import the controller.
SCHEDULER_PIDS_ARG = "scheduler_pids"


@dataclasses.dataclass
class SchedulerInitResult:
    """Result from launching schedulers."""

    scheduler_infos: List[Dict[str, Any]]
    all_child_pids: List[int] = dataclasses.field(default_factory=list)
    wait_for_ready: Callable[[], None] = lambda: None
    block_until_scheduler_exits: Callable[[], None] = lambda: None
    engine_info_bootstrap_server: Optional[Any] = None


def _set_envs_and_config(server_args: ServerArgs):

    cfg = resolving_view(server_args)
    # Set global environments
    # MNNVL fabric (GB200/GB300) multi-node: cross-node NVLink needs NCCL's
    # cuMem-based buffers and MNNVL transport. Default them on (user-set
    # values win; the symm-mem override below only fires when unset).
    if cfg.nnodes > 1 and is_mnnvl_fabric_device():
        os.environ.setdefault("NCCL_CUMEM_ENABLE", "1")
        os.environ.setdefault("NCCL_MNNVL_ENABLE", "1")
    if "NCCL_CUMEM_ENABLE" not in os.environ or cfg.enable_symm_mem:
        os.environ["NCCL_CUMEM_ENABLE"] = str(int(cfg.enable_symm_mem))
    if (
        "NCCL_NVLS_ENABLE" not in os.environ
        or cfg.enable_nccl_nvls
        or cfg.enable_symm_mem
    ):
        os.environ["NCCL_NVLS_ENABLE"] = str(
            int(cfg.enable_nccl_nvls or cfg.enable_symm_mem)
        )
    if "NCCL_GRAPH_MIXING_SUPPORT" not in os.environ or cfg.enable_symm_mem:
        # Note(wh): NCCL_GRAPH_MIXING_SUPPORT=0 can help improve performance for symmetric kernels.
        # details in https://github.com/NVIDIA/nccl-tests/issues/333#issuecomment-3103636985
        if cfg.dcp_size > 1:
            os.environ["NCCL_GRAPH_MIXING_SUPPORT"] = "0"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "8"

    if os.environ.get("TRTLLM_ENABLE_PDL", "1") != "0":
        # flashinfer uses this environment variable for various kernels from MoE to quant kernels
        os.environ["TRTLLM_ENABLE_PDL"] = "1"

    if os.environ.get("CUTE_DSL_LOG_LEVEL") is None:
        # Default to warning level, to avoid too many logs
        os.environ["CUTE_DSL_LOG_LEVEL"] = "30"

    if os.environ.get("CUTE_DSL_LOG_TO_CONSOLE") is None:
        # Need to set log to console, otherwise the log level won't take effect
        os.environ["CUTE_DSL_LOG_TO_CONSOLE"] = "1"

    # Can also be passed as argument
    os.environ["SGLANG_RUN_ID"] = (
        f"sglang-run-{time.time()}-{random.randint(0, 100000000)}"
    )

    # Set prometheus env vars
    if cfg.enable_metrics:
        set_prometheus_multiproc_dir()

    # Set ulimit
    set_ulimit()

    # Check flashinfer version
    if not get_bool_env_var("SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK"):
        if "flashinfer" in attention_backends_of(resolved_view(cfg)):
            assert_pkg_version(
                "flashinfer_python",
                "0.6.18",
                "Please uninstall the old version and "
                "reinstall the latest version by following the instructions "
                "at https://docs.flashinfer.ai/installation.html.",
            )
        if _is_cuda:
            assert_pkg_version(
                "sglang-kernel",
                "0.4.6.post1",
                "Please reinstall the latest version with `pip install sglang-kernel --force-reinstall`",
            )

    # Signal handlers can only be registered from the main thread.
    if threading.current_thread() is threading.main_thread():
        if cfg.custom_sigquit_handler is None:
            # Register the signal handler.
            # The child processes will send SIGQUIT to this process when any error happens
            # This process then clean up the whole process tree
            # Note: This sigquit handler is used in the launch phase, and may be replaced by
            # the running_phase_sigquit_handler in the tokenizer manager after the grpc server is launched.
            def launch_phase_sigquit_handler(signum, frame):
                logger.error(
                    "Received sigquit from a child process. It usually means the child failed."
                )
                kill_process_tree(os.getpid())

            signal.signal(signal.SIGQUIT, launch_phase_sigquit_handler)
        else:
            # Allow users to register a custom SIGQUIT handler for things like crash dump
            logger.error(f"Using custom SIGQUIT handler: {cfg.custom_sigquit_handler}")
            signal.signal(signal.SIGQUIT, cfg.custom_sigquit_handler)
    else:
        logger.warning(
            "Signal handler is not added because the engine is not in the "
            "main thread. This disables the SIGQUIT handler for cleaning up "
            "the process tree when a child process fails."
        )

    # Set mp start method
    # SGLANG_MP_START_METHOD (experiment): spawn (default) | fork | forkserver.
    mp.set_start_method(os.environ.get("SGLANG_MP_START_METHOD", "spawn"), force=True)

    # Set gc threshold
    if gc_threshold := cfg.gc_threshold:
        gc.set_threshold(*gc_threshold)

    _log_legacy_kernel_cache_dirs()


def _log_legacy_kernel_cache_dirs():
    """Note the pre-SGLANG_CACHE_DIR cache dirs without touching them: other
    frameworks on the box may still be using them."""
    # TODO(shuwang21): drop once SGLANG_CACHE_DIR has been the default for a
    # few releases.
    legacy_dirs = [
        d
        for d in (
            os.path.expanduser("~/.triton"),
            os.path.expanduser("~/.cache/flashinfer"),
            os.path.expanduser("~/.cache/deep_gemm"),
        )
        if os.path.isdir(d)
    ]
    if not legacy_dirs:
        return
    logger.info(
        "Compiled-kernel caches now live under SGLANG_CACHE_DIR (%s). These "
        "older directories are no longer used by sglang, but may still be "
        "used by other frameworks on this machine, so they were left alone: "
        "%s. Remove them yourself if nothing else needs them.",
        envs.SGLANG_CACHE_DIR.get(),
        ", ".join(legacy_dirs),
    )


def _scheduler_died_error(rank: int, proc) -> RuntimeError:
    """Build a descriptive error for a scheduler process that died during init."""
    proc.join(timeout=10)
    return RuntimeError(
        f"Rank {rank} scheduler died during initialization "
        f"(exit code: {proc.exitcode}). "
        f"If exit code is -9 (SIGKILL), a common cause is the OS OOM killer. "
        f"Run `dmesg -T | grep -i oom` to check."
    )


def _wait_for_scheduler_ready(
    scheduler_pipe_readers: List,
    scheduler_procs: List,
) -> List[Dict]:
    """Wait for the model to finish loading and return scheduler infos.

    Uses poll() with timeout instead of blocking recv(), so that child process
    death (e.g. OOM SIGKILL) is detected promptly instead of hanging forever.
    """
    scheduler_infos = []
    for i in range(len(scheduler_pipe_readers)):
        while True:
            if scheduler_pipe_readers[i].poll(timeout=5.0):
                try:
                    data = scheduler_pipe_readers[i].recv()
                except EOFError:
                    raise _scheduler_died_error(i, scheduler_procs[i])
                if data["status"] != "ready":
                    raise RuntimeError(
                        "Initialization failed. Please see the error messages above."
                    )
                scheduler_infos.append(data)
                break

            # Poll timed out — check all processes for early death
            for j in range(len(scheduler_procs)):
                if not scheduler_procs[j].is_alive():
                    raise _scheduler_died_error(j, scheduler_procs[j])

    return scheduler_infos


def _calculate_rank_ranges(
    nnodes: int, pp_size: int, tp_size: int, node_rank: int
) -> Tuple[range, range, int, int]:
    """Calculate pp_rank_range and tp_rank_range for a given node.

    Args:
        nnodes: Total number of nodes.
        pp_size: Pipeline parallel size.
        tp_size: Tensor parallel size.
        node_rank: The rank of the node to compute ranges for.

    Returns:
        A tuple of (pp_rank_range, tp_rank_range, pp_size_per_node, tp_size_per_node):
        - pp_rank_range: range of pipeline-parallel ranks assigned to this node.
        - tp_rank_range: range of tensor-parallel ranks assigned to this node.
        - pp_size_per_node: number of PP ranks per node.
        - tp_size_per_node: number of TP ranks per node.
    """
    pp_size_per_node = max(pp_size // nnodes, 1)
    nnodes_per_pp_rank = max(nnodes // pp_size, 1)
    pp_rank_range = range(
        pp_size_per_node * (node_rank // nnodes_per_pp_rank),
        pp_size_per_node * (node_rank // nnodes_per_pp_rank + 1),
    )

    nnodes_per_tp_group = nnodes_per_pp_rank
    tp_size_per_node = tp_size // nnodes_per_tp_group
    tp_rank_range = range(
        tp_size_per_node * (node_rank % nnodes_per_tp_group),
        tp_size_per_node * (node_rank % nnodes_per_tp_group + 1),
    )

    return pp_rank_range, tp_rank_range, pp_size_per_node, tp_size_per_node


def _compute_parallelism_ranks(
    server_args: ServerArgs, tp_rank: int
) -> Tuple[int, int, int]:
    """Compute attention-CP, MoE-DP, and MoE-EP ranks for a TP rank.

    Called while the launcher is deciding what to spawn, so the sizes are the
    configured ones -- the groups this is laying out do not exist yet.
    """
    attn_dp_size = get_parallel().dp_size if get_parallel().enable_dp_attention else 1
    tp_size = server_args.tp_size
    attn_cp_size = get_parallel().attn_cp_size
    moe_dp_size = get_parallel().moe_dp_size

    # Parallelism hierarchy (outermost to innermost):
    # - Attention: Global(TP) -> DP -> ATTN_CP -> ATTN_TP (innermost)
    # - MoE: Global(TP) -> MOE_DP -> EP -> MOE_TP (innermost)
    attn_tp_size = tp_size // attn_dp_size // attn_cp_size
    attn_cp_rank = (tp_rank // attn_tp_size) % attn_cp_size
    moe_dp_rank = tp_rank // (tp_size // moe_dp_size)
    moe_ep_rank = (
        tp_rank
        % (tp_size // moe_dp_size)
        // (tp_size // moe_dp_size // get_parallel().ep_size)
    )
    return attn_cp_rank, moe_dp_rank, moe_ep_rank


def launch_scheduler_processes(
    server_args: ServerArgs,
    port_args: PortArgs,
    run_scheduler_process_func: Callable,
    run_dp_controller_func: Callable = None,
) -> Tuple[SchedulerInitResult, Optional[List]]:
    """Launch scheduler processes using multiprocessing.
    Override in subclasses for different backends (e.g. Ray).

    Returns:
        Tuple of (SchedulerInitResult, scheduler_procs).
        scheduler_procs is None for RayEngine (uses Ray actors instead).
    """
    scheduler_procs = []
    use_dp_controller = (
        get_parallel().dp_size > 1 or get_exec().moe.ep_join_mode == "scale"
    )

    if not use_dp_controller:
        # Launch tensor parallel scheduler processes
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=server_args.enable_memory_saver
        )
        scheduler_pipe_readers = []

        pp_rank_range, tp_rank_range, pp_size_per_node, tp_size_per_node = (
            _calculate_rank_ranges(
                server_args.nnodes,
                get_parallel().pp_size,
                server_args.tp_size,
                server_args.node_rank,
            )
        )

        for pp_rank in pp_rank_range:
            for tp_rank in tp_rank_range:
                reader, writer = mp.Pipe(duplex=False)
                gpu_id = (
                    server_args.base_gpu_id
                    + ((pp_rank % pp_size_per_node) * tp_size_per_node)
                    + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
                )
                attn_cp_rank, moe_dp_rank, moe_ep_rank = _compute_parallelism_ranks(
                    server_args, tp_rank
                )

                with maybe_reindex_device_id(gpu_id) as gpu_id:
                    proc = mp.Process(
                        target=run_scheduler_process_func,
                        args=(
                            server_args,
                            port_args,
                            gpu_id,
                            tp_rank,
                            attn_cp_rank,
                            moe_dp_rank,
                            moe_ep_rank,
                            pp_rank,
                            None,
                            writer,
                        ),
                    )
                    with (
                        memory_saver_adapter.configure_subprocess(),
                        numa_utils.configure_subprocess(server_args, gpu_id),
                    ):
                        proc.start()

                scheduler_procs.append(proc)
                scheduler_pipe_readers.append(reader)
    else:
        # Launch the data parallel controller
        reader, writer = mp.Pipe(duplex=False)
        scheduler_pipe_readers = [reader]
        proc = mp.Process(
            target=run_dp_controller_func,
            kwargs=dict(
                server_args=server_args,
                port_args=port_args,
                pipe_writer=writer,
                run_scheduler_process_func=run_scheduler_process_func,
            ),
        )
        proc.start()
        scheduler_procs.append(proc)

    all_child_pids = [proc.pid for proc in scheduler_procs]
    scheduler_infos = []

    def wait_for_ready():
        infos = _wait_for_scheduler_ready(scheduler_pipe_readers, scheduler_procs)
        scheduler_infos.extend(infos)
        if use_dp_controller:
            for info in infos:
                if SCHEDULER_PIDS_ARG in info:
                    all_child_pids.extend(info[SCHEDULER_PIDS_ARG])

    def block_until_scheduler_exits():
        for proc in scheduler_procs:
            proc.join()
            logger.error(
                f"Scheduler or DataParallelController {proc.pid} "
                f"terminated with {proc.exitcode}"
            )

    return (
        SchedulerInitResult(
            scheduler_infos=scheduler_infos,
            all_child_pids=all_child_pids,
            wait_for_ready=wait_for_ready,
            block_until_scheduler_exits=block_until_scheduler_exits,
        ),
        scheduler_procs,
    )
