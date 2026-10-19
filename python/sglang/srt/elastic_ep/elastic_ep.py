from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional

import msgspec
import torch

from sglang.srt.distributed import parallel_state
from sglang.srt.distributed.utils import get_global_tcp_store
from sglang.srt.eplb.expert_location import (
    broadcast_global_expert_location_metadata,
    get_global_expert_location_metadata,
)
from sglang.srt.runtime_context import (
    get_exec,
    get_parallel,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import is_cpu, is_cuda

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.eplb.eplb_manager import EPLBManager

logger = logging.getLogger(__name__)

_SCALE_COHORT_KEY_PREFIX = "elastic_ep/scale_cohort"


class ScaleCohort(msgspec.Struct, frozen=True, kw_only=True):
    target_ep_size: int
    cuda_graph_enabled: bool


def register_scale_cohort(
    rank_offset: int, target_ep_size: int, cuda_graph_enabled: bool
) -> None:
    store = get_global_tcp_store()
    if store is None:
        raise RuntimeError("Elastic EP scale-up requires the global TCPStore.")
    payload = msgspec.json.encode(
        ScaleCohort(
            target_ep_size=target_ep_size,
            cuda_graph_enabled=cuda_graph_enabled,
        )
    )
    store.set(f"{_SCALE_COHORT_KEY_PREFIX}/{rank_offset}", payload)


def get_scale_cohort(rank_offset: int) -> Optional[ScaleCohort]:
    store = get_global_tcp_store()
    if store is None:
        return None
    key = f"{_SCALE_COHORT_KEY_PREFIX}/{rank_offset}"
    if not store.check([key]):
        return None
    return msgspec.json.decode(store.get(key), type=ScaleCohort)


@dataclass
class ElasticEPState:
    active_ranks: Optional[torch.Tensor]
    last_active_ranks: Optional[torch.Tensor]
    active_ranks_cpu: Optional[torch.Tensor]
    effective_ep_size: int = 0
    pending_ep_size: Optional[int] = None
    scale_phase: str = "idle"
    last_error: Optional[str] = None
    pending_since: Optional[float] = None
    original_ep_size: int = 0
    has_scaled: bool = False
    ep_join_rank_offset: int = 0

    def is_active_equal_last(self) -> bool:
        return torch.equal(self.active_ranks, self.last_active_ranks)

    def sync_active_to_cpu(self):
        if self.active_ranks is not None:
            self.active_ranks_cpu = self.active_ranks.detach().cpu().clone()

    def snapshot_active_to_last(self):
        if self.active_ranks is not None:
            self.last_active_ranks = self.active_ranks.clone()

    def reset(self):
        if self.active_ranks is not None:
            # Reserved slots stay inactive until their ranks join.
            self.active_ranks.zero_()
            self.active_ranks[: self.effective_ep_size] = 1
            self.snapshot_active_to_last()
            self.sync_active_to_cpu()


class ElasticEPStateManager:
    _instance: Optional[ElasticEPState] = None
    _on_scale: Optional[Callable[[int, int], None]] = None

    @classmethod
    def instance(cls) -> ElasticEPState:
        return cls._instance

    @classmethod
    def init(cls, server_args: ServerArgs):
        if cls._instance is not None:
            return cls._instance

        if get_exec().moe.elastic_ep_backend is not None:
            world_size = torch.distributed.get_world_size()
            active_rank_capacity = get_parallel().max_world_size
            assert active_rank_capacity >= world_size, (
                f"--max-ep-size ({active_rank_capacity}) must be >= "
                f"world_size ({world_size})."
            )

            inst = cls._build_state(ep_size=active_rank_capacity, device=None)
            inst.effective_ep_size = world_size
            inst.original_ep_size = world_size
            if active_rank_capacity > world_size:
                inst.active_ranks[world_size:].zero_()
                inst.snapshot_active_to_last()
                inst.sync_active_to_cpu()

            if get_exec().moe.moe_a2a_backend == "nixl":
                cls._on_scale = cls._on_scale_nixl

            inst.ep_join_rank_offset = get_parallel().ep_join_rank_offset
            if get_exec().moe.is_ep_joiner:
                cls._init_joiner_state(inst)

            cls._instance = inst

        return cls._instance

    @classmethod
    def _init_joiner_state(cls, inst: ElasticEPState) -> None:
        global_rank = torch.distributed.get_rank()
        inst.active_ranks.zero_()
        inst.active_ranks[global_rank] = 1
        inst.snapshot_active_to_last()
        inst.sync_active_to_cpu()

        if get_exec().moe.ep_join_mode == "scale":
            inst.effective_ep_size = (
                get_parallel().ep_join_rank_offset + get_parallel().tp_size
            )
            inst.original_ep_size = (
                get_parallel().elastic_ep_initial_size
                or get_parallel().ep_join_rank_offset
            )
            inst.has_scaled = True
        else:
            world_size = torch.distributed.get_world_size()
            inst.effective_ep_size = world_size
            inst.original_ep_size = world_size

    @staticmethod
    def _select_device() -> torch.device:
        if is_cuda():
            return torch.device("cuda")
        elif is_cpu():
            return torch.device("cpu")
        else:
            raise NotImplementedError("Only CUDA and CPU support elastic ep now.")

    @classmethod
    def _build_state(
        cls, *, ep_size: Optional[int] = None, device: Optional[torch.device] = None
    ) -> ElasticEPState:
        active = cls.healthy_rank_state(ep_size=ep_size, device=device)
        return ElasticEPState(
            active_ranks=active,
            last_active_ranks=active.clone(),
            active_ranks_cpu=active.detach().cpu().clone(),
        )

    @classmethod
    def healthy_rank_state(
        cls, *, ep_size: Optional[int] = None, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        size = ep_size if ep_size is not None else torch.distributed.get_world_size()
        dev = device if device is not None else cls._select_device()

        return torch.ones(size, dtype=torch.int32, device=dev)

    @classmethod
    def request_scale(cls, n: int) -> bool:
        inst = cls._instance
        if inst is None:
            return False
        if (
            inst.pending_ep_size is not None
            or inst.scale_phase == "recovery_unsupported"
        ):
            return False
        inst.pending_ep_size = n
        inst.scale_phase = "waiting_for_cohort"
        inst.last_error = None
        inst.pending_since = time.monotonic()
        return True

    @classmethod
    def begin_scale(cls) -> bool:
        inst = cls._instance
        if (
            inst is None
            or inst.pending_ep_size is None
            or inst.scale_phase != "waiting_for_cohort"
        ):
            return False
        inst.scale_phase = "pending"
        return True

    @classmethod
    def mark_joining(cls) -> None:
        cls._mark_phase("joining")

    @classmethod
    def mark_configuring_data_plane(cls) -> None:
        cls._mark_phase("configuring_data_plane")

    @classmethod
    def mark_syncing_new_world(cls) -> None:
        cls._mark_phase("syncing_new_world")

    @classmethod
    def _mark_phase(cls, phase: str) -> None:
        inst = cls._instance
        if inst is not None and inst.pending_ep_size is not None:
            inst.scale_phase = phase

    @classmethod
    def commit_scale(cls) -> None:
        inst = cls._instance
        if inst is None or inst.pending_ep_size is None:
            return
        inst.effective_ep_size = inst.pending_ep_size
        inst.pending_ep_size = None
        inst.has_scaled = True
        inst.scale_phase = "serving_expanded"
        inst.last_error = None
        inst.pending_since = None
        inst.reset()

    @classmethod
    def fail_scale(cls, error: str) -> None:
        inst = cls._instance
        if inst is None:
            return
        inst.pending_ep_size = None
        inst.scale_phase = "failed"
        inst.last_error = error
        inst.pending_since = None
        inst.reset()

    @classmethod
    def fail_recovery(cls, error: str) -> None:
        inst = cls._instance
        if inst is None:
            return
        inst.scale_phase = "recovery_unsupported"
        inst.last_error = error

    @classmethod
    def get_effective_ep_size(cls) -> int:
        inst = cls._instance
        assert inst is not None, "Elastic EP state is not initialized."
        return inst.effective_ep_size

    @classmethod
    def get_pending_ep_size(cls) -> Optional[int]:
        inst = cls._instance
        if inst is None:
            return None
        return inst.pending_ep_size

    @classmethod
    def get_data_plane_ep_size(cls) -> int:
        inst = cls._instance
        assert inst is not None, "Elastic EP state is not initialized."
        if inst.pending_ep_size is not None and inst.scale_phase in (
            "configuring_data_plane",
            "syncing_new_world",
        ):
            return inst.pending_ep_size
        return inst.effective_ep_size

    @classmethod
    def get_scale_phase(cls) -> str:
        inst = cls._instance
        if inst is None:
            return "disabled"
        return inst.scale_phase

    @classmethod
    def get_last_error(cls) -> Optional[str]:
        inst = cls._instance
        if inst is None:
            return None
        return inst.last_error

    @classmethod
    def get_ep_join_rank_offset(cls) -> int:
        inst = cls._instance
        if inst is None:
            return 0
        return inst.ep_join_rank_offset

    @classmethod
    def on_scale(cls, from_ep_size: int, to_ep_size: int) -> None:
        if cls._on_scale is not None:
            cls._on_scale(from_ep_size, to_ep_size)

    @staticmethod
    def _on_scale_nixl(from_ep_size: int, to_ep_size: int) -> None:
        from sglang.srt.layers.moe.token_dispatcher.nixl import NixlEPBuffer

        NixlEPBuffer.on_scale(from_ep_size, to_ep_size)

    @classmethod
    def is_scaling(cls) -> bool:
        """Return whether a scale or recovery operation is pending.

        The CPU snapshot is authoritative because rank polling uses it too.
        """
        inst = cls._instance
        if inst is None or inst.active_ranks_cpu is None:
            return False
        if inst.scale_phase == "recovery_unsupported":
            return False
        if inst.pending_ep_size is not None:
            return True
        active_count = int(inst.active_ranks_cpu[: inst.effective_ep_size].sum().item())
        return active_count < inst.effective_ep_size


def elastic_expanded_world_enabled() -> bool:
    """Return whether execution uses ranks admitted after server launch.

    Launch-time TP groups exclude ranks admitted during scale-up.
    """
    inst = ElasticEPStateManager.instance()
    if inst is None:
        return False
    if get_parallel().max_ep_size is None:
        return False
    return ElasticEPStateManager.get_data_plane_ep_size() > inst.original_ep_size


def _refresh_ep_members() -> None:
    from sglang.srt.layers.moe.token_dispatcher.mooncake import EPBuffer

    buffer = EPBuffer.get_existing_buffer()
    if buffer is not None:
        buffer.update_ep_member()


_PEER_STATE_POLL_INTERVAL_SEC = 0.01


def _iter_live_parallel_groups() -> Iterator[parallel_state.GroupCoordinator]:
    groups = []
    for group_ref in parallel_state._groups.values():
        group = group_ref()
        if group is not None:
            groups.append(group)
    yield from sorted(groups, key=lambda group: group.unique_name)


def _map_global_to_group_local_ranks(
    group_ranks: List[int], global_ranks: List[int]
) -> List[int]:
    rank_to_local = {rank: index for index, rank in enumerate(group_ranks)}
    return [rank_to_local[rank] for rank in global_ranks if rank in rank_to_local]


def _wait_for_peer_state(backend, ranks: List[int]) -> None:
    from mooncake.pg import get_peer_state

    while not all(get_peer_state(backend, ranks)):
        time.sleep(_PEER_STATE_POLL_INTERVAL_SEC)


def _maybe_create_message_queue(group) -> None:
    if not group.use_message_queue_broadcaster or group.world_size <= 1:
        return

    from sglang.srt.distributed.device_communicators.shm_broadcast import MessageQueue

    group.mq_broadcaster = MessageQueue.create_from_process_group(
        group.cpu_group, 1 << 22, 6
    )


def _try_recover_world(global_ranks: List[int]) -> bool:
    from mooncake.pg import get_peer_state, recover_ranks

    world_backend = torch.distributed.group.WORLD
    if not all(get_peer_state(world_backend, global_ranks)):
        return False

    recover_ranks(world_backend, global_ranks)
    logger.debug("[Elastic EP][recover] WORLD recover_ranks(%s) done", global_ranks)
    return True


def try_admit_scale_ranks(global_ranks: List[int]) -> bool:
    """Admit append-only ranks into the expandable WORLD group."""
    if not _try_recover_world(global_ranks):
        return False

    _refresh_ep_members()
    return True


def try_recover_ranks(global_ranks: List[int]) -> bool:
    """Recover ranks in WORLD and every launch-time parallel group."""
    if not _try_recover_world(global_ranks):
        return False

    from mooncake.pg import recover_ranks

    for group in _iter_live_parallel_groups():
        local_ranks = _map_global_to_group_local_ranks(group.ranks, global_ranks)
        if not local_ranks:
            continue

        _wait_for_peer_state(group.device_group, local_ranks)
        recover_ranks(group.device_group, local_ranks)
        _wait_for_peer_state(group.cpu_group, local_ranks)
        recover_ranks(group.cpu_group, local_ranks)
        _maybe_create_message_queue(group)

    _refresh_ep_members()
    return True


def _join_world_group() -> None:
    from mooncake.pg import join_group

    join_group(torch.distributed.group.WORLD)


def join_scale_process_group() -> None:
    """Join the expandable WORLD group for an append-only scale operation."""
    _join_world_group()
    _refresh_ep_members()


def ensure_mooncake_ep_buffer(model) -> bool:
    """Create lazy eager EP state before the membership IPC all-gather.

    Opt-in recovery recipes use this alongside post-recovery layout repair.
    Captured/Foundry paths already own a buffer and do not allocate another one.
    """
    if os.environ.get("SGLANG_ELASTIC_REBALANCE_AFTER_RECOVER", "0") not in (
        "1",
        "true",
    ):
        return False
    if get_exec().moe.elastic_ep_backend != "mooncake":
        return False
    from sglang.srt.layers.moe.token_dispatcher.mooncake import (
        EPBuffer,
        MooncakeEPDispatcher,
    )

    if EPBuffer.get_existing_buffer() is not None:
        return True
    for module in model.modules():
        dispatcher = getattr(module, "dispatcher", None)
        if dispatcher is None:
            continue
        for candidate in (dispatcher, *getattr(dispatcher, "_inners", [])):
            if isinstance(candidate, MooncakeEPDispatcher):
                impl = getattr(candidate, "_low_latency_dispatcher", None)
                if impl is not None:
                    impl._get_buffer()
                    if EPBuffer.get_existing_buffer() is None:
                        raise RuntimeError(
                            "Mooncake dispatcher did not create its recovery EP buffer"
                        )
                    logger.info(
                        "[Elastic EP] created the mooncake EP buffer before joining"
                    )
                    return True
    raise RuntimeError(
        "Mooncake recovery requires an EP dispatcher buffer before membership join"
    )


def _recovery_layout_summary(layout: str) -> dict:
    """One-time evidence of installed slots and this rank's actual dispatch map."""
    metadata = get_global_expert_location_metadata()
    if metadata is None:
        raise RuntimeError("Recovery layout metadata is missing")
    mapping = metadata.physical_to_logical_map_cpu.tolist()
    ep_size = metadata.ep_size
    local_slots = metadata.num_local_physical_experts
    slots_by_rank = [
        sum(
            value >= 0
            for layer in mapping
            for value in layer[rank * local_slots : (rank + 1) * local_slots]
        )
        for rank in range(ep_size)
    ]
    dispatch = metadata.logical_to_rank_dispatch_physical_map
    dispatch_by_rank = None
    if dispatch is not None:
        dispatch_by_rank = [0] * ep_size
        for layer in dispatch.detach().cpu().tolist():
            for physical_slot in layer:
                if not 0 <= physical_slot < ep_size * local_slots:
                    raise RuntimeError(
                        "Recovery dispatch map references an invalid expert slot"
                    )
                dispatch_by_rank[physical_slot // local_slots] += 1
    return dict(
        layout=layout,
        tp_rank=get_parallel().tp_rank + get_parallel().ep_join_rank_offset,
        ep_size=ep_size,
        physical_slots_per_rank=local_slots,
        populated_slots_across_layers_by_rank=slots_by_rank,
        ranks_with_experts=[rank for rank, count in enumerate(slots_by_rank) if count],
        dispatch_entries_by_rank=dispatch_by_rank,
        physical_mapping_sha256=hashlib.sha256(
            json.dumps(mapping, separators=(",", ":")).encode()
        ).hexdigest(),
    )


def schedule_post_recovery_rebalance(eplb_manager) -> None:
    """Opt-in collective layout restoration on survivors AND recovered ranks."""
    if os.environ.get("SGLANG_ELASTIC_REBALANCE_AFTER_RECOVER", "0") not in (
        "1",
        "true",
    ):
        return
    layout = os.environ.get("SGLANG_ELASTIC_POST_RECOVERY_LAYOUT", "trivial")
    if layout not in ("trivial", "eplb"):
        raise ValueError(f"Unsupported post-recovery expert layout: {layout}")
    if eplb_manager is None:
        raise RuntimeError("Post-recovery expert layout requires EPLB")
    logger.info("[Elastic EP] post-recovery EPLB rebalance (layout=%s)", layout)
    started = time.perf_counter()
    for _ in eplb_manager.rebalance(restore_trivial=layout == "trivial"):
        pass
    # P2P completion alone does not synchronize ranks that have no expert moves.
    # All recovered workers must install their metadata before service is ready.
    torch.distributed.barrier()
    eplb_manager.reset_generator()
    summary = _recovery_layout_summary(layout)
    logger.info(
        "[Elastic EP] post-recovery EPLB rebalance done in %.3f s",
        time.perf_counter() - started,
    )
    logger.info(
        "[Elastic EP] recovery layout ready %s", json.dumps(summary, sort_keys=True)
    )


def join_process_groups() -> None:
    """Rejoin WORLD and every launch-time parallel group after recovery."""
    from mooncake.pg import join_group

    _join_world_group()
    for group in _iter_live_parallel_groups():
        if group.world_size <= 1:
            continue
        join_group(group.device_group)
        join_group(group.cpu_group)
        _maybe_create_message_queue(group)

    _refresh_ep_members()


def get_healthy_expert_location_src_rank(
    *, invoked_in_elastic_ep_rejoin_path: bool
) -> int:
    world_group = get_parallel().world_group
    # NOTE: do not key off the launch-time `ep_join_mode` here.
    # A rank that was started as a rejoin rank may later act as a healthy
    # rank in a subsequent recovery cycle.
    local_rejoin_flag = bool(invoked_in_elastic_ep_rejoin_path)
    gathered_rejoin_flags = world_group.all_gather_object(local_rejoin_flag)

    for rank_in_group, is_rejoin_rank in enumerate(gathered_rejoin_flags):
        if not is_rejoin_rank:
            return world_group.ranks[rank_in_group]

    raise RuntimeError(
        "No healthy rank found for broadcasting expert location metadata. "
        "All ranks are marked as elastic_ep_rejoin."
    )


def maybe_recover_ep_ranks(
    *,
    tp_group: parallel_state.GroupCoordinator,
    eplb_manager: EPLBManager,
    model_config: ModelConfig,
    moe_ep_rank: int,
) -> bool:
    # TODO(perf): `active_ranks.all()` on a CUDA tensor triggers host-device
    # synchronization, and this function is on the forward-path.
    # This check only runs when `--elastic-ep-backend` is enabled, so the
    # synchronization overhead does not propagate to other configs.
    # Leave for future optimization of the elastic EP path.
    if tp_group.active_ranks.all() and tp_group.active_ranks_cpu.all():
        return False

    tp_active_ranks = tp_group.active_ranks.detach().cpu().numpy()
    tp_active_ranks_cpu = tp_group.active_ranks_cpu.detach().numpy()
    tp_active_ranks &= tp_active_ranks_cpu
    # NOTE: `ranks_to_recover` uses indices in `tp_group`. For the current
    # Mooncake elastic EP implementation we assume `--pp-size=1`, so the
    # tp-group index is the same as the global rank index.
    ranks_to_recover = [
        i for i in range(len(tp_active_ranks)) if not tp_active_ranks[i]
    ]

    # try_recover_ranks polls peer state via Mooncake EP backend.
    # Mooncake's internal semantics guarantee that all ranks observe
    # consistent peer readiness state, so collective operations below
    # are safe even though polling appears local.
    if ranks_to_recover and try_recover_ranks(ranks_to_recover):
        eplb_manager.reset_generator()
        broadcast_global_expert_location_metadata(
            model_config=model_config,
            moe_ep_rank=moe_ep_rank,
            src_rank=get_healthy_expert_location_src_rank(
                invoked_in_elastic_ep_rejoin_path=False
            ),
        )
        ElasticEPStateManager.instance().reset()
        if os.environ.get("SGLANG_ELASTIC_REBALANCE_AFTER_RECOVER", "0") in (
            "1",
            "true",
        ):
            logger.info(f"recover ranks {ranks_to_recover} membership restored")
        schedule_post_recovery_rebalance(eplb_manager)
        logger.info(f"recover ranks {ranks_to_recover} done")
        return True

    return False


def _wait_for_pg_fault_membership(elastic_ep_state, tp_group) -> None:
    """Recipe-only gate: do not move experts against stale PG membership.

    EP kernels and PG have separate active masks. Activation-ready peer state
    is not a substitute for the authoritative CPU/device group masks here.
    """
    from fault_timing import get_config

    if tp_group is None:
        raise RuntimeError("EEP PG fault gate requires the TP group")
    config = get_config()
    timeout_s = (
        2
        * (
            config["pg_fault_window_us"]
            + max(config["pg_collective_timeout_us"], config["pg_p2p_timeout_us"])
        )
        / 1_000_000
    )
    current = [bool(x) for x in elastic_ep_state.active_ranks.detach().cpu().tolist()]
    previous = [
        bool(x) for x in elastic_ep_state.last_active_ranks.detach().cpu().tolist()
    ]
    if len(previous) != len(current):
        raise RuntimeError("EEP PG fault gate: EP mask shape changed")
    newly_dead = [
        i for i, (old, new) in enumerate(zip(previous, current)) if old and not new
    ]
    if not newly_dead:
        return
    local_rank = tp_group.rank_in_group
    if not 0 <= local_rank < len(current):
        raise RuntimeError("EEP PG fault gate: invalid local group rank")
    started = time.monotonic()
    logger.info(
        "[Elastic EP] PG fault gate begin dead=%s timeout_s=%.3f",
        newly_dead,
        timeout_s,
    )
    while True:
        device = [bool(x) for x in tp_group.active_ranks.detach().cpu().tolist()]
        cpu = [bool(x) for x in tp_group.active_ranks_cpu.detach().cpu().tolist()]
        if len(device) != len(current) or len(cpu) != len(current):
            raise RuntimeError("EEP PG fault gate: PG/EP mask shape mismatch")
        if not (device[local_rank] and cpu[local_rank]):
            raise RuntimeError(
                f"EEP PG fault gate: PG removed this process's rank {local_rank}; "
                "refusing expert movement from an inactive member"
            )
        elapsed = time.monotonic() - started
        if device == cpu and all(not device[i] for i in newly_dead):
            # PG can observe the second failed rank before EP does. Merge only
            # removals into the existing tensor; graph-held pointers stay valid
            # and this fault path never activates a reserved/rejoining slot.
            extra_dead = [
                i for i, alive in enumerate(current) if alive and not device[i]
            ]
            for rank in extra_dead:
                elastic_ep_state.active_ranks[rank] = 0
            logger.info(
                "[Elastic EP] PG fault gate end dead=%s additional_dead=%s duration_s=%.6f",
                newly_dead,
                extra_dead,
                elapsed,
            )
            return
        if elapsed >= timeout_s:
            raise TimeoutError(
                f"EEP PG fault gate timed out after {elapsed:.3f}s: dead={newly_dead} "
                f"device={device} cpu={cpu}; refusing expert movement"
            )
        time.sleep(min(0.1, timeout_s - elapsed))


def maybe_rebalance_after_rank_fault(
    *, eplb_manager: EPLBManager, tp_group=None
) -> bool:
    elastic_ep_state = ElasticEPStateManager.instance()
    if elastic_ep_state is None or elastic_ep_state.is_active_equal_last():
        return False
    if os.environ.get("EEP_FAULT_TIMING") == "1":
        _wait_for_pg_fault_membership(elastic_ep_state, tp_group)
    elastic_ep_state.snapshot_active_to_last()
    elastic_ep_state.sync_active_to_cpu()
    logger.info("EPLB due to rank faults")
    gen = eplb_manager.rebalance()
    while True:
        try:
            next(gen)
        except StopIteration:
            break
    return True
