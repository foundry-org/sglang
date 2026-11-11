from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional

import msgspec
import torch

from sglang.srt.distributed import parallel_state
from sglang.srt.distributed.utils import get_global_tcp_store
from sglang.srt.eplb.expert_location import broadcast_global_expert_location_metadata
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


def diag_recover_launch_map() -> Optional[torch.Tensor]:
    """DIAGNOSTIC (SGLANG_DIAG_ELASTIC_NIXL_RECOVER): a restarted rank's expert map before the
    recovery broadcast; its expert weights were loaded for this map."""
    from sglang.srt.environ import envs
    from sglang.srt.eplb.expert_location import get_global_expert_location_metadata

    if not envs.SGLANG_DIAG_ELASTIC_NIXL_RECOVER.get():
        return None
    return get_global_expert_location_metadata().physical_to_logical_map.clone()


def diag_recover_expert_reload_filter(
    *, launch_map: Optional[torch.Tensor]
) -> Optional[Callable[[str], bool]]:
    """DIAGNOSTIC (SGLANG_DIAG_ELASTIC_RECOVER_RELOAD_EXPERTS): a routed-expert weight-name filter
    when the recovery broadcast changed the logical experts of this rank's slots, else None."""
    from sglang.srt.environ import envs
    from sglang.srt.eplb.expert_location import get_global_expert_location_metadata

    if launch_map is None or not envs.SGLANG_DIAG_ELASTIC_RECOVER_RELOAD_EXPERTS.get():
        return None
    p2l = get_global_expert_location_metadata().physical_to_logical_map
    slots = p2l.shape[1] // get_parallel().tp_size
    own = slice(get_parallel().tp_rank * slots, (get_parallel().tp_rank + 1) * slots)
    if torch.equal(p2l[:, own], launch_map[:, own]):
        return None
    logger.info(
        "[DIAG] recover: reloading routed experts from disk for the broadcast map"
    )
    return lambda name: ".experts." in name


def diag_nixl_recover_connect(
    *, restarted_ranks: Optional[List[int]], launch_map: Optional[torch.Tensor] = None
) -> None:
    """DIAGNOSTIC (SGLANG_DIAG_ELASTIC_NIXL_RECOVER): NIXL connections after a PG recovery;
    survivors pass the restarted ranks, a restarted rank passes None. Both sides rendezvous here."""
    from sglang.srt.environ import envs

    if (
        not envs.SGLANG_DIAG_ELASTIC_NIXL_RECOVER.get()
        or get_exec().moe.moe_a2a_backend != "nixl"
    ):
        return
    from sglang.srt.layers.moe.token_dispatcher.nixl import NixlEPBuffer

    t0 = time.perf_counter()
    if restarted_ranks is None:
        ranks, masks = NixlEPBuffer.diag_connect_recover_pending()
        role = "restarted"
    else:
        masks = NixlEPBuffer.diag_reconnect_restarted_ranks(restarted_ranks)
        ranks, role = restarted_ranks, "survivor"
    from sglang.srt.eplb.expert_location import get_global_expert_location_metadata

    if launch_map is not None:
        # Per rank: (layer, slot) entries whose logical expert differs between the broadcast map
        # and the map this rank loaded its weights for.
        p2l = get_global_expert_location_metadata().physical_to_logical_map
        moved = (
            (p2l != launch_map).sum(dim=0).view(get_parallel().tp_size, -1).sum(dim=1)
        )
        logger.info(
            "[DIAG] NIXL recover: map entries changed vs this rank's load-time map, per rank=%s",
            moved.tolist(),
        )
    logger.info(
        "[DIAG] NIXL recover: %s connected ranks=%s mask before/after clean=%s/%s in %.3f s",
        role,
        ranks,
        masks[0],
        masks[1],
        time.perf_counter() - t0,
    )


class _DiagPgFaultWatch:
    last_mirror: Optional[List[int]] = None


def _diag_pg_active_mirror() -> List[int]:
    """Active ranks as the PG sees them, host memory only. The launch-time TP group holds only the launch ranks;
    after a scale-up the admitted ranks are only in WORLD, whose Mooncake size is the upper bound of the active
    rank space (`MooncakeCommunicator::getSize`), so there only an eviction of the highest ranks is seen."""
    mirror = get_parallel().tp_group.active_ranks_cpu.tolist()
    if not elastic_expanded_world_enabled() or ElasticEPStateManager.is_scaling():
        return mirror
    world_size = torch.distributed.get_world_size()
    return [
        mirror[r] if r < len(mirror) else int(r < world_size)
        for r in range(ElasticEPStateManager.get_effective_ep_size())
    ]


def diag_pg_fault_unblock(*, send_to_tokenizer: Callable) -> None:
    """DIAGNOSTIC (SGLANG_DIAG_ELASTIC_PG_FAULT_UNBLOCK): act on a Mooncake PG eviction before any NIXL timeout.

    Called once per scheduler loop iteration; reads the PG's CPU active-rank mirror (host memory, no GPU sync).
    On a 1 -> 0 transition the NIXL EP buffer masks those ranks, so the next dispatch / combine skips them
    instead of waiting for the kernel timeout, and the DP controller is told to stop routing to their DP ranks.
    The rank-fault rebalance (the next combine reads the mask) and the recovery then run unchanged."""
    from sglang.srt.environ import envs

    if not envs.SGLANG_DIAG_ELASTIC_PG_FAULT_UNBLOCK.get():
        return
    mirror = _diag_pg_active_mirror()
    last = _DiagPgFaultWatch.last_mirror
    _DiagPgFaultWatch.last_mirror = mirror
    if last is None:
        return
    evicted = [r for r, (was, now) in enumerate(zip(last, mirror)) if was and not now]
    if not evicted:
        return
    t0 = time.perf_counter()
    if get_exec().moe.moe_a2a_backend == "nixl":
        from sglang.srt.layers.moe.token_dispatcher.nixl import NixlEPBuffer

        NixlEPBuffer.diag_mask_ranks(evicted)
    from sglang.srt.elastic_ep.topology import (
        attn_replica_size,
        collapse_physical_rank_status,
    )
    from sglang.srt.managers.io_struct import ActiveRanksOutput

    physical = ElasticEPStateManager.instance().active_ranks_cpu.tolist()
    for r in evicted:
        physical[r] = 0
    dp_status = collapse_physical_rank_status(physical, attn_replica_size())
    send_to_tokenizer(ActiveRanksOutput(status=dp_status))
    logger.info(
        "[DIAG] PG fault unblock: ranks %s evicted by the PG (mirror %s) -> NIXL masked, DP status %s, %.3f s",
        evicted,
        mirror,
        dp_status,
        time.perf_counter() - t0,
    )


def diag_scale_timeline(step: str) -> None:
    """DIAGNOSTIC (SGLANG_DIAG_ELASTIC_SCALE_TIMELINE): one wall-clock line per scale step."""
    from sglang.srt.environ import envs

    if not envs.SGLANG_DIAG_ELASTIC_SCALE_TIMELINE.get():
        return
    from sglang.srt.model_executor.model_runner_components.capture_state import (
        DIAG_CAPTURE_COUNTS,
    )

    role = "joiner" if get_exec().moe.is_ep_scale_joiner else "primary"
    logger.info(
        "[DIAG timeline] %s tp_rank=%d step=%s t=%.3f counts=%s",
        role,
        get_parallel().tp_rank,
        step,
        time.time(),
        DIAG_CAPTURE_COUNTS,
    )


class ScaleCohort(msgspec.Struct, frozen=True, kw_only=True):
    target_ep_size: int
    cuda_graph_enabled: bool
    attn_tp_size: int
    attn_cp_size: int


def register_scale_cohort(
    rank_offset: int, target_ep_size: int, cuda_graph_enabled: bool
) -> None:
    store = get_global_tcp_store()
    if store is None:
        raise RuntimeError("Elastic EP scale-up requires the global TCPStore.")
    parallel = get_parallel()
    payload = msgspec.json.encode(
        ScaleCohort(
            target_ep_size=target_ep_size,
            cuda_graph_enabled=cuda_graph_enabled,
            attn_tp_size=parallel.attn_tp_size,
            attn_cp_size=parallel.attn_cp_size,
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
    if _diag_hide_cohort_once():
        return None
    return msgspec.json.decode(store.get(key), type=ScaleCohort)


_DIAG_COHORT_HIDDEN = [False]


def _diag_hide_cohort_once() -> bool:
    """DIAGNOSTIC (SGLANG_DIAG_ELASTIC_DELAY_COHORT_TP_RANK): that TP rank misses a present cohort once."""
    from sglang.srt.environ import envs

    if (
        _DIAG_COHORT_HIDDEN[0]
        or envs.SGLANG_DIAG_ELASTIC_DELAY_COHORT_TP_RANK.get() != get_parallel().tp_rank
    ):
        return False
    _DIAG_COHORT_HIDDEN[0] = True
    logger.info("[DIAG] cohort hidden from this rank for one forward")
    return True


def diag_agree_scale_progress(
    *,
    local_timeout: bool,
    waiting_for_cohort: bool,
    effective_size: int,
    pending_size: int,
    like: torch.Tensor,
) -> tuple:
    """DIAGNOSTIC (SGLANG_DIAG_ELASTIC_SCALE_REQ_CONSENSUS): one WORLD all-reduce (MAX) agrees on the scale
    timeout, the joining cohort's visibility and the joiners' peer readiness, so every rank takes the same
    branch of the scale step in the same forward. Returns (timed_out, cohort, peers_ready)."""
    from mooncake.pg import get_peer_state

    cohort = get_scale_cohort(effective_size) if waiting_for_cohort else None
    not_ready = 1
    if cohort is not None or not waiting_for_cohort:
        ranks = list(range(effective_size, pending_size))
        world = torch.distributed.group.WORLD
        not_ready = int(not (ranks and all(get_peer_state(world, ranks))))
    flags = like.new_tensor([int(local_timeout), int(cohort is not None), not_ready])
    torch.distributed.all_reduce(
        flags, op=torch.distributed.ReduceOp.MAX, group=torch.distributed.group.WORLD
    )
    timed_out, any_cohort, any_not_ready = flags.tolist()
    if any_cohort and waiting_for_cohort and cohort is None:
        # Another rank saw the cohort key: it exists, so this blocking read returns at once.
        key = f"{_SCALE_COHORT_KEY_PREFIX}/{effective_size}"
        cohort = msgspec.json.decode(get_global_tcp_store().get(key), type=ScaleCohort)
    return bool(timed_out), cohort, not any_not_ready


def validate_scale_cohort_topology(cohort: ScaleCohort) -> None:
    from sglang.srt.elastic_ep.topology import validate_matching_attention_topology

    parallel = get_parallel()
    validate_matching_attention_topology(
        primary_attn_tp_size=parallel.attn_tp_size,
        primary_attn_cp_size=parallel.attn_cp_size,
        joining_attn_tp_size=cohort.attn_tp_size,
        joining_attn_cp_size=cohort.attn_cp_size,
    )


def update_dp_attention_for_elastic_ep(
    *, physical_ep_size: int, physical_ep_rank: int
) -> int:
    """Apply a physical EP topology to the logical DP-attention runtime."""
    from sglang.srt.elastic_ep.topology import (
        attn_replica_size,
        physical_ep_rank_to_dp_rank,
        physical_ep_size_to_dp_size,
    )
    from sglang.srt.layers.dp_attention import update_dp_attention_post_scale

    replica_size = attn_replica_size()
    dp_size = physical_ep_size_to_dp_size(physical_ep_size, replica_size)
    update_dp_attention_post_scale(
        new_dp_size=dp_size,
        new_dp_rank=physical_ep_rank_to_dp_rank(physical_ep_rank, replica_size),
    )
    get_parallel().override_permanently(num_dp_ranks=dp_size)
    return dp_size


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
        # In place: a fresh clone per snapshot moves later small blocks, so the init-time
        # snapshot (capacity > world size only) would make the layout depend on the EP size.
        if self.active_ranks is not None:
            self.last_active_ranks.copy_(self.active_ranks)

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
    # DIAGNOSTIC (SGLANG_DIAG_ELASTIC_SCALE_REQ_CONSENSUS): a scale request this rank received but has
    # not activated; the MLP sync carries it, so every rank activates it in the same iteration.
    _noted_ep_size: Optional[int] = None
    # DIAGNOSTIC (SGLANG_DIAG_ELASTIC_DELAY_SCALE_REQ_TP_RANK): a request held back until after the next forward.
    _diag_deferred_ep_size: Optional[int] = None

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
    def note_scale_request(cls, n: int) -> None:
        cls._noted_ep_size = n

    @classmethod
    def noted_scale_request(cls) -> int:
        return cls._noted_ep_size or 0

    @classmethod
    def apply_agreed_scale_request(cls, n: int) -> None:
        """Activate the largest scale request any rank has noted (the MLP-sync maximum)."""
        inst = cls._instance
        if n <= 0 or inst is None:
            return
        if inst.pending_ep_size is None and n > inst.effective_ep_size:
            cls.request_scale(n)
            logger.info("[DIAG] scale request %d activated by the MLP sync", n)
        noted = cls._noted_ep_size
        if noted is not None and inst.pending_ep_size == noted:
            cls._noted_ep_size = None

    @classmethod
    def diag_release_deferred_scale_request(cls) -> None:
        n = cls._diag_deferred_ep_size
        if n is None:
            return
        cls._diag_deferred_ep_size = None
        from sglang.srt.environ import envs

        if envs.SGLANG_DIAG_ELASTIC_SCALE_REQ_CONSENSUS.get():
            cls.note_scale_request(n)
        else:
            cls.request_scale(n)
        logger.info("[DIAG] deferred scale request %d released after a forward", n)

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
        diag_nixl_recover_connect(restarted_ranks=ranks_to_recover)
        logger.info(f"recover ranks {ranks_to_recover} done")
        return True

    return False


def maybe_rebalance_after_rank_fault(*, eplb_manager: EPLBManager) -> bool:
    elastic_ep_state = ElasticEPStateManager.instance()
    if elastic_ep_state is None or elastic_ep_state.is_active_equal_last():
        return False
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
