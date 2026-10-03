"""CPU-only admission for the bounded, balanced DP2/attention-TP1 probe.

This module neither imports torch nor issues collectives. The caller MUST gather
one offer from every TP/EP participant on every common replay boundary, before
returning for an unarmed/seen phase, an idle rank, or a local batch mismatch.
``replay_index`` counts those boundaries, including skipped probes. A watchdog
is still required: this protocol cannot gather a rank that never calls it.

``phase['batch']`` retains the suite's LOCAL captured-batch meaning. DP runs must
also supply ``phase['global_batch'] == phase['batch'] * dp_size``. The original
ForwardBatch passed to DecodeCudaGraphRunner.backend.replay has live local
``batch_size``; runner.bs and ShapeKey.size describe the padded capture. Read
``original_global_num_tokens_cpu``, not its possibly padded/scaled replacement.
Despite its name this field is a LOCAL singleton when require_mlp_tp_gather is
false; the complete vector is then reconstructed from the gathered offers.
Only ordinary one-token-per-request DECODE, exact capture buckets, and balanced
DP2 batches are admitted. Prefill, IDLE, speculation and ragged/padded batches
are deliberately skipped. The helper does not mark a phase seen or launch work.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence


PROTOCOL = "foundry-balanced-dp-probe-v1"


def _integer(value, name, minimum=0):
    # Do not call int(tensor): it could introduce a device synchronization.
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be a Python int >= {minimum}")
    return value


def make_offer(*, rank, world_size, dp_rank, dp_size, attn_tp_size,
               replay_index, phase, seen_phase_ids, forward_mode, raw_batch,
               runner_raw_batch, padded_batch, capture_batch, global_num_tokens,
               can_run_decode_cuda_graph, runner_name="DecodeCudaGraphRunner",
               require_mlp_tp_gather=True):
    """Snapshot already-CPU metadata, retaining local errors for consensus.

    Caller should pass mode.name, e.g. ``DECODE`` or ``IDLE``. On a local parse
    error, return an error offer and still participate in the same gather. A
    phase-file read error can be represented as ``phase={'read_error': repr(e)}``
    and will reject collectively. No handles, tensor contents, or pointers are
    included. Offer dictionaries are suitable for Gloo all_gather_object/JSON.
    """
    offer = {"protocol": PROTOCOL, "rank": rank, "world_size": world_size,
             "replay_index": replay_index, "error": None}
    try:
        for name, value, minimum in (
            ("rank", rank, 0), ("world_size", world_size, 1),
            ("dp_rank", dp_rank, 0), ("dp_size", dp_size, 1),
            ("attn_tp_size", attn_tp_size, 1), ("replay_index", replay_index, 0),
            ("raw_batch", raw_batch, 0), ("runner_raw_batch", runner_raw_batch, 0),
            ("padded_batch", padded_batch, 0), ("capture_batch", capture_batch, 0),
        ):
            offer[name] = _integer(value, name, minimum)
        if not isinstance(runner_name, str) or not isinstance(forward_mode, str):
            raise ValueError("runner_name and forward_mode must be strings")
        if type(can_run_decode_cuda_graph) is not bool:
            raise ValueError("can_run_decode_cuda_graph must be a Python bool")
        if type(require_mlp_tp_gather) is not bool:
            raise ValueError("require_mlp_tp_gather must be a Python bool")
        if phase is None:
            phase = {"armed": False}
        if not isinstance(phase, Mapping):
            raise ValueError("phase must be a mapping or None")
        phase = json.loads(json.dumps(dict(phase), sort_keys=True, allow_nan=False))
        if "read_error" in phase:
            raise ValueError(f"phase read error: {phase['read_error']}")
        if type(phase.get("armed")) is not bool:
            raise ValueError("phase.armed must be a bool")
        if phase["armed"]:
            _integer(phase.get("id"), "phase.id")
            _integer(phase.get("batch"), "phase.batch", 1)
            _integer(phase.get("global_batch"), "phase.global_batch", 1)
        if global_num_tokens is not None:
            if not isinstance(global_num_tokens, (list, tuple)):
                raise ValueError("global_num_tokens must be a CPU list/tuple or None")
            global_num_tokens = [_integer(x, "global_num_tokens member")
                                 for x in global_num_tokens]
        offer.update(dp_rank=dp_rank, dp_size=dp_size, attn_tp_size=attn_tp_size,
                     phase=phase, seen=phase.get("id") in seen_phase_ids,
                     forward_mode=forward_mode, runner_name=runner_name,
                     global_num_tokens=global_num_tokens,
                     require_mlp_tp_gather=require_mlp_tp_gather,
                     can_run_decode_cuda_graph=can_run_decode_cuda_graph)
    except Exception as exc:
        offer["error"] = f"{type(exc).__name__}: {exc}"
    return offer


def decide(offers):
    """Return the same probe/skip/reject decision from the gathered rank set.

    ``reject`` is a fail-stop metadata/protocol fault, never permission to issue
    CUDA cleanup. ``skip`` means all ranks continue their ordinary engine path.
    Phase publication races skip as a group; a partially consumed phase rejects.
    Admission checks preserve all offers in the caller's report for diagnosis.
    """
    def result(action, reason, **fields):
        return {"protocol": PROTOCOL, "action": action, "reason": reason, **fields}

    try:
        if not isinstance(offers, Sequence) or isinstance(offers, (str, bytes)) or not offers:
            return result("reject", "empty_or_invalid_rank_set")
        if any(not isinstance(x, Mapping) for x in offers):
            return result("reject", "malformed_offer")
        if any(x.get("protocol") != PROTOCOL for x in offers):
            return result("reject", "protocol_mismatch")
        if any(x.get("error") for x in offers):
            return result("reject", "local_metadata_error",
                          errors=[x.get("error") for x in offers])
        n = len(offers)
        if any(type(x.get("rank")) is not int for x in offers):
            return result("reject", "invalid_rank_set")
        offers = sorted(offers, key=lambda x: x["rank"])
        if [x["rank"] for x in offers] != list(range(n)):
            return result("reject", "invalid_rank_set")
        if any(x["world_size"] != n for x in offers):
            return result("reject", "incomplete_rank_set")
        first = offers[0]
        if any(x["replay_index"] != first["replay_index"] for x in offers):
            return result("reject", "replay_boundary_disagreement")
        if n != 2 or any(x["dp_size"] != 2 or x["attn_tp_size"] != 1 for x in offers):
            return result("reject", "unsupported_parallel_layout")
        if [x["dp_rank"] for x in offers] != [0, 1]:
            return result("reject", "unexpected_dp_rank_mapping")
        if any(x["runner_name"] != first["runner_name"] for x in offers):
            return result("reject", "runner_disagreement")
        if any(x["phase"] != first["phase"] for x in offers):
            return result("skip", "phase_publication_race")
        phase = first["phase"]
        if not phase["armed"]:
            return result("skip", "unarmed")
        if any(x["seen"] for x in offers):
            if not all(x["seen"] for x in offers):
                return result("reject", "partially_consumed_phase")
            return result("skip", "already_probed")
        if phase["global_batch"] != phase["batch"] * n:
            return result("reject", "phase_local_global_batch_conflict")
        if first["runner_name"] != "DecodeCudaGraphRunner":
            return result("skip", "not_decode_runner")
        if any(x["forward_mode"] != "DECODE" for x in offers):
            return result("skip", "not_all_ranks_ordinary_decode")
        if not all(x["can_run_decode_cuda_graph"] for x in offers):
            return result("reject", "missing_global_decode_graph_admission")
        if any(x["raw_batch"] != x["runner_raw_batch"] for x in offers):
            return result("reject", "runner_forward_batch_disagreement")
        if any(x["capture_batch"] != x["padded_batch"] for x in offers):
            return result("reject", "runner_capture_key_disagreement")
        gathered_counts = first["require_mlp_tp_gather"]
        if any(x["require_mlp_tp_gather"] != gathered_counts for x in offers):
            return result("reject", "count_representation_disagreement")
        if gathered_counts:
            counts = first["global_num_tokens"]
            if counts is None or len(counts) != n:
                return result("reject", "missing_or_invalid_global_counts")
            if any(x["global_num_tokens"] != counts for x in offers):
                return result("reject", "global_counts_disagreement")
            if any(x["raw_batch"] != counts[x["dp_rank"]] for x in offers):
                return result("reject", "local_global_counts_disagreement")
            counts_source = "shared_global_vector"
        else:
            if any(x["global_num_tokens"] != [x["raw_batch"]] for x in offers):
                return result("reject", "local_count_singleton_disagreement")
            counts = [x["raw_batch"] for x in offers]
            counts_source = "cpu_offer_gather_of_local_singletons"
        if counts != [phase["batch"]] * n:
            return result("skip", "not_target_balanced_live_batch",
                          observed_global_batch=sum(counts), dp_local_batches=counts)
        if any(x["capture_batch"] != phase["batch"] for x in offers):
            return result("skip", "not_exact_target_capture_bucket")
        return result("probe", "all_ranks_agree", phase_id=phase["id"],
                      replay_index=first["replay_index"], local_batch=phase["batch"],
                      global_batch=phase["global_batch"], dp_local_batches=counts,
                      counts_source=counts_source,
                      capture_batches=[x["capture_batch"] for x in offers])
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        return result("reject", "malformed_offer", error=f"{type(exc).__name__}: {exc}")


def gather_and_decide(local_offer, gather):
    """Always call a supplied CPU gather, even when this rank would skip.

    ``gather(offer)`` returns the complete offer list. A Gloo adapter can create
    a world-size list, call all_gather_object, and return that list. Exceptions
    propagate to the worker's fail-stop handler; there are no CUDA finalizers.
    """
    offers = gather(local_offer)
    return decide(offers), offers
