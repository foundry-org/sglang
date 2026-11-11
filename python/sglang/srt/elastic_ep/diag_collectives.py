"""DIAGNOSTIC (SGLANG_DIAG_COLLECTIVE_COUNTS=1): count torch.distributed collectives per
(op, group backend/size, tensor device, bytes, first sglang caller frame); one log block per 20 s."""

import logging
import sys
import time

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

logger = logging.getLogger(__name__)

_OPS = (
    "all_reduce",
    "all_gather",
    "all_gather_into_tensor",
    "all_gather_object",
    "broadcast",
    "broadcast_object_list",
    "gather",
    "reduce_scatter_tensor",
    "all_to_all_single",
    "barrier",
    "send",
    "recv",
    "isend",
    "irecv",
)
_PERIOD_S = 20.0
_counts: dict = {}
_last = [time.monotonic()]
_installed = [False]


def _caller() -> str:
    f = sys._getframe(2)
    while f is not None:
        name = f.f_code.co_filename
        if "/sglang/" in name and "diag_collectives" not in name:
            return f"{name.split('/sglang/', 1)[1]}:{f.f_lineno}"
        f = f.f_back
    return "?"


def _describe(args, kwargs):
    group = kwargs.get("group")
    if group is None:
        for a in args:
            if isinstance(a, dist.ProcessGroup):
                group = a
    tensor = next((a for a in args if isinstance(a, torch.Tensor)), None)
    try:
        g = group or c10d._get_default_group()
        gname = f"{dist.get_backend(g)}/{dist.get_world_size(g)}/{g.group_desc}"
    except Exception:
        gname = "?"
    if tensor is None:
        return gname, "-", 0
    return gname, tensor.device.type, tensor.numel() * tensor.element_size()


def _wrap(name, fn):
    def wrapper(*args, **kwargs):
        if not _in_wrapper[0]:
            _in_wrapper[0] = True
            try:
                key = (name, *_describe(args, kwargs), _caller())
                _counts[key] = _counts.get(key, 0) + 1
                now = time.monotonic()
                if now - _last[0] > _PERIOD_S:
                    _dump(now)
            finally:
                _in_wrapper[0] = False
        return fn(*args, **kwargs)

    wrapper.__wrapped__ = fn
    return wrapper


_in_wrapper = [False]


def _dump(now: float) -> None:
    dt = now - _last[0]
    _last[0] = now
    rows = sorted(_counts.items(), key=lambda kv: -kv[1])
    _counts.clear()
    lines = [
        f"  {n:7d} ({n / dt:7.1f}/s) {op} group={g} dev={d} bytes={b} at {c}"
        for (op, g, d, b, c), n in rows
    ]
    logger.info(
        "[DIAG] collectives in the last %.1f s:\n%s", dt, "\n".join(lines) or "  none"
    )


def install() -> None:
    if _installed[0]:
        return
    _installed[0] = True
    replaced = {}
    for name in _OPS + ("all_gather_single", "reduce_scatter_single"):
        orig = getattr(c10d, name, None) or getattr(dist, name, None)
        if orig is None:
            continue
        w = _wrap(name, orig)
        replaced[id(orig)] = w
        if getattr(c10d, name, None) is orig:
            setattr(c10d, name, w)
        if getattr(dist, name, None) is orig:
            setattr(dist, name, w)
    # Names bound at import time (e.g. distributed.utils.all_gather_single) in loaded sglang modules.
    n = 0
    for mod_name, mod in list(sys.modules.items()):
        if not mod_name.startswith("sglang") or mod is None:
            continue
        for attr, val in list(vars(mod).items()):
            if callable(val) and id(val) in replaced:
                setattr(mod, attr, replaced[id(val)])
                n += 1
    logger.info("[DIAG] collective counters installed (%d module bindings)", n)
