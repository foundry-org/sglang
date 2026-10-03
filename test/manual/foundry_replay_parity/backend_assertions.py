"""Observe SGLang communication selection while CUDA graphs are captured.

No hooks run during raw cuGraphLaunch replay. The hooks call their originals
exactly once and preserve results/exceptions. They never read device tensors,
synchronize, or change the selected communication backend. Call the assertion
only after capture, at a rank-consistent CPU control boundary.
"""
from __future__ import annotations

from collections import Counter
import copy
import functools
import hashlib
import importlib
import inspect
from pathlib import Path
import threading


class BackendAssertions:
    def __init__(self, require_symmem: bool, require_deepep: bool, forbid_captured_nccl: bool = False):
        self.require_symmem = bool(require_symmem)
        self.require_deepep = bool(require_deepep)
        self.forbid_captured_nccl = bool(forbid_captured_nccl)
        self.counts = Counter()
        self.details = Counter()
        self.violations = Counter()
        self.sources = {}
        self.instrumented = []
        self.capture_receipts = []
        self._local = threading.local()
        self._torch = None

    def _capturing(self):
        return bool(self._torch.cuda.is_current_stream_capturing())

    @staticmethod
    def _tensor_info(args, kwargs):
        tensor = args[0] if args else kwargs.get("input_", kwargs.get("inp", kwargs.get("x")))
        if tensor is None:
            return "unknown"
        shape = getattr(tensor, "shape", None)
        return f"shape={tuple(shape) if shape is not None else None};dtype={getattr(tensor, 'dtype', None)};device={getattr(tensor, 'device', None)}"

    def _module_source(self, module):
        path = getattr(module, "__file__", None)
        info = {"path": path}
        if path and Path(path).is_file():
            info["sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        self.sources[module.__name__] = info

    def _hook(self, owner, method, make_wrapper, *, required=True):
        original = getattr(owner, method, None)
        if not callable(original):
            if required:
                raise RuntimeError(f"Backend assertion hook missing: {owner.__name__}.{method}")
            return
        if getattr(original, "_foundry_backend_observer", False):
            raise RuntimeError(f"Backend assertion already installed: {owner.__name__}.{method}")
        wrapper = functools.wraps(original)(make_wrapper(original))
        wrapper._foundry_backend_observer = True
        setattr(owner, method, wrapper)
        self.instrumented.append({"owner": owner.__name__, "method": method,
                                  "signature": str(inspect.signature(original))})

    def _operation(self, name, *, forbidden=False):
        def build(original):
            def call(instance, *args, **kwargs):
                captured = self._capturing()
                if captured:
                    self.counts[name + ".attempted"] += 1
                    detail = name + ";" + self._tensor_info(args, kwargs)
                    self.details[detail] += 1
                    if forbidden:
                        self.violations[name + ".captured_fallback"] += 1
                    if name.startswith("deepep.low_latency") and not getattr(instance, "low_latency_mode", False):
                        self.violations[name + ".buffer_not_low_latency"] += 1
                try:
                    result = original(instance, *args, **kwargs)
                except BaseException:
                    if captured:
                        self.counts[name + ".failed"] += 1
                    raise
                if captured:
                    self.counts[name + ".completed"] += 1
                    if name == "symmem.all_reduce":
                        if result is None or getattr(instance, "disabled", True):
                            self.violations["symmem.returned_none_or_disabled"] += 1
                        else:
                            self._local.symmem_completed = getattr(self._local, "symmem_completed", 0) + 1
                            self.details[f"symmem.world_size={instance.world_size};max_size={instance.max_size}"] += 1
                return result
            return call
        return build

    def _group_allreduce(self, original):
        def call(instance, *args, **kwargs):
            captured = self._capturing() and getattr(instance, "world_size", 1) > 1
            before = getattr(self._local, "symmem_completed", 0)
            if captured:
                self.counts["group.all_reduce.attempted"] += 1
                self.details[f"group={getattr(instance, 'unique_name', 'unknown')};" + self._tensor_info(args, kwargs)] += 1
            try:
                result = original(instance, *args, **kwargs)
            except BaseException:
                if captured:
                    self.counts["group.all_reduce.failed"] += 1
                raise
            if captured:
                self.counts["group.all_reduce.completed"] += 1
                used = getattr(self._local, "symmem_completed", 0) - before
                if used == 1:
                    self.counts["group.all_reduce.confirmed_symmem"] += 1
                elif self.require_symmem:
                    self.violations[f"group.all_reduce.symmem_calls={used}"] += 1
            return result
        return call

    def _distributed_operation(self, distributed, method):
        """Observe ProcessGroup fallbacks without changing their dispatch."""
        def build(original):
            signature = inspect.signature(original)
            def call(*args, **kwargs):
                captured = self._capturing()
                name = None
                if captured:
                    try:
                        bound = signature.bind_partial(*args, **kwargs)
                        group = bound.arguments.get("group")
                        backend = str(distributed.get_backend(group)).lower()
                        if "nccl" in backend:
                            backend = "nccl"
                        elif backend != "gloo":
                            self.details[f"torchdist.{method}.unclassified_backend={backend}"] += 1
                            backend = "unknown"
                    except Exception as exc:
                        backend = "unknown"
                        self.details[f"torchdist.{method}.backend_lookup_error={type(exc).__name__}"] += 1
                    # ProcessGroupNCCL may bypass PyNccl. CPU Gloo control is
                    # observed separately, never mislabeled as GPU NCCL work.
                    name = ("nccl" if backend == "nccl" else "torchdist." + backend) + ".torchdist_" + method
                    self.counts[name + ".attempted"] += 1
                    self.details[name + ";" + self._tensor_info(args, kwargs)] += 1
                    if self.forbid_captured_nccl and backend in ("nccl", "unknown"):
                        self.violations[name + ".captured_forbidden_or_unknown_backend"] += 1
                try:
                    result = original(*args, **kwargs)
                except BaseException:
                    if captured:
                        self.counts[name + ".failed"] += 1
                    raise
                if captured:
                    self.counts[name + ".completed"] += 1
                return result
            return call
        return build

    def _capture_one(self, original):
        """Associate capture-only counter deltas with the actual retained graph."""
        def call(instance, *args, **kwargs):
            key = args[0] if args else kwargs.get("shape_key")
            before = (self.counts.copy(), self.details.copy(), self.violations.copy())
            receipt = {"index": len(self.capture_receipts), "shape_key": repr(key),
                       "backend_id": id(instance),
                       "runner": type(getattr(instance, "_cuda_graph_runner", None)).__name__,
                       "status": "started", "raw_graph": None}
            try:
                result = original(instance, *args, **kwargs)
            except BaseException as exc:
                receipt.update(status="capture_failed", error=repr(exc))
                raise
            else:
                try:
                    graph = instance._graphs[key]
                    receipt.update(status="complete", raw_graph=int(graph.raw_cuda_graph()))
                except BaseException as exc:
                    # Observation must not replace a successful original call
                    # with an exception. Reject later at the CPU consensus.
                    receipt.update(status="identity_unavailable", error=repr(exc))
                    self.violations["capture_receipt.identity_unavailable"] += 1
                return result
            finally:
                for name, current, old in zip(("counts", "details", "violations"),
                                               (self.counts, self.details, self.violations), before):
                    receipt[name] = dict(sorted((current - old).items()))
                self.capture_receipts.append(receipt)
        return call

    def snapshot(self):
        return copy.deepcopy({
            "required": {"torch_symmetric_memory": self.require_symmem,
                         "deepep_low_latency": self.require_deepep,
                         "no_captured_nccl": self.forbid_captured_nccl},
            "counts": dict(sorted(self.counts.items())),
            "details": dict(sorted(self.details.items())),
            "violations": dict(sorted(self.violations.items())),
            "sources": self.sources,
            "instrumented": self.instrumented,
            "capture_receipts": self.capture_receipts,
            "scope": "Python calls observed during actual CUDA stream capture, associated with retained raw graph handles by FullCudaGraphBackend.capture_one; no GPU timing hooks. " + (
                "All observed PyNccl and monitored torch.distributed NCCL collectives are forbidden, including all-gather and reduce-scatter; unknown ProcessGroup backends fail closed. Gloo control remains allowed. Native extensions that bypass these Python entrypoints require independent raw kernel inventory corroboration."
                if self.forbid_captured_nccl else
                "NCCL non-allreduce collectives are reported separately and are not asserted absent."),
        })

    def _errors(self, counts, violations):
        errors = []
        if violations:
            errors.append(f"captured backend violations: {dict(violations)}")
        if self.forbid_captured_nccl:
            nccl = {k: v for k, v in counts.items() if k.startswith("nccl.") and k.endswith(".attempted") and v}
            if nccl:
                errors.append(f"NCCL operations captured although forbidden: {nccl}")
        if self.require_symmem:
            if counts.get("symmem.all_reduce.completed", 0) == 0:
                errors.append("no successful torch symmetric-memory all-reduce captured")
            if counts.get("group.all_reduce.confirmed_symmem", 0) == 0:
                errors.append("no SGLang group all-reduce confirmed to use symmetric memory")
        if self.require_deepep:
            dispatch = counts.get("deepep.low_latency_dispatch.completed", 0)
            combine = counts.get("deepep.low_latency_combine.completed", 0)
            if dispatch == 0 or dispatch != combine:
                errors.append(f"DeepEP captured low-latency dispatch/combine counts not positive and balanced: {dispatch}/{combine}")
        failed = {k: v for k, v in counts.items() if k.endswith(".failed") and v}
        if failed:
            errors.append(f"captured communication calls raised: {failed}")
        return errors

    def assert_captured_backend(self):
        if self._capturing():
            raise RuntimeError("Assert backend only after CUDA capture has ended")
        errors = self._errors(self.counts, self.violations)
        for receipt in self.capture_receipts:
            if receipt["status"] != "complete":
                errors.append(f"capture receipt {receipt['index']} did not complete: {receipt['status']}")
            errors.extend(f"graph {receipt['raw_graph']} {receipt['shape_key']}: {error}"
                          for error in self._errors(receipt["counts"], receipt["violations"]))
        if errors:
            raise RuntimeError("; ".join(errors))
        return self.snapshot()

    def assert_graph_backend(self, raw_graph: int):
        """Return evidence for exactly this retained raw CUgraph, outside capture."""
        if self._capturing():
            raise RuntimeError("Assert backend only after CUDA capture has ended")
        receipts = [r for r in self.capture_receipts if r["raw_graph"] == int(raw_graph)]
        if len(receipts) != 1 or receipts[0]["status"] != "complete":
            raise RuntimeError(f"Expected one complete capture receipt for graph {raw_graph}, got {len(receipts)}")
        receipt = receipts[0]
        errors = self._errors(receipt["counts"], receipt["violations"])
        if errors:
            raise RuntimeError("; ".join(errors))
        return copy.deepcopy(receipt)


def _install_on_classes(state, torch, group_class, symmem_class, pynccl_class, deepep_class=None):
    """Dependency injection permits CPU-only protocol tests without CUDA imports."""
    state._torch = torch
    state._hook(group_class, "all_reduce", state._group_allreduce)
    state._hook(symmem_class, "all_reduce", state._operation("symmem.all_reduce"))
    for method in ("all_reduce", "outplace_all_reduce"):
        state._hook(pynccl_class, method, state._operation("nccl." + method,
                    forbidden=state.require_symmem or state.forbid_captured_nccl))
    # These may be required by logits gathering or metadata exchange and are
    # evidence of incidental NCCL, not evidence of the target all-reduce path.
    for method in ("all_gather", "all_gather_into_tensor", "reduce_scatter", "reduce_scatter_tensor",
                   "all_to_all", "all_to_all_single", "broadcast", "send", "recv"):
        state._hook(pynccl_class, method, state._operation("nccl." + method,
                    forbidden=state.forbid_captured_nccl), required=False)
    if deepep_class is not None:
        for method in ("low_latency_dispatch", "low_latency_combine"):
            state._hook(deepep_class, method, state._operation("deepep." + method))
        for method in ("dispatch", "combine"):
            state._hook(deepep_class, method, state._operation("deepep.normal_" + method, forbidden=state.require_deepep))
    return state


def _install_torch_distributed(state, distributed, sglang_group_module):
    """Cover pinned SGLang's ProcessGroup collective fallback entrypoints."""
    for method in ("all_reduce", "all_reduce_coalesced", "all_gather", "all_gather_into_tensor",
                   "all_gather_single", "_all_gather_base", "all_gather_coalesced",
                   "reduce_scatter", "reduce_scatter_tensor", "reduce_scatter_single",
                   "_reduce_scatter_base", "all_to_all", "all_to_all_single", "broadcast",
                   "reduce", "gather", "scatter", "send", "recv", "isend", "irecv", "barrier"):
        state._hook(distributed, method, state._distributed_operation(distributed, method), required=False)
    # SGLang binds these free-function aliases at import time, before the
    # observer installation, so modifying torch.distributed alone misses them.
    for method in ("all_gather_single", "reduce_scatter_single"):
        state._hook(sglang_group_module, method,
                    state._distributed_operation(distributed, method), required=False)


def install_backend_assertions(require_symmem: bool, require_deepep: bool,
                               forbid_captured_nccl: bool = False):
    """Install once in each SGLang worker before model construction/capture."""
    import torch

    group = importlib.import_module("sglang.srt.distributed.parallel_state")
    symmem = importlib.import_module("sglang.srt.distributed.device_communicators.torch_symm_mem")
    pynccl = importlib.import_module("sglang.srt.distributed.device_communicators.pynccl")
    deep = importlib.import_module("deep_ep") if require_deepep else None
    fullgraph = importlib.import_module("sglang.srt.model_executor.runner_backend.full_cuda_graph_backend")
    state = BackendAssertions(require_symmem, require_deepep, forbid_captured_nccl)
    for module in (group, symmem, pynccl, deep, fullgraph):
        if module is not None:
            state._module_source(module)
    if deep is not None:
        state._module_source(importlib.import_module(deep.Buffer.__module__))
    _install_on_classes(state, torch, group.GroupCoordinator,
                        symmem.TorchSymmMemCommunicator, pynccl.PyNcclCommunicator,
                        deep.Buffer if deep is not None else None)
    if forbid_captured_nccl:
        state._module_source(torch.distributed)
        _install_torch_distributed(state, torch.distributed, group)
    state._hook(fullgraph.FullCudaGraphBackend, "capture_one", state._capture_one)
    return state
