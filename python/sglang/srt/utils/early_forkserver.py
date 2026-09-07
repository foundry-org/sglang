"""Experimental: start the multiprocessing forkserver early, with the worker
modules preloaded, so worker processes fork from an already-imported server
instead of re-importing sglang/torch (7-9 s each) under ``spawn``.

Enabled by ``SGLANG_EARLY_FORKSERVER=1``. Called from the CLI entry before the
launcher's own heavy imports, so the server's import overlaps the launcher's.

This module is also in the forkserver preload list: importing it inside the
forkserver installs the child-side env patch (forkserver children inherit the
server's environment, not the launcher's, so the launcher snapshots
``os.environ`` at ``Process.start`` and the child re-applies it in ``run``).
"""

import multiprocessing as mp
import multiprocessing.process as mpp
import os

PRELOAD = [
    "sglang.srt.utils.early_forkserver",
    "sglang.srt.managers.scheduler",
    "sglang.srt.managers.detokenizer_manager",
    "sglang.srt.managers.data_parallel_controller",
]

_ENV_ATTR = "_sglang_env_snapshot"
_STDIO_ATTR = "_sglang_stdio"


class _FdHandle:
    """Pickles as a duplicated fd via multiprocessing.reduction (sent over the
    forkserver socket), so the child can dup2() the launcher's stdout/stderr."""

    def __init__(self, fd):
        self.fd = fd
        self._dup = None

    def __reduce__(self):
        from multiprocessing import reduction

        return (_FdHandle._rebuild, (reduction.DupFd(self.fd),))

    @staticmethod
    def _rebuild(dup):
        h = _FdHandle(None)
        h._dup = dup
        return h

    def detach(self):
        return self._dup.detach() if self._dup is not None else None


def _install_process_patches():
    if getattr(mpp.BaseProcess, "_sglang_env_patched", False):
        return
    orig_start = mpp.BaseProcess.start
    orig_run = mpp.BaseProcess.run

    def start(self):
        setattr(self, _ENV_ATTR, dict(os.environ))
        # Forkserver children inherit the *server's* stdio. With a shared daemon
        # that is the daemon's log, so hand the child our own stdout/stderr.
        try:
            setattr(self, _STDIO_ATTR, (_FdHandle(1), _FdHandle(2)))
        except Exception:
            pass
        return orig_start(self)

    def run(self):
        env = getattr(self, _ENV_ATTR, None)
        if env:
            os.environ.update(env)
        stdio = getattr(self, _STDIO_ATTR, None)
        if stdio:
            try:
                for target, handle in zip((1, 2), stdio):
                    fd = handle.detach()
                    if fd is not None:
                        os.dup2(fd, target)
                        os.close(fd)
            except Exception:
                pass
        _restore_torch_cuda()
        _reuse_forkserver()
        return orig_run(self)

    mpp.BaseProcess.start = start
    mpp.BaseProcess.run = run
    mpp.BaseProcess._sglang_env_patched = True


def _reuse_forkserver():
    """Child side: point multiprocessing at the launcher's forkserver instead of
    starting a nested one. A child forked by the server already holds a valid
    alive fd (handed over by the server), only the address/pid are missing.
    ensure_running() would waitpid() the server, which is our parent, so it is
    disabled on this instance."""
    addr = os.environ.get("SGLANG_FORKSERVER_ADDRESS")
    if not addr:
        return
    import multiprocessing.forkserver as fs

    inst = fs._forkserver
    # forkserver.main() already stores the listener address in children it
    # forks, but not the pid, and ensure_running() keys on the pid.
    if inst._forkserver_pid is None:
        inst._forkserver_address = addr
        inst._forkserver_pid = int(os.environ.get("SGLANG_FORKSERVER_PID", "0")) or None
        inst.ensure_running = lambda: None


def enabled() -> bool:
    return os.environ.get("SGLANG_EARLY_FORKSERVER") == "1"


DEFAULT_DAEMON_FILE = "/tmp/sglang_forkserver.json"


def _try_attach_daemon(fs) -> bool:
    """Reuse a forkserver started by `python -m sglang.srt.utils.early_forkserver`
    (SGLANG_FORKSERVER_FILE, default /tmp/sglang_forkserver.json). The server has
    already imported the worker stack, so workers fork immediately instead of
    waiting ~7 s for a fresh preload. The launcher keeps one end of a pipe as the
    'alive fd' the protocol hands to each forked child."""
    import json

    path = os.environ.get("SGLANG_FORKSERVER_FILE", DEFAULT_DAEMON_FILE)
    if not os.path.exists(path):
        return False
    try:
        info = json.load(open(path))
        addr, pid = info["address"], int(info["pid"])
        for check_pid in (pid, int(info.get("daemon_pid", pid))):
            os.kill(check_pid, 0)  # alive?
            with open(f"/proc/{check_pid}/status") as st:
                if any(l.startswith("State:") and "Z" in l for l in st):
                    raise ProcessLookupError(f"pid {check_pid} is a zombie")
        # Do NOT probe-connect: the forkserver main loop treats a connection
        # that sends nothing as a fatal EOF and exits.
        if not os.path.exists(addr):
            raise FileNotFoundError(addr)
    except Exception as e:  # stale file, dead daemon, permission
        logger_warning = getattr(__import__("logging").getLogger(__name__), "warning")
        logger_warning(
            "[early_forkserver] daemon at %s unusable (%s); starting a private one",
            path,
            e,
        )
        return False
    inst = fs._forkserver
    inst._forkserver_address = addr
    inst._forkserver_pid = pid
    r, w = os.pipe()
    inst._forkserver_alive_fd = w
    inst._keep_r = r  # never closed on purpose
    inst.ensure_running = lambda: None
    os.environ["SGLANG_FORKSERVER_ADDRESS"] = addr
    os.environ["SGLANG_FORKSERVER_PID"] = str(pid)
    return True


def start_early():
    """Parent side: call once at CLI entry, before heavy imports."""
    if not enabled():
        return
    import multiprocessing.forkserver as fs

    # The launcher-side engine code re-applies the start method from this env
    # var; the numactl spawn wrapper asserts 'spawn', so it is disabled here.
    os.environ["SGLANG_MP_START_METHOD"] = "forkserver"
    os.environ.setdefault("SGLANG_NUMA_BIND_V2", "0")
    # torch.cuda.is_available() goes through cudaGetDeviceCount and marks the
    # process as unsafe to fork (poison_fork) even without a context; the
    # NVML-based check does not. sglang calls is_available() at import time.
    os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
    mp.set_start_method("forkserver", force=True)
    mp.set_forkserver_preload(PRELOAD)
    _install_process_patches()
    if _try_attach_daemon(fs):
        return
    fs.ensure_running()
    # Children forked from the server (e.g. the DP controller) would otherwise
    # start a *second* forkserver with the default preload when they call
    # Process.start(); publish this one so they reuse it (see _reuse_forkserver).
    os.environ["SGLANG_FORKSERVER_ADDRESS"] = fs._forkserver._forkserver_address
    os.environ["SGLANG_FORKSERVER_PID"] = str(fs._forkserver._forkserver_pid)


_TORCH_CUDA_ORIG = {}


def _nvml_cc(index: int = 0):
    import pynvml
    import torch

    idx = index
    getter = getattr(torch.cuda, "_get_nvml_device_index", None)
    if getter is not None:
        try:
            idx = getter(index)
        except Exception:
            idx = index
    pynvml.nvmlInit()
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(idx)
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(h)
        name = pynvml.nvmlDeviceGetName(h)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h).total
        try:
            cores = pynvml.nvmlDeviceGetNumGpuCores(h)
        except Exception:
            cores = 0
    finally:
        pynvml.nvmlShutdown()
    return major, minor, name, mem, cores


def _install_import_time_cuda_shim():
    """Forkserver side: third-party kernel packages (sgl_kernel, flashinfer)
    probe the GPU at import via torch.cuda.get_device_capability /
    get_device_properties / current_device, which initializes a CUDA context
    and makes fork() unusable. Answer those probes from NVML instead while the
    preload imports run; children restore the real functions in run()."""
    import types

    import torch

    if _TORCH_CUDA_ORIG:
        return
    for name in ("get_device_capability", "get_device_properties", "current_device"):
        _TORCH_CUDA_ORIG[name] = getattr(torch.cuda, name)

    def get_device_capability(device=None):
        if torch.cuda.is_initialized():
            return _TORCH_CUDA_ORIG["get_device_capability"](device)
        idx = device if isinstance(device, int) else 0
        major, minor, *_ = _nvml_cc(idx)
        return (major, minor)

    def get_device_properties(device=None):
        if torch.cuda.is_initialized():
            return _TORCH_CUDA_ORIG["get_device_properties"](device)
        idx = device if isinstance(device, int) else 0
        major, minor, name, mem, cores = _nvml_cc(idx)
        return types.SimpleNamespace(
            major=major,
            minor=minor,
            name=name,
            total_memory=mem,
            # 128 FP32 cores per SM on Hopper/Ada/Blackwell; import-time users
            # only size grids with it.
            multi_processor_count=(cores // 128) if cores else 0,
        )

    def current_device():
        if torch.cuda.is_initialized():
            return _TORCH_CUDA_ORIG["current_device"]()
        return 0

    torch.cuda.get_device_capability = get_device_capability
    torch.cuda.get_device_properties = get_device_properties
    torch.cuda.current_device = current_device


def _restore_torch_cuda():
    if not _TORCH_CUDA_ORIG:
        return
    import torch

    for name, fn in _TORCH_CUDA_ORIG.items():
        setattr(torch.cuda, name, fn)
    _TORCH_CUDA_ORIG.clear()


# Forkserver / child side (module preloaded): make nested Process() calls (the
# DP controller launching schedulers) reuse the same forkserver and carry env.
if enabled():
    try:
        mp.set_start_method("forkserver", force=True)
    except RuntimeError:
        pass
    _install_process_patches()
    mp.set_forkserver_preload(PRELOAD)  # fallback if a nested server is ever started
    os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
    # flashinfer enumerates device capabilities at import unless told the arch.
    os.environ.setdefault(
        "FLASHINFER_CUDA_ARCH_LIST", os.environ.get("SGLANG_FLASHINFER_ARCH", "9.0a")
    )
    _install_import_time_cuda_shim()


def run_daemon():
    """`python -m sglang.srt.utils.early_forkserver`: start a preloaded forkserver
    that outlives individual `sglang serve` launches and publish its address in
    SGLANG_FORKSERVER_FILE (default /tmp/sglang_forkserver.json). Must be
    started from the same environment/code as the servers that will use it."""
    import json
    import multiprocessing.forkserver as fs
    import signal
    import time

    os.environ["SGLANG_EARLY_FORKSERVER"] = "1"
    os.environ["SGLANG_MP_START_METHOD"] = "forkserver"
    os.environ.setdefault("SGLANG_NUMA_BIND_V2", "0")
    os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
    mp.set_start_method("forkserver", force=True)
    mp.set_forkserver_preload(PRELOAD)
    _install_process_patches()
    t0 = time.time()
    fs.ensure_running()
    # Force the preload now (the server imports lazily on first connect otherwise).
    p = mp.Process(target=_noop)
    p.start()
    p.join()
    inst = fs._forkserver
    path = os.environ.get("SGLANG_FORKSERVER_FILE", DEFAULT_DAEMON_FILE)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(
            {
                "address": inst._forkserver_address,
                "pid": inst._forkserver_pid,
                "daemon_pid": os.getpid(),
            },
            f,
        )
    os.replace(tmp, path)
    print(
        f"[early_forkserver] daemon ready in {time.time() - t0:.1f} s: "
        f"address={inst._forkserver_address} pid={inst._forkserver_pid} file={path}",
        flush=True,
    )

    def _cleanup(*_):
        try:
            os.unlink(path)
        except OSError:
            pass
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _cleanup)
    signal.signal(signal.SIGINT, _cleanup)
    while True:
        time.sleep(3600)


def _noop():
    pass


if __name__ == "__main__":
    run_daemon()
