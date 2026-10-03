"""Process-start repair loading for this isolated experiment's spawn children.

Placed on PYTHONPATH only by run_sglang.py. It is not installed anywhere. The
patch is loaded before torch/numpy thread creation, with its normal build and
single-thread guards intact. The update gate remains closed until ServingSuite
explicitly opens it in a scheduler worker.
"""
import os
import sys

_library = os.environ.get("FOUNDRY_SG_SUITE_PATCH_LIBRARY")
if _library and "--multiprocessing-fork" in sys.argv:
    import ctypes

    if len(os.listdir("/proc/self/task")) != 1:
        raise RuntimeError("Experiment repair must load before worker threads")
    os.environ["QMDREPAIR_ENABLE"] = "1"
    _foundry_repair_library = ctypes.CDLL(_library, mode=ctypes.RTLD_GLOBAL)
    os.environ["FOUNDRY_SG_SUITE_PATCH_LOADED"] = _library
