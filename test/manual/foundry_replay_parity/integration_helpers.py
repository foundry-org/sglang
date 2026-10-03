"""CPU-only utilities retained from the standalone replay protocol."""
import json
import math
from pathlib import Path
import random
import statistics
import subprocess

def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)

def prefill_defaults(target_backend):
    # SGLang 0.5.21 divides chunked_prefill_size by DP size during resolution.
    # DP2 therefore needs 32K global to retain 16K per-rank prefill admission.
    budget = 32768 if target_backend == "deepep_dp" else 16384
    return {"chunked_prefill_size": budget, "max_prefill_tokens": budget}

def required_backend_options(target_backend):
    required = dict(enable_torch_symm_mem=target_backend != "single_gpu",
                    enable_symm_mem=False, enable_mscclpp=False,
                    disable_custom_all_reduce=True,
                    enforce_disable_flashinfer_allreduce_fusion=True)
    if target_backend.startswith("deepep"):
        required.update(moe_a2a_backend="deepep", deepep_mode="auto",
                        moe_runner_backend="deep_gemm", ep_size=2, tp_size=2)
    else:
        required.update(moe_a2a_backend="none", ep_size=1,
                        tp_size=1 if target_backend == "single_gpu" else 2)
    if target_backend == "deepep_dp":
        required.update(dp_size=2, enable_dp_attention=True,
                        enable_dp_lm_head=True, enable_tp_lm_head_all_to_all=False,
                        attn_cp_size=1, moe_dp_size=1)
    elif target_backend == "single_gpu":
        required.update(dp_size=1, enable_dp_attention=False, enable_dp_lm_head=False)
    return required

def resolved_prefill_snapshot(engine):
    """Record normalized values, rather than reconstructing them from inputs."""
    resolved = engine.server_args.resolved_dict()
    required = ("chunked_prefill_size", "max_prefill_tokens", "dp_size",
                "enable_dp_attention", "max_running_requests")
    snapshot = {name: resolved[name] for name in required}
    for name in ("cuda_graph_bs", "tp_size", "ep_size", "enable_dp_lm_head"):
        if name in resolved:
            snapshot[name] = resolved[name]
    return {"source": "Engine.server_args.resolved_dict() after Engine initialization",
            "values": snapshot}

def gpu_state():
    return {
        "gpus": subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu,temperature.gpu,clocks.sm,clocks.mem,power.draw",
            "--format=csv,noheader,nounits"], text=True),
        "apps": subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"], text=True)}

def generation_ids(outputs):
    if isinstance(outputs, dict):
        outputs = [outputs]
    values = []
    for result in outputs:
        if "output_ids" not in result:
            raise RuntimeError(f"skip_tokenizer_init output lacks token IDs: {list(result)}")
        values.append(result["output_ids"])
    return values

def paired_ratios(timings, numerator="updated", denominator="fresh", seed=20260926):
    """Per-block ratios; percentile bootstrap is descriptive, not independence."""
    a = {s["block"]: s["gpu_per_launch_us"] for s in timings[numerator]}
    b = {s["block"]: s["gpu_per_launch_us"] for s in timings[denominator]}
    if set(a) != set(b) or not a:
        raise ValueError("Timing variants must have the same nonempty paired blocks")
    ratios = [a[k] / b[k] for k in sorted(a)]
    if not all(math.isfinite(x) and x > 0 for x in ratios):
        raise ValueError("Nonpositive or nonfinite paired timing")
    rng = random.Random(seed)
    boots = sorted(statistics.median(rng.choices(ratios, k=len(ratios)))
                   for _ in range(2000))
    upper = boots[int(len(boots) * 0.975)]
    return {"numerator": numerator, "denominator": denominator,
            "ratios": ratios, "median_ratio": statistics.median(ratios),
            "median_overhead_percent": (statistics.median(ratios) - 1) * 100,
            "percentile_bootstrap_95ci_ratio": [boots[int(len(boots) * 0.025)], upper],
            "median_within_2_percent": statistics.median(ratios) <= 1.02,
            "bootstrap_upper_within_2_percent": upper <= 1.02,
            "uncertainty_note": "Within-process paired-block descriptive bootstrap; blocks need not be independent. Independent process replications remain required."}
