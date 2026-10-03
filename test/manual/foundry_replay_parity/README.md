# Actual Foundry SAVE/LOAD replay experiment

This manual test runs the current SGLang Foundry adapter. SAVE executes its normal capture/archive path; a separate LOAD process reconstructs those archived graphs, including the actual manifest's templates and on-demand members. LOAD invokes the opt-in Foundry research bridge through the ordinary backend graph replay entrypoint.

It requires the companion Foundry research branch with `foundry.research_qmd` and its C++ research bindings. The standard adapter needs no runtime modification. The two checkouts, all dependencies, model configs, caches and the prefetch-repair library must remain in an isolated workspace/venv. Only the exact supported driver build may enable the repair library.

## What this test checks

- Every SAVE capture batch receives a deterministic prompt set and a full first-decode logits reference. Prompts depend on the batch, not the order of the test phases.
- LOAD with `--save-reference` compares all logits and greedy generation tokens against that independent SAVE process. Outputs must match bitwise; no tolerance silently changes the criterion.
- Each measured LOAD phase calls the real backend replay first, records the bridge state before and after it, and checks that a changed member has an actual update receipt. Same-member replays are explicitly distinguishable from switches.
- A never-updated fresh exec is instantiated from the current target graph reconstructed by Foundry. The actual persistent candidate and fresh exec use the same instantiate flags, current inputs and output addresses. NaN poisoning and complete logits comparisons surround timing.
- GPU event timing uses balanced paired blocks and reports the slowest rank for each variant/block. The Python transaction, fresh instantiation, upload, validation and barriers are outside the timing interval.
- SAVE records per-shape communication calls, including symmetric memory / DeepEP low-latency dispatch and combine, and rejects observed captured NCCL. LOAD uses that saved archive. DP2 probes require a common CPU admission decision before any extra graph launches.
- `--manifest-sequence` reads the real SAVE manifest and walks every directed pair within every group. It does not assume adjacent batch sizes share a template.

## Example

Paths below are placeholders. Use an external bounded launcher, a shared GPU lock, and an occupancy check before either process. Never run competing GPU jobs.

```bash
export PYTHONPATH="$TEST/startup:$TEST:$SGLANG/python:$FOUNDRY/python"

# SAVE: actual archives and independent output references for all eight batches.
python "$TEST/run_foundry_integration.py" \
  --mode save --output "$SAVE_REPORT" --archive "$ARCHIVE" \
  --model "$VENV/models/Qwen--Qwen3-8B" --target-backend single_gpu \
  --gpu-uuids "$GPU_UUID" --capture-batches 2,4,8,16,30,31,32,64 \
  --blocks 3 --launches 8

# LOAD: patch is installed only at scheduler process start, before threads.
export FOUNDRY_SG_SUITE_PATCH_LIBRARY="$EXACT_BUILD_REPAIR_SO"
export QMDREPAIR_HELPER_ROLES=1 QMDREPAIR_V3_WRITE=1
export FOUNDRY_QMD_REPAIR=1 FOUNDRY_QMD_RESEARCH_MODE=repair
export FOUNDRY_QMD_RECEIPT_DIR="$LOAD_REPORT/bridge_receipts"
python "$TEST/run_foundry_integration.py" \
  --mode load --output "$LOAD_REPORT" --archive "$ARCHIVE" \
  --save-reference "$SAVE_REPORT" --manifest-sequence \
  --model "$VENV/models/Qwen--Qwen3-8B" --target-backend single_gpu \
  --gpu-uuids "$GPU_UUID" --capture-batches 2,4,8,16,30,31,32,64 \
  --blocks 11 --launches 32
```

`FOUNDRY_QMD_RESEARCH_MODE=unpatched` selects the same actual LOAD/execUpdate path without field repair. Do not preload the repair SO for this control. `--target-backend symmem` selects TP2 Qwen3-8B; `deepep_dp` selects EP2/DP2 and requires an appropriate MoE model such as Qwen3-30B-A3B and two GPU UUIDs. Each configuration needs its own matching SAVE archive and reference directory.

CPU-only tests:

```bash
python -m unittest discover -s "$TEST" -p 'test_*.py' -v
```

## Scope

This is a bounded synchronous correctness/replay experiment. It does not prove scheduler overlap, optimize host-side validation, or establish performance with real weights or arbitrary driver builds. A failed graph guard or numerical check is fatal: the worker exits without attempting GPU cleanup. The outer launcher must bound and reap the complete experiment process tree.

SAVE/capture timing and LOAD timing occur in separate processes, so they are context rather than a paired speed ratio. The primary performance comparison is the same-process updated-vs-fresh pair; the independent SAVE output comparison prevents two equally incorrect LOAD variants from passing.

All ranks reach a CPU admission decision before any early return or extra launch. At each selected phase the ordinary backend replay is NaN-poisoned before launch and checked against the independent SAVE tensor before timing. The subsequent bounded candidate and fresh validation launches are also NaN-poisoned before and after timing.

The preserved standalone protocol supplied `backend_assertions.py`, `dp_probe.py`, the bootstrap loader and CPU helpers. These local copies keep the manual test reproducible without requiring a separate experiment directory on `PYTHONPATH`.

## Explicit mismatch diagnostic

`--diagnostic-save-mismatch` preserves full actual/SAVE CPU logits, row-level differences and top-two margins, exact row-permutation evidence, and the live ForwardBatch / captured input-buffer values before replay. Requests receive deterministic IDs so live rows can be associated with submitted prompts. This helps distinguish a different selected decode position or request order from an execUpdate error.

Only in this explicit mode may the run continue past a cross-process SAVE mismatch to collect a fresh-LOAD comparison and generated-token evidence. Candidate/fresh poison validation remains strictly bitwise. Cross-process and repeated-generation mismatches remain recorded, and the run always ends with `diagnostic_complete_not_accepted`, never `passed`; no performance acceptance is inferred. The default strict behavior is unchanged.
