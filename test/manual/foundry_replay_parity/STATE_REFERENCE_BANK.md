# State-keyed independent SAVE reference protocol

Use `--state-reference-bank NEW_DIRECTORY` in a new SAVE run, then pass the same
bank directory plus `--save-reference NEW_SAVE_REPORT` to independent LOAD runs.
Keep the same model config, seed, capture batches, prompt length, generation
length, backend options, and actual archive. A SAVE bank directory must not exist;
LOAD requires its sealed manifest. Do not combine with diagnostic continuation.
For example, append these arguments to the existing SAVE/LOAD commands:

```
--state-reference-bank /private/experiment/new_bank \
--prompt-tokens 128 --generate-tokens 8 --seed 20261004
```

SAVE records complete logits after each naturally occurring, armed, exact-batch
decode replay, including the existing repeat generation and steps after timing
has completed for the phase. The collector calls no graph launch and makes no
additional generation request. Full tensor copies, hashes, filesystem writes,
and CPU metadata gathers are outside event timing, but can change scheduling and
cache/thermal history; new series must be compared against their own paired
fresh baseline. Old archives/results and default strict/diagnostic paths remain
unchanged.

A row key includes rank, local capture batch, global request count, canonical
request index, complete prompt, consumed generated prefix, actual position,
sequence length, and live input token. Physical request/KV/cache indices and the
whole live global batch are retained as provenance, not assumed identical across
processes. For prompt length P, position P+j consumes generated token j; the
prefix includes tokens 0 through j and this logit's next token is j+1. SAVE
validates the actual next generated token against the natural logit's argmax.
The bank only seals after complete actual/repeat generation equality. A repeated
key with different complete logit bytes fails, including across global contexts.

LOAD assembles complete vocabulary rows from the independently captured SAVE
bank. All ranks agree before selecting a measured phase. Missing rows defer the
probe as a group and run only the ordinary engine replay; they never pass or
mark the phase seen. Invalid inputs/state fail. Deferred ordinary replays can
perform the actual execUpdate; `bank_deferred.jsonl` and each selected phase's
`deferred_natural_replays` retain before/after Foundry states and receipt sequence
numbers for transition audits. If no compatible state occurs, missing phase
coverage fails the run.

Before timing, the selected actual replay is poisoned and its complete logits
must equal the assembled reference bitwise on every rank. The existing strict
within-process updated/fresh comparisons remain unchanged. Prefixes selected
using SAVE outputs are initially **provisional**: acceptance requires the actual
LOAD's complete generated sequence to equal independent SAVE and repeat, then
recomputes each local/global consumed prefix using the actual LOAD generation.
Worker completion alone is insufficient; only final metadata plus this final
prefix verification closes the correctness protocol.

`per_request_prefix_bitwise_agreement` and `global_logical_state_identity` are
separate fields. The former can be true while the latter is false. Row assembly
is explicitly scoped to measured causal Qwen3/Qwen3MoE configurations, fixed
seed/dummy BF16 weights, no EPLB, no redundant experts, trivial expert placement,
no uniform/round-robin routing simulation, and adequate observed DeepEP dispatch
capacity. This is a direct numeric comparison under those conditions, **not** a
proof that arbitrary MoE packing, other global contexts, batch-coupled models,
capacity dropping, or other models are invariant. Full global provenance allows
identical global states to be identified when they occur naturally.

The bank retains `sealed.json` (config/archive/evidence signatures), per-rank
`frames.jsonl` (one record per natural replay), complete `.pt` tensors with file
and row checksums, and `index.json` mapping exact semantic keys to observations.
Checksums detect artifact changes. Failed/unsealed banks and mismatches remain
available for diagnosis; never reinterpret them as accepted evidence.
