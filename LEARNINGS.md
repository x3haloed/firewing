# Firewing decision frontier

This is the smallest current state needed to choose the next useful experiment.
It is not the experiment history. Immutable observations, failed attempts,
reversals, commands, and evidence hashes live in [`experiments/`](experiments/);
machine-readable constants and provenance live in
[`spec/throughput-model.json`](spec/throughput-model.json).

Read a branch conclusion as scoped to its named premises. “Rejected” never
means universally impossible, and a component result never implies endpoint
throughput or fidelity.

## Outcome

Build the full local Qwen3.8-Flash-Next system defined by
[`TARGET.md`](TARGET.md), without crossing [`RED_LINES.md`](RED_LINES.md). The
primary completion gate remains a full-capability, near-equivalent native
runtime passing every target gate, including reproducible batch-one decode at
4 median accepted TPS on the 16 GB M1-centered system.

## Goal invariants

- The qualifying system is the 16 GB Apple M1 Mac mini and its internal SSD.
  Companion compute, memory, storage, and networking hardware are outside the
  completion target.
- The target checkpoint is Qwen/Qwen3.8-Flash-Next. OpenRouter currently serves
  `qwen/qwen3.8-flash`; it cannot become the external whole-model behavioral
  authority unless the independent equivalence experiment in
  `docs/EXPERIMENTS.md` passes. Until then, hosted parity is unresolved rather
  than incomplete.
- The completion threshold is 4 median and 3 p10 accepted TPS after an 8K text
  prefill. The 8/6 result is the stretch target.
- No runtime, checkpoint census, model lock, local endpoint, or accepted-TPS
  measurement exists yet. Prospective budgets are hypotheses, not results.

## Prediction errors

These unresolved distinctions can still change the next decision:

- Whether hosted Qwen3.8-Flash is distributionally and semantically equivalent
  enough to the open Qwen3.8-Flash-Next checkpoint to serve as its behavioral
  reference.
- The exact source tensor census, active executable bytes, n-gram lookup
  traffic, MTP acceptance, expert union, and cold internal-SSD demand.

When evidence resolves one of these items, update this frontier in place:
replace the affected belief, retain the smallest evidence pointer needed to
justify it, and leave experiment chronology in `experiments/`. Do not append a
new diary entry here.
