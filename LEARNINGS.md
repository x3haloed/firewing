# Firemwing decision frontier

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

## Prediction errors

These unresolved distinctions can still change the next decision:


When evidence resolves one of these items, update this frontier in place:
replace the affected belief, retain the smallest evidence pointer needed to
justify it, and leave experiment chronology in `experiments/`. Do not append a
new diary entry here.
