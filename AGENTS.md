# Project instructions for coding agents

Read `TARGET.md`, `RED_LINES.md`, `LEARNINGS.md`, and `docs/WORKFLOW.md` before
changing runtime, evaluation, checkpoint, or performance code.

## Required discipline

- Preserve the distinction between target-faithful and modified modes in names,
  configuration, artifacts, and reports.
- Do not weaken acceptance thresholds or omit required slices as part of an
  implementation change. A target change must be explicit and isolated.
- Add a correctness fixture before or with every new kernel or model semantic.
- Benchmark end to end. Storage-only, decompression-only, and kernel-only
  numbers are diagnostic, not accepted TPS.
- Record cold and warm state, batch size, concurrency, accepted tokens, `A`,
  `U`, bytes moved, hardware, and commit for every performance claim.
- Never commit model weights, credentials, private fixtures, or raw licensed
  media.
- Keep large generated evidence out of Git; commit schemas, manifests, hashes,
  and small representative fixtures.
- Fail closed on unknown model revisions, layouts, tensor shapes, processor
  configurations, or evidence schemas.
- Prefer a cheap falsification experiment from `docs/EXPERIMENTS.md` before a
  large implementation or purchase.
- Give each experiment a stable `FW-NNNN` record under `experiments/`. Preserve
  rejected and reversed results; do not turn a plan or microbenchmark into a
  reported endpoint result.
- Build semantic changes up the correctness ladder: deterministic tiny fixture,
  slow reference, layer-local real fixture, accelerated parity, whole-model
  parity, then hosted-reference parity.
- Promote a performance default only after a repeatable full-path gain. Use
  interleaved control/candidate runs when cache state, warm-up, or thermals can
  bias order.

## Documentation contract

When an experiment changes a belief:

1. Add the raw-evidence manifest or its content hash.
2. Update `LEARNINGS.md`, marking the old assumption superseded rather than
   erasing its history.
3. Update throughput-model constants and their provenance.
4. State which experiment or architecture branch was promoted or killed.
5. Assign a disposition using `docs/WORKFLOW.md` and preserve links to any
   superseded or reversing evidence.
