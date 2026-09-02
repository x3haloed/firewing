# Research and delivery workflow

Firewing combines two observable development patterns:

- Swiftlet's **reference-first vertical slice** gets to a slow, correct endpoint
  before optimizing it.
- TurboFieldfare's **measured optimization loop** promotes only repeatable
  end-to-end gains and preserves failed and reversed experiments.

The public Git histories do not reveal every iteration. TurboFieldfare's core
research appears largely consolidated before its first public commit, and
Swiftlet's implementation arrived in one large commit. Their plans, fixtures,
tests, experiment inventory, and optimization journal are better process
evidence than commit archaeology.

## 1. Establish the answer keys

Before runtime construction:

1. Pin the Qwen checkpoint, tokenizer, processor, published official
   implementation semantics, and hosted-reference corpus.
2. Build source-derived deterministic component oracles and select an
   independent executable architecture reference where possible. A
   whole-model official-framework run is not assumed available; every report
   preserves that evidence limitation. Independent implementations help expose
   mistaken assumptions shared with the published code.
3. Maintain a source ledger with exact repository commits, files, papers,
   kernels, and which decision each source informed—including rejected work.
4. Write an explicit reuse map: code or patterns safe to reuse, code requiring
   semantic adaptation, and model-specific code that must not be ported.
5. Make a capacity, traffic, compute, and latency budget before implementation.
   Treat every number in that budget as a hypothesis until measured.

Swiftlet's `PLAN.md` follows this pattern: it names multiple answer keys,
specifies what to copy and what not to port, budgets memory up front, and makes
the correctness harness non-negotiable.

## 2. Climb the correctness ladder

Build the smallest falsifiable artifact at each level. Do not begin the next
level while an unexplained mismatch remains below it.

1. **Tiny deterministic fixtures.** Seeded synthetic tensors and tokens for
   every primitive, expert equation, attention variant, router, KV update, MTP
   path, and modality projector.
2. **Slow scalar or CPU oracle.** A readable implementation of a complete
   forward pass with intermediate capture.
3. **Layer-local real fixtures.** Feed each layer the reference input rather
   than the previous local layer's output. This localizes semantic errors that
   accumulated floating-point drift can hide.
4. **Accelerated primitive parity.** Compare every Metal, ANE, CPU-vector, I/O,
   and codec path against its direct reference using production shapes,
   offsets, alignment, metadata, and state transitions.
5. **Whole-model parity.** Test end-to-end drift, greedy token equality,
   incremental versus whole-sequence evaluation, and teacher-forced logprobs.
6. **Native modality parity.** Add image, multi-image, video, and mixed
   inputs through their real processors and encoders.
7. **Hosted-reference parity.** Run the frozen protocol in
   `docs/VALIDATION_PROTOCOL.md`.

Swiftlet's fixture generator and tests embody this ladder: deterministic tiny
models, per-layer captures, CPU references, Metal-versus-scalar kernels, and
separate local and accumulated tolerances. Its plan deliberately puts a CPU
endpoint before Metal optimization.

## 3. Deliver endpoints in narrow milestones

An endpoint is a complete, inspectable path, not a throughput promise.

| Milestone | Required endpoint |
| --- | --- |
| M0 | Checkpoint census, pinned references, tiny fixtures, tokenizer/processor |
| M1 | Verifiable repacker/container and lossless tensor round trip |
| M2 | Slow complete text forward pass and incremental decode |
| M3 | Minimum accelerated text runtime with bounded expert streaming |
| M4 | Native image, video, and mixed-modality paths |
| M5 | Hosted parity and reproducible local baseline |
| M6 | Evidence-backed performance passes toward Firewing 1/2/4 |

Use the simplest correct implementation first. A serial prefill, scalar
reduction, or conservative synchronization path is acceptable as an oracle.
Optimize it only after the complete endpoint identifies the actual bottleneck.

## 4. Run the optimization loop

Every performance change follows the same loop:

1. Profile a clean, complete request and identify the largest measured share.
2. Open a stable experiment record with a hypothesis, mechanism, exactness
   class, success threshold, kill threshold, and expected system-level effect.
3. Reproduce the bottleneck in isolation using real matrix sizes, layouts,
   offsets, memory limits, synchronization, and cold/warm states.
4. Make one focused change.
5. Apply the correctness gate appropriate to the claim.
6. Measure isolated attribution, then return to the clean endpoint.
7. Interleave control and candidate runs when warm-up, page cache, memory
   pressure, or thermals can bias order.
8. Test short, medium, long, multimodal, and holdout workloads as relevant.
9. Assign a disposition. The default changes only after a repeatable full-path
   gain; an inconclusive result leaves it unchanged.
10. Update the evidence ledger, throughput model, and affected beliefs.

This is the central TurboFieldfare lesson. Its journal records attractive
microbenchmarks that lost end to end, first-run gains that disappeared under
interleaving, policies that failed on holdouts, and optimizations later
reversed. The full token path and the appropriate correctness gate made the
decision.

## 5. Match the correctness gate to the claim

| Change class | Minimum gate |
| --- | --- |
| Claimed lossless storage, caching, loading, scheduling, or layout | Exact bytes and exact outputs |
| Algebraically exact but reordered floating-point execution | Reference deltas, top-k stability, logprob/KL limits, and endpoint quality |
| Quantization or other acknowledged approximation | Full near-equivalence suite and slice-specific regression limits |
| Draft-only approximation under exact verification | Statistical target-distribution tests plus accepted-output parity |
| Modified weights, routing, topology, or expert count | Explicit `modified` mode and all gates in `TARGET.md`; never describe it as stock MiMo |

Exact token identity is not the universal oracle: floating-point near-ties can
change a greedy token without a material distribution change. Conversely, a
storage optimization claiming identical mathematics has no reason to relax to
a distributional test.

## 6. Keep an append-only experiment ledger

Create one Markdown record under `experiments/` for every executed or seriously
considered experiment. IDs are sequential (`FW-0001`, `FW-0002`, …) and never
reused. Preserve negative results and later reversals.

Allowed dispositions:

- `production` — repeatable gain, correctness passed, default enabled.
- `conditional` — useful only under named hardware/workload conditions.
- `rejected` — tested and not promoted.
- `correctness-repair` — fixes semantics rather than performance.
- `reversed` — an earlier decision changed after stronger evidence.
- `scope-decision` — intentionally outside the active architecture.
- `unexecuted` — documented hypothesis, not evidence.

The record must contain:

- hypothesis and causal mechanism;
- exactness class and red-line check;
- baseline, candidate, commands, commits, checkpoint hashes, and environment;
- production-shaped fixture and workload slices;
- isolated measurements and complete endpoint measurements;
- correctness results and raw-evidence manifest/hash;
- short/medium/long/holdout behavior;
- decision, confidence, limitations, reusable lesson, and follow-up;
- links to any superseded or reversing record.

`docs/EXPERIMENTS.md` is the prospective program. `experiments/` is the
historical record of what actually happened. Plans must not silently become
results.

## 7. Measurement hygiene

- Use release builds and record the exact commit, dirty state, OS, compiler,
  device, RAM, storage, checkpoint hash, command, exit code, and protocol
  deviations.
- Run one full model process at a time on the 16 GB machine. Record memory
  pressure and conflicting processes. Run shared-Metal-state tests serially.
- Discard a declared number of warm-up runs; do not quietly discard outliers.
- Distinguish cold storage, warm OS cache, and warm application cache.
- Report accepted TPS from complete wall time. Profiler TPS, kernel bandwidth,
  allocation count, and dispatch count explain mechanisms but cannot promote a
  change.
- Freeze prompts, seeds, sampling settings, stop conditions, and generated
  token counts. Compare rows only when the protocol says they are comparable.
- Inspect generated outputs for repetition loops, premature stops, and modality
  failures; a fast broken decode is not a benchmark result.
- Train cache, layout, router, or compression policies on one trace set and
  require holdout traces before promotion.

## 8. Reversal is part of the method

Do not erase a rejected idea when a new implementation makes it viable. Add a
new record, link the old evidence, explain which premise changed, and mark the
old disposition superseded. A living project should preserve how it learned,
not manufacture a clean hindsight narrative.

Planning estimates are also revisable. Swiftlet's plan anticipated much higher
throughput than its first published endpoint achieved; the working endpoint
correctly replaced the estimate. Firewing budgets are decision aids, never
reported measurements.

## 9. Primary workflow evidence

TurboFieldfare:

- [Optimization journey](https://github.com/drumih/turbo-fieldfare/blob/main/docs/OPTIMIZATION_JOURNEY.md)
- [Experiment inventory](https://github.com/drumih/turbo-fieldfare/blob/main/docs/experiments/EXPERIMENT_INVENTORY.md)
- [Validation and measurement lessons](https://github.com/drumih/turbo-fieldfare/blob/main/docs/experiments/summaries/09-validation-and-measurement-lessons.md)
- [Community benchmark protocol](https://github.com/drumih/turbo-fieldfare/blob/main/docs/COMMUNITY_BENCHMARKS.md)
- [Contributor protocol](https://github.com/drumih/turbo-fieldfare/blob/main/CONTRIBUTING.md)
- [Agent/test-environment rules](https://github.com/drumih/turbo-fieldfare/blob/main/AGENTS.md)
- [Pinned implementation references](https://github.com/drumih/turbo-fieldfare/blob/main/docs/IMPLEMENTATION_REFERENCES.md)

Swiftlet:

- [Implementation plan and milestone ladder](https://github.com/leonickson1/Swiftlet/blob/main/PLAN.md)
- [Deterministic fixture generator](https://github.com/leonickson1/Swiftlet/blob/main/scripts/gen_fixtures.py)
- [Layer-local and whole-model fixture tests](https://github.com/leonickson1/Swiftlet/blob/main/Tests/FixtureForwardTests.swift)
- [Metal-versus-CPU kernel tests](https://github.com/leonickson1/Swiftlet/blob/main/Tests/MetalKernelTests.swift)
- [Correctness contract](https://github.com/leonickson1/Swiftlet/blob/main/README.md#correctness)
