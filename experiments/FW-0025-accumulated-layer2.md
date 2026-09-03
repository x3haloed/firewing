# FW-0025 - Accumulated linear-attention layer 2

- Status: planned
- Disposition: pending
- Date: 2026-09-03
- Parent experiment: FW-0024
- Exactness: L0 bit-identical component semantics
- Hardware/runtime: Apple M1 Mac mini (`Macmini9,1`), 16 GiB, internal SSD

## Question and hypothesis

Can a reusable arbitrary-layer linear decoder extend FW-0024's exact outputs
through layer 2 across an initial and cached token, using layer 2's own
attention, recurrent state, router, experts, and residual weights?

The hypothesis is that layer 2 will preserve the already established linear
decoder semantics but produce routes and states that discriminate its actual
upstream activation and tensor ownership. This experiment must remove the
remaining layer-0/layer-1 specialization needed before a 48-layer walk.

## Frozen authority and baseline

- Checkpoint revision:
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Model lock SHA-256:
  `f87399e8659ab3274601fcd455b78b73c600f57e5fc1e91499eec3ac1f4b9444`
- Baseline commit: `9f0b85c`
- Framework reference: Transformers 5.16.1
  `Qwen4ExpTextGatedDeltaNet.forward`, `Qwen4ExpTextGatedResidual.forward`,
  `Qwen4ExpTextSparseMoeBlock.forward`, and
  `Qwen4ExpTextDecoderLayer.forward`

## Method and commands

Regenerate FW-0024 exactly and retain its two final layer-1 outputs in process.
Feed those BF16 states into layer 2's attention hyper-connection and Gated
DeltaNet with a fresh layer-2 recurrent cache, then through layer 2's MLP
hyper-connection, dynamic top-10 MoE, shared expert, and final residual. Freeze
all attention states and complete-layer boundaries, selected expert payloads,
weighted outputs, and accumulated layer-2 outputs. Every tensor identity must
carry the layer-2 prefix.

```shell
.venv/bin/python tools/generate_accumulated_layer2_fixture.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  --model-lock spec/model.lock.json \
  --layers01-fixture fixtures/accumulated/qwen3_8_flash_next_layers0_1.json \
  --output fixtures/accumulated/qwen3_8_flash_next_layer2.json

cargo run --release -- verify-accumulated-layer2 \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  fixtures/accumulated/qwen3_8_flash_next_layers0_1.json \
  fixtures/accumulated/qwen3_8_flash_next_layer2.json \
  /Users/chad/Models/firewing/evidence/FW-0025/accumulated-layer2.json
```

The abbreviated commands may resolve already locked FW-0024 component paths
from the committed fixture; final commands will list any additional explicit
authorities required by the verifier. Batch size and concurrency are one.
Accepted tokens, `A`, `U`, and measured TPS are zero because this is an
accumulated correctness fixture.

## Gates

- Fixture: exact checkpoint and FW-0024 identity; deterministic regeneration;
  layer-2 prefix on every tensor; selected expert hashes; no payload committed.
- Correctness: every BF16 and F32 attention/state boundary, MoE capture,
  weighted expert output, and final layer-2 result matches independently.
- State: layer 2 owns a new convolution and recurrent state; it must not alias
  layer 0 or layer 1 state.
- Routing: dynamic top-10 selection is recomputed from the actual accumulated
  activation; only equal-logit internal permutations are semantically neutral.
- Safety: load only bounded ordinary tensors and selected experts; evidence
  remains outside Git.
- Continuation: exact parity unlocks accumulated layer-3 full attention and
  then repeated execution over the remaining schedule.
- Kill/repair: stop at the first upstream-link, ownership, attention, state,
  route, expert, or residual mismatch and preserve it.

Excluded claims: layer 3 or later, full attention, logits, an endpoint, real
prefill, modality processing, latency, and TPS.

## Result

Pending.

## Decision

Pending.
