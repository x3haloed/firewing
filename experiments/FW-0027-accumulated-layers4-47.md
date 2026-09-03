# FW-0027 - Accumulated decoder layers 4 through 47

- Status: planned
- Disposition: pending
- Date: 2026-09-03
- Parent experiment: FW-0026
- Exactness: L0 bit-identical component semantics
- Hardware/runtime: Apple M1 Mac mini (`Macmini9,1`), 16 GiB, internal SSD

## Question and hypothesis

Can one data-driven source and native walker extend FW-0026's exact outputs
through all remaining decoder layers while preserving every layer's private
cache, tensor ownership, dynamic route, selected expert payload, and residual
boundary?

The hypothesis is that the reusable linear and full-attention paths already
established through layer 3 are sufficient for the checkpoint's periodic
schedule: 36 linear-attention layers and 12 full-attention layers, with full
attention at layers `3, 7, ..., 47`. Repetition must still be verified rather
than inferred because each layer has independent weights, activations, routes,
and cache state.

## Frozen authority and baseline

- Checkpoint revision:
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Model lock SHA-256:
  `f87399e8659ab3274601fcd455b78b73c600f57e5fc1e91499eec3ac1f4b9444`
- Baseline commit: `33e501f`
- Framework reference: Transformers 5.16.1 Qwen4-Exp attention, gated
  residual, sparse-MoE, and decoder-layer semantics

## Method

Regenerate FW-0026 exactly and retain both layer-3 outputs. Iterate layers 4
through 47 according to the pinned `layer_types` array. For each linear layer,
create fresh convolution and recurrent state, execute the two consecutive
tokens, and retain the resulting state only for that layer. For each full
attention layer, create an empty indexer/K/V cache for the initial token and
reuse its exact result at position one. Feed both attention residuals through
that layer's dynamic MoE and final residual before advancing.

The fixture stores one flat hash-only record per layer rather than recursively
embedding prior layers. It freezes input/output links, attention and cache
boundaries, selected routes, selected expert payload hashes, weighted outputs,
and per-layer byte ledgers. The verifier must replay the chain from FW-0026;
it may not trust fixture-supplied intermediate activations or routes.

Planned commands:

```shell
.venv/bin/python tools/generate_accumulated_layers4_47_fixture.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  --model-lock spec/model.lock.json \
  --layer3-fixture fixtures/accumulated/qwen3_8_flash_next_layer3.json \
  --output fixtures/accumulated/qwen3_8_flash_next_layers4_47.json

cargo run --release -- verify-accumulated-layers4-47 \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  fixtures/accumulated/qwen3_8_flash_next_layer3.json \
  fixtures/accumulated/qwen3_8_flash_next_layers4_47.json \
  /Users/chad/Models/firewing/evidence/FW-0027/accumulated-layers4-47.json
```

Final commands may enumerate the committed FW-0026 component authorities
explicitly. Batch size and concurrency are one. Accepted tokens, `A`, `U`, and
measured TPS are zero because this is accumulated correctness evidence.

## Gates

- Identity: exact model/config/index/lock and exact 48-entry layer schedule;
  all tensor names carry their current layer prefix.
- Correctness: every input link, attention/cache boundary, dynamic route,
  expert output, MoE boundary, and final layer output matches independently.
- State: every linear layer owns distinct convolution/recurrent state; every
  full-attention layer owns distinct indexer/K/V state across the two tokens.
- Routing: recompute all 880 remaining expert selections from actual upstream
  activations; authenticate every selected slice and weighted output.
- Representation: commit only hashes, tensor metadata, and a bounded fixture;
  keep weights and raw evidence outside Git.
- Safety: process one layer at a time and release layer-scoped tensors before
  advancing; no whole-checkpoint mapping or unbounded residency.
- Continuation: exact parity unlocks token embedding, final normalization,
  LM-head/logit semantics, and a slow complete text endpoint.
- Kill/repair: stop at the earliest layer and boundary mismatch and preserve
  the exact discrepancy; do not weaken later comparisons to hide drift.

Excluded claims: embedding, final normalization, logits, MTP, a text endpoint,
real prefill, modality processing, latency, and TPS.

## Result

Pending.

## Decision

Pending.
