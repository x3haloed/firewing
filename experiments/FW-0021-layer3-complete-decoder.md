# FW-0021 - Complete layer-3 full-attention decoder

- Status: planned
- Disposition: unexecuted
- Date: 2026-09-03
- Parent experiments: FW-0017, FW-0020
- Exactness: L0 bit-identical component semantics
- Hardware/runtime: Apple M1 Mac mini (`Macmini9,1`), 16 GiB, internal SSD

## Question and hypothesis

Can the exact layer-3 full-attention residual feed that layer's independently
parameterized MLP hyper-connection, routed and shared experts, and final
four-stream residual with complete native parity?

The hypothesis is that the empty-cache and 2,080-position active-QSA cases
from FW-0020 will expose both wrapper composition and input-dependent routing.
The expert selections must be derived from the actual post-attention states;
layer 0's routes, weights, or fixture outputs are not reusable authority.

## Frozen authority and baseline

- Checkpoint revision:
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Model lock SHA-256:
  `f87399e8659ab3274601fcd455b78b73c600f57e5fc1e91499eec3ac1f4b9444`
- Baseline commit: `a2abfaf`
- Framework reference: Transformers 5.16.1
  `Qwen4ExpTextDecoderLayer.forward`, `Qwen4ExpTextGatedResidual.forward`, and
  the source-derived sparse-MoE equations already validated by FW-0017

## Method and commands

Reproduce FW-0020's two post-attention residuals from the pinned checkpoint.
For each, run layer 3's real MLP hyper-connection, 512-way router with
normalized top-10 selection, only the ten selected expert slices in ascending
expert-ID execution order, and the shared expert with sigmoid gate. Compose
the MLP result into all four residual streams.

Freeze layer-3 MLP tensor identities, both expert-bank descriptors, actual
routes and selected payload hashes, every weighted expert result, all shared
expert boundaries, injection products, and final layer output. Require the
native verifier to call the completed FW-0020 path rather than accepting a
stored post-attention activation.

```shell
.venv/bin/python tools/generate_full_decoder_layer3_fixture.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  --model-lock spec/model.lock.json \
  --full-attention-fixture fixtures/full_attention/qwen3_8_flash_next_layer3.json \
  --attention-residual-fixture fixtures/attention_residual/qwen3_8_flash_next_layer3.json \
  --output fixtures/decoder_layer/qwen3_8_flash_next_layer3.json

cargo run --release -- verify-decoder-layer3 \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  fixtures/full_attention/qwen3_8_flash_next_layer3.json \
  fixtures/attention_residual/qwen3_8_flash_next_layer3.json \
  fixtures/decoder_layer/qwen3_8_flash_next_layer3.json \
  /Users/chad/Models/firewing/evidence/FW-0021/decoder-layer3.json
```

Batch size and concurrency are one. Accepted tokens, `A`, `U`, and measured
TPS are zero because this is a layer-local correctness fixture.

## Gates

- Fixture: exact model/config/index/lock identity; FW-0019 and FW-0020 parent
  hashes; nine MLP tensor identities; two expert-bank descriptors; deterministic
  regeneration; and no expert payload committed.
- Correctness: exact post-attention, hyper-connection, router, top-10 scores,
  each selected weighted expert, routed mixture, shared expert, injection, and
  final layer-output hashes for both cases.
- State: the active-QSA case must retain FW-0020's independent raw-indexer/K/V
  cache semantics; MLP execution must not mutate attention state.
- Safety: read only selected expert slices; remain below the 13 GiB process
  ceiling and do not commit checkpoint-derived payloads.
- Continuation: exact parity establishes complete layer-local semantics for
  both decoder architectures and unlocks accumulated multi-layer execution.
- Kill/repair: fail at the earliest parent, hyper, routing, expert, shared,
  injection, or output boundary mismatch and preserve the discrepancy.

Excluded claims: accumulated layer parity, real prefill, logits, generation,
endpoint behavior, modalities, latency, physical-I/O performance, and TPS.

## Result

Pending execution.

## Decision

Pending. No performance default follows from layer-local correctness.
