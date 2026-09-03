# FW-0021 - Complete layer-3 full-attention decoder

- Status: completed
- Disposition: correctness-repair
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

The reference fixture passes and regenerates byte-identically. It reproduces
FW-0020 from the checkpoint rather than loading a stored activation, then
freezes all 16 MLP/MoE and final-residual captures for each case. The initial
and active-QSA states select disjoint top-10 expert sets, exercising twenty
distinct layer-3 expert slices. Nine ordinary tensors and both full expert-bank
descriptors are bound to the model lock; selected expert payloads are committed
only as hashes. Fixture SHA-256:
`c41e5db1d9cd4678f08e3fdac82f5ce569d4e49ab2e9d4bef050b76466f1f5a9`.
All 41 Python tests pass.

At commit `bd74783`, the release-mode native verifier recomputed the complete
FW-0020 parent path, then exactly matched all 32 layer-local BF16 captures and
all twenty selected weighted-expert hashes. It authenticated 116,102,656 bytes
through the attention residual, 25,666,560 bytes of layer-3 MLP
hyper/router/shared tensors, and 196,608,000 bytes from twenty distinct expert
slices: 338,377,216 logical payload bytes total. Both dynamic routes, ascending
expert execution order, routed mixtures, shared-expert boundaries, four MLP
injection products, and final 10,240-wide layer outputs match.

The final receipt is
`/Users/chad/Models/firewing/evidence/FW-0021/decoder-layer3.json`, SHA-256
`4aee8ec5ed398f15c18a4349f25ee3451744242010fd4c85c555ae3a5b59c760`.
All 41 Python and 32 Rust tests pass, and Clippy passes with warnings denied.
No accepted tokens, physical-I/O timing, or TPS were measured.

## Decision

Pass as a correctness repair. A complete full-attention decoder layer now has
exact native parity under empty-cache and active-QSA conditions, complementing
FW-0017's complete linear-attention layer. Proceed to PLE-bearing layer-1
composition and accumulated multi-layer execution. No performance default or
endpoint claim follows from layer-local parity.
