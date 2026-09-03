# FW-0024 - Accumulated layers 0 through 1

- Status: planned
- Disposition: pending
- Date: 2026-09-03
- Parent experiments: FW-0017, FW-0023
- Exactness: L0 bit-identical component semantics
- Hardware/runtime: Apple M1 Mac mini (`Macmini9,1`), 16 GiB, internal SSD

## Question and hypothesis

Can the exact native layer-0 decoder output feed the complete PLE-bearing
layer-1 decoder across an initial and cached token without relying on a
layer-local reference input?

The hypothesis is that accumulating two real layers is sufficient to expose
hidden assumptions in PLE query construction, cache ownership, route identity,
or BF16 boundary staging that layer-local fixtures cannot detect. Layer 1 must
receive layer 0's actual 10,240-wide output for the same token, not FW-0023's
standalone deterministic input.

## Frozen authority and baseline

- Checkpoint revision:
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Model lock SHA-256:
  `f87399e8659ab3274601fcd455b78b73c600f57e5fc1e91499eec3ac1f4b9444`
- Baseline commit: `884f460`
- Framework reference: Transformers 5.16.1
  `Qwen4ExpTextDecoderLayer.forward` and its layer-0/layer-1 submodules

## Method and commands

For token 42 and then cached token 43, begin with FW-0017's deterministic
four-stream layer-0 input and execute the complete real layer-0 decoder with
persistent DeltaNet state. Feed its exact output directly into layer 1. Perform
the sparse PLE lookup for the same token and persistent n-gram context, add the
PLE output at BF16, and execute layer 1's complete attention and MLP/MoE paths
with independent recurrent states. Freeze each layer output, both layers'
stateful boundaries, the accumulated layer-1 PLE output, dynamic routes,
selected expert hashes, and final two-layer output.

```shell
.venv/bin/python tools/generate_accumulated_layers01_fixture.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  --model-lock spec/model.lock.json \
  --ngram-fixture fixtures/ngram/qwen3_8_flash_next.json \
  --ngram-row-fixture fixtures/ngram/qwen3_8_flash_next_row_hashes.json \
  --layer0-attention-fixture fixtures/attention_residual/qwen3_8_flash_next_layer0.json \
  --layer0-sparse-moe-fixture fixtures/sparse_moe/qwen3_8_flash_next_layer0.json \
  --layer0-fixture fixtures/decoder_layer/qwen3_8_flash_next_layer0.json \
  --ple-fixture fixtures/ple/qwen3_8_flash_next_layer1_decode.json \
  --layer1-attention-fixture fixtures/attention_residual/qwen3_8_flash_next_layer1_ple.json \
  --layer1-fixture fixtures/decoder_layer/qwen3_8_flash_next_layer1_ple.json \
  --output fixtures/accumulated/qwen3_8_flash_next_layers0_1.json

cargo run --release -- verify-accumulated-layers01 \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  fixtures/ngram/qwen3_8_flash_next.json \
  fixtures/ngram/qwen3_8_flash_next_row_hashes.json \
  fixtures/decoder_layer/qwen3_8_flash_next_layer0.json \
  fixtures/decoder_layer/qwen3_8_flash_next_layer1_ple.json \
  fixtures/accumulated/qwen3_8_flash_next_layers0_1.json \
  /Users/chad/Models/firewing/evidence/FW-0024/accumulated-layers01.json
```

Batch size and concurrency are one. Accepted tokens, `A`, `U`, and measured
TPS are zero because this is an accumulated correctness fixture.

## Gates

- Fixture: exact model/config/index/lock identity; both parent fixture hashes;
  deterministic regeneration; exact layer ownership for every tensor; and no
  checkpoint-derived payload committed.
- Correctness: the source-derived and official component paths agree at every
  layer-local boundary; each accumulated layer-0 and layer-1 output hash
  matches the independent native execution.
- State: layers 0 and 1 retain distinct DeltaNet convolution and recurrent
  states; layer 1 additionally retains independent PLE context and dilated
  convolution state across tokens 42 and 43.
- Routing: both layers recompute top-10 routes from their actual accumulated
  activations and authenticate every selected expert slice and weighted output.
- Safety: only selected PLE rows, selected expert slices, and bounded ordinary
  tensors are read; generated evidence remains outside Git.
- Continuation: exact parity unlocks accumulation through layer 3, including
  the first full-attention/QSA boundary, then the remaining 48-layer walk.
- Kill/repair: stop at the earliest upstream output, PLE, state, route, expert,
  or residual mismatch and preserve the discrepancy.

Excluded claims: layers 2 through 47, embedding or final normalization, logits,
an endpoint, real prefill, modality processing, latency, and TPS.

## Result

The source-derived accumulated fixture passes and regenerates byte-identically.
Layer 0 retains the FW-0017 routes. Its outputs change layer 1's initial route
from `[495, 40, 7, 110, 113, 450, 241, 252, 236, 503]` to
`[40, 495, 7, 110, 450, 113, 503, 370, 241, 236]` and the cached route from
`[469, 60, 456, 259, 80, 202, 453, 245, 176, 186]` to
`[60, 469, 259, 80, 456, 453, 468, 202, 176, 186]`. Thus both calls
discriminate the accumulated input from FW-0023's layer-local authority.

The fixture embeds hash-only authorities for the accumulated PLE, attention,
and complete layer-1 stages. Each PLE hidden-state hash exactly equals the
corresponding layer-0 output hash. Its SHA-256 is
`d1b204354dddf606ad1156f558bb9656f57a81a79f9ae35733127947dc4d2e0b`.
All 48 Python tests pass. Native verification is pending.

## Decision

Pending.
