# FW-0026 - Accumulated full-attention layer 3

- Status: planned
- Disposition: pending
- Date: 2026-09-03
- Parent experiments: FW-0021, FW-0025
- Exactness: L0 bit-identical component semantics
- Hardware/runtime: Apple M1 Mac mini (`Macmini9,1`), 16 GiB, internal SSD

## Question and hypothesis

Can the exact accumulated layer-2 outputs from FW-0025 cross the model's first
linear-to-full-attention boundary and complete layer 3 for the same initial and
cached tokens?

The hypothesis is that the layer-local full-attention semantics established by
FW-0021 remain exact when query, key, value, indexer, router, and MoE inputs all
come from the real preceding layers. This is the last distinct decoder
transition before a generalized remaining-layer walk.

## Frozen authority and baseline

- Checkpoint revision:
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Model lock SHA-256:
  `f87399e8659ab3274601fcd455b78b73c600f57e5fc1e91499eec3ac1f4b9444`
- Baseline commit: `8c28cd9`
- Framework reference: Transformers 5.16.1
  `Qwen4ExpTextAttention.forward`, `Qwen4ExpTextQSAIndexer.forward`,
  `Qwen4ExpTextGatedResidual.forward`, and
  `Qwen4ExpTextDecoderLayer.forward`

## Method

Regenerate FW-0025 exactly and retain both complete layer-2 outputs. Feed the
first into layer 3 with an empty full-attention cache. Retain its indexer, key,
and value entries, then feed the second layer-2 output at position one through
that same layer-3 cache. Execute layer 3's attention hyper-connection, full
attention, four-stream residual, MLP hyper-connection, dynamic top-10 MoE,
shared expert, and final residual. Freeze every full-attention cache and
selection boundary, both dynamic routes, selected expert hashes, weighted
outputs, and complete layer-3 outputs. Every tensor identity must carry the
layer-3 prefix.

The cached case has one prior token and therefore exercises real incremental
cache ownership without activating long-context QSA pruning. FW-0019 and
FW-0021 remain the independent exact authorities for the 2,080-position active
QSA path; this experiment must not replace that slice with a synthetic history
that did not pass through layers 0 through 2.

Planned commands:

```shell
.venv/bin/python tools/generate_accumulated_layer3_fixture.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  --model-lock spec/model.lock.json \
  --ngram-fixture fixtures/ngram/qwen3_8_flash_next.json \
  --ngram-row-fixture fixtures/ngram/qwen3_8_flash_next_row_hashes.json \
  --layer0-hyper-fixture fixtures/hyper_connection/qwen3_8_flash_next_layer0.json \
  --layer0-deltanet-fixture fixtures/deltanet/qwen3_8_flash_next_layer0_decode.json \
  --layer0-attention-fixture fixtures/attention_residual/qwen3_8_flash_next_layer0.json \
  --layer0-sparse-moe-fixture fixtures/sparse_moe/qwen3_8_flash_next_layer0.json \
  --layer0-fixture fixtures/decoder_layer/qwen3_8_flash_next_layer0.json \
  --ple-fixture fixtures/ple/qwen3_8_flash_next_layer1_decode.json \
  --layer1-attention-fixture fixtures/attention_residual/qwen3_8_flash_next_layer1_ple.json \
  --layer1-fixture fixtures/decoder_layer/qwen3_8_flash_next_layer1_ple.json \
  --layers01-fixture fixtures/accumulated/qwen3_8_flash_next_layers0_1.json \
  --layer2-fixture fixtures/accumulated/qwen3_8_flash_next_layer2.json \
  --full-attention-fixture fixtures/full_attention/qwen3_8_flash_next_layer3.json \
  --attention-residual-fixture fixtures/attention_residual/qwen3_8_flash_next_layer3.json \
  --output fixtures/accumulated/qwen3_8_flash_next_layer3.json

cargo run --release -- verify-accumulated-layer3 \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  fixtures/accumulated/qwen3_8_flash_next_layer2.json \
  fixtures/accumulated/qwen3_8_flash_next_layer3.json \
  /Users/chad/Models/firewing/evidence/FW-0026/accumulated-layer3.json
```

The final commands may list the committed FW-0025 component authorities
explicitly rather than resolving them transitively. Batch size and concurrency
are one. Accepted tokens, `A`, `U`, and measured TPS are zero because this is an
accumulated correctness fixture.

## Gates

- Fixture: exact checkpoint and FW-0025 identity; deterministic regeneration;
  layer-3 prefix on every tensor; selected expert hashes; no payload committed.
- Correctness: every BF16/F32/integer/boolean attention and cache boundary,
  MoE capture, weighted expert output, and final layer-3 result matches the
  source-derived reference.
- State: the second step must consume exactly the layer-3 indexer, key, and
  value state produced by the first step; no synthetic long cache is allowed.
- Routing: top-10 selection is recomputed from the actual accumulated
  post-attention state; only equal-logit internal permutations are neutral.
- Safety: load only bounded ordinary tensors and selected experts; generated
  evidence remains outside Git.
- Continuation: exact parity unlocks a data-driven 48-layer accumulated walk,
  embedding/final-normalization/logit semantics, and then a slow endpoint.
- Kill/repair: stop at the first upstream link, position, cache, attention,
  route, expert, or residual mismatch and preserve it.

Excluded claims: active long-context QSA within the accumulated walk, layers 4
through 47, embedding or final normalization, logits, an endpoint, real
prefill, modality processing, latency, and TPS.

## Result

The source-derived fixture executes successfully and regenerates
byte-identically. Layer 3 selects
`[208, 282, 174, 419, 343, 106, 250, 25, 38, 140]` for the initial token and
`[360, 65, 448, 357, 327, 170, 436, 298, 82, 213]` for the cached token; the
routes are disjoint. Both attention hyper-input hashes exactly equal FW-0025's
corresponding final layer-2 output hashes. The cached case freezes an indexer
cache of shape `[1, 2, 128]`, key and value caches of shape `[1, 2, 2, 256]`,
and selection of both visible tokens with no excluded blocks.

The fixture SHA-256 is
`5b457ee60daafb8a69093e3177e2e56896cea0348bdf9b8c5d876860ae28794f`.
All 55 Python tests pass. The generalized generators also reproduce FW-0019's
full-attention and FW-0020's residual fixtures byte-for-byte. Independent
native verification is pending.

## Decision

Pending.
