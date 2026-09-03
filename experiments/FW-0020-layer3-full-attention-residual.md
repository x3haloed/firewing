# FW-0020 - Layer-3 full-attention residual composition

- Status: in progress
- Disposition: reference fixture passed; native composition pending
- Date: 2026-09-03
- Parent experiments: FW-0014, FW-0019
- Exactness: L0 bit-identical component semantics
- Hardware/runtime: Apple M1 Mac mini (`Macmini9,1`), 16 GiB, internal SSD

## Question and hypothesis

Can the exact layer-3 gated hyper-connection and full-attention/QSA primitives
compose through the decoder's four-stream residual update without changing
cache ownership, tensor layout, or BF16 operation staging?

The hypothesis is that an empty-cache token and one token over the same
deterministic 2,080-position active-pruning cache used by FW-0019 are sufficient
to expose wrapper and state errors. Unlike FW-0019, attention receives the
actual 2,560-wide mixed output of layer 3's real 10,240-wide hyper-connection,
so its prior hashes cannot be reused as output authority.

## Frozen authority and baseline

- Checkpoint revision:
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Model lock SHA-256:
  `f87399e8659ab3274601fcd455b78b73c600f57e5fc1e91499eec3ac1f4b9444`
- Baseline commit: `618735f`
- Framework reference: Transformers 5.16.1
  `Qwen4ExpTextGatedResidual.forward`, `Qwen4ExpTextAttention.forward`, and
  `Qwen4ExpTextDecoderLayer.forward`

## Method and commands

Load layer 3's four attention-hyper-connection tensors and nine full-attention
tensors. For each deterministic four-stream BF16 input, run the official gated
residual module, then run both the official full-attention module and the
source-derived explicit attention path against independent caches. Compose the
attention output with the four injection weights exactly as the decoder does.

The initial case uses position zero and empty caches. The long case starts at
position 2,080 with independently regenerated raw-indexer, rotated-key, and
value history. It must select exactly 512 of 520 complete QSA blocks with an
untied boundary and retain the current token. Freeze every hyper-connection and
FW-0019 attention boundary plus the injection products and composed residual.

```shell
.venv/bin/python tools/generate_full_attention_residual_fixture.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  --model-lock spec/model.lock.json \
  --full-attention-fixture fixtures/full_attention/qwen3_8_flash_next_layer3.json \
  --output fixtures/attention_residual/qwen3_8_flash_next_layer3.json

cargo run --release -- verify-full-attention-residual \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  fixtures/full_attention/qwen3_8_flash_next_layer3.json \
  fixtures/attention_residual/qwen3_8_flash_next_layer3.json \
  /Users/chad/Models/firewing/evidence/FW-0020/full-attention-residual.json
```

Batch size and concurrency are one. Accepted tokens, `A`, `U`, and measured
TPS are zero because this is a component-composition correctness fixture.

## Gates

- Fixture: exact model/config/index/lock identity; parent FW-0019 fixture hash;
  13 tensor identities; explicit-versus-official attention agreement; and
  deterministic regeneration.
- Correctness: every BF16, F32, integer, and boolean capture matches, including
  mixed attention input, QSA selection, all cache appends, injection products,
  and the final 10,240-wide residual.
- State: wrapper composition must not alias raw indexer keys with rotated K/V
  state or mutate the preserved hyper input.
- Safety: generation and verification remain below 1 GiB; only hashes and
  compact arithmetic specifications are committed.
- Continuation: exact parity unlocks layer-3 MLP/MoE composition and then a
  complete decoder fixture containing the model's full-attention variant.
- Kill/repair: stop at the earliest hyper, attention, cache, injection, or
  residual boundary mismatch and preserve the discrepancy.

Excluded claims: complete layer 3, accumulated layers, real 2K prefill,
million-token behavior, endpoint behavior, modality processing, latency, and
TPS.

## Result

The reference fixture passes and regenerates byte-identically. Across the
empty-cache and active-pruning cases, the source-derived attention path exactly
matches the official module's output, selected-token mask, raw indexer cache,
rotated key cache, and value cache after receiving layer 3's actual gated
hyper-connection output. The fixture binds 13 real tensors and 36 captures per
case, including the 10,240-wide injection products and composed residual. Its
SHA-256 is
`21c7dbf9b34af540ef424044a797a880674752574a820d727759ff1a9ea51035`.
All 40 Python tests pass. Native composition remains pending.

## Decision

Continue by extracting FW-0019's exact attention execution into a reusable
native case runner, then compose it with layer 3's hyper-connection. No
performance default follows from reference composition.
