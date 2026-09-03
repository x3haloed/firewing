# FW-0019 - Real layer-3 full attention and QSA

- Status: in progress
- Disposition: reference fixture passed; native integration pending
- Date: 2026-09-03
- Parent experiments: FW-0014, FW-0017, FW-0018
- Exactness: L0 bit-identical component semantics
- Hardware/runtime: Apple M1 Mac mini (`Macmini9,1`), 16 GiB, internal SSD

## Question and hypothesis

Can a bounded native implementation exactly reproduce layer 3's full-attention
path, including QSA indexer selection, partial multimodal RoPE, grouped-query
KV repetition, query gating, and KV/indexer cache updates?

The hypothesis is that two cases cover the distinct semantics without a large
prefill: a one-token initial call validates empty-cache construction, and one
decode token over a deterministic 2,080-position synthetic cache crosses QSA's
actual pruning threshold. With compression ratio four and token budget 2,048,
the long case forms 520 complete blocks; QSA must select 512 blocks and exclude
eight complete four-token blocks while retaining the current tail token. The
extra blocks keep the top-k boundary out of the indexer's exact-zero ReLU tail;
a 2,052-position draft produced three tied zero-score blocks at a one-block
exclusion boundary and was rejected rather than made tie-dependent.

## Frozen authority and baseline

- Checkpoint revision:
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Model lock SHA-256:
  `f87399e8659ab3274601fcd455b78b73c600f57e5fc1e91499eec3ac1f4b9444`
- Baseline commit: `9c8b12a`
- Framework reference: Transformers 5.16.1
  `Qwen4ExpTextAttention.forward`, `Qwen4ExpTextQSAIndexer.forward`,
  `Qwen4ExpTextRotaryEmbedding.forward`, `apply_rotary_pos_emb`,
  `eager_attention_forward`, and `DynamicCache`

## Method and commands

Load layer 3's nine real BF16 attention/indexer tensors. Use deterministic BF16
hidden inputs and the pinned default partial-RoPE configuration: 24 query
heads, two KV heads, 256-wide heads, 64 rotary dimensions, theta 10,000,000,
four 128-wide indexer query heads, and one indexer key head.

Case 1 starts from empty `DynamicCache` and one visible token at position zero.
Case 2 preloads independently specified deterministic BF16 raw-indexer, rotated
key, and value caches for positions 0 through 2,079, then evaluates one current
token at position 2,080 with an all-visible causal row. Run the isolated official
module and a source-derived explicit path against separate caches and require
exact agreement.

Freeze exact hashes for projections, normalized and rotated Q/K, gate, cache
states, QSA pooled keys and scores, selected/excluded blocks and token mask,
attention scores/probabilities/value result, gated result, and output. The
native verifier must regenerate—not embed—the synthetic cache payload and fail
closed on QSA top-k boundary ties.

```shell
.venv/bin/python tools/generate_full_attention_fixture.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  --model-lock spec/model.lock.json \
  --output fixtures/full_attention/qwen3_8_flash_next_layer3.json

cargo run --release -- verify-full-attention \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  fixtures/full_attention/qwen3_8_flash_next_layer3.json \
  /Users/chad/Models/firewing/evidence/FW-0019/full-attention.json
```

Batch size and concurrency are one. Accepted tokens, `A`, `U`, and measured
TPS are zero because this is a stateful component correctness fixture.

## Gates

- Fixture: exact model/config/index/lock identity; nine tensor identities and
  payload hashes; deterministic cache regeneration; explicit-versus-official
  agreement; and deterministic fixture regeneration.
- Correctness: every declared BF16, F32, integer, and boolean capture matches;
  the long case selects exactly 512 of 520 complete blocks, excludes 32
  tokens, and retains the current tail.
- State: raw indexer keys, rotated main keys, and values append exactly one
  position without aliasing or confusing their coordinate systems.
- Safety: peak generation and verification remain below 1 GiB; no large cache,
  activation, or checkpoint payload is committed.
- Continuation: exact parity unlocks layer-3 attention residual composition and
  a decoder path covering all three text-layer semantic variants.
- Kill/repair: fail at the earliest projection, normalization, RoPE, selection,
  cache, attention, gate, or output boundary and preserve the mismatch.

Excluded claims: a real 2K prefill, complete layer 3, accumulated multi-layer
parity, million-token QSA behavior, endpoint behavior, modality preprocessing,
latency, and TPS.

## Result

The two-case reference fixture passes. A source-derived explicit path exactly
matches the official Transformers module's output, selected-token mask, raw
indexer cache, rotated main-key cache, and value cache. It freezes nine real
tensor identities and 31 captures per case. Regeneration is byte-identical;
39 Python tests pass.

The first 2,052-position draft falsified the assumption that a one-block
exclusion would be unambiguous: three blocks had exact zero scores after the
indexer's ReLU aggregation, tying across the top-k boundary. The retained
2,080-position case excludes eight blocks and has an untied boundary. Native
work has begun with independently tested 128/256-wide RMSNorm, BF16-staged
partial RoPE, and fail-closed top-k selection primitives; full fixture parity
is pending.

At commit `7f21b79`, the release-mode native projection verifier authenticated
all nine real layer-3 tensors (102,893,056 payload bytes) and exactly matched
12 BF16 captures across the two cases: deterministic hidden state, index-QK
projection, raw-indexer cache append, Q projection, per-head gate extraction,
K projection, and V projection. The long case regenerated 532,480 bytes of
synthetic raw-indexer history rather than storing it. Its report is
`/Users/chad/Models/firewing/evidence/FW-0019/projections.json`, SHA-256
`08e50d070c080648a6dee80c5bb4075bcf53f8176c0c7530becb54cf2bf030bd`.
This remains a projection diagnostic, not complete attention parity.

## Decision

Continue to native checkpoint integration. No performance default follows
from reference parity or isolated semantic primitives.
