# FW-0027 - Accumulated decoder layers 4 through 47

- Status: completed
- Disposition: correctness-repair
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
  --layer3-fixture fixtures/accumulated/qwen3_8_flash_next_layer3.json \
  --full-attention-fixture fixtures/full_attention/qwen3_8_flash_next_layer3.json \
  --attention-residual-fixture fixtures/attention_residual/qwen3_8_flash_next_layer3.json \
  --output fixtures/accumulated/qwen3_8_flash_next_layers4_47.json

cargo run --release -- verify-accumulated-layers4-47 \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  fixtures/ngram/qwen3_8_flash_next.json \
  fixtures/ngram/qwen3_8_flash_next_row_hashes.json \
  fixtures/hyper_connection/qwen3_8_flash_next_layer0.json \
  fixtures/deltanet/qwen3_8_flash_next_layer0_decode.json \
  fixtures/attention_residual/qwen3_8_flash_next_layer0.json \
  fixtures/sparse_moe/qwen3_8_flash_next_layer0.json \
  fixtures/decoder_layer/qwen3_8_flash_next_layer0.json \
  fixtures/ple/qwen3_8_flash_next_layer1_decode.json \
  fixtures/attention_residual/qwen3_8_flash_next_layer1_ple.json \
  fixtures/decoder_layer/qwen3_8_flash_next_layer1_ple.json \
  fixtures/accumulated/qwen3_8_flash_next_layers0_1.json \
  fixtures/accumulated/qwen3_8_flash_next_layer2.json \
  fixtures/accumulated/qwen3_8_flash_next_layer3.json \
  fixtures/accumulated/qwen3_8_flash_next_layers4_47.json \
  /Users/chad/Models/firewing/evidence/FW-0027/accumulated-layers4-47.json
```

Batch size and concurrency are one. Accepted tokens, `A`, `U`, and measured
TPS are zero because this is accumulated correctness evidence.

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

The source-derived walker completes all 44 remaining layers and regenerates
byte-identically. It follows the exact 33-linear/11-full-attention suffix,
freezes all 880 expert selections, and links every layer input hash to the
preceding real output. Across the two steps, 27 selections reuse an expert
within the same layer, leaving 853 layer-scoped unique expert payloads for the
native byte ledger. The final layer-47 output hashes are
`7a2a93674993a3720bf921db9c97b9e81c38db83860086d8f6988fb6b51d9c9c`
and `3c63332aa6808d414ca198e88f52ee5c16c70f681753c2aa6ad85ac9e50c0714`.

The 2.0 MiB hash-only fixture has SHA-256
`6ed2e16da1e64fb8001c26608d7972f4910190f74768055b5778dc7891ebf525`.
All 59 Python tests pass, and the generalized full-attention source path still
regenerates FW-0020's committed layer-3 fixture byte-for-byte. Independent
native verification initially stopped at layer 5's cached MLP input. Every
intermediate hash through the 10,240 hyper-connection products matched; only
the four-stream mean differed. The old native pair tree produced
`135d7cf3...baf37`, while PyTorch's sequential F32 reduction produced the
expected `7603e3b8...6b4bf`. A cancellation-sensitive unit fixture now
distinguishes those orders.

After that repair, layer 5 exposed an equal-BF16-logit tie at the top-10
selection boundary. This cannot be treated as a neutral permutation because
it changes the selected expert set. The native path now reproduces the pinned
PyTorch 2.14.0 CPU algorithm: libc++ `nth_element` at `k - 1`, followed by
sorting the first `k - 1` entries for this 512-by-10 shape. A synthetic
boundary-tie fixture matches PyTorch's exact selected IDs. The installed
`ATen/native/TopKImpl.h` has SHA-256
`1ff24ba878ccb3816511ba34609d7247225342c6aa61740b51917c8ca79407ab`.

At commit `1e49846`, the release-mode native verifier passes the complete
48-layer replay. The suffix matches 1,144 BF16 and 88 F32 attention captures,
66 int64 and 22 boolean full-attention selection captures, 1,408 decoder BF16
captures, and 880 weighted-expert hashes across 853 layer-scoped unique expert
payloads. Layers 4 through 47 authenticate 15,052,975,040 logical payload
bytes; including FW-0026, the complete stack authenticates 16,511,246,080
bytes. Both final native output hashes equal the source fixture.

The final receipt is
`/Users/chad/Models/firewing/evidence/FW-0027/accumulated-layers4-47.json`,
SHA-256
`cae2c5657138d721ce8195455dd76e0550dbd61d7763faddaa22bc1362430cc7`.
The preserved prediction-error record is
`/Users/chad/Models/firewing/evidence/FW-0027/prediction-errors.json`, SHA-256
`41a1d20b300cfc382f942f69d51d3b4efb4d00453a69c020f47c6f27b24de3df`.
The final suite has 59 Python and 39 Rust tests; Clippy passes with warnings
denied. FW-0020's active-QSA residual and FW-0021's complete layer-3 native
commands also pass after the generalized changes.

## Decision

Pass as a correctness repair. Exact native execution now covers every decoder
layer for an initial and cached token, including all private recurrent/KV
states, dynamic routes, selected experts, and residual boundaries. Proceed to
token embedding, final normalization, LM-head/logits, and the first slow text
endpoint. This result does not establish MTP, real prefill, modality behavior,
latency, or TPS.
