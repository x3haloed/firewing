# FW-0017 - Complete layer-0 cached decoder composition

- Status: planned
- Disposition: unexecuted
- Date: 2026-09-03
- Parent experiments: FW-0009, FW-0012, FW-0016
- Exactness: L0 bit-identical component semantics
- Hardware/runtime: Apple M1 Mac mini (`Macmini9,1`), 16 GiB, internal SSD

## Question and hypothesis

Can the native layer-0 attention residual feed the independently parameterized
MLP hyper-connection, dynamic top-10 router, selected routed experts, shared
expert, and final four-stream residual update exactly for both the initial and
cached decode tokens?

The hypothesis is that a source-derived two-step fixture can bind the actual
post-attention state to each step's routes and selected expert payloads without
materializing the complete 512-expert layer. This is the first fixture whose
final capture is a complete real Qwen decoder layer rather than an isolated
primitive. Following Prismwing's real base-layer pattern, the MoE authority
must name and hash the post-attention input it consumes, and the verifier must
defer success until the final residual is checked.

## Frozen authority and baseline

- Checkpoint revision:
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Model lock SHA-256:
  `f87399e8659ab3274601fcd455b78b73c600f57e5fc1e91499eec3ac1f4b9444`
- Baseline commit: `1325438`
- Attention-residual fixture SHA-256:
  `d4f19fd28cc0a56fbbbb64cdfd494b8a9ff886f397d95a45b21a26a959c7427e`
- Sparse-MoE fixture SHA-256:
  `a6a706f93b8e97603574594ec7a100b9d44090de06db7e72d41092d49d447990`
- Framework reference: Transformers 5.16.1
  `Qwen4ExpTextDecoderLayer.forward`, `Qwen4ExpTextGatedResidual.forward`,
  `Qwen4ExpTextTopKRouter.forward`, `Qwen4ExpTextExperts.forward`, and
  `Qwen4ExpTextSparseMoeBlock.forward`

## Method and commands

Reuse FW-0016's two deterministic 10,240-wide BF16 inputs and stateful
attention computation. For each resulting post-attention residual, load the
four real `mlp_hyper_connection` tensors, derive its 2,560-wide MoE input and
four injection weights, then execute the source equations against the real
router, only the ten selected routed experts, and all four shared-expert
tensors. Record selection order, source expert execution order, BF16 normalized
route weights, every selected payload hash, and exact captures through the
final 10,240-wide layer output.

The generator must independently reproduce the official isolated gated
residual and router results. It must also reproduce the published expert and
shared-expert equations with explicit BF16 boundaries. The fixture is bound to
the FW-0016 and FW-0012 fixture hashes, but its real post-attention inputs and
routes are new authority rather than borrowed outputs from the affine-mod MoE
fixture.

```shell
.venv/bin/python tools/generate_decoder_layer_fixture.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  --model-lock spec/model.lock.json \
  --attention-fixture fixtures/attention_residual/qwen3_8_flash_next_layer0.json \
  --sparse-moe-fixture fixtures/sparse_moe/qwen3_8_flash_next_layer0.json \
  --output fixtures/decoder_layer/qwen3_8_flash_next_layer0.json

cargo run --release -- verify-decoder-layer \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  fixtures/hyper_connection/qwen3_8_flash_next_layer0.json \
  fixtures/deltanet/qwen3_8_flash_next_layer0_decode.json \
  fixtures/attention_residual/qwen3_8_flash_next_layer0.json \
  fixtures/sparse_moe/qwen3_8_flash_next_layer0.json \
  fixtures/decoder_layer/qwen3_8_flash_next_layer0.json \
  /Users/chad/Models/firewing/evidence/FW-0017/decoder-layer.json
```

Batch size and concurrency are one. Accepted tokens, `A`, `U`, and TPS are
zero because this is a correctness fixture, not an endpoint benchmark.

## Gates

- Fixture: exact model/config/index/lock identity; both parent fixture hashes;
  exact dense tensor and selected expert payload identities; deterministic
  regeneration; and no checkpoint-derived payload committed.
- Correctness: exact BF16 hashes for both MLP hyper-connection boundaries,
  router logits and normalized selected weights, routed and shared results,
  MoE output, injection products, and final layer residual at both steps.
- State: the attention cache persists across the two calls, while MoE execution
  is stateless and independently routed for each actual attention result.
- Safety: only selected expert slices are resident; generation and verification
  stay bounded well below the 13 GiB process ceiling.
- Continuation: exact parity unlocks sequential real-layer composition and
  layer-local fixtures for PLE and full-attention layers.
- Kill/repair: fail at the earliest named boundary. Preserve any mismatch in
  reduction order, top-k tie behavior, softmax, expert execution order, or
  residual rounding before proceeding.

Excluded claims: multi-token prefill beyond one initial token, PLE, full/QSA
attention, multiple accumulated layers, logits, endpoint behavior, modality
paths, latency, and accepted TPS.

## Result

Pending execution.

## Decision

Pending. No performance default follows from component parity.
