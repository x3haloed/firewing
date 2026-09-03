# FW-0017 - Complete layer-0 cached decoder composition

- Status: completed
- Disposition: correctness-repair
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
- Baseline commit: `13254380b04dded8dcf311ad472be41b142ff4cf`
- Fixture commit: `c27ebb5417caff00b6569cb79d35e89e7c078149`
- Candidate commit: `5b5e105a72968891cf09d62c47eb2b7d930b74b8`
- Frozen fixture SHA-256:
  `f9529e94abd581956419cd28afebeade7f5a321776ab33bcefb60fba5c6311c0`
- Raw evidence SHA-256:
  `993edda73ce6106fa280b9db966afe898a8f652059d78356d5855464bf0bb232`
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

The source-derived fixture regenerated byte-identically and produced different
top-10 routes for the two real post-attention activations:

- step 0: `308, 7, 247, 139, 343, 17, 479, 489, 105, 381`;
- step 1: `446, 203, 74, 211, 414, 254, 138, 246, 12, 503`.

The two selections are disjoint, so the native run authenticated and executed
20 unique experts. It first reran the complete FW-0016 attention authority,
then matched all 16 declared BF16 captures at each step: post-attention state,
MLP mixed input and injection weights, all 512 router logits, selected route
weights, routed mixture, seven shared-expert boundaries, combined MoE output,
four-stream injection products, and the final 10,240-wide layer output. It also
matched every selected expert payload pair and all 20 weighted expert outputs.

The report accounts for 129,126,848 attention tensor bytes, 25,666,560 dense
MLP/router/shared tensor bytes, and 196,608,000 selected expert bytes, totaling
351,401,408 unique verified payload bytes. The standalone FW-0012 sparse-MoE
verifier remained exact after runtime helper extraction. The complete suites
passed with 34 Python and 24 Rust tests, and Clippy passed with warnings denied.
The router fails closed if a future input has a tie at the top-10 boundary,
whose platform-specific `torch.topk` ordering is not yet independently pinned.

No endpoint tokens were accepted and no TPS was measured. This is a complete
real layer-0 correctness result, not a multi-layer or endpoint result.

## Decision

Pass as a correctness repair. Firewing now has an exact, stateful native
execution of one complete real Qwen decoder layer, including activation-derived
dynamic routing and both hyper-residual compositions. This promotes sequential
layer composition, with PLE at layer 1 and full attention at layer 3 remaining
separate semantic gates. No performance default follows from layer-local
correctness.
