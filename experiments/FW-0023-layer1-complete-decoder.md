# FW-0023 - Complete layer-1 PLE-bearing decoder

- Status: planned
- Disposition: pending
- Date: 2026-09-03
- Parent experiments: FW-0012, FW-0017, FW-0022
- Exactness: L0 bit-identical component semantics
- Hardware/runtime: Apple M1 Mac mini (`Macmini9,1`), 16 GiB, internal SSD

## Question and hypothesis

Can FW-0022's exact PLE-bearing attention residual compose through layer 1's
separately parameterized MLP hyper-connection, dynamically routed top-10 MoE,
shared expert, and final four-stream residual?

The hypothesis is that the initial token 42 and cached token 43 produce
nontrivial real layer-1 routes and expose any wrapper, expert-slice, execution
order, or BF16 accumulation error. No route or activation from layer 0 is
reused as output authority.

## Frozen authority and baseline

- Checkpoint revision:
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Model lock SHA-256:
  `f87399e8659ab3274601fcd455b78b73c600f57e5fc1e91499eec3ac1f4b9444`
- Baseline commit: `ea629c1`
- Framework reference: Transformers 5.16.1
  `Qwen4ExpTextGatedResidual.forward`, `Qwen4ExpTextSparseMoeBlock.forward`,
  and `Qwen4ExpTextDecoderLayer.forward`

## Method and commands

Regenerate and require byte-identical FW-0022 parent evidence. For each of its
two post-attention states, load layer 1's real MLP hyper-connection, router,
shared expert, and shared-expert gate tensors. Compute the dynamic top-10 route,
read only those experts from the source banks, execute active experts in source
order, and freeze every MoE and final residual boundary plus selected expert
payload and weighted-output hashes.

```shell
.venv/bin/python tools/generate_full_decoder_layer1_fixture.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  --model-lock spec/model.lock.json \
  --ngram-fixture fixtures/ngram/qwen3_8_flash_next.json \
  --ngram-row-fixture fixtures/ngram/qwen3_8_flash_next_row_hashes.json \
  --ple-fixture fixtures/ple/qwen3_8_flash_next_layer1_decode.json \
  --attention-residual-fixture fixtures/attention_residual/qwen3_8_flash_next_layer1_ple.json \
  --output fixtures/decoder_layer/qwen3_8_flash_next_layer1_ple.json

cargo run --release -- verify-decoder-layer1 \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  fixtures/ngram/qwen3_8_flash_next.json \
  fixtures/ngram/qwen3_8_flash_next_row_hashes.json \
  fixtures/ple/qwen3_8_flash_next_layer1_decode.json \
  fixtures/attention_residual/qwen3_8_flash_next_layer1_ple.json \
  fixtures/decoder_layer/qwen3_8_flash_next_layer1_ple.json \
  /Users/chad/Models/firewing/evidence/FW-0023/decoder-layer1.json
```

Batch size and concurrency are one. Accepted tokens, `A`, `U`, and measured
TPS are zero because this is a complete-layer correctness fixture.

## Gates

- Fixture: exact model/config/index/lock identity; exact FW-0022 parent hash;
  deterministic regeneration; nine dense tensor identities; both expert-bank
  identities; selected expert payload hashes; and no payload bytes committed.
- Correctness: every BF16 MLP hyper-connection, router, routed mixture, shared
  expert, injection, and final residual capture must match independently.
- Routing: top-10 selection is recomputed from the actual post-attention state;
  route weights and source-ordered BF16 accumulation must match.
- State: the two calls retain FW-0022's independent PLE and DeltaNet cache
  evolution and must not reuse output authority from another layer.
- Safety: only selected expert slices and bounded ordinary tensors are read;
  generated evidence remains outside Git.
- Continuation: exact parity completes every decoder-layer wrapper variant and
  unlocks accumulated multi-layer execution.
- Kill/repair: stop at the earliest parent, hyper, route, expert, shared,
  injection, or residual mismatch and preserve the discrepancy.

Excluded claims: accumulated layers, embedding or final normalization, logits,
real prefill, endpoint behavior, modality processing, latency, and TPS.

## Result

Pending.

## Decision

Pending.
