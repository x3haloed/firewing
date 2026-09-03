# FW-0022 - Layer-1 PLE and attention residual composition

- Status: planned
- Disposition: pending
- Date: 2026-09-03
- Parent experiments: FW-0015, FW-0018, FW-0021
- Exactness: L0 bit-identical component semantics
- Hardware/runtime: Apple M1 Mac mini (`Macmini9,1`), 16 GiB, internal SSD

## Question and hypothesis

Can layer 1's exact position-local embedding (PLE) injection compose with its
gated hyper-connection, Gated DeltaNet, and four-stream residual update while
preserving both independent recurrent states?

The hypothesis is that FW-0018's initial token 42 and cached token 43 are
sufficient to expose BF16 staging, wrapper ordering, or cache-ownership errors.
The PLE output must be added to the 10,240-wide hidden state before attention's
hyper-connection; the resulting 2,560-wide mixed input must then drive layer
1's real DeltaNet tensors and update a distinct convolution and recurrent
state.

## Frozen authority and baseline

- Checkpoint revision:
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Model lock SHA-256:
  `f87399e8659ab3274601fcd455b78b73c600f57e5fc1e91499eec3ac1f4b9444`
- Baseline commit: `3dd668e`
- Framework reference: Transformers 5.16.1
  `Qwen4ExpTextPLELayer.forward`, `Qwen4ExpTextGatedResidual.forward`,
  `Qwen4ExpTextGatedDeltaNet.forward`, and
  `Qwen4ExpTextDecoderLayer.forward`

## Method and commands

Regenerate FW-0018's sparse PLE rows and exact two-step output, add that output
to the same deterministic BF16 hidden state, and feed the result through layer
1's real attention hyper-connection and Gated DeltaNet. Compare the explicit
DeltaNet path with the official module and freeze the PLE output, post-PLE
hidden state, mixed attention input, injection weights, attention output, both
attention states, injection products, and composed residual.

```shell
.venv/bin/python tools/generate_ple_attention_residual_fixture.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  --model-lock spec/model.lock.json \
  --ngram-fixture fixtures/ngram/qwen3_8_flash_next.json \
  --ngram-row-fixture fixtures/ngram/qwen3_8_flash_next_row_hashes.json \
  --ple-fixture fixtures/ple/qwen3_8_flash_next_layer1_decode.json \
  --output fixtures/attention_residual/qwen3_8_flash_next_layer1_ple.json

cargo run --release -- verify-ple-attention-residual \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  fixtures/ngram/qwen3_8_flash_next.json \
  fixtures/ngram/qwen3_8_flash_next_row_hashes.json \
  fixtures/ple/qwen3_8_flash_next_layer1_decode.json \
  fixtures/attention_residual/qwen3_8_flash_next_layer1_ple.json \
  /Users/chad/Models/firewing/evidence/FW-0022/ple-attention-residual.json
```

Batch size and concurrency are one. Accepted tokens, `A`, `U`, and measured
TPS are zero because this is a component-composition correctness fixture.

## Gates

- Fixture: exact model/config/index/lock identity; exact FW-0018 parent hash;
  deterministic regeneration; and layer-1 tensor identities.
- Correctness: every frozen BF16 and F32 boundary matches, including PLE
  output, post-PLE addition, hyper-connection outputs, DeltaNet output, and the
  final 10,240-wide attention residual.
- State: PLE token context and dilated-convolution state and DeltaNet
  convolution and recurrent state must evolve independently across both steps.
- Safety: only the 32 sparse PLE rows selected by the frozen tokens may be read
  from the 128 large n-gram tables; generated evidence stays outside Git.
- Continuation: exact parity unlocks layer 1's MLP/MoE wrapper, then accumulated
  multi-layer execution.
- Kill/repair: stop at the earliest PLE, addition, hyper, attention, cache,
  injection, or residual boundary mismatch and preserve the discrepancy.

Excluded claims: complete layer 1, accumulated layers, real prefill,
million-token behavior, endpoint behavior, modality processing, latency, and
TPS.

## Result

Pending.

## Decision

Pending.
