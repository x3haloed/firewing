# FW-0015 - Real Gated DeltaNet cached decode

- Status: planned
- Disposition: unexecuted
- Date: 2026-09-03
- Parent experiments: FW-0001, FW-0014
- Exactness: L0 bit-identical component semantics
- Hardware/runtime: Apple M1 Mac mini (`Macmini9,1`), 16 GiB, internal SSD

## Question and hypothesis

Can a bounded native implementation reproduce Qwen's layer-0 Gated DeltaNet
for both the first token and the subsequent cached decode token, including the
depthwise convolution and recurrent-state transitions? Thirty-six of the 48
main decoder layers use this attention type, so a slow exact implementation is
required before a complete text endpoint or real expert-reuse trace is valid.

The hypothesis is that two deterministic BF16 inputs and an initially empty
official `DynamicCache` expose all state needed to implement the decode path:
the first call creates the four-token convolution state and F32 delta-rule
state, and the second call exercises `causal_conv1d_update` plus the recurrent
gated-delta rule. The fixture must localize any mismatch at the earliest
captured boundary; final-output-only agreement is insufficient.

## Frozen authority and baseline

- Checkpoint revision:
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Model lock SHA-256:
  `f87399e8659ab3274601fcd455b78b73c600f57e5fc1e91499eec3ac1f4b9444`
- Baseline commit: `3deee662b9fdb7f7d03b765b5684ce6a2592d578`
- Framework reference: Transformers 5.16.1
  `Qwen4ExpTextGatedDeltaNet.forward`, fallback causal-convolution functions,
  chunk/recurrent gated-delta rules, RMSNormGated, and `DynamicCache`

## Method and commands

Load only the nine layer-0 `linear_attn` tensors from the indexed checkpoint.
Generate two distinct deterministic BF16 inputs of shape `[1,1,2560]`. Run an
explicit source-derived capture path and the isolated official module with
separate initially empty caches; require exact agreement on outputs and cache
states after each call.

Freeze hashes for every weight and for both steps' input projections,
convolution result/state, split and repeated Q/K/V, beta and F32 decay, normalized
Q/K, delta-rule output/state, gated normalization, and output projection. The
native verifier must rederive all tensor shapes, dtypes, state dimensions, and
BF16/F32 boundaries and fail at the first mismatch.

```shell
.venv/bin/python tools/generate_deltanet_fixture.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  --model-lock spec/model.lock.json \
  --output fixtures/deltanet/qwen3_8_flash_next_layer0_decode.json

cargo run --release -- verify-deltanet \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  fixtures/deltanet/qwen3_8_flash_next_layer0_decode.json \
  /Users/chad/Models/firewing/evidence/FW-0015/deltanet.json
```

Batch size and concurrency are one. Accepted tokens and measured TPS are zero;
this is a stateful component correctness fixture, not an endpoint.

## Gates

- Fixture: nine exact tensor identities and payload hashes; two deterministic
  inputs; exact explicit-versus-official output, convolution-state, and
  recurrent-state equality at both steps; deterministic regeneration.
- Correctness: every declared BF16 and F32 capture hash matches exactly. State
  mutation order and cached versus uncached branches are explicit.
- Safety: peak fixture generation and verification remain bounded below 1 GiB;
  no checkpoint-derived payload is committed.
- Continuation: exact parity unlocks layer-0 attention residual composition and
  the complete real decoder-block slice.
- Kill/repair: preserve the first mismatching capture and resolve its reduction,
  layout, precision, or mutation semantics before proceeding.

Excluded claims: multi-token prefill/chunk parity beyond one initial token,
full-attention/QSA layers, complete decoder parity, endpoint behavior, and TPS.

## Result

Pending execution.

## Decision

Pending. No performance default follows from component parity.
