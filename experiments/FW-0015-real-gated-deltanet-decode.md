# FW-0015 - Real Gated DeltaNet cached decode

- Status: in progress
- Disposition: partial parity; cached recurrent reduction unresolved
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
- Frozen fixture SHA-256:
  `d2cb94f6a6d08896cf836efe84f8a5effa9eb17603f077e44cf9ee9bbcc3c93f`
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

The native verifier now validates all nine real tensor identities and payloads.
For step 0 it reproduces all 20 captures exactly, including projection,
depthwise convolution and state, SLEEF-based decay, repeated and normalized
Q/K, padded chunk attention, the 3 MiB F32 recurrent state, configured sigmoid
gating, and the final output projection.

Failed localization attempts were preserved as observations rather than
reported results:

- scalar Rust `exp`/`log1p` first failed at step-0 decay;
- Accelerate `vvexpf`/`vvlog1pf` also failed there, with 24 of 48 decay lanes
  differing from the PyTorch/SLEEF result;
- SLEEF 3.9.0 matched the frozen SLEEF-3.8-derived decay capture exactly;
- ATen cascade reduction plus the two BF16-rounded `sqrt`/reciprocal operations
  resolved Q/K normalization;
- reproducing the actual 64-by-128 padded Accelerate SGEMM resolved two
  low-bit-sensitive chunk-attention outputs;
- the configured output gate is sigmoid, not the model's hidden SiLU
  activation; correcting that resolved gated normalization and step-0 output;
- step 1 currently fails closed at `recurrent_state`. Its retention vector and
  decayed incoming state match exactly, while the strided F32 memory reduction
  does not. Regenerating with `OMP_NUM_THREADS=1` produced the identical full
  fixture, excluding thread partitioning as the cause.

No report artifact is emitted on failure. Accepted tokens and TPS remain zero.

## Decision

Continue by reproducing the cached path's strided F32 outer reduction exactly.
Step-0 parity is a useful implementation milestone, but FW-0015 remains open
and no performance default follows from it.
