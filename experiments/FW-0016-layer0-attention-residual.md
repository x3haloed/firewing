# FW-0016 - Layer-0 attention residual composition

- Status: planned
- Disposition: unexecuted
- Date: 2026-09-03
- Parent experiments: FW-0001, FW-0014, FW-0015
- Exactness: L0 bit-identical component semantics
- Hardware/runtime: Apple M1 Mac mini (`Macmini9,1`), 16 GiB, internal SSD

## Question and hypothesis

Can the native FW-0014 gated hyper-connection and FW-0015 Gated DeltaNet
implementations compose into the complete attention half of Qwen's real
layer-0 decoder residual for both an initial token and a cached recurrent
token? Layer 0 has no PLE injection, so this slice is exactly the source path
from the 10,240-wide four-stream input through attention and back to the
10,240-wide residual state.

The hypothesis is that two distinct deterministic BF16 hyper-inputs expose the
complete composition: hyper-connection mixing and injection weights,
stateful DeltaNet execution, per-stream BF16 injection products, and the final
BF16 residual addition. Existing component fixtures are necessary evidence but
do not prove their dtype boundaries, state ownership, or layout compose.

## Frozen authority and baseline

- Checkpoint revision:
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Model lock SHA-256:
  `f87399e8659ab3274601fcd455b78b73c600f57e5fc1e91499eec3ac1f4b9444`
- Baseline commit: `06e5a4c`
- Hyper-connection fixture SHA-256:
  `3615d35c75ed25fc7e81f5b82712017a1948260e3fe7f4b0e7cc8c92ead65503`
- DeltaNet fixture SHA-256:
  `d2cb94f6a6d08896cf836efe84f8a5effa9eb17603f077e44cf9ee9bbcc3c93f`
- Framework reference: Transformers 5.16.1
  `Qwen4ExpTextDecoderLayer.forward`, `Qwen4ExpTextGatedResidual.forward`, and
  `Qwen4ExpTextGatedDeltaNet.forward`

## Method and commands

Generate two deterministic BF16 hyper-inputs of shape `[1,1,10240]` using
separate affine modular sequences. Load only layer 0's four
`attn_hyper_connection` tensors and nine `linear_attn` tensors from the locked
checkpoint. Layer 0 must be confirmed as `linear_attention`, and
`ple_layer_ids` must exclude layer 1 so no omitted PLE semantic is hidden.

For each token, run the official gated-residual module to produce the
2,560-wide mixed input and four injection weights. Run the official DeltaNet
module against a shared initially empty `DynamicCache`. Then reproduce the
decoder source operations exactly:

```text
injection = attention_output.unsqueeze(-2) * injection_weights.unsqueeze(-1)
composed = hyper_input + injection.flatten(-2)
```

Freeze exact BF16 hashes for hyper-input, mixed input, injection weights,
attention output, injection products, and composed output, plus exact BF16
convolution-state and F32 recurrent-state hashes after each step. Bind the new
fixture to both parent fixture hashes. The native verifier must load each real
tensor once, maintain one explicit attention state across both calls, and fail
closed at the first mismatching boundary.

```shell
.venv/bin/python tools/generate_attention_residual_fixture.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  --model-lock spec/model.lock.json \
  --hyper-fixture fixtures/hyper_connection/qwen3_8_flash_next_layer0.json \
  --deltanet-fixture fixtures/deltanet/qwen3_8_flash_next_layer0_decode.json \
  --output fixtures/attention_residual/qwen3_8_flash_next_layer0.json

cargo run --release -- verify-attention-residual \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  fixtures/hyper_connection/qwen3_8_flash_next_layer0.json \
  fixtures/deltanet/qwen3_8_flash_next_layer0_decode.json \
  fixtures/attention_residual/qwen3_8_flash_next_layer0.json \
  /Users/chad/Models/firewing/evidence/FW-0016/attention-residual.json
```

Batch size and concurrency are one. Accepted tokens, `A`, `U`, and measured
TPS are zero because this is a stateful component correctness fixture.

## Gates

- Fixture: exact model/config/index/lock identity; both parent fixture hashes;
  13 exact real tensor identities and payload hashes; two deterministic inputs;
  and deterministic regeneration.
- Correctness: all declared BF16 and F32 capture hashes match exactly for both
  steps, including the post-attention 10,240-wide residual state.
- State: convolution and recurrent state persist across the two calls and no
  MLP, PLE, QSA, or unrelated cache state is smuggled into the slice.
- Safety: peak fixture generation and verification remain bounded below 1 GiB;
  no checkpoint-derived payload or large evidence is committed.
- Continuation: exact parity unlocks the layer-0 MLP hyper-connection and full
  sparse-MoE residual composition using the real attention output as input.
- Kill/repair: any mismatch stops at the earliest named boundary. A parent
  primitive may be refactored for arbitrary inputs but its existing fixture
  must continue to pass unchanged.

Excluded claims: the MLP half of layer 0, PLE at layer 1, QSA/full-attention
layers, complete decoder parity, endpoint behavior, multimodal behavior,
latency, and accepted TPS.

## Result

Pending execution.

## Decision

Pending. No performance default follows from component parity.
