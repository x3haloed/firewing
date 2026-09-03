# FW-0014 - Real gated hyper-connection semantics

- Status: planned
- Disposition: unexecuted
- Date: 2026-09-03
- Parent experiments: FW-0001, FW-0009, FW-0012
- Exactness: L0 bit-identical component semantics
- Hardware/runtime: Apple M1 Mac mini (`Macmini9,1`), 16 GiB, internal SSD

## Question and hypothesis

Can an independent native implementation reproduce Qwen's real layer-0
attention gated-residual primitive exactly at every BF16 boundary? This
primitive mixes four 2,560-wide hyper-connection streams into one decoder
input and produces four block-injection weights. It wraps both attention and
MoE in every decoder layer, so resolving it is required before real activation
traces or a complete decoder block are meaningful.

The hypothesis is that the existing PyTorch aarch64 BF16 GEMV reduction from
FW-0010 plus Prismwing's independently tested PyTorch aarch64 F32 RMS reduction
topology can reproduce the pinned framework captures exactly. No approximate
mode, changed weight, or relaxed comparison is allowed.

## Frozen authority and baseline

- Checkpoint revision:
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Model lock SHA-256:
  `f87399e8659ab3274601fcd455b78b73c600f57e5fc1e91499eec3ac1f4b9444`
- Frozen fixture SHA-256:
  `3615d35c75ed25fc7e81f5b82712017a1948260e3fe7f4b0e7cc8c92ead65503`
- Baseline commit: `37c9451f439644e6b0a3a539531295be15da15c0`
- Framework reference: Transformers 5.16.1
  `Qwen4ExpTextGatedResidual.forward` and `Qwen4ExpTextRMSNorm.forward`
- Reduction reference: Prismwing commit
  `c87d0c1aa2c118f71ca5348434be35d02f62f031`, `src/lib.rs` SHA-256
  `2f2de84115cc99bcf2bca8714682fd374582a671df3254841f6adaa64b3d6717`

## Method and commands

Generate one deterministic 10,240-element BF16 hyper-input. Load only the four
real layer-0 attention hyper-connection tensors from their indexed checkpoint
shard. Capture hashes for input, grouped RMS normalization, down projection,
division by four, SiLU, up projection, sigmoid mix weights, mixed input, block
injection projection, division by four, and doubled sigmoid injection weights.
Commit tensor identities and hashes, never payload bytes.

The native verifier must independently re-read exact BF16 tensors from the
locked checkpoint, rederive every shape and byte count, reproduce every capture
hash, and fail closed on an unknown schema, revision, tensor, shape, reduction,
or dtype.

```shell
.venv/bin/python tools/generate_hyper_connection_fixture.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  --model-lock spec/model.lock.json \
  --output fixtures/hyper_connection/qwen3_8_flash_next_layer0.json

cargo run --release -- verify-hyper-connection \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  fixtures/hyper_connection/qwen3_8_flash_next_layer0.json \
  /Users/chad/Models/firewing/evidence/FW-0014/hyper-connection.json
```

Batch size and concurrency are one. Accepted tokens, `A`, and measured TPS are
zero because this is a component correctness fixture.

## Gates

- Fixture: exact model/config/index/lock identities, four tensor payload
  hashes, and deterministic input/capture regeneration.
- Correctness: every declared BF16 capture hash matches exactly; no numerical
  tolerance substitutes for the L0 claim.
- Safety: only the four bounded tensors and intermediate vectors are loaded;
  model weights and large evidence remain outside Git.
- Continuation: exact parity permits reuse in the subsequent real attention or
  complete decoder-block slice.
- Kill/repair: any mismatch stops at the first capture and remains unresolved
  until the arithmetic or reduction topology is explained.

Excluded claims: attention, MoE, complete residual injection, accumulated-layer
parity, modality behavior, endpoint latency, and accepted TPS.

## Result

Pending execution.

## Decision

Pending. No runtime default or performance claim follows from this fixture.
