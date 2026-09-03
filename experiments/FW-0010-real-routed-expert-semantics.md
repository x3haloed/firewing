# FW-0010 - Real routed-expert semantics

- Status: complete
- Disposition: correctness-repair — source-derived ARM reduction retained
- Date: 2026-09-03
- Parent experiments: FW-0009
- Exactness: L0 for the six tested BF16 capture identities
- Hardware/runtime: Apple M1 Mac mini (`Macmini9,1`), 16 GiB, internal SSD

## Question and hypothesis

Can an independent native implementation reproduce one real Qwen4-Exp routed
expert from deterministic BF16 input through gate/up projection, SiLU product,
down projection, and route weighting? The initial hypothesis was that forward
F32 accumulation with BF16 rounding at every published tensor boundary would
be sufficient.

This is the cheapest useful step into the dominant decode component: one
ordinary token selects ten 9,830,400-byte routed experts in every one of 48
layers before caching or speculative union.

## Frozen authority and baseline

- Checkpoint revision:
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Model lock SHA-256:
  `f87399e8659ab3274601fcd455b78b73c600f57e5fc1e91499eec3ac1f4b9444`
- Router fixture SHA-256:
  `bbf73e4361d81b60e8ab49898422ba5fd0d2a90f312433ac43a013e48c89d39b`
- Reference: Transformers 5.16.1 `Qwen4ExpTextExperts.forward`
- Baseline commit: `0b10d359232b972bb82f2006c3b4d9e6b81c4bca`
- Python 3.11, Torch 2.14.0, Rust 1.96.0, macOS 26.6.2 build 25G83

Protocol deviation: the fixture, native verifier, prediction-error probe, and
repair were implemented in one dirty worktree rather than after a separate
protocol-freeze commit. This is a correctness-only slice with no timed claim;
the immutable fixture, exact commands, failed hash, and final receipt hash are
preserved.

## Method and commands

FW-0009's layer-0 deterministic input routes first to expert 376 with BF16
weight `0.11962890625`. The generator reads only expert 376 from the real
three-dimensional gate/up and down tensors, then records payload and capture
SHA-256 identities. It commits no checkpoint weights or complete outputs.

The native verifier independently regenerates the input, reads exactly the
selected 6,553,600-byte gate/up slice and 3,276,800-byte down slice, and checks
six exact BF16 capture hashes: combined gate/up, gate, up, SwiGLU, down, and
route-weighted down.

```shell
.venv/bin/python tools/generate_expert_fixture.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  --model-lock spec/model.lock.json \
  --output fixtures/expert/qwen3_8_flash_next_real.json

cargo run --release -- verify-expert \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json fixtures/router/qwen3_8_flash_next_real.json \
  fixtures/expert/qwen3_8_flash_next_real.json \
  /Users/chad/Models/firewing/evidence/FW-0010/expert-verification-de4b8e4d.json
```

Batch size and concurrency are one. Accepted tokens, `A`, `U`, physical SSD
bytes, endpoint wall time, and TPS are zero, unmeasured, or not applicable.
Cache state is uncontrolled because this is not a performance claim.

## Gates

- Exact checkpoint, model-lock, router-fixture, shard, tensor, shape, dtype,
  selected-expert, route-weight, and payload identities.
- Exact hashes for all six BF16 intermediate and output captures.
- Deterministic fixture regeneration and tiny synthetic equation coverage.
- Fail closed rather than relaxing equality on any numerical mismatch.
- Excluded: the other nine selected experts, mixture accumulation, real layer
  activations, SSD traffic, accelerated kernels, whole-model parity, and TPS.

## Result

The initial forward-sum verifier passed gate/up and SwiGLU but failed the down
capture: Torch expected
`e4977de49cca592bb33a6eb2e5c3ce6cb9a6cf291c426efd04513838c7fb274d`
while native produced
`c695503a6402fe6a7d50b5a8690ade3976546408f801f5b7ca41d3b165041198`.
This falsified accumulation order, not the expert equation or layout.

Prismwing PW-0073 and `text_endpoint.rs` at commit
`c87d0c1aa2c118f71ca5348434be35d02f62f031` supplied the source-derived
PyTorch aarch64 BF16 GEMV discriminator: eight four-lane F32 accumulators over
32-value blocks, a fixed pairwise register tree, then horizontal reduction.
Using that schedule made all six real capture hashes exact without weakening
the gate. The fixture confirms 9,830,400 selected source bytes, exactly the
prior ledger constant.

Twelve Rust tests, twenty Python tests, strict Clippy, deterministic fixture
regeneration, and the release verifier passed.

- Fixture SHA-256:
  `10315f99986464e85e186cc32d55488d9c68f7db0979f5cef1411c6b7e8a4752`
- External receipt SHA-256:
  `23d8a3b01c18695a4d3d2245427cb56ffa260d87cbb35d18dbd8f13b137f62c6`

## Decision

Retain the readable expert loader and the source-derived ARM reduction as the
native BF16 correctness oracle. Confidence is high for this expert/input pair
and the tested boundary sequence, but it does not establish mixture or
real-activation behavior. The next slice should execute all ten selected
experts and their F32 mixture accumulation for one token, reusing these exact
per-expert semantics before any accelerated kernel or storage optimization.
