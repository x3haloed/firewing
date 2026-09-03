# FW-0012 - Real sparse-MoE block

- Status: complete
- Disposition: correctness-repair — exact shared/routed block retained
- Date: 2026-09-03
- Parent experiments: FW-0009, FW-0010, FW-0011
- Exactness: L0 for the nine tested BF16 capture identities
- Hardware/runtime: Apple M1 Mac mini (`Macmini9,1`), 16 GiB, internal SSD

## Question and hypothesis

Can the exact routed mixture from FW-0011 be combined with Qwen4-Exp's real
shared expert and sigmoid gate to reproduce the complete sparse-MoE block for
one layer/token input? The hypothesis is that the shared MLP uses the same BF16
projection and SwiGLU boundaries, its scalar gate is sigmoid-rounded to BF16,
and the gated shared result is added to the BF16 routed mixture in BF16.

This is the narrowest complete MoE-block semantic needed before layer residual
integration, and it accounts for all ordinary expert payload rather than only
the routed bank.

## Frozen authority and baseline

- Checkpoint revision:
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Model lock SHA-256:
  `f87399e8659ab3274601fcd455b78b73c600f57e5fc1e91499eec3ac1f4b9444`
- FW-0011 mixture fixture SHA-256:
  `975a9982919297d37dd077f774693c782295cba496542c6adf278182e27b4d89`
- Reference: Transformers 5.16.1 `Qwen4ExpTextSparseMoeBlock.forward`
- Baseline commit: `86fdc45decf26f611fdf4166a19347eaa9cd5c4d`
- Python 3.11, Torch 2.14.0, Rust 1.96.0, macOS 26.6.2 build 25G83

Protocol deviation: implementation and execution occurred in one dirty
worktree rather than after a separate freeze commit. This is exact component
correctness evidence with no timing or endpoint claim.

## Method and commands

The generator recomputes FW-0011's routed mixture, reads the real layer-0
shared gate/up/down matrices and scalar shared-expert gate, and captures hashes
for shared gate, up, SwiGLU, down, gate logit, sigmoid, gated shared output,
routed mixture, and their final sum. Only identities are committed.

Native verification replays the already-gated routed mixture and independently
reads the four shared tensors from their exact safetensors extents. It uses the
source-derived PyTorch aarch64 BF16 reduction topology and rounds each published
BF16 operation boundary.

```shell
.venv/bin/python tools/generate_sparse_moe_block_fixture.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  --model-lock spec/model.lock.json \
  --output fixtures/sparse_moe/qwen3_8_flash_next_layer0.json

cargo run --release -- verify-sparse-moe \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json fixtures/router/qwen3_8_flash_next_real.json \
  fixtures/expert/qwen3_8_flash_next_real.json \
  fixtures/mixture/qwen3_8_flash_next_real.json \
  fixtures/sparse_moe/qwen3_8_flash_next_layer0.json \
  /Users/chad/Models/firewing/evidence/FW-0012/sparse-moe-verification-de4b8e4d.json
```

Batch size and concurrency are one. Accepted tokens and endpoint `A` are zero;
the routed component has `U=10`. Cache state, physical SSD bytes, endpoint wall
time, and TPS are unmeasured and no performance claim is made.

## Gates

- Exact checkpoint, model-lock, prior-fixture, shard, tensor, shape, dtype,
  input, and payload identities.
- Exact hashes for seven shared-path captures, the routed mixture, and final
  combined BF16 output.
- Deterministic regeneration of FW-0011 and FW-0012 fixtures.
- Excluded: layer normalization, hyper connections, attention/DeltaNet,
  residual integration, real layer activations, physical I/O, and TPS.

## Result

All seven shared-path capture hashes, the routed mixture, and final combined
output matched exactly. The routed experts contribute 98,304,000 logical
bytes; the shared MLP contributes 9,830,400 bytes; and the shared gate adds
5,120 bytes. The complete tested MoE block therefore consumes 108,139,520
logical source bytes. These are verified payload extents, not measured physical
I/O.

Fourteen Rust tests, twenty-three Python tests, strict Clippy, deterministic
regeneration of both dependent fixtures, and the release verifier passed.

- Fixture SHA-256:
  `a6a706f93b8e97603574594ec7a100b9d44090de06db7e72d41092d49d447990`
- External receipt SHA-256:
  `7455004941385b9476f058f1e363f3e2695f70f739861f31903b7928e018f891`

## Decision

Retain the exact sparse-MoE block as the slow native oracle. The next slice
must integrate the block with Qwen's layer-local norm, hyper-connection, and
residual semantics against a real reference input; no transport or accelerated
default is promoted here.
