# FW-0009 - Real-checkpoint top-10 router semantics

- Status: complete
- Disposition: correctness-repair — independent scalar router retained
- Date: 2026-09-03
- Parent experiments: FW-0001, FW-0008
- Exactness: L0 ordered expert IDs and selected BF16 outputs
- Hardware/runtime: Apple M1 Mac mini (`Macmini9,1`), 16 GiB, internal SSD

## Question and hypothesis

Can an independent native implementation reproduce Qwen4-Exp's router
precision path and ordered top-10 selections against real checkpoint gate
matrices? The hypothesis was that BF16 inputs and weights, F32 accumulation,
BF16 logits, F32 softmax, top-10 selection, selected-probability
renormalization, and BF16 scores are sufficient to match the pinned executable
reference.

This is the next high-leverage correctness slice because routed-expert traffic,
not n-gram rows, dominates the source-byte hypothesis.

## Frozen authority and baseline

- Checkpoint revision:
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Model lock SHA-256:
  `f87399e8659ab3274601fcd455b78b73c600f57e5fc1e91499eec3ac1f4b9444`
- Reference: Transformers 5.16.1
  `Qwen4ExpTextTopKRouter`
- Baseline commit:
  `25d6f76ddd75b35a25f0c6705503c730ea618d9a`
- Python 3.11, Torch 2.14.0, Rust 1.96.0, macOS 26.6.2 build 25G83

Protocol deviation: the fixture and verifier were implemented and executed in
one dirty worktree rather than a prior protocol-freeze commit. This is a
correctness slice with no timed comparison; identities and raw receipt are
preserved.

## Method and commands

Three deterministic, exactly reproducible BF16 input formulas exercise real
512-by-2,560 BF16 gate matrices from layers 0, 1, and 47. The third input is
sparse to broaden accumulation behavior. The Python fixture generator runs the
pinned router and stores only hashes, selected IDs, selected logits, and
normalized scores. It commits no checkpoint weight bytes.

Rust independently regenerates and hashes each input, reads and hashes each
2,621,440-byte matrix directly from safetensors, executes scalar routing, and
requires the exact ordered top-10 list. Frozen maximum absolute tolerances were
one BF16-scale step: 0.00390625 for selected logits and 0.001953125 for scores.

```shell
.venv/bin/python tools/generate_router_fixture.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  --model-lock spec/model.lock.json \
  --output fixtures/router/qwen3_8_flash_next_real.json

cargo run --release -- verify-router \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json fixtures/router/qwen3_8_flash_next_real.json \
  /Users/chad/Models/firewing/evidence/FW-0009/router-verification-de4b8e4d.json
```

Batch size and concurrency are one. Accepted tokens, `A`, `U`, endpoint bytes,
and TPS are zero or not applicable. Cache state is uncontrolled and no timing
claim is made.

## Gates

- Exact fixture/checkpoint/model-lock identity and tensor payload hashes.
- Exact ordered top-10 expert IDs for all three cases.
- Selected logit absolute error at most 0.00390625.
- Normalized score absolute error at most 0.001953125.
- Fail closed on unknown precision, dimensions, tensor names, input formulas,
  hashes, or numerical mismatch.
- Excluded: real hidden activations, all-layer coverage, expert MLP outputs,
  route union, SSD demand, whole-model parity, and endpoint TPS.

## Result

All three ordered expert lists matched exactly. Both the maximum selected-logit
error and maximum normalized-score error were `0.0`. Nine Rust tests and
eighteen Python tests passed; strict Clippy passed.

- Fixture SHA-256:
  `bbf73e4361d81b60e8ab49898422ba5fd0d2a90f312433ac43a013e48c89d39b`
- External receipt SHA-256:
  `37dd7916d2c15bb3f2c221b551a9b9f2a60251ed2efeca85703c42043c6a3d0c`

## Decision

Retain the scalar router as the readable real-weight oracle and loader seed.
Confidence is high for the tested precision and selection semantics, but the
three synthetic hidden states do not establish production route frequencies or
expert-set union. The next correctness step is a routed expert MLP fixture or a
real layer-local activation fixture; no performance branch is promoted.
