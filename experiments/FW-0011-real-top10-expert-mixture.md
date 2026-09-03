# FW-0011 - Real top-10 expert mixture

- Status: complete
- Disposition: correctness-repair — exact source schedule retained
- Date: 2026-09-03
- Parent experiments: FW-0009, FW-0010
- Exactness: L0 for ten weighted expert outputs and the tested mixture
- Hardware/runtime: Apple M1 Mac mini (`Macmini9,1`), 16 GiB, internal SSD

## Question and hypothesis

Can the native path extend FW-0010 from one selected expert to Qwen4-Exp's
complete top-10 mixture for one real layer/token input? The hypothesis is that
the source loop executes active experts in ascending expert ID—not router rank
order—and accumulates each weighted BF16 contribution into a BF16 destination.

This closes the smallest remaining semantic gap between a validated router and
a complete routed MLP result while exercising the full 98,304,000-byte
layer/token source payload implied by top-10 routing.

## Frozen authority and baseline

- Checkpoint revision:
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Model lock SHA-256:
  `f87399e8659ab3274601fcd455b78b73c600f57e5fc1e91499eec3ac1f4b9444`
- Router fixture SHA-256:
  `bbf73e4361d81b60e8ab49898422ba5fd0d2a90f312433ac43a013e48c89d39b`
- Single-expert fixture SHA-256:
  `10315f99986464e85e186cc32d55488d9c68f7db0979f5cef1411c6b7e8a4752`
- Reference: Transformers 5.16.1 `Qwen4ExpTextExperts.forward`
- Baseline commit: `9859b7f97d34ca068309992ed306c5185acb20d2`
- Python 3.11, Torch 2.14.0, Rust 1.96.0, macOS 26.6.2 build 25G83

Protocol deviation: implementation and execution occurred in one dirty
worktree rather than after a separate freeze commit. This is an exact
correctness slice without timing or promotion of a performance default.

## Method and commands

The generator reuses FW-0009's layer-0 deterministic BF16 input, selected
expert IDs, and normalized BF16 weights. It reads only the ten selected slices,
executes each expert using FW-0010's boundary semantics, and reproduces the
published `expert_hit.nonzero()` schedule: ascending expert IDs. Each weighted
result is accumulated by `index_add_` into the BF16 output.

The fixture stores payload and output hashes only. The native verifier checks
the authority chain, reads and hashes all twenty selected tensor slices,
recomputes ten weighted outputs, and rounds the mixture after every expert in
the same source order.

```shell
.venv/bin/python tools/generate_mixture_fixture.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  --model-lock spec/model.lock.json \
  --output fixtures/mixture/qwen3_8_flash_next_real.json

cargo run --release -- verify-mixture \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json fixtures/router/qwen3_8_flash_next_real.json \
  fixtures/expert/qwen3_8_flash_next_real.json \
  fixtures/mixture/qwen3_8_flash_next_real.json \
  /Users/chad/Models/firewing/evidence/FW-0011/mixture-verification-de4b8e4d.json
```

Batch size and concurrency are one. Accepted tokens and endpoint `A` are zero;
the component expert union is `U=10`. Cache state, physical SSD bytes, endpoint
wall time, and TPS are unmeasured and no performance claim is made.

## Gates

- Exact checkpoint, model-lock, router, single-expert, shard, tensor, input,
  selected-ID, route-weight, and per-expert payload identities.
- Exact hashes for all ten weighted BF16 outputs and the final BF16 mixture.
- Exact source expert execution order and deterministic regeneration.
- Tiny fixtures must prove that BF16 mixture accumulation is order-sensitive.
- Excluded: real layer activations, shared expert, residuals, SSD traffic,
  accelerated kernels, other layers, whole-model parity, and TPS.

## Result

All ten weighted expert hashes and the final mixture hash matched exactly. The
router selection order
`[376,349,384,191,211,363,337,206,247,295]` becomes source execution order
`[191,206,211,247,295,337,349,363,376,384]`. Tiny fixtures demonstrate that
changing this order can change the BF16 result.

The verifier consumed exactly 65,536,000 gate/up bytes and 32,768,000 down
bytes, totaling 98,304,000 selected expert bytes. This is logical source
payload, not measured physical I/O. Thirteen Rust tests, twenty-two Python
tests, strict Clippy, deterministic regeneration, and the release verifier
passed.

- Fixture SHA-256:
  `975a9982919297d37dd077f774693c782295cba496542c6adf278182e27b4d89`
- External receipt SHA-256:
  `6a6053c922f706663d240de93577ab35822a1689b42a7d2970f694b2af64eb59`

## Decision

Retain the exact top-10 mixture path as the slow native routed-MLP oracle. The
next correctness slice should add the layer's shared expert and published
shared/routed combination, then residual integration. This result promotes no
transport or acceleration branch and makes no endpoint claim.
