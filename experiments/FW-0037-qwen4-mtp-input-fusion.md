# FW-0037 - Qwen4-Exp MTP source lock and input fusion

- Status: completed
- Disposition: correctness milestone
- Date: 2026-09-03
- Parent experiments: FW-0029, FW-0036
- Exactness: source-derived Qwen4-Exp MTP semantics with real BF16 checkpoint weights
- Hardware: Apple M1 Mac mini, 16 GiB, internal SSD, no companion hardware

## Question

What exact hidden state enters Qwen3.8-Flash-Next's trained MTP layer, and can
Firewing reproduce that boundary before implementing speculative acceptance or
making an `A/U` claim?

Transformers 5.16.1 implements the target model but intentionally ignores
`mtp.*` weights during ordinary loading. The similar Qwen3-Next MTP path is not
an adequate Qwen4-Exp authority because the target has four hyper-connection
streams. SGLang's official Qwen3.8-Flash-Next cookbook instead identifies model
support PR #36497. Firewing pins its immutable head commit and exact source-file
hashes in `spec/sglang-qwen4-exp-mtp.lock.json`.

## Frozen authority and method

- Clean implementation commit:
  `3244bac498d13100bbe3b608ed55f002bc54ee32`
- SGLang source commit:
  `78c5024e9d9f589dcb4deb7f4ba4fb23f7e85385`
- Qwen4-Exp MTP source SHA-256:
  `2b2ec09230875279a75ae651a1d9e1d88999bc89748e9d0cb6b4a768ffc0e54e`
- Source-lock SHA-256:
  `8160eed0480d8a5bacad0803569f2031626dde26a87eff2b79e62058f7699282`
- Checkpoint revision:
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Boundary dtype: BF16
- Batch size: 1 deterministic row
- Concurrency: 1
- Accepted tokens: 0
- `A=0`, `U=0`, and `performance_claim=null`

The source specifies this Qwen4-specific fusion:

1. normalize the next-token 2,560-wide embedding with
   `mtp.pre_fc_norm_embedding`;
2. normalize the complete 10,240-wide target hidden state with
   `mtp.pre_fc_norm_hidden` as one vector, not as four separate norms;
3. project the embedding with `mtp.fc_embedding`;
4. view the target hidden state as four 2,560-wide streams and apply the same
   `mtp.fc_hidden` matrix to each stream; and
5. add the embedding projection to each projected target stream at a BF16
   boundary.

The Python generator explicitly evaluates that equation with PyTorch 2.14.0
and the four real checkpoint tensors. It commits only payload and capture
hashes. The Rust verifier independently reads bounded tensor payloads, uses the
previously verified PyTorch-aarch64 reduction topology, and fails closed on the
model, source lock, source blob, checkpoint metadata, tensor shapes, shard
identity, or any capture mismatch.

```shell
git -C /tmp/firewing-sglang.ilGHpN fetch --depth=1 \
  origin pull/36497/head:refs/remotes/origin/pr-36497

.venv/bin/python tools/generate_mtp_input_fusion_fixture.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  --model-lock spec/model.lock.json \
  --source-lock spec/sglang-qwen4-exp-mtp.lock.json \
  --output fixtures/mtp/qwen3_8_flash_next_input_fusion.json

target/release/firewing verify-mtp-input-fusion \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  spec/sglang-qwen4-exp-mtp.lock.json \
  fixtures/mtp/qwen3_8_flash_next_input_fusion.json \
  /Users/chad/Models/firewing/evidence/FW-0037/mtp-input-fusion-3244bac.json
```

## Result

The independent verifier matches seven exact BF16 hashes: both deterministic
inputs, both normalized inputs, both projections, and the final four-stream
fused hidden state. It authenticates all four real tensors and 26,240,000
logical payload bytes. The committed tests also distinguish the 10,240-wide
target norm from four grouped norms and verify that the embedding projection
is shared across every stream.

Raw receipt:
`/Users/chad/Models/firewing/evidence/FW-0037/mtp-input-fusion-3244bac.json`

Receipt SHA-256:
`4de435309c92acd7ce6e6665042409a5551ddf0c599b61789012ad861eed26d1`

The repository has 51 Rust tests and strict Clippy passes.

## Decision and follow-up

Promote the source lock and real-weight input fusion as the MTP correctness
boundary. This resolves the initial semantic-authority blocker and rules out
copying Qwen3-Next's two-vector concatenation into Qwen4-Exp.

This is not an MTP-layer, proposal-quality, acceptance, route-union, or TPS
result. The next correctness step is the complete checkpoint MTP decoder layer
and shared target LM head on a target-derived hidden state. Only after that
path matches may Firewing collect recursive proposals, verifier acceptance
`A`, and target-plus-draft expert union `U`. Prismwing's useful transferable
lesson is the ordering—prove native proposal semantics, then measure causal
windows and exact verifier-only commit—not its MiMo recurrence, tensor values,
or acceptance rates.
