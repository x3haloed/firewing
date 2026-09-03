# FW-0005 - Checkpoint-backed n-gram addressing

- Status: complete
- Disposition: correctness-repair — target semantic and physical layout retained
- Date: 2026-09-03
- Parent experiments: FW-0001, FW-0004
- Exactness: L0 integer addresses and checkpoint metadata bytes
- Hardware/runtime: Apple M1 Mac mini (`Macmini9,1`), 16 GiB, internal SSD

## Question and hypothesis

Can a small independent native implementation reproduce Qwen4-Exp's 16
bigram/trigram table addresses and resolve each global address to the original
128-part checkpoint layout? The hypothesis was that the official equations,
configuration defaults, three stored int64 buffers, and numeric shard
concatenation provide an exact, allocation-free address oracle before any
102.4 GB table reader is built.

This follows Prismwing PW-0003's reusable method: a readable Python oracle,
multiple deterministic cases, a separate scalar implementation, and strict
schema identity. No MiMo model semantics or fixture values were copied.

## Frozen authority and baseline

- Checkpoint: `Qwen/Qwen3.8-Flash-Next` revision
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Model lock SHA-256:
  `f87399e8659ab3274601fcd455b78b73c600f57e5fc1e91499eec3ac1f4b9444`
- Tensor index SHA-256:
  `99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de`
- Reference: Transformers 5.16.1
  `transformers.models.qwen4_exp.modeling_qwen4_exp`
- Reference source SHA-256:
  `77fec77d87f2a0eb23b95fa04276fb5779698a7c7f523cf5061e49c118bcc459`
- Transformers conversion mapping SHA-256:
  `319e24abfac50cd2464dfc25c336c0f71dc08b6273fef65dfdf9feb414608577`
- Torch 2.14.0; Rust 1.96.0; macOS 26.6.2 build 25G83
- Baseline commit:
  `009e55161d7ba5b99dd62ed3981f7d0db7469622`

Protocol deviation: this correctness slice was implemented and executed in one
dirty worktree rather than freezing the hypothesis in a preceding clean
commit. The fixture, commands, identities, and resulting receipt are preserved
here; there was no performance comparison whose order could be biased.

## Method and commands

The Python generator imports the pinned upstream multiplier and prime helpers,
reproduces the published EOS-aware shift and hash equations with Torch int64
operations, and refuses any unexpected configuration. It compares all
generated metadata values to the actual checkpoint payloads. It also validates
the index and safetensors headers for table parts 0 through 127 without loading
their BF16 payloads.

Five cases cover initial EOS context, an ordinary sequence, an EOS segment
boundary, incremental context, and token IDs at the vocabulary ceiling. Rust
independently implements SplitMix64, prime selection, signed wrapping hash/XOR,
EOS segmentation, remainder, and global-to-part mapping.

```shell
.venv/bin/python tools/generate_ngram_address_fixture.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  --model-lock spec/model.lock.json \
  --output fixtures/ngram/qwen3_8_flash_next.json

cargo run --release -- verify-ngram \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json fixtures/ngram/qwen3_8_flash_next.json \
  /Users/chad/Models/firewing/evidence/FW-0005/ngram-verification-de4b8e4d.json
```

This was a cold/warm-neutral correctness run at batch size one and concurrency
one. It accepted zero output tokens; `A`, `U`, endpoint bytes, TPS, memory,
power, and thermal behavior are not applicable and no table payload pages were
read.

## Gates

- Exact equality for all generated and checkpoint-stored metadata values.
- Exact equality for every reference and native global row.
- Exact equality for every logical-to-physical table-part address.
- Fail closed on model, revision, dependency version, hashes, configuration,
  tensor names, dtype, shape, offsets, shard size, or fixture dimensions.
- Excluded: embedding values, PLE projection, model output, filesystem read
  amplification, cache behavior, and endpoint throughput.

## Result

All five cases passed: 14 token positions and 224 exact head addresses. The
native verifier also matched the three int64 checkpoint buffers and validated
all 128 BF16 table descriptors. The table contains 320,001,536 logical padded
rows, with 2,500,012 rows of width 160 in each physical part.

- Fixture SHA-256:
  `cdfd44ad62dc8fe60219b1f97e966faf776e49f30e7f46fb11f07d7e913a1430`
- External verification receipt SHA-256:
  `b91fe54d13d18f9aa3dec94e1687dee598ea2d5d795c48de5612f8fe32a3c442`
- Python tests: 14 passed
- Rust tests: 5 passed
- `cargo clippy --all-targets -- -D warnings`: passed

## Decision

Retain the native address calculation and descriptor verification as the
semantic front end for a future sparse table reader. Confidence is high for
integer address and original-checkpoint part selection, bounded by the five
fixture cases and the pinned Transformers implementation. No branch is
promoted for storage or performance: actual row bytes, PLE math, cold page
amplification, and cache behavior remain follow-up work. No prior experiment is
superseded or reversed.
