# FW-0006 - Exact sparse n-gram row reader

- Status: complete
- Disposition: correctness-repair — bounded target-faithful reader retained
- Date: 2026-09-03
- Parent experiment: FW-0005
- Exactness: L0 checkpoint row bytes, compared by SHA-256
- Hardware/runtime: Apple M1 Mac mini (`Macmini9,1`), 16 GiB, internal SSD

## Question and hypothesis

Can Firewing retrieve only the 16 Qwen n-gram rows selected for a token, from
the original 128-part safetensors layout, without mapping or materializing the
102.4 GB table? The hypothesis was that FW-0005's verified descriptors make an
exact bounded positional reader possible with one 320-byte logical request per
head.

## Frozen authority and baseline

- Checkpoint: `Qwen/Qwen3.8-Flash-Next` revision
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Address fixture SHA-256:
  `cdfd44ad62dc8fe60219b1f97e966faf776e49f30e7f46fb11f07d7e913a1430`
- Model lock SHA-256:
  `f87399e8659ab3274601fcd455b78b73c600f57e5fc1e91499eec3ac1f4b9444`
- Baseline commit:
  `c4eef4d04aa6edb9c9b0a20243487dd2504c9f17`
- Python 3.11, Torch 2.14.0, Transformers 5.16.1, Rust 1.96.0
- macOS 26.6.2 build 25G83

As in FW-0005, the correctness implementation and record were completed in one
dirty worktree. There was no ordered performance comparison.

## Method and commands

The Python oracle opens the verified shard files and seeks to each selected
row using the safetensors payload start, tensor-relative data offset, and
320-byte row stride. It commits only SHA-256 row identities. Qwen weight bytes
are neither printed nor stored in Git.

The Rust verifier first reruns FW-0005's complete identity, metadata, address,
and 128-part layout checks. It then independently seeks and reads each row,
hashes exactly 320 bytes, and compares the result. A committed synthetic
fixture made from invented bytes covers nonzero tensor offsets, every row, and
the out-of-bounds failure.

```shell
.venv/bin/python tools/generate_ngram_row_hash_fixture.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  fixtures/ngram/qwen3_8_flash_next.json \
  --output fixtures/ngram/qwen3_8_flash_next_row_hashes.json

cargo run --release -- verify-ngram-rows \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json fixtures/ngram/qwen3_8_flash_next.json \
  fixtures/ngram/qwen3_8_flash_next_row_hashes.json \
  /Users/chad/Models/firewing/evidence/FW-0006/ngram-row-verification-de4b8e4d.json
```

Batch size and concurrency were one. The run verified 14 token positions,
requested 71,680 logical table bytes, accepted zero output tokens, and did not
measure `A`, `U`, physical disk bytes, power, memory, or endpoint time. The OS
cache state was uncontrolled because this is a correctness result.

## Gates

- Exact SHA-256 equality for all 224 selected rows.
- Read exactly 320 logical bytes per requested row.
- Fail closed on every FW-0005 identity/layout condition, row-fixture identity,
  case dimensions, hashes, address mapping, bounds, truncation, or offset
  overflow.
- Commit no checkpoint weight bytes.
- Excluded: BF16 decoding, embedding concatenation/projection, measured storage
  amplification, cache policy, and endpoint TPS.

## Result

All 224 real row hashes matched. The verifier requested 71,680 bytes, exactly
`224 * 320`, and never allocated or mapped a table tensor. The six Rust tests
and sixteen Python tests passed, including the synthetic offset/bounds fixture;
strict Clippy also passed.

- Real row-hash fixture SHA-256:
  `8896518e313ff0cb9d847fe5f6170b8f56ec168196c50d18a527ef89e3e2ffce`
- Synthetic fixture SHA-256:
  `126c356c3a175af12c16bded5cd4369ae1589431c226506ea36583596ea14aa9`
- External verification receipt SHA-256:
  `e4d3afd1e972f60a67aa44a82a2024fab70056b778ec92ee045e21473c5c868a`

## Decision

Retain the exact sparse reader as the table-access correctness baseline. This
promotes no storage transport or performance default: 320 requested bytes may
cause much larger physical reads, and the current seek/read path is deliberately
simple. Confidence is high for byte selection and bounds behavior under the
pinned layout. The next experiment must measure page-aligned buffered and
uncached transports with Darwin disk counters before making an SSD-demand
claim.
