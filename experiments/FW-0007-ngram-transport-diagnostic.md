# FW-0007 - N-gram sparse transport diagnostic

- Status: planned and frozen
- Disposition: unexecuted
- Date: 2026-09-03
- Parent experiment: FW-0006
- Exactness: L0 row bytes
- Hardware/runtime: Apple M1 Mac mini (`Macmini9,1`), 16 GiB, internal SSD

## Question and hypothesis

What latency and physical disk-read amplification does the verified Qwen
n-gram schedule exhibit under warm cacheable 320-byte positional reads and a
page-aligned uncached transport? The hypothesis is that widening each request
to its containing 16 KiB page or pages will keep uncached n-gram demand small
relative to Firewing's 250 ms/token completion budget, while producing honest
Darwin disk counters absent from FW-0006.

This is the cheapest useful falsification of the assumption that the 102.4 GB
table is operationally cheap because each token selects only 5,120 useful
bytes. It cannot establish endpoint TPS.

## Frozen authority and baseline

- Checkpoint: `Qwen/Qwen3.8-Flash-Next` revision
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Address fixture SHA-256:
  `cdfd44ad62dc8fe60219b1f97e966faf776e49f30e7f46fb11f07d7e913a1430`
- Row-hash fixture SHA-256:
  `8896518e313ff0cb9d847fe5f6170b8f56ec168196c50d18a527ef89e3e2ffce`
- Model lock SHA-256:
  `f87399e8659ab3274601fcd455b78b73c600f57e5fc1e91499eec3ac1f4b9444`
- Baseline implementation commit: to be filled by the protocol-freeze commit
- Rust 1.96.0; macOS 26.6.2 build 25G83

Generic Darwin transport mechanics were adapted from Prismwing commit
`c87d0c1aa2c118f71ca5348434be35d02f62f031`; Qwen addresses and hashes remain
independently derived.

## Method and commands

The fixed trace is FW-0005/FW-0006's five cases: 14 token positions, 224 rows,
and 71,680 useful bytes per trial. Before timing, the executable reruns all
identity, descriptor, address, and row-hash checks. The cacheable control issues
exact 320-byte `pread` calls after that cache-influencing validation. The
uncached candidate opens separate descriptors, enables `F_NOCACHE`, disables
automatic read-ahead, and widens every request to 16 KiB boundaries using one
reused aligned buffer.

Each transport receives five unreported warmups followed by 30 serialized
measurements. Every timed row is SHA-256 checked. Each trial records complete
transport wall time, logical and widened bytes, calls, stream hash, and the
Darwin `proc_pid_rusage` physical disk-read delta.

```shell
cargo run --release -- bench-ngram-transport \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json fixtures/ngram/qwen3_8_flash_next.json \
  fixtures/ngram/qwen3_8_flash_next_row_hashes.json \
  IMPLEMENTATION_COMMIT \
  /Users/chad/Models/firewing/evidence/FW-0007/ngram-transport.json
```

Batch size and concurrency are one. Accepted tokens, `A`, and `U` are zero.
The run is not a complete inference request and does not exercise PLE math.

## Gates

- All 13,440 timed row reads (`224 * 30 * 2`) must match exact SHA-256 values.
- Each transport must retain all 30 declared trials; no outlier deletion.
- Uncached reports must show nonzero physical disk reads and exact widened-byte
  accounting. A zero counter makes physical amplification inconclusive.
- Continue with this transport if uncached median is at most 14 ms per
  14-position trace (1 ms/token) and p90 is at most 28 ms.
- Kill the claim that the n-gram table is a minor cold-path cost if uncached
  median exceeds 70 ms per trace (5 ms/token) or physical reads exceed twice
  the declared widened bytes without an explained counter granularity effect.
- No result promotes a runtime default or counts as accepted TPS without a
  complete inference endpoint.

## Result

Unexecuted. Preserve raw output and fill this section only after running the
committed protocol from a clean worktree.

## Decision

Unexecuted. The sparse row reader remains the correctness baseline; no
transport is promoted yet.
