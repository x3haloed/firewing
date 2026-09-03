# FW-0044 - Width-two raw-BF16 overlap bound

- Status: implementation ready; measurement pending
- Disposition: pending
- Date: 2026-09-03
- Parent experiments: FW-0036, FW-0040, FW-0043
- Exactness: exact source payloads, exact accepted transaction, and exact routed
  Metal arithmetic; deliberately favorable target-verification contention bound
- Hardware: Apple M1 Mac mini (`Macmini9,1`), 16 GiB, internal SSD, no
  companion hardware

## Question and falsification rule

Can the exact first width-two MTP transaction preserve a path to Firewing 4
when its target-verification expert misses overlap routed Metal compute?

Reject raw-BF16 q2 residency for this frozen transaction if accepted-token
throughput is below the 3-TPS p10 gate or 4-TPS median gate even after every
unmeasured cost is made free. Passing only authorizes implementation of a real
stateful q2 path; it is not endpoint TPS or a production default.

## Frozen authority and method

- Endpoint fixture SHA-256:
  `e2ccf01a37cc5cb2cf44a30185850b8910b06233bc32d7ddaaeb537204daa899`
- Transaction fixture SHA-256:
  `9954668a28b64944c0830760a799383082e834be22106ec1613df12d748b9757`
- Model lock SHA-256:
  `f87399e8659ab3274601fcd455b78b73c600f57e5fc1e91499eec3ac1f4b9444`
- Exact target steps: ordinals 2 and 3 of the four-token endpoint
- Cache capacity: 433 exact 9,830,400-byte experts
- Exact Belady misses by target row: 47 and 207
- Logical miss bytes by target row: 462,028,800 and 2,034,892,800
- Batch size: 1
- Concurrency: 1
- Sampling: greedy
- Verification width: `q=2`
- Accepted tokens: `A=2`
- Target expert-union rows: 687
- Draft expert rows: 10
- `U=697/480=1.4520833333333334`
- `A/U=1.3773314203730271`
- `performance_claim=null`

The benchmark preserves FW-0036's interleaved cold physical-read controls and
overlap candidates. It authenticates every miss payload before timing, evicts
every timed aligned range, requires nonzero process physical reads, copies
payloads into bounded install staging, and overlaps each target row with 48
exact layer-0 top-10 Metal executions.

The bound grants the candidate a free future-aware initial cache, free cache
hits and fixed matrices, free MTP drafting (including its ten expert rows),
free eviction and cache-slot binding, foreknowledge of both target routes, and
free attention, shared experts, routers, n-gram work, output projection,
sampling, rollback, and synchronization. Thus failure is decisive for this
raw-BF16 branch, while success is only permission for a fuller implementation.

```shell
cargo build --release
target/release/firewing bench-q2-exact-overlap-bound \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  fixtures/router/qwen3_8_flash_next_real.json \
  fixtures/expert/qwen3_8_flash_next_real.json \
  fixtures/mixture/qwen3_8_flash_next_real.json \
  fixtures/endpoint/qwen3_8_flash_next_firewing_four_token.json \
  fixtures/mtp/qwen3_8_flash_next_first_transaction.json \
  kernels/bf16_gemv.metal \
  IMPLEMENTATION_COMMIT \
  /Users/chad/Models/firewing/evidence/FW-0044/q2-overlap-bound-COMMIT.json
```

## Result

Pending a clean-commit measurement.
