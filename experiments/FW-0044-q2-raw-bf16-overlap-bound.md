# FW-0044 - Width-two raw-BF16 overlap bound

- Status: completed
- Disposition: rejected for raw-BF16 q2 residency on the frozen transaction
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

- Clean implementation commit:
  `e5012d3cae49c4a7e32ccb60d4d7e61b8af79f8d`
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
  e5012d3cae49c4a7e32ccb60d4d7e61b8af79f8d \
  /Users/chad/Models/firewing/evidence/FW-0044/q2-overlap-bound-e5012d3.json
```

## Result

| Target row | Misses | Physical bytes/trial | Storage-only median | Overlap median |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 47 | 463,568,896 | 133.818 ms | 190.750 ms |
| 1 | 207 | 2,041,675,776 | 581.664 ms | 580.886 ms |

The three paired accepted-throughput bounds were 2.583371, 2.601109, and
2.598053 TPS. Their p10/median/p90 are **2.583371 / 2.598053 / 2.601109**.
All miss payloads authenticated exactly, every timed read reported the expected
nonzero widened physical-byte count, every cold preparation reached zero
resident page instances, and exact routed Metal ran concurrently.

The first target row is compute-bound at about 191 ms. The 207-miss second row
is SSD-bound at about 581 ms: overlapping roughly 204 ms of exact routed Metal
does not improve its storage interval. Together they require about 770 ms to
commit `A=2`, before charging any MTP or fixed endpoint work.

Host safety passed with 52% system memory free at completion, no swap growth,
no new throttled pages, and a 190.6-MB final process physical footprint. The
measurement ran on macOS 26.6.2 build 25G83 with Rust 1.96.0.

Raw receipt:
`/Users/chad/Models/firewing/evidence/FW-0044/q2-overlap-bound-e5012d3.json`

Receipt SHA-256:
`6aa3f7cc04d35a8686ceb6c3c0b55f22b548129b67db242975b6693c79d5d6f9`

The repository has 63 passing Rust tests and strict Clippy passes.

## Decision, confidence, and follow-up

Reject raw-BF16 q2 residency on this frozen accepted transaction. It misses
both Firewing 4 gates under grants strictly more favorable than a realizable
runtime, so implementing cache slots, drafting, dense work, and synchronization
cannot rescue it. Confidence is high for the frozen route and hardware state;
one prompt prefix still does not establish a production route distribution.

This result supersedes FW-0043's raw-BF16 q2 runtime follow-up, not its width
comparison: q2 remains materially better than q4, but neither raw-BF16 branch
is viable here. The next branch must reduce expert bytes per accepted token.
Test a source-faithful lossless expert representation with exact byte/output
round trips and a cheap capacity/transport bound before building a repeated
endpoint. Approximate formats remain explicitly modified until they pass the
full required fidelity suite.

Reusable lesson: speculative acceptance and route union must be converted into
physical miss traffic. `A/U > 1` did not imply viable accepted TPS once the
remaining raw expert union was issued against the SSD.
