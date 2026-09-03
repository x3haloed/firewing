# FW-0036 - Exact storage/compute overlap bound

- Status: completed
- Disposition: rejected for the 12-GiB raw-BF16 residency branch
- Date: 2026-09-03
- Parent experiments: FW-0013, FW-0034, FW-0035
- Exactness: exact source payloads and exact routed Metal arithmetic; favorable
  two-position contention proxy
- Hardware: Apple M1 Mac mini (`Macmini9,1`), 16 GiB, internal SSD, no
  companion hardware

## Question and hypothesis

Can perfect overlap between FW-0034's exact 12-GiB Belady miss schedule and
FW-0035's exact routed Metal compute preserve a path to Firewing 4?

FW-0034 projected the second position's 3,725,721,600 miss bytes at about
294 ms by scaling FW-0013's 372.548-ms slowest-worker `pread` interval for a
4,718,592,000-byte trace. FW-0035 then showed that routed compute alone costs
about 175 ms per token. The hypothesis was that aggressive overlap would
expose the larger interval and narrowly retain the 3-TPS p10 gate, leaving the
aggregate near 4 TPS before fixed work.

This experiment gives the branch every important advantage that can be tested
without allocating the unsafe 12.881-GB analytical resident set:

- all fixed matrices and all simulated cache hits are free;
- the 433-expert initial cache is free and future-aware;
- exact Belady eviction, cache metadata, and victim handling are free;
- all misses for a token may begin before layer dependencies reveal its route;
- eight workers read page-aligned ranges with `F_NOCACHE=1` and
  `F_RDAHEAD=0`;
- each miss is copied into preallocated page-aligned installation staging;
- exact layer-0 top-10 Metal work repeats 48 times as the routed-compute load;
  and
- attention, shared experts, routers, n-gram lookup, final projection,
  sampling, and every other endpoint cost are free.

Failure under these grants rejects this raw-BF16 residency design on the
frozen trace. Passing would only justify a real bounded cache experiment.

## Frozen authority and method

- Clean implementation commit:
  `905db3bee2b53b325510610ae6c9298e601eeb0d`
- Endpoint fixture SHA-256:
  `2eb3712c7959837a24fe6db5b5d7b1f87c9926b4dd80eae524590e43287331ca`
- Model lock SHA-256:
  `f87399e8659ab3274601fcd455b78b73c600f57e5fc1e91499eec3ac1f4b9444`
- Cache capacity: 433 exact 9,830,400-byte experts
- Misses by position: 47 and 379
- Logical miss bytes by position: 462,028,800 and 3,725,721,600
- Unique payload extents authenticated before timing: 852
- Unique payload bytes authenticated before timing: 4,187,750,400
- Persistent exact Metal contention workload: 98,398,736 bytes
- Maximum phase-scoped read/install staging: 105,119,744 bytes
- Batch size: 1
- Concurrency: 1
- Accepted tokens: 0
- `A=0`, `U=0`, and `performance_claim=null`

For each position, three storage-only controls and three overlap candidates
run in the interleaved order control/candidate/candidate/control/control/
candidate. Every trial invalidates every selected aligned range and requires
zero resident page instances afterward. Eight worker threads then read the
exact real extents and copy their logical payloads into bounded installation
staging. Candidate trials concurrently execute and re-verify the exact
FW-0035 mixture 48 times. Process physical reads must be nonzero and reconcile
with the widened request ledger.

```shell
cargo build --release
target/release/firewing bench-exact-overlap-bound \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  fixtures/router/qwen3_8_flash_next_real.json \
  fixtures/expert/qwen3_8_flash_next_real.json \
  fixtures/mixture/qwen3_8_flash_next_real.json \
  fixtures/endpoint/qwen3_8_flash_next_firewing_two_token.json \
  kernels/bf16_gemv.metal \
  905db3bee2b53b325510610ae6c9298e601eeb0d \
  /Users/chad/Models/firewing/evidence/FW-0036/exact-overlap-bound-905db3be.json
```

## Result

| Position | Misses | Physical bytes/trial | Storage-only median | Overlap median |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 47 | 463,568,896 | 133.227 ms | 189.433 ms |
| 1 | 379 | 3,738,140,672 | 1,063.343 ms | 1,064.075 ms |

The three paired two-position overlap rates were 1.593380, 1.600681, and
1.596034 diagnostic TPS. Their p10/median/p90 are **1.593380 / 1.596034 /
1.600681**, all far below Firewing 4's 3/4-TPS gates before any fixed endpoint
work. The second-position storage interval completely dominates its 204–207 ms
concurrent routed compute. Position zero exposes the 189-ms compute interval
instead of its 133-ms storage control.

Every trial reported exactly 463,568,896 or 3,738,140,672 process physical read
bytes, matching its widened request ledger. Installation copies were not
optimized away: the maximum worker spent about 6.7–14.2 ms installing position
zero and 55.3–68.8 ms installing position one. The final run retained 57%
system memory free, added no swap or throttled pages, and peaked at about
190.7 MB process physical footprint at the phase checkpoints.

### Resolved decision and open causal detail

Expected: FW-0013's paced slowest-reader interval implied about 294 ms for the
second-position miss bytes.

Observed: the actual unpaced endpoint miss order takes 1,063 ms storage-only
and 1,064 ms while overlapped, with exact physical-byte accounting.

The prior proxy is therefore invalid for projecting an unpaced cache-refill
critical path. FW-0013 measured each worker's `pread` time inside a loop whose
per-extent SHA-256 work dominated and paced subsequent reads; removing that
work changes the issued I/O workload. The exact share attributable to pacing
versus this endpoint's extent order remains unresolved, but cannot rescue the
measured branch: the production-relevant no-hot-path-hash schedule is the slow
one.

Raw receipt:
`/Users/chad/Models/firewing/evidence/FW-0036/exact-overlap-bound-905db3be.json`

Receipt SHA-256:
`967e7444c64d54e905b9e9487055efe13b33175f06691cd7fa17409409b7e8ac`

The repository has 49 Rust tests and strict Clippy passes.

## Decision, confidence, and follow-up

Reject the 12-GiB raw-BF16 exact-residency branch represented by FW-0034. Do
not allocate the multi-gigabyte cache and do not use FW-0013's slowest-reader
proxy for unpaced refill projections. Confidence is high for this frozen
two-position route: exact miss payloads, physical reads, installation traffic,
and concurrent exact compute all repeat tightly under deliberately favorable
conditions.

This does not reject Firewing 4 globally. Two positions do not establish the
long-run route distribution, and trained MTP may amortize a different expert
union across accepted tokens. The next useful branches must reduce bytes per
accepted token—first an exact MTP route-union/acceptance oracle, then lossless
expert representation if MTP cannot close the gap. A longer ordinary-decode
route trace becomes worthwhile only when a viable runtime can generate it; it
must not reinterpret this two-position rejection as a production hit-rate
claim.

Reusable lesson: measure the whole unpaced refill critical path. Summing or
scaling `pread` intervals collected inside a hash-paced diagnostic can
materially overstate achievable transport.
