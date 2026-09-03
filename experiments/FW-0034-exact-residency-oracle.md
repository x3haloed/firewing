# FW-0034 - Exact future-aware residency oracle

- Status: completed
- Disposition: 10 GiB rejected; 12 GiB analytical survivor superseded and
  rejected by FW-0036
- Date: 2026-09-03
- Parent experiments: FW-0013, FW-0033
- Exactness: L0 source-byte and route accounting; impossible-favorable model
- Hardware premise: Apple M1 Mac mini, 16 GiB, internal SSD, no companion
  hardware

## Question

Can a perfect exact-BF16 resident set reduce Firewing's authenticated
two-position route trace enough to retain any storage-only path to Firewing 4?

The oracle pins all 8,623,999,000 ordinary fixed bytes, gives the expert cache
free future-aware initial contents, uses exact Belady eviction after every
ten-expert layer demand, and charges misses at FW-0013's optimistic
12.665740063 GB/s eight-reader transport-only proxy. Compute, synchronization,
n-gram traffic, allocation granularity, installation, prefetch, and runtime
buffers are free. These grants make the result an upper bound, not projected
endpoint performance.

## Frozen authority and method

- Endpoint fixture SHA-256:
  `2eb3712c7959837a24fe6db5b5d7b1f87c9926b4dd80eae524590e43287331ca`
- FW-0001 census SHA-256:
  `043b5b45edd1f4aeb628a66b00fec60c035204c30a48d34d0a95f3e10d0bd937`
- FW-0013 acquisition report SHA-256:
  `b8e5a175c0402bced494ebb1cc4a61f903f2ff8a1a094fa8a17d043311f942b5`
- Clean implementation commit:
  `2fd14bc5df0e20b37f2ac0bdfeaa49396fa37fac`

The analyzer reconstructs two token-major positions, each containing 48
ordered layer demands of ten distinct `(layer, expert)` identities. It validates
every route against the unchanged FW-0029 fixture and finds 859 distinct
identities among 960 accesses. It independently sums the seven ordinary fixed
categories from the checkpoint census and derives the optimistic transport
rate from FW-0013's cold eight-worker median slowest-reader interval.

```shell
.venv/bin/python tools/analyze_exact_residency_oracle.py \
  --endpoint fixtures/endpoint/qwen3_8_flash_next_firewing_two_token.json \
  --census /Users/chad/Models/firewing/evidence/FW-0001/final-census-20260903T063239Z.json \
  --acquisition /Users/chad/Models/firewing/evidence/FW-0013/expert-acquisition.json \
  --implementation-commit 2fd14bc5df0e20b37f2ac0bdfeaa49396fa37fac \
  --output /Users/chad/Models/firewing/evidence/FW-0034/exact-residency-oracle-2fd14bc5.json
```

Batch and concurrency are one. This is a two-position authority, not accepted
generation, so `A=0`, `U=0`, and performance claim is null.

## Result

| Resident allowance | Fixed bytes | Expert slots | Misses | Miss bytes | Aggregate storage-only TPS | Token TPS | Decision |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 10 GiB | 8.624 GB | 214 | 645 | 6.341 GB | 3.995118 | 4.844, 3.400 | reject 4-TPS aggregate |
| 12 GiB | 8.624 GB | 433 | 426 | 4.188 GB | 6.048947 | 27.413, 3.400 | analytical survivor |

The 10-GiB scenario fails Firewing 4 before any compute. The 12-GiB scenario
passes the analogous 4-TPS aggregate and 3-TPS minimum-token screens, but only
with 12,880,562,200 resident source bytes and 4,339,688 bytes unallocated.
It therefore leaves essentially no residency room, only about 1 GiB below the
13-GiB process peak ceiling, and no charged time for execution. The second
token still incurs 379 misses because only 101 of the 960 accesses reuse a
layer-qualified identity across the pair.

Raw receipt:
`/Users/chad/Models/firewing/evidence/FW-0034/exact-residency-oracle-2fd14bc5.json`

Receipt SHA-256:
`19bd38ecc103a80fafc0085063123b86ddaa2aa5365c2fbdf147dae73c6168da`

The analyzer has three focused tests and the repository now has 72 Python and
46 Rust tests; strict Clippy passes.

## Decision

Do not implement the rejected 10-GiB cache and do not promote the 12-GiB
oracle into a runtime. Preserve 12 GiB as a fragile survivor whose next gate is
an enclosing compute-and-buffer bound. A real cache would also require the
Darwin pressure observer and warning eviction required by `TARGET.md`.

The cheapest next test is an exact production-shaped top-10 Metal MoE path for
one real layer. FW-0031's serialized single-expert result extrapolates to about
739 ms per token across 48 layers before fixed attention/shared work, already
above the 250-ms target, but a fused or concurrent ten-expert implementation
could change that premise. Measure it before building multi-gigabyte residency.

FW-0036 subsequently measured the exact miss payloads, bounded installation
copies, and exact routed Metal load concurrently. Its favorable two-position
bound reached only 1.596 diagnostic TPS, so the 12-GiB raw-BF16 survivor is now
rejected. This record remains the analytical precursor rather than being
reinterpreted as runtime evidence.
