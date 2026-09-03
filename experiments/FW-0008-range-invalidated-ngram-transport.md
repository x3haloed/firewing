# FW-0008 - Range-invalidated n-gram transport

- Status: complete
- Disposition: conditional — valid fixed-trace diagnostic; transport not promoted
- Date: 2026-09-03
- Parent experiment: FW-0007
- Exactness: L0 row bytes
- Hardware/runtime: Apple M1 Mac mini (`Macmini9,1`), 16 GiB, internal SSD

## Question and hypothesis

Does explicit `MS_INVALIDATE` plus `MADV_DONTNEED` range preparation make the
same aligned Qwen row reads nonresident and observable in Darwin's physical
disk counter? FW-0007 falsified the assumption that `F_NOCACHE` alone bypasses
pages already warmed by correctness preflight. The successor hypothesis is
that explicit invalidation before every uncached timed trial closes that gap.

## Frozen authority and baseline

The checkpoint, model lock, address fixture, row-hash fixture, toolchains, and
hardware are identical to FW-0007. FW-0007 raw evidence SHA-256 is
`9b9b313b2a4b731a09d865035bbc8416a09fa288de7fce1fc33662eaf8277fb7`.
Implementation/protocol commit:
`5e2c84a7b3789e87df19ac976eff917c2f0f79b2`.

Range invalidation, residency probing, aligned buffers, and Darwin counters are
adapted from Prismwing commit
`c87d0c1aa2c118f71ca5348434be35d02f62f031`; Qwen addresses and correctness
hashes remain Firewing-specific.

## Method and commands

Retain FW-0007's exact fixed trace, two transports, five warmups, 30 measured
trials, serialization, counters, and correctness checks. Before each uncached
trial only:

1. map each 16 KiB-aligned read range and count resident page instances with
   `mincore`;
2. apply `MS_INVALIDATE` and `MADV_DONTNEED` to every range;
3. repeat `mincore` and record the post-invalidation count;
4. enable `F_NOCACHE` and `F_RDAHEAD=0` on all source descriptors;
5. start the disk counter and timer, then perform the aligned reads.

Cold preparation time is reported but remains outside transport wall time.

```shell
cargo run --release -- bench-ngram-transport \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json fixtures/ngram/qwen3_8_flash_next.json \
  fixtures/ngram/qwen3_8_flash_next_row_hashes.json \
  IMPLEMENTATION_COMMIT \
  /Users/chad/Models/firewing/evidence/FW-0008/ngram-transport.json
```

Batch size and concurrency are one. Accepted tokens, `A`, and `U` are zero.

## Gates

- All 13,440 timed row reads must match exact SHA-256 values; preserve all 60
  measured trials.
- Every uncached trial must report zero resident page instances after
  invalidation. Otherwise the cold-state mechanism is rejected.
- Every uncached trial must report nonzero physical disk bytes. A zero makes
  SSD amplification inconclusive and rejects the mechanism.
- Continue if uncached median is at most 14 ms per 14-position trace and p90 is
  at most 28 ms.
- Reject the minor-cost hypothesis if median exceeds 70 ms per trace or median
  physical bytes exceed twice the declared widened bytes without an explained
  counter granularity effect.
- Do not promote endpoint TPS, a runtime transport default, or production-trace
  representativeness from this diagnostic.

## Result

All 13,440 timed row reads matched. Every one of the 30 invalidated trials
reported zero resident page instances before timing and exactly 3,719,168
physical disk bytes, equal to the declared widened reads. Explicit invalidation
therefore repaired FW-0007's cache-state ambiguity.

| Transport | Median | p10 | p90 | Physical bytes median |
| --- | ---: | ---: | ---: | ---: |
| Warm cacheable exact `pread` | 0.7365 ms | 0.6449 ms | 0.8984 ms | 0 |
| Invalidated aligned uncached | 22.0785 ms | 21.9508 ms | 22.2274 ms | 3,719,168 |

Across 14 positions this is 0.0526 warm ms/token and 1.5770 uncached
ms/token. The uncached transport moved 265,654.9 physical bytes/token for
5,120 useful bytes/token, a 51.886x observed amplification on this trace.
Median excluded cold preparation was 0.5122 ms per trace.

The uncached p90 passed the 28 ms gate, but the median missed the 14 ms
continuation threshold. It remained well below the 70 ms minor-cost kill
threshold, and physical bytes exactly matched rather than exceeding the
widened declaration.

- Raw report SHA-256:
  `5cc08a817b3ec711e48cedc76ed72c8e36bdfbba38b7bab12a50af502909a562`
- Stream SHA-256 in every trial:
  `95129dd9c62501a44f1c987c8ac5d871011c59b3cea2d5579a99a3789ba07c31`
- Accepted tokens: 0; `A=0`; `U=0`; performance claim: none

## Decision

Retain the measurement as conditional evidence for this exact five-case trace,
M1 host, internal SSD, and serialized 224-read schedule. Do not promote the
transport unchanged because it missed the frozen median continuation gate.
The evidence rejects neither the minor-cost hypothesis nor Firewing 4: 1.58
ms/token is a small component of a 250 ms/token endpoint budget, but this trace
is not a production decode distribution and excludes address computation, BF16
conversion, PLE math, synchronization, and the rest of the model.

The next transport experiment should preserve exact rows while increasing I/O
parallelism or coalescing shared pages. Long-context and holdout schedules must
be measured after real tokenizer traces exist; no result here covers them.
