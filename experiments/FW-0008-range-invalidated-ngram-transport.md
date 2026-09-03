# FW-0008 - Range-invalidated n-gram transport

- Status: planned and frozen
- Disposition: unexecuted
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
The implementation/protocol commit will be filled after this frozen record is
committed.

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

Unexecuted. Preserve the raw report before changing this section.

## Decision

Unexecuted. FW-0007 remains rejected as physical-I/O evidence.
