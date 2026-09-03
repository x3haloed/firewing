# FW-0046 - Parallel zstd physical overlap

- Status: container implementation ready; measurement pending
- Disposition: unexecuted
- Date: 2026-09-03
- Parent experiments: FW-0044, FW-0045
- Exactness: L1 source-exact page-aligned independent expert frames
- Hardware/runtime: Apple M1 Mac mini, 16 GiB, internal SSD

## Question and hypothesis

Can bounded parallel cold reads and exact decompression turn FW-0045's 22.24%
zstd-1 byte reduction into a q2 accepted-throughput path that preserves the
Firewing 4 gates while contending with exact target Metal work?

## Frozen construction and gates

Materialize the 687 authenticated q2 target experts as independent zstd-1
frames. Each page-aligned record contains one unchanged 9,830,400-byte BF16
gate/up-plus-down payload. Keep the approximately 5.3-GB container and manifest
outside Git, authenticate every source and encoded frame, and record their
content hashes.

The native benchmark will use a realizable whole-frame, future-known initial
cache under the same 12-GiB favorable residency allowance as FW-0044. It will
invalidate every miss range, require zero resident pages, use `F_NOCACHE=1`
and `F_RDAHEAD=0`, verify nonzero physical reads, decompress into bounded
preallocated worker buffers, and establish exact round trips before timing.

Interleave at least three cold controls and three candidates. Candidates
overlap the complete exact 96-layer routed Metal proxy with all scheduled
compressed reads and decompression. Report worker count, compressed/logical/
physical bytes, cold and warm state, `A=2`, `U=697/480`, rollback zero, batch
one, concurrency one, hardware, commit, and host-safety telemetry.

- Kill this zstd embodiment if the favorable candidate median is below 4
  accepted TPS or p10 is below 3 TPS.
- Passing authorizes integration into a repeated stateful q2 runtime only; it
  is not endpoint TPS or a production default.
- No model value, route, capability, precision, or arithmetic may change.

## Result

Pending clean-commit container construction and native measurement.
