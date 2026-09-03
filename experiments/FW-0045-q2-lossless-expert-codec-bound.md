# FW-0045 - Width-two lossless expert codec bound

- Status: implementation ready; measurement pending
- Disposition: unexecuted
- Date: 2026-09-03
- Parent experiments: FW-0044
- Exactness: L1 source-exact independent whole-expert frames; favorable analytical bound
- Hardware/runtime: Apple M1 Mac mini, 16 GiB, internal SSD

## Question and hypothesis

Can fast lossless whole-expert compression reduce the exact first q2
transaction's executable bytes enough to reopen a path to Firewing 4 after
FW-0044 rejected raw BF16?

## Frozen authority and method

Compress each of the 687 distinct exact target expert records independently
with zstandard 0.25.0 level 1. Each frame concatenates its unchanged BF16
gate/up and down payloads. Authenticate source payloads against the endpoint
fixture and require every decompression to reproduce all 9,830,400 source
bytes exactly.

Compute an impossible-favorable 12-GiB bound: subtract the unchanged
8,623,999,000 fixed bytes, grant fractional future-aware compressed residency,
charge only compressed union bytes beyond that free cache, scale FW-0044's
measured physical bandwidth without page/frame amplification, and perfectly
overlap storage, proportional measured decompression, and exact target Metal.
All MTP and remaining endpoint work is free.

## Gates

- Exactness: all 687 independent frames round-trip byte exactly.
- Firewing 4 continuation: favorable accepted throughput must reach 4 TPS.
- Codec continuation: compressed bytes must fall at least 10% and the favorable
  complete bound must pass before any container or inline decoder is built.
- Safety: process one expert at a time; do not materialize or commit the union.

Failure rejects this exact zstd-1 representation, not every possible lossless
transform. Passing authorizes a physical compressed-read/decode experiment; it
does not establish endpoint TPS or a runtime default.

## Result

Pending a clean-commit run.
