# FW-0048 - BF16 byte-shuffle zstd oracle

- Status: implementation ready; measurement pending
- Disposition: unexecuted
- Date: 2026-09-03
- Parent experiment: FW-0047
- Exactness: L1 reversible BF16 byte transform plus lossless zstd-1
- Hardware/runtime: Apple M1 Mac mini, 16 GiB, internal SSD

## Question and hypothesis

Can separating each BF16 record's even and odd bytes expose enough exact
structure for zstd-1 to move the two-transaction q2 union below FW-0047's
approximately 72.0% storage-survival threshold?

The transform is deterministic and source exact: serialize every BF16 low-byte
lane followed by every high-byte lane inside each independently addressable
expert frame. Decompression followed by inverse interleaving must reproduce all
9,830,400 original bytes for every one of the 1,097 experts.

## Gates

Use FW-0047's unchanged impossible-favorable sequential cache/storage oracle.
The transformed union must:

- round-trip every expert exactly;
- fit at or below the 4-TPS storage threshold under one 4.26-GB fractional
  future-known cache; and
- improve untransformed zstd-1 by enough to justify measuring physical inverse
  shuffle and parallel decode.

Passing authorizes a page-aligned transformed container and physical overlap
experiment. It is not endpoint TPS, causal cache evidence, or a runtime
default. Failure rejects this transform without changing model precision.

## Result

Pending a clean-commit full-union run.
