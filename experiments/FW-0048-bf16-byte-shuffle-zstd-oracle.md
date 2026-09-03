# FW-0048 - BF16 byte-shuffle zstd oracle

- Status: completed
- Disposition: conditional sequential storage survivor
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

At clean commit `80f40104ad44aa1a8250eb1a9116e12cc6f93995`, all
1,097 independent transformed frames decompress and inverse-shuffle to their
exact 9,830,400-byte source records. The 10,783,948,800-byte sequential union
compresses to 7,381,296,763 bytes, or **68.4471%** of source. This is
1,005,105,747 bytes smaller than FW-0047's untransformed zstd-1 union and clears
the approximately 72.0% threshold.

After one free 4,260,902,888-byte fractional cache, 3,120,393,875 compressed
bytes remain. At the same favorable 3.501-GB/s rate, storage takes 891.164 ms
for aggregate `A=4`, a **4.488514 accepted-TPS storage-only ceiling**.

Raw receipt:
`/Users/chad/Models/firewing/evidence/FW-0048/sequential-byte-shuffle-zstd-oracle-80f4010.json`

Receipt SHA-256:
`592d5b4e4c45f3733977a9a068c660dd23f90c877f5df1e0afa960841f1f1e89`

## Decision

Promote BF16 byte shuffle plus zstd-1 as a conditional sequential storage
survivor. The exact reversible transform changes neither values nor arithmetic
and provides the additional compression FW-0047 required. Its 12% storage-only
headroom is too narrow for promotion: decompression, inverse shuffle, physical
page amplification, and four target rows of Metal work remain uncharged.

Build one external page-aligned two-transaction transformed container and run a
parallel physical read/decode/inverse-shuffle/Metal overlap bound next. Passing
still will not establish causal cache behavior or endpoint TPS.
