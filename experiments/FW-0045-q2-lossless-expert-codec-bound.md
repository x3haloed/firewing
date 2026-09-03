# FW-0045 - Width-two lossless expert codec bound

- Status: completed
- Disposition: conditional for compression; rejected for serial CPU decoding
- Date: 2026-09-03
- Parent experiments: FW-0044
- Exactness: L1 source-exact independent whole-expert frames; favorable analytical bound
- Hardware/runtime: Apple M1 Mac mini, 16 GiB, internal SSD

## Question and hypothesis

Can fast lossless whole-expert compression reduce the exact first q2
transaction's executable bytes enough to reopen a path to Firewing 4 after
FW-0044 rejected raw BF16?

## Frozen authority and method

- Clean implementation commit:
  `c50966e853457f7aad94a8f9781b612544f35ff2`

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

All 687 independent expert frames reproduced their exact 9,830,400-byte source
records. The 6,753,484,800-byte union compressed to 5,251,840,172 bytes, a
ratio of 77.7649% and a reduction of 22.2351%. Individual frames ranged from
7,632,089 to 7,741,149 bytes, with a 7,642,827-byte median.

After granting 4,260,902,888 bytes of free fractional compressed residency,
only 990,937,284 compressed miss bytes remain. At FW-0044's measured
3.501-GB/s physical rate, their impossible-favorable storage time is 283.005
ms. Exact target Metal totals 394.951 ms, so storage and compute alone would
retain a 5.064-TPS ceiling under perfect overlap.

Serial zstd decompression changes the decision. Decoding the full union takes
6.090 seconds. Even charging only the missing compressed fraction yields
1,149.072 ms, which dominates both storage and Metal despite granting perfect
three-way overlap. The resulting accepted-throughput bound is **1.740535
TPS**.

Raw receipt:
`/Users/chad/Models/firewing/evidence/FW-0045/q2-zstd1-bound-c50966e.json`

Receipt SHA-256:
`0f2b29af1fc6e42869a9a311f869042fe7671ef34a89d8f7c9f889807c08804c`

The first clean run exited before emitting a receipt because the analyzer
expected throughput-model convenience fields in the raw FW-0044 evidence.
Commit `c50966e` instead derives and validates physical bytes, storage medians,
and Metal medians from the raw trial ledger; a focused regression test freezes
that schema boundary. No failed-run output was used as evidence.

## Decision

Promote per-expert zstd-1 only as a conditional byte-capacity lead: its 22.24%
exact reduction is large enough to reopen the storage/compute ceiling. Reject
the measured serial CPU decoder as an executable path. This neither promotes a
container nor rejects parallel or fused exact decoding.

The next cheap experiment should materialize the independently addressable
frames outside Git and measure cold compressed reads plus bounded parallel
decompression with exact round trips. It must beat 500 ms per `A=2`
transaction before endpoint integration; decompression-only throughput remains
diagnostic.
