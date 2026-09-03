# FW-0047 - Sequential q2 zstd storage oracle

- Status: completed
- Disposition: rejected for untransformed zstd-1 on the frozen sequential prefix
- Date: 2026-09-03
- Parent experiments: FW-0043, FW-0046
- Exactness: L1 source-exact zstd-1 sizes; impossible-favorable fractional cache oracle
- Hardware/runtime: Apple M1 Mac mini, 16 GiB, internal SSD

## Question and hypothesis

Does one 4.26-GB compressed cache preserve a Firewing 4 storage path across
both exact sequential q2 transactions, without resetting its free contents for
each transaction as FW-0046 did?

The first and second transactions contain 687 and 731 target expert rows, with
321 shared layer-expert identities and 1,097 in their sequential union. Measure
the exact independent zstd-1 size of every union expert, reusing the immutable
FW-0046 sizes and compressing only the 410 new experts.

## Falsification rule

Grant the candidate the complete future, free fractional initial contents,
one load at most for every distinct nonresident expert, zero framing/page
amplification, FW-0044's favorable raw physical bandwidth, and free
decompression, Metal, MTP, fixed work, cache management, and synchronization.

Reject zstd-1 as sufficient on this frozen sequential prefix if even this
storage-only oracle misses 4 accepted TPS for aggregate `A=4`. Passing would
only authorize a causal cache-policy experiment. Neither outcome is a
production acceptance distribution or endpoint TPS.

## Result

The two transactions share 321 layer-expert identities and contain 1,097
distinct experts. The analyzer reuses all 687 immutable FW-0046 frame sizes,
then compresses and exactly round-trips the remaining 410 source experts.
Their combined 10,783,948,800 source bytes become 8,386,402,510 zstd-1 bytes,
a stable 77.7675% ratio.

One impossible-favorable 4,260,902,888-byte fractional cache leaves
4,125,499,622 bytes outside residency. At FW-0044's favorable 3.501-GB/s raw
physical rate, those bytes alone require 1.178215 seconds. Aggregate `A=4`
therefore has a storage-only ceiling of **3.394966 accepted TPS**, below the
4-TPS median gate before decompression, compute, or any endpoint work.

Raw receipt:
`/Users/chad/Models/firewing/evidence/FW-0047/sequential-zstd-oracle-1a635e5.json`

Receipt SHA-256:
`b9dea3412892e0970195b32dc54c2dce9508cc52ec71d3987538c7bb45a8fc2c`

## Decision

Reject untransformed per-expert zstd-1 as sufficient for the frozen sequential
prefix. This reverses FW-0046's single-transaction survivor: resetting free
future-known contents per transaction hid the cross-transaction capacity
deficit. It does not reject zstd-1 as a codec component after a different exact
byte transform, nor does two transactions establish the sustained route
distribution.

To reach four storage-only TPS under the same favorable assumptions, the
1,097-expert union must fit within approximately 7.762 GB—an overall ratio near
72.0%, about 7.4% smaller than the current compressed representation. Test an
exact BF16 byte-shuffle transform before any causal cache or resident runtime.
