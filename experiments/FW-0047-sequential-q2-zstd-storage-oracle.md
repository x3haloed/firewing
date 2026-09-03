# FW-0047 - Sequential q2 zstd storage oracle

- Status: implementation ready; measurement pending
- Disposition: unexecuted
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

Pending a clean-commit run.
