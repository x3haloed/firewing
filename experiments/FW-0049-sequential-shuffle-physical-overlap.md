# FW-0049 - Sequential transformed physical overlap

- Status: container builder ready; measurement pending
- Disposition: unexecuted
- Date: 2026-09-03
- Parent experiment: FW-0048
- Exactness: L1 reversible BF16 byte transform plus lossless zstd-1
- Hardware/runtime: Apple M1 Mac mini, 16 GiB, internal SSD

## Question and hypothesis

Does FW-0048's narrow 4.4885-TPS analytical storage survivor remain above 4
accepted TPS after real 16-KiB page amplification, parallel zstd-1 decode,
inverse BF16 shuffle, and the exact four routed rows of Metal work are measured
together?

## Method and gates

Build one externally stored container over the exact 1,097-expert union of the
two independently verified q2 transactions. Each expert is independently byte
shuffled, zstd-1 compressed, padded to a 16-KiB boundary, and bound to both its
compressed-frame and reconstructed-source SHA-256.

Before timing, the native benchmark must authenticate the manifest and whole
container, decompress and inverse-shuffle every frame to its exact source hash,
and fail closed on any authority, layout, route, byte-ledger, or host-safety
mismatch. Give the candidate a free future-known initial cache selected from
whole frames under the unchanged 4,260,902,888-byte limit. Cold-invalidate and
physically read every remaining frame once while eight workers decode and
inverse-shuffle concurrently with 192 exact routed-expert Metal executions.

Use three interleaved cold control/candidate pairs. Passing requires both median
and p10 accepted throughput at or above 4 TPS with exact physical-byte ledgers
and no host-safety violation. This remains an impossible-favorable component
bound: cache construction, causality, eviction, fixed endpoint work, and runtime
integration are free.

## Result

Pending a clean-commit container build and native measurement.
