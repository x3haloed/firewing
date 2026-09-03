# FW-0049 - Sequential transformed physical overlap

- Status: completed
- Disposition: favorable physical overlap survivor
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

The builder at clean commit `6271f3dd845f00d183d5b76053718859be0f14bd`
produced 1,097 exact independent frames. The 7,381,296,763 compressed bytes
occupy 7,388,381,184 physical bytes after 16-KiB padding, only 7,084,421 bytes
of whole-container page amplification.

Manifest:
`/Users/chad/Models/firewing/evidence/FW-0049/q2-sequential-bf16-shuffle-zstd1-manifest-6271f3d.json`

Manifest SHA-256:
`6759e772d2c9a4560ab39ae80a3b4f4e1a24552adafbf30a396e84166b9c71ca`

Container SHA-256:
`b14d0f9827a001b97495b97f11d111495f94e8c7392e0ec7d9e7f39095a372bb`

At benchmark commit `fa7e4dfeb689b6507ccc6fd39c157ab8137bba8f`,
the largest-fitting whole-frame initial cache holds 633 frames and
4,260,632,646 compressed bytes. Its 270,242-byte unused tail leaves 464 miss
frames containing 3,120,664,117 compressed bytes and 3,124,494,336 physical
bytes. Every trial's process disk counter equals that physical ledger exactly;
zero pages remain resident after each range invalidation.

One, two, and four worker controls take 2,953.342, 1,675.973, and 1,016.993 ms.
Eight-worker interleaved controls have a 924.591-ms median. Candidate wall
times are 932.048, 953.073, and 935.310 ms while their 192 Metal executions
take 931.974, 950.068, and 929.180 ms. Metal therefore remains the observed
overlap boundary despite real reads, decompression, and inverse shuffle.

The candidate reaches **4.196951 / 4.276656 / 4.291624 accepted TPS** at
p10/median/p90, passing both declared 4-TPS gates. Host safety records 52--53%
free memory, no swap growth, no throttled pages, and a 343.0-MB final physical
footprint.

Raw receipt:
`/Users/chad/Models/firewing/evidence/FW-0049/sequential-shuffle-overlap-fa7e4df.json`

Receipt SHA-256:
`31f6e89a27dfe87a5f80ac1125a8020d6be58b51c57d7a1e4d59b507797d3266`

## Decision

Promote byte-shuffled zstd-1 as a favorable physical-overlap survivor, not a
runtime default or endpoint result. FW-0049 resolves FW-0048's page,
decompression, inverse-transform, and routed-Metal questions on the frozen
two-transaction trace.

The remaining 4.9% p10 margin depends on an impossible cache: future-known
contents are installed for free and all 1,097 identities may effectively be
retained after first acquisition despite the 4.26-GB capacity. Measure an
explicit capacity-respecting sequential cache oracle next, before building a
causal runtime cache.
