# FW-0050 - Capacity-respecting sequential cache oracle

- Status: completed
- Disposition: conditional offline cache survivor
- Date: 2026-09-03
- Parent experiment: FW-0049
- Exactness: L1 whole-frame cache scheduling over exact lossless records
- Hardware/runtime: Apple M1 Mac mini, 16 GiB; SciPy 1.16.2 HiGHS MILP

## Question and hypothesis

Does FW-0049 remain above the 4-TPS storage threshold when its free
future-known compressed cache obeys the 4,260,902,888-byte capacity at every
one of the 192 ordered target layer events?

The earlier physical bound selected a legal initial cache, then charged each
other union identity only once without evicting newly acquired records. That
was deliberately favorable but not capacity respecting. A whole-record
retention schedule can answer the narrower capacity question before a physical
cache runtime is justified.

## Method and gates

Represent every possible hit as a binary interval from an identity's previous
access (or pre-run initial state) through its next access. Its compressed size
consumes capacity at every intervening event boundary and its physical frame
size is the avoided-read objective. Solve the resulting interval-packing MILP
with a deterministic 10,000-node limit, preserve the incumbent and
solver-reported dual bound, then independently replay every selected interval
to verify all capacities, hits, misses, and byte ledgers. The replayed incumbent
is the decision authority; a node-limited dual bound is diagnostic unless the
wrapper reports an optimal solve.

A feasible incumbent at or above 4 TPS under FW-0048's favorable measured SSD
rate promotes a capacity-respecting offline physical replay. A certified
optimistic bound below 4 TPS rejects the representation. An unresolved bound
with no passing incumbent is inconclusive. Passing does not prove a causal
policy, runtime residency, physical overlap, or endpoint TPS.

## Result

At clean commit `4fa77fd3ed24171b862914f826c958279110acb7`, the
10,000-node MILP incumbent selects 1,456 hit-producing retention intervals and
leaves 464 misses across 1,920 accesses. Independent replay shows a maximum of
4,258,752,496 resident compressed bytes across all 192 boundaries, 2,150,392
bytes below the fixed capacity, with zero capacity violations.

Every miss is the first access to one distinct identity. The free initial cache
contains 633 future-selected frames, and all 823 subsequent reuses remain
resident until their next access. This explains the prediction error from the
pre-experiment farthest-future probe: that probe initialized the cache with the
largest frames, which optimized static byte fill but discarded future reuse
value and caused 58 avoidable reloads.

The feasible schedule reads 3,122,618,255 compressed bytes occupying
3,124,527,104 physical bytes. At the unchanged favorable measured SSD rate,
storage takes 892.344 ms and retains a **4.482576 accepted-TPS storage-only
rate** for aggregate `A=4`.

The node-limited solve stops with a 0.0233911% relative gap. SciPy maps HiGHS'
solution-limit status to generic status 4, so the 3,122,233,344-byte dual-side
miss estimate is retained only as a diagnostic. The passing incumbent itself
is the authority: it is integral, its objective matches the independent byte
replay within 0.5 byte, and it violates no capacity boundary.

Raw receipt:
`/Users/chad/Models/firewing/evidence/FW-0050/capacity-cache-milp-4fa77fd.json`

Receipt SHA-256:
`ed4ae0d2137bde9393b9aad1556910360bf1e5689de56df5c1f32a82a691159e`

## Decision

Promote this exact schedule to a capacity-respecting offline physical replay.
Do not promote it as a runtime cache: its 633 initial frames are installed for
free with full route knowledge, and its retention choices are noncausal. The
next benchmark must replay its exact miss list with page-aligned reads,
decompression, inverse shuffle, and four-row Metal overlap.
