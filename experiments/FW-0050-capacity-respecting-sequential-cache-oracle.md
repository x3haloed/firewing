# FW-0050 - Capacity-respecting sequential cache oracle

- Status: implementation ready; measurement pending
- Disposition: unexecuted
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

Pending a clean-commit run.
