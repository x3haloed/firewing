# FW-0051 - Capacity-cache physical overlap

- Status: implementation ready; measurement pending
- Disposition: unexecuted
- Date: 2026-09-03
- Parent experiment: FW-0050
- Exactness: L1 lossless transformed frames and exact offline cache schedule
- Hardware/runtime: Apple M1 Mac mini, 16 GiB, internal SSD

## Question and hypothesis

Does FW-0050's capacity-respecting offline cache schedule retain at least 4
accepted TPS when its exact page-aligned misses are physically read,
decompressed, inverse-shuffled, and overlapped with all four routed Metal rows?

## Method and gates

The native benchmark must authenticate the FW-0049 manifest and container plus
the FW-0050 cache receipt. It independently reconstructs all 1,920 possible
retention intervals, replays the 1,456 selected intervals across 192 event
boundaries, proves the 4,260,902,888-byte capacity is never exceeded, and
derives the 464 declared misses as the exact complement.

After authenticating and exact-round-tripping all 1,097 transformed frames,
run one-, two-, and four-worker diagnostics followed by three interleaved cold
eight-worker control/candidate pairs. Every trial must physically read exactly
the scheduled page ledger. Candidate timing includes parallel zstd decode,
inverse shuffle, and 192 exact routed-expert Metal executions.

Passing requires p10 and median accepted throughput at least 4 TPS, exact byte
ledgers, and host safety. Passing remains an offline, future-known component
bound—not a causal cache, endpoint TPS, or runtime default.

## Result

Pending a clean-commit native run.
