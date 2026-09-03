# FW-0051 - Capacity-cache physical overlap

- Status: completed
- Disposition: favorable capacity-cache physical survivor
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

At clean commit `d671bd10aa5e50aaf4705dc38f02cad0af71acc4`, the
native replay independently reconstructs FW-0050's 1,456 retained intervals,
633 free initial frames, 192 capacity boundaries, and 464 exact misses. The
initial frames occupy 4,258,678,508 compressed bytes. Scheduled misses contain
3,122,618,255 compressed bytes in 3,124,527,104 requested physical bytes.

The first attempts correctly failed closed when macOS reported more physical
bytes than the 16-KiB request ledger. Instrumentation localized this to the
process I/O counter: it advances in 4-KiB accounting quanta, and the less
sequential schedule incurred 199--201 extra quanta on early cold trials. The
final implementation rejects under-reads, accepts only 4-KiB-aligned over-read,
and reports that amplification separately. In the accepted series the
one-worker diagnostic charges 823,296 extra bytes; the other eight trials have
zero amplification.

One-, two-, and four-worker controls take 3,022.229, 1,606.838, and 995.287 ms.
The eight-worker control median is 920.533 ms. Candidate wall times are
959.540, 933.286, and 919.095 ms, producing **4.168663 / 4.285932 / 4.352106
accepted TPS** at p10/median/p90. Both declared gates pass.

All 1,097 transformed frames authenticate and exact-round-trip before timing.
Every timed trial reconstructs the expected 4,561,305,600 source bytes. Host
safety records no swap growth, throttling, or lost protected service, 52% final
free memory, and a 343.4-MB final physical footprint.

Raw receipt:
`/Users/chad/Models/firewing/evidence/FW-0051/capacity-cache-overlap-d671bd1.json`

Receipt SHA-256:
`59dcef0b2c78da0dbb7521ce0c824632b86d894bc9db8b6140a0ef24294d0644`

## Decision

Promote the transformed, capacity-respecting **offline** cache branch to causal
cache investigation. This is not a runtime or endpoint promotion. The complete
future and initial contents remain free, all misses launch without layer
dependencies, and fixed model work remains omitted. The next cheap experiment
must measure causal initial-cache and eviction policies from prior route history
before any production cache implementation.
