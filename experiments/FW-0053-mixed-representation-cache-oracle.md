# FW-0053 - Mixed-representation cache oracle

- Status: completed
- Disposition: conditional offline representation survivor
- Date: 2026-09-03
- Parent experiments: FW-0050, FW-0051, FW-0052
- Exactness: L1 compressed frames or exact decoded BF16 expert payloads
- Hardware/rates: Apple M1 Mac mini, 16 GiB, internal SSD

## Question and hypothesis

Does any capacity-respecting cache remain above the Firewing-4 rate after
retained data is charged in the representation actually held? FW-0051 charged
capacity in compressed bytes, decoded only SSD misses, and granted retained
hit traffic for free. A real implementation must either keep a compressed
frame and decode it on each hit, or retain its 9,830,400-byte executable BF16
payload and avoid that decode.

The hypothesis was that an offline mixed-representation schedule might balance
those costs: compact initial contents preserve physical hit rate, while later
high-value intervals are promoted to decoded BF16.

## Frozen authority and method

- Implementation commit: `5f69eee3ca1db5c4b7285a376a9c9b95f3325f60`
- Sequential manifest SHA-256:
  `6759e772d2c9a4560ab39ae80a3b4f4e1a24552adafbf30a396e84166b9c71ca`
- FW-0051 receipt SHA-256:
  `59dcef0b2c78da0dbb7521ce0c824632b86d894bc9db8b6140a0ef24294d0644`
- FW-0052 receipt SHA-256:
  `d8561eae477282e59cf1ed32828f993ef0f99bafd6457f053e53f6df3221100b`
- Mixed-representation capacity: 4,260,902,888 bytes
- Events/accesses: 192 / 1,920
- Batch size/concurrency: 1 / 1
- `q=2`, `A=4`, `sum_equivalent_U=2.995833`

For each access interval, a binary MILP chooses exactly one of compressed
retention, decoded-BF16 retention, or absence. Each of 192 event boundaries
charges compressed size or the full 9,830,400 source bytes. Absence charges a
page-aligned physical miss and a decode; compressed retention charges a decode
but no SSD read; decoded retention charges neither.

The objective minimizes the larger of optimistic physical seconds and ideal
eight-worker decode seconds. Physical bandwidth remains FW-0044's favorable
3,501,482,752.689 bytes/s. Decode capacity is the fastest FW-0051 control's
aggregate decompression-plus-inverse-shuffle CPU work divided ideally across
eight workers: 12,221,212,890.578 source bytes/s. FW-0052 contributes 192 exact
Metal executions at its 3.164709-ms median. The three resources overlap
perfectly; fixed work, installation copies, metadata, and synchronization are
free.

The analyzer has a tiny exhaustive unequal-size fixture. The full solver stops
at a deterministic 10,000-node limit; its integer incumbent is independently
replayed for exclusivity, every capacity boundary, access partition, resource
ledgers, and objective. Its 0.056371% dual gap is diagnostic, not the authority
for the feasible result.

```shell
.venv/bin/python tools/analyze_executable_cache_milp.py \
  /Users/chad/Models/firewing/evidence/FW-0049/q2-sequential-bf16-shuffle-zstd1-manifest-6271f3d.json \
  /Users/chad/Models/firewing/evidence/FW-0051/capacity-cache-overlap-d671bd1.json \
  /Users/chad/Models/firewing/evidence/FW-0052/metal-swiglu-c2bac85.json \
  5f69eee3ca1db5c4b7285a376a9c9b95f3325f60 \
  /Users/chad/Models/firewing/evidence/FW-0053/executable-cache-5f69eee.json
```

## Gates

Reject the current lossless representation branch if even the offline mixed
incumbent exceeds one second for four accepted tokens. A survivor only permits
a physical replay that decodes compressed hits and installs the scheduled
representation; it cannot promote a causal cache or endpoint.

## Result

The independently replayed incumbent selects 645 compressed and 811 decoded
retention intervals, leaving 464 misses. All 633 free initial frames are
compressed; only 12 later intervals remain compressed, while 811 later
intervals retain decoded BF16. Maximum combined residency is 4,259,199,939
bytes, 1,702,949 bytes below capacity.

| Resource | Charged work | Optimistic time |
| --- | ---: | ---: |
| Physical SSD | 3,124,527,104 bytes | 892.344 ms |
| Decode + inverse shuffle | 1,109 accesses / 10,901,913,600 source bytes | 892.048 ms |
| Exact routed Metal | 192 executions | 607.624 ms |

With perfect three-way overlap, physical SSD remains the 892.344-ms boundary:
**4.482576 accepted TPS**, leaving only 107.656 ms before the four-TPS limit.
The representation correction therefore does not kill the branch, but it
reduces the prior free-hit abstraction to a specific mixed schedule with very
little fixed-work headroom.

Raw receipt:
`/Users/chad/Models/firewing/evidence/FW-0053/executable-cache-5f69eee.json`

Receipt SHA-256:
`a61d498af5512c3fcbdc3447d8217383ddacf39361cdea553a41a79d9a10cb3f`

The repository has 84 Python tests; the new exhaustive fixture passes.

## Decision

Retain the exact transformed representation only as a conditional offline
survivor. FW-0051's direct promotion to causal-policy investigation is
superseded: first physically replay this mixed schedule, including all 1,109
decodes, representation transitions, actual bounded buffers, and Metal
installation. A failure there kills the current representation before route
causality is relevant; a pass then returns the branch to causal-cache work.
