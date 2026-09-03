# FW-0055 - Mixed-cache capacity frontier

- Status: completed
- Disposition: conditional
- Date: 2026-09-03
- Parent experiments: FW-0053, FW-0054
- Exactness: L1 compressed frames or exact decoded BF16 payloads
- Hardware/rates: Apple M1 Mac mini, 16 GiB, internal SSD

## Question and hypothesis

After FW-0054 stopped on swap growth, how far can the mixed cache shrink before
four accepted TPS becomes impossible even in FW-0053's favorable offline
model? The hypothesis was that a materially smaller capacity could retain
enough physical and decode headroom to justify a safer installation probe.

## Frozen authority and method

- Implementation commit:
  `6bae8dcf7bfb87625f3c6e35787553aa9431489d`
- Sequential manifest SHA-256:
  `6759e772d2c9a4560ab39ae80a3b4f4e1a24552adafbf30a396e84166b9c71ca`
- FW-0051 receipt SHA-256:
  `59dcef0b2c78da0dbb7521ce0c824632b86d894bc9db8b6140a0ef24294d0644`
- FW-0052 receipt SHA-256:
  `d8561eae477282e59cf1ed32828f993ef0f99bafd6457f053e53f6df3221100b`
- Batch size/concurrency: 1 / 1
- `q=2`, `A=4`, `sum_equivalent_U=2.995833`

The FW-0053 analyzer now accepts only a positive capacity no larger than the
4,260,902,888-byte source-manifest budget. Every result still independently
replays interval exclusivity, 192 capacity boundaries, all 1,920 accesses,
physical and decode ledgers, and the integer incumbent objective. Both points
use the same deterministic 10,000-node limit. The solver dual bound is retained
so a failing incumbent can be distinguished from a proven failing optimum.

```shell
.venv/bin/python tools/analyze_executable_cache_milp.py \
  /Users/chad/Models/firewing/evidence/FW-0049/q2-sequential-bf16-shuffle-zstd1-manifest-6271f3d.json \
  /Users/chad/Models/firewing/evidence/FW-0051/capacity-cache-overlap-d671bd1.json \
  /Users/chad/Models/firewing/evidence/FW-0052/metal-swiglu-c2bac85.json \
  6bae8dcf7bfb87625f3c6e35787553aa9431489d \
  REPORT_JSON \
  --capacity-bytes CAPACITY_BYTES
```

## Gates

- Reject a capacity when the solver dual bound is greater than one second;
  even the optimal favorable schedule then cannot reach four accepted TPS.
- Retain a point only as an offline survivor when an independently replayed
  incumbent is at most one second.
- Do not allocate another multi-gigabyte cache unless the retained point has
  enough headroom to justify the omitted fixed work and host-safety risk.

## Result

| Capacity | Misses | Physical bytes | Decodes | Physical / decode | Accepted TPS | Solver gap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4.00 GB | 503 | 3,387,195,392 | 1,202 | 967.360 / 966.855 ms | 4.134964 | 0.102676% |
| 3.75 GB | 540 | 3,636,330,496 | 1,291 | 1,038.512 / 1,038.444 ms | 3.851666 | 0.068875% |

At 3.75 GB the dual bound is 1.037796 seconds. This proves that the point is
below four TPS within the favorable model; it is not merely a poor incumbent.
At 4.00 GB the replayed incumbent remains above four TPS, but its complete
headroom is only 32.640 ms. It still grants free initial contents, perfect
three-way overlap, no causal barriers, and every fixed endpoint operation.

Raw receipts and hashes:

- `executable-cache-4000000000-6bae8dc.json`:
  `7c9136e878cefa5c1689285b167b25f3ffd340a808ea3282f0caae84946c100f`
- `executable-cache-3750000000-6bae8dc.json`:
  `0491debb18eec433ac77d97da2fe31a1f02671b65d9a0b0ad33552459a4557f4`

Both live under `/Users/chad/Models/firewing/evidence/FW-0055/`. The analyzer
has a new fail-closed capacity regression; all 85 Python tests pass.

## Decision

Reject 3.75 GB and every smaller cache under this monotone capacity model.
Retain 4.00 GB only as a razor-thin offline survivor. The hypothesis that a
materially smaller cache would leave useful four-TPS headroom is rejected.

Do not immediately allocate 4.00 GB: it is only 259 MB smaller than the
FW-0054 capacity that caused 734 MB of swap growth, and its 32.640-ms optimistic
headroom excludes all fixed work. The next higher-leverage falsification is to
charge an authenticated lower bound for the omitted fixed matrices and work.
Only if that still fits should a cache-only safety probe isolate the actual
residency ceiling before Metal overlap.
