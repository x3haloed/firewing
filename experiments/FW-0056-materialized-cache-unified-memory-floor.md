# FW-0056 - Materialized-cache unified-memory floor

- Status: completed
- Disposition: rejected
- Date: 2026-09-03
- Parent experiments: FW-0034, FW-0053, FW-0055
- Exactness: L1 byte-accounting lower bound for the named materialized-BF16 representation
- Hardware: Apple M1 Mac mini, 16 GiB unified memory, internal SSD

## Question and hypothesis

Can FW-0053's full-budget schedule or FW-0055's 4.00-GB schedule reach four
accepted TPS after their omitted fixed matrices and shared memory fabric are
charged? The prior optimistic model took the maximum of physical SSD, CPU
decode, and Metal time as if those paths consumed independent bandwidth. The
hypothesis was that its remaining 107.656 or 32.640 ms might absorb fixed work.

## Frozen authority and method

- Implementation commit:
  `4bee6582c675cab7697dad9904b72cc744e746dd`
- FW-0034 fixed-byte receipt SHA-256:
  `19bd38ecc103a80fafc0085063123b86ddaa2aa5365c2fbdf147dae73c6168da`
- FW-0053 schedule SHA-256:
  `a61d498af5512c3fcbdc3447d8217383ddacf39361cdea553a41a79d9a10cb3f`
- FW-0055 4.00-GB schedule SHA-256:
  `7c9136e878cefa5c1689285b167b25f3ffd340a808ea3282f0caae84946c100f`
- Sequential manifest SHA-256:
  `6759e772d2c9a4560ab39ae80a3b4f4e1a24552adafbf30a396e84166b9c71ca`
- `q=2`, `A=4`, `sum_equivalent_U=2.995833`, batch/concurrency 1/1

For four target rows the lower bound charges only:

1. Four reads of FW-0034's authenticated 8,623,999,000 fixed matrix bytes.
2. The 1,920 routed expert accesses at 9,830,400 BF16 bytes each.
3. One read of each compressed frame entering the 1,109 or 1,202 decodes.
4. One write of each materialized decoded BF16 expert payload.

Everything else is free, including SSD DMA writes, activations and outputs,
drafter work, cache metadata, synchronization, and all non-weight traffic.
Every charged transfer may overlap perfectly, but the aggregate cannot exceed
one shared fabric ceiling.

Apple states that M2 provides 100 GB/s unified-memory bandwidth, 50% more than
M1, implying about 66.667 decimal GB/s for M1. The analysis instead grants
68.25 GB/s continuously. It also subtracts one decimal GB of cross-row cache
reuse at each of three row transitions. This is deliberately impossible-
favorable: ideal 6T SRAM for that cache alone requires 48 billion transistors,
three times Apple's published 16-billion-transistor count for the entire M1.

```shell
.venv/bin/python tools/analyze_materialized_memory_floor.py \
  /Users/chad/Models/firewing/evidence/FW-0049/q2-sequential-bf16-shuffle-zstd1-manifest-6271f3d.json \
  /Users/chad/Models/firewing/evidence/FW-0034/exact-residency-oracle-2fd14bc5.json \
  /Users/chad/Models/firewing/evidence/FW-0053/executable-cache-5f69eee.json \
  /Users/chad/Models/firewing/evidence/FW-0055/executable-cache-4000000000-6bae8dc.json \
  4bee6582c675cab7697dad9904b72cc744e746dd \
  /Users/chad/Models/firewing/evidence/FW-0056/materialized-memory-floor-4bee658.json
```

## Gate and result

Reject a schedule when adjusted mandatory bytes exceed the granted fabric's
one-second capacity. This is a necessary condition for `A=4`, not a measured
endpoint rate.

| Schedule | Fixed reads | Routed reads | Compressed reads | Decoded writes | After 3-GB grant | Minimum time | Maximum TPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FW-0053 4.261 GB | 34.496 GB | 18.874 GB | 7.462 GB | 10.902 GB | 68.734 GB | 1.007095 s | 3.971818 |
| FW-0055 4.00 GB | 34.496 GB | 18.874 GB | 8.088 GB | 11.816 GB | 70.274 GB | 1.029658 s | 3.884785 |

Both fail despite the intentionally overstated bandwidth and cache grants.
Using Apple's implied 66.667 GB/s would make the rejection stronger. The
result is independent of CPU/GPU scheduling efficiency, SSD latency, thermal
behavior, and the FW-0054 swap failure because all are granted away.

Raw receipt:
`/Users/chad/Models/firewing/evidence/FW-0056/materialized-memory-floor-4bee658.json`

Receipt SHA-256:
`f4d6964c87266af64e5774268b5fbfa1168f7230fcccb824a63f14299f64aa41`

The new arithmetic and fail-closed tests bring the Python suite to 87 passing
tests.

## Decision

Reject the materialized-BF16 mixed-cache representation for Firewing 4 on the
frozen two-transaction trace. This reverses FW-0053's conditional survivor and
closes FW-0055's 4.00-GB point; a physical retry or causal cache policy cannot
repair a necessary fabric-byte failure. FW-0054's exact safe-residency ceiling
no longer changes this branch decision.

This does not prove the full Firewing objective impossible. It does not reject
an exact compressed-weight compute kernel that avoids materializing every BF16
payload, a stronger exact representation that reduces both compressed reads
and decoded writes, or a different production acceptance/route distribution.
Those are new architecture branches and must receive their own correctness and
end-to-end evidence.
