# FW-0046 - Parallel zstd physical overlap

- Status: completed
- Disposition: conditional favorable-bound survivor
- Date: 2026-09-03
- Parent experiments: FW-0044, FW-0045
- Exactness: L1 source-exact page-aligned independent expert frames
- Hardware/runtime: Apple M1 Mac mini, 16 GiB, internal SSD

## Question and hypothesis

Can bounded parallel cold reads and exact decompression turn FW-0045's 22.24%
zstd-1 byte reduction into a q2 accepted-throughput path that preserves the
Firewing 4 gates while contending with exact target Metal work?

## Frozen construction and gates

Materialize the 687 authenticated q2 target experts as independent zstd-1
frames. Each page-aligned record contains one unchanged 9,830,400-byte BF16
gate/up-plus-down payload. Keep the approximately 5.3-GB container and manifest
outside Git, authenticate every source and encoded frame, and record their
content hashes.

- Builder commit: `a782e771f3ec4067ad4430865938defcc591108b`
- Manifest SHA-256:
  `893fa5739e4d4e22f23f5306d2e32ef33bb17af54a7e631fdf5b1286e63cc863`
- Container SHA-256:
  `bcc410a162445937641f4b5c894eccab9547c23e2cf4e9a3bf233a41edb93b87`
- Source bytes: 6,753,484,800
- Compressed bytes: 5,251,840,172
- Page-aligned physical bytes: 5,257,854,976
- Benchmark implementation commit:
  `121dd5a2d5a34be4ab0198e1045e317051241b31`

The native benchmark will use a realizable whole-frame, future-known initial
cache under the same 12-GiB favorable residency allowance as FW-0044. It will
invalidate every miss range, require zero resident pages, use `F_NOCACHE=1`
and `F_RDAHEAD=0`, verify nonzero physical reads, decompress into bounded
preallocated worker buffers, and establish exact round trips before timing.

Interleave at least three cold controls and three candidates. Candidates
overlap the complete exact 96-layer routed Metal proxy with all scheduled
compressed reads and decompression. Report worker count, compressed/logical/
physical bytes, cold and warm state, `A=2`, `U=697/480`, rollback zero, batch
one, concurrency one, hardware, commit, and host-safety telemetry.

- Kill this zstd embodiment if the favorable candidate median is below 4
  accepted TPS or p10 is below 3 TPS.
- Passing authorizes integration into a repeated stateful q2 runtime only; it
  is not endpoint TPS or a production default.
- No model value, route, capability, precision, or arithmetic may change.

## Result

The whole-frame cache construction fits all 480 first-row experts and the 77
largest fitting second-only experts in 4,258,605,775 compressed bytes. Of 207
second-only experts, 130 remain misses: 993,234,397 compressed bytes,
994,623,488 page-aligned physical bytes, and 1,277,952,000 decompressed source
bytes. All 687 container frames reproduced their exact source hashes before
timing.

Parallel cold read plus decompression scales as follows:

| Workers | Complete wall | Slowest-worker read | Slowest-worker decode |
| ---: | ---: | ---: | ---: |
| 1 | 1,453.408 ms | 321.684 ms | 1,131.615 ms |
| 2 | 865.949 ms | 284.204 ms | 584.272 ms |
| 4 | 439.910 ms | 132.349 ms | 310.727 ms |
| 8 control median | 327.379 ms | 127–135 ms | 195–201 ms |

Every trial reports exactly 994,623,488 physical bytes and zero resident page
instances after cold preparation. The interleaved 8-worker control/candidate
series gives:

- storage-plus-decode control median: 327.379 ms;
- exact 96-execution Metal-overlap median: 401.772 ms;
- candidate p10/median/p90 wall: 388.160 / 401.772 / 407.665 ms; and
- favorable accepted TPS p10/median/p90: **4.905990 / 4.977945 / 5.152521**.

The candidate wall matches its 388–408-ms Metal interval, so parallel physical
read and exact decompression are fully hidden on this schedule. Host safety
passes with 51% free memory, no swap growth or throttled pages, and a 256.7-MB
final process physical footprint. The run used macOS 26.6.2 build 25G83 and
Rust 1.96.0.

```shell
target/release/firewing bench-parallel-zstd-overlap \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  fixtures/mixture/qwen3_8_flash_next_real.json \
  kernels/bf16_gemv.metal \
  /Users/chad/Models/firewing/evidence/FW-0046/q2-zstd1-manifest-a782e77.json \
  /Users/chad/Models/firewing/evidence/FW-0046/q2-zstd1.fwz \
  121dd5a2d5a34be4ab0198e1045e317051241b31 \
  /Users/chad/Models/firewing/evidence/FW-0046/parallel-zstd-overlap-121dd5a.json
```

Raw receipt SHA-256:
`fa27310db856c9a2ef2cde1ce2f1a66e0be29f7db1a0b0dfe8357f83216c2c51`

The repository has 65 passing Rust tests, five focused Python tests, and strict
Clippy passes.

## Decision

Promote parallel page-aligned zstd acquisition as a conditional architecture
survivor. It reverses FW-0045's serial-decoder rejection only for bounded
parallel decoding; it does not promote a production runtime or establish
endpoint TPS.

The remaining favorable assumptions are now decisive: the initial 4.26-GB
compressed cache is free and future-known, all misses launch before causal
layer routing, layer-0 Metal stands in for every routed layer, and all MTP,
fixed, attention, routing, sampling, synchronization, and cache-management work
is free. The next experiment must carry a realizable compressed cache across
both exact q2 transactions and compare causal policies against the oracle. A
favorable single transaction is not enough to justify allocating the full
resident system.
