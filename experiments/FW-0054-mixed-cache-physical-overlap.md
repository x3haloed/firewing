# FW-0054 - Mixed-cache physical overlap

- Status: completed
- Disposition: rejected
- Date: 2026-09-03
- Parent experiment: FW-0053
- Exactness: L1 compressed frames and exact decoded BF16 payloads
- Hardware: Apple M1 Mac mini, 16 GiB, internal SSD

## Question and hypothesis

Can FW-0053's favorable offline mixed-representation schedule be physically
installed and replayed above four accepted TPS while respecting the target's
host-safety contract? The hypothesis was that its 4,259,199,939-byte maximum
residency would fit safely and that real cold reads, all 1,109 decodes, inverse
shuffle, installation copies, and 192 exact fused Metal executions could still
finish inside one second.

## Frozen method

- Implementation commit:
  `233958f200e3c15c24917c5abdee32ca521f85b0`
- FW-0053 receipt SHA-256:
  `a61d498af5512c3fcbdc3447d8217383ddacf39361cdea553a41a79d9a10cb3f`
- Sequential container SHA-256:
  `b14d0f9827a001b97495b97f11d111495f94e8c7392e0ec7d9e7f39095a372bb`
- Batch size/concurrency: 1 / 1
- `q=2`, `A=4`, `sum_equivalent_U=2.995833`
- macOS 26.6.2 (25G83), Rust 1.96.0

The implementation independently authenticates and replays every FW-0053
interval and capacity boundary. It preloads the 633 initial compressed frames,
cold-invalidates the 464 physical misses, and uses eight workers for every
required compressed miss or hit decode, exact BF16 inverse shuffle, and a full
9,830,400-byte host installation copy. Candidate trials concurrently execute
the exact FW-0052 fused Metal workload 192 times. Controls and candidates are
predeclared in interleaved order. The complete future and decoded-hit residency
remain favorable grants; causal layer barriers and fixed endpoint work remain
free.

```shell
target/release/firewing bench-executable-cache-overlap \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  fixtures/mixture/qwen3_8_flash_next_real.json \
  kernels/bf16_gemv.metal \
  /Users/chad/Models/firewing/evidence/FW-0049/q2-sequential-bf16-shuffle-zstd1-manifest-6271f3d.json \
  /Users/chad/Models/firewing/evidence/FW-0049/q2-sequential-bf16-shuffle-zstd1.fwz \
  /Users/chad/Models/firewing/evidence/FW-0053/executable-cache-5f69eee.json \
  233958f200e3c15c24917c5abdee32ca521f85b0 \
  /Users/chad/Models/firewing/evidence/FW-0054/executable-overlap-233958f.json
```

## Gate and result

The target requires an immediate stop on any swap growth. No performance gate
was reached. At `cache_and_metal_install_complete`, before warmups or a timed
trial, swap had grown by **733,741,056 bytes** from the run baseline. The
process physical footprint was 4,409,634,304 bytes, system-free memory was 34%,
and no new throttled pages appeared. The runner stopped with exit code 1; zero
timed samples and no TPS result were produced.

This is a host-safety rejection, not a slow-performance result. The 4.259-GB
mixed cache cannot be promoted on the observed shared target host merely
because its logical capacity fits below a nominal memory budget. FW-0052 had
already installed the much smaller exact Metal runner without swap growth, but
this checkpoint combines cache and runner installation, so the precise
allocation boundary remains unresolved.

Raw failure receipt:
`/Users/chad/Models/firewing/evidence/FW-0054/safety-stop-233958f.json`

Receipt SHA-256:
`2a6442a92213d3f9a304762ede3ea3dd19485fb89ca72fbc35363bb8df6d2203`

The implementation passed strict Clippy, 68 Rust tests, 84 Python tests, and a
release build before execution.

## Decision

Reject FW-0053's 4.259-GB physical instantiation as the next runtime default on
the observed shared host. Preserve its mathematical bound, but do not report
or infer physical TPS from it. The next cheap experiment should reduce the
mixed-cache capacity and solve the resulting traffic/decode frontier before
allocating another multi-gigabyte cache. A smaller schedule must first retain
a greater-than-four-TPS optimistic bound, then demonstrate zero swap growth in
a separately checkpointed cache-only installation before repeating Metal
overlap.

This result does not reject exact BF16 shuffle/zstd, smaller caches, a cleaner
but still target-compliant host state, or another exact executable
representation. It does reject treating nominal free percentage or logical
cache capacity as sufficient residency evidence.
