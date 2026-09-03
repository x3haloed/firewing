# FW-0013 - All-layer source expert acquisition

- Status: completed
- Disposition: rejected for raw all-miss Firewing 4; retained diagnostic
- Date: 2026-09-03
- Parent experiments: FW-0001, FW-0008, FW-0009, FW-0012
- Exactness: L0 source payload acquisition; no arithmetic change
- Hardware/runtime: Apple M1 Mac mini (`Macmini9,1`), 16 GiB, internal SSD

## Question and hypothesis

Can the pinned raw BF16 checkpoint supply one ordinary token's complete routed
expert payload quickly enough to remain a plausible direct-streaming path?
Every token selects ten 9,830,400-byte experts in each of 48 layers, so the
logical source trace is exactly 4,718,592,000 bytes before cache hits,
filesystem widening, shared/fixed weights, compute, or MTP union.

The hypothesis is that parallel positional reads improve over serialization
but remain far below the 18,874,368,000 logical bytes/s required to supply four
all-miss tokens/s. This cheap test can reject only direct raw-source all-miss
streaming. It cannot reject expert residency, lossless recoding, cache hits,
MTP amortization, or Firewing 4 as a whole.

## Frozen authority and baseline

- Checkpoint revision:
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Model lock SHA-256:
  `f87399e8659ab3274601fcd455b78b73c600f57e5fc1e91499eec3ac1f4b9444`
- Baseline commit: `4e08302ac24cdbdd577afe8a8702589cd7913dc4`
- Frozen acquisition fixture SHA-256:
  `06847b6729503e47bfc87eccc35099327bd92370e421a141c370178c921aea08`
- Semantic references: Transformers 5.16.1 top-10 router; FW-0009 router
  precision; FW-0010 bounded expert tensor layout
- Transport references: FW-0008 and Prismwing's bounded `pread`, Darwin
  process-I/O counter, `F_NOCACHE`, `F_RDAHEAD=0`, range invalidation, and
  interleaved worker-count patterns pinned in `docs/SOURCES.md`

- Implementation commit:
  `b0e5f72836c08dd565d9fe5ea8242ab3690527f4`
- Toolchain: Rust/Cargo 1.96.0; macOS 26.6.2 (`25G83`)
- Raw external receipt:
  `/Users/chad/Models/firewing/evidence/FW-0013/expert-acquisition.json`
- Raw receipt SHA-256:
  `b8e5a175c0402bced494ebb1cc4a61f903f2ff8a1a094fa8a17d043311f942b5`

## Method and commands

Generate one deterministic BF16 router input per layer using a frozen affine
formula derived from the layer index. Run the pinned framework router for all
48 layers. For each selected expert, record the exact gate/up and down tensor
name, shard, absolute safetensors payload offset, logical byte count, and
SHA-256. Commit only metadata and hashes, never weight bytes.

The native benchmark must verify the model lock, fixture identity, tensor
shapes and offsets, unique top-10 list per layer, every selected payload hash,
and the 4,718,592,000-byte total before reporting measurements. It may reuse
bounded buffers by layer; it must not allocate one 4.7 GB destination.

Run worker counts 1, 2, 4, and 8 in these interleaved orders:
`[1,2,4,8]`, `[8,4,2,1]`, and `[2,8,1,4]`. For each cold trial, map only the
selected page-aligned ranges, apply `MS_INVALIDATE` and `MADV_DONTNEED`, verify
their resident-page count before and after invalidation, then use descriptors
configured with `F_NOCACHE=1` and `F_RDAHEAD=0`. Post-prefault trials use
cacheable descriptors after one declared full-trace prefault. Hash verification
is recorded separately from positional-read time and included in complete
diagnostic wall time.

```shell
.venv/bin/python tools/generate_expert_acquisition_fixture.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  --model-lock spec/model.lock.json \
  --output fixtures/expert_acquisition/qwen3_8_flash_next_all_layers.json

cargo run --release -- bench-expert-acquisition \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  fixtures/expert_acquisition/qwen3_8_flash_next_all_layers.json \
  IMPLEMENTATION_COMMIT \
  /Users/chad/Models/firewing/evidence/FW-0013/expert-acquisition.json
```

Batch size and concurrency are one. Accepted tokens and `A` are zero because
this is a component diagnostic; `U=480` selected layer/expert identities per
trace (ten within each layer). No measured rate is accepted TPS.

## Gates

- Fixture: exactly 48 layers, ten unique selected experts per layer, 960
  nonempty bounded extents, 4,718,592,000 logical bytes, exact per-extent hashes,
  and deterministic regeneration.
- Correctness: every trial returns the exact requested bytes and matches all
  960 payload hashes. Unknown tensors, shapes, offsets, revisions, or schemas
  fail closed.
- Cache state: cold preparation and resident-page observations are reported;
  any cold trial with zero process physical bytes is invalid, preserved, and
  cannot support a cold conclusion.
- Safety: maximum reusable destination capacity is below 256 MiB; no checkpoint
  or evidence file is modified; all large evidence remains outside Git.
- Direct-stream continuation: a robust cold result at or below 250 ms/trace
  keeps raw all-miss Firewing-4 supply plausible for endpoint testing.
- Direct-stream kill: if every valid worker configuration has cold median above
  250 ms/trace, reject raw all-miss direct streaming for Firewing 4. Above
  1,000 ms/trace also rejects it for Firewing 1.
- No configuration becomes a runtime default without a complete endpoint gain.

Excluded claims: shared/fixed traffic, arithmetic, output correctness, expert
cache hit rate, real-activation expert identity or union, MTP `A/U`, endpoint
latency, full capability, and accepted TPS.

## Result

All 24 measured trials verified all 960 extent hashes and the exact
4,718,592,000 logical-byte ledger. The destination upper bound was 52,559,872
bytes. Every cold trial reported 4,734,296,064 process physical bytes (median
for every worker count), while the aligned request ledger was 4,734,288,216
bytes. Cold preparation observed zero selected resident-page instances after
invalidation in every trial.

| Workers | Cold complete median | Cold slowest-reader median | Logical GB/s, complete |
| ---: | ---: | ---: | ---: |
| 1 | 15,036.522 ms | 1,636.904 ms | 0.314 |
| 2 | 10,175.853 ms | 1,111.395 ms | 0.464 |
| 4 | 5,303.798 ms | 551.411 ms | 0.890 |
| 8 | 3,289.722 ms | 372.548 ms | 1.434 |

The slowest worker's timed `pread` total is the conservative transport-only
critical-path proxy; complete wall time additionally includes mandatory
per-extent SHA-256 verification. Even the best transport-only proxy is 1.490x
the 250 ms Firewing-4 source budget (12.666 logical GB/s versus the required
18.874 GB/s). Complete diagnostic wall time is 13.159x the budget.

The intended warm-state prediction failed. The prefault itself physically read
4,808,843,264 bytes and, immediately afterward, only 172,044 of 288,960
selected page instances (59.539%) were resident. Subsequent cacheable trials
continued to report substantial physical I/O, so they are preserved and named
`post_prefault_cacheable_exact_pread`; they are not reported as warm-OS-cache
measurements. This establishes that one full trace pass does not produce the
assumed resident state on this host, without assigning an unevidenced cache
replacement cause.

Protocol deviation: the first clean implementation run at commit `c0c1544`
was overwritten when the receipt schema was amended to expose the unexpected
post-prefault residency and per-worker read critical path. Its receipt hash was
`2fcb092556f1d94f990a74c5627d64c4a04186b7cca16751e44a50e510eb9259`;
the final clean rerun above is the retained raw artifact. This preservation
failure does not affect the final run's gates, but future reruns must use unique
receipt names.

## Decision

Reject direct raw-BF16 all-miss expert streaming as the Firewing-4 runtime path
under this schedule and hardware. Eight readers improve acquisition materially,
but do not reach the 250 ms transport budget before any expert arithmetic,
shared/fixed traffic, n-gram traffic, or MTP union is added. The
transport-only critical path remains below 1,000 ms, so this result does not
conservatively reject a one-token/s raw transport path; the slower complete
diagnostic includes integrity work that a production runtime would not perform.

Firewing 4 itself remains open. The next useful branch must reduce physical
expert demand through measured reuse/residency, lossless recoding, or MTP union
amortization rather than optimizing this all-miss raw schedule as the default.
