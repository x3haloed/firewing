# FW-0031 - Exact Metal BF16 real expert

- Status: completed
- Disposition: conditional primitive; endpoint integration deferred
- Date: 2026-09-03
- Parent experiment: FW-0030
- Exactness: L1 function-preserving execution with exact tested BF16 captures
- Hardware/runtime: Apple M1 Mac mini (`Macmini9,1`), 16 GiB, internal SSD,
  macOS 26.6.2 (`25G83`)

## Hypothesis and prediction error

FW-0030 attributed 90.36% of endpoint wall time to projection-heavy decoder
and attention calls. The initial hypothesis was that accelerating their BF16
GEMVs would therefore be the first dominant endpoint optimization.

The real-expert result contradicts that causal interpretation. Warm in-memory
CPU execution is already milliseconds, while repeated source loading and hash
verification is an order of magnitude slower. FW-0030's outer intervals
conflated arithmetic with checkpoint acquisition and integrity work. The Metal
kernel remains useful, but immediate whole-endpoint kernel substitution is no
longer the highest-value next action.

## Authority, implementation, and mechanism

- Checkpoint revision:
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Model-lock SHA-256:
  `f87399e8659ab3274601fcd455b78b73c600f57e5fc1e91499eec3ac1f4b9444`
- Real expert fixture SHA-256:
  `10315f99986464e85e186cc32d55488d9c68f7db0979f5cef1411c6b7e8a4752`
- Clean implementation commit:
  `cc412d40f35b4b46dd26a7e2e41764fd5a2d7544`
- Kernel SHA-256:
  `ef9e463a6204298a0c098fb15fd8f863fd67b6a0bb7c294b0a1a298d0eca64d6`
- Rust/Cargo 1.96.0, Metal crate 0.33.0, release LTO

The kernel assigns one 32-thread group to each output row. Its 32 F32 partial
sums reproduce Firewing's source-derived PyTorch-aarch64 accumulation topology:
16-, 8-, and 4-wide pairwise register reductions followed by
`(partial[0] + partial[1]) + (partial[2] + partial[3])`. Fast math is disabled
and output uses explicit round-to-nearest-even BF16 staging.

Each measured candidate execution covers both real expert projections,
CPU BF16-staged SwiGLU, route weighting, command encoding, two synchronous
completion waits, output readback, and fresh copies of the 9,830,400 source
bytes into bounded shared Metal buffers. It does not hide the expert buffers in
compile time. Three candidate warmups precede five interleaved groups, each
containing one scalar control followed by six Metal candidates, for five
control and 30 candidate measurements.

```shell
cargo run --release -- bench-metal-bf16-gemv \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  fixtures/router/qwen3_8_flash_next_real.json \
  fixtures/expert/qwen3_8_flash_next_real.json \
  kernels/bf16_gemv.metal \
  cc412d40f35b4b46dd26a7e2e41764fd5a2d7544 \
  /Users/chad/Models/firewing/evidence/FW-0031/metal-bf16-gemv-cc412d40.json
```

Batch and concurrency are one. Cache state is
`warm_application_bf16_expert_weights_copied_into_bounded_metal_buffers_each_projection`.
Accepted tokens, `A`, and endpoint TPS are zero/not applicable.

## Gates

- Run the complete FW-0010 real-expert authority before the candidate.
- Authenticate the real layer-0 expert-376 gate/up and down payloads and the
  deterministic BF16 input.
- Match exact BF16 hashes for combined gate/up, SwiGLU, down, and route-weighted
  output on every warmup and measured candidate.
- Use both production Qwen shapes: 1,280 by 2,560 and 2,560 by 640.
- Predeclare the 9,830,400-byte resident expert and enforce all host-safety
  thresholds before authority loading, after compilation, after warmup, after
  measurement, and after release.
- Treat timing as an isolated warm-application component result, never TPS.

Excluded: checkpoint streaming, cold storage, all ten routed experts, mixture
reduction, shared expert, complete layer, endpoint gain, MTP, accepted tokens,
modalities, and sustained performance.

## Result

All 33 Metal executions match all four tested BF16 capture hashes exactly. The
scalar controls also retain the complete FW-0010 six-capture authority.

| Path | p10 | Median | p90 |
| --- | ---: | ---: | ---: |
| Scalar/vectorized CPU control | — | 3.546 ms | — |
| Exact Metal candidate | 1.454 ms | 1.540 ms | 1.702 ms |

The candidate is 2.301836x faster at median. Runtime Metal compilation takes
1.130 ms in the clean run. However, re-reading and hashing the same warm source
expert immediately after authority verification takes 31.064 ms, 8.76x the
CPU compute median and 20.17x the Metal median. Complete authority verification
takes 37.932 ms. This directly falsifies the assumption that arithmetic was the
main cost inside FW-0030's coarse projection intervals.

The process records zero physical read bytes during the clean warm-cache
run, so the 31.064-ms source interval is not a cold-SSD result. It includes file
open/header parsing, copied range reads from resident pages, and SHA-256. It
does not identify their individual shares.

All safety gates pass: system-free memory remains at least 63%, physical
footprint stays at or below 30,016,960 bytes, peak RSS stays at or below
46,301,184 bytes, swap and throttled pages do not grow, and protected services
remain live. The process writes no bytes during the measured run.

Raw receipt:
`/Users/chad/Models/firewing/evidence/FW-0031/metal-bf16-gemv-cc412d40.json`

Receipt SHA-256:
`25c5efdab0cc55156139408601ee8d2620a03acb1bb8413da48d6ab41733d287`

Two earlier receipts are preserved but excluded from the result. The dirty
exploratory run used an all-zero placeholder commit and hashes to
`8220fba2a08e475940fd069fe0ede290ffb2b835c199b4722e109fb80ae48008`.
The first clean rerun was accidentally passed a fabricated expansion of the
seven-character commit prefix rather than the actual full hash; its otherwise
passing receipt hashes to
`88408fd4940a45f6a72a707242836580b2607486e2fe3862b6292ecbe37b5dd5`.
Neither receipt contributes reported timing. The final rerun above binds the
actual clean commit returned by Git.

The repository gate has 65 Python and 44 Rust tests, and strict Clippy passes.
The `block` transitive dependency emits a future-incompatibility notice under
Rust 1.96.0; it is not a current build or test failure.

## Decision

Retain the exact Metal GEMV as a conditional production-shaped primitive, but
do not promote it into the endpoint yet. First replace repeated per-use
open/header/read/hash work with a fail-closed tensor catalog authenticated once
at startup and bounded views during inference. Reprofile the unchanged exact
endpoint to separate remaining storage transport from arithmetic. Only then can
an interleaved endpoint Metal integration test estimate system-level value.

This correction does not reverse FW-0013: raw all-miss expert streaming remains
rejected for Firewing 4. It narrows the next question to how much current wall
time is verifier overhead versus unavoidable source traffic, after which route
reuse, residency, lossless recoding, and MTP union remain the candidate paths.
