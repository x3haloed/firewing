# FW-0035 - Exact resident top-10 Metal MoE

- Status: completed
- Disposition: compute survivor; serial storage-plus-compute rejected
- Date: 2026-09-03
- Parent experiments: FW-0031, FW-0034
- Exactness: real layer-0 top-10 BF16 route and source-order mixture
- Hardware: Apple M1 Mac mini, 16 GiB, internal SSD, no companion hardware

## Question

Can a production-shaped, exact, resident ten-expert Metal transaction execute
the routed MoE portion of all 48 layers inside Firewing 4's 250 ms/token
budget?

FW-0034 left a 12-GiB analytical storage survivor, but it charged no execution
or runtime buffers. FW-0031's individually synchronized expert primitive could
not answer whether ten dispatches amortized together would saturate the GPU or
serialize. This experiment installs the real layer-0 route's 98,304,000 BF16
source bytes into persistent shared Metal buffers and dispatches all ten
gate/up projections in one command buffer and all ten down projections in a
second. Exact CPU BF16-staged SwiGLU lies between the transactions; exact route
weighting and source-order BF16 mixture accumulation follow the second. All are
inside each candidate timing.

The scheduling pattern—persistent source bindings and amortized dispatch in a
small number of Metal transactions—was adapted from Prismwing's
`wide_metal_moe.rs` at commit
`c87d0c1aa2c118f71ca5348434be35d02f62f031`, SHA-256
`ad3362042331a09a4f9077d9e3fba84d555c4d667bd6fa5b28ddfc32840612a0`.
Firewing independently binds Qwen's tensors, BF16 reduction, route, and output
hashes; no MiMo or FP8 semantics transfer.

## Frozen method

- Clean implementation commit:
  `1690332743a221611ddf59c84686dfda82e3b42d`
- Kernel SHA-256:
  `ef9e463a6204298a0c098fb15fd8f863fd67b6a0bb7c294b0a1a298d0eca64d6`
- Persistent Metal allocation declaration: 98,398,736 bytes
- Batch size: 1
- Concurrency: 1
- Warmups: 3
- Interleaving: five scalar controls, each followed by six candidates
- Candidate synchronization: two command buffers and two completion waits
- Cache state:
  `warm_application_exact_top10_bf16_weights_persistent_in_shared_metal_buffers_install_excluded`

The complete FW-0011 mixture verifier runs first. Every warmup and measured
candidate must match all ten route-weighted expert hashes and the final
source-order mixture hash. Weight loading, payload hashing, Metal compilation,
and installation are reported separately and excluded from the candidate
interval.

```shell
cargo build --release
target/release/firewing bench-metal-top10-moe \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  fixtures/router/qwen3_8_flash_next_real.json \
  fixtures/expert/qwen3_8_flash_next_real.json \
  fixtures/mixture/qwen3_8_flash_next_real.json \
  kernels/bf16_gemv.metal \
  1690332743a221611ddf59c84686dfda82e3b42d \
  /Users/chad/Models/firewing/evidence/FW-0035/metal-top10-moe-16903327.json
```

Accepted tokens are zero, and `performance_claim` is null. This is a routed
component experiment, not endpoint TPS.

## Result

| Measurement | Result |
| --- | ---: |
| Scalar control median | 38.555 ms |
| Metal candidate p10 | 3.051 ms |
| Metal candidate median | 3.639 ms |
| Metal candidate p90 | 4.795 ms |
| Median speedup | 10.593738x |
| 48-layer projection at candidate median | 174.690 ms |
| Routed-only projected rate | 5.724426 TPS |
| Median budget remaining before all other work | 75.310 ms/token |

The exact routed compute path survives the isolated Firewing-4 screen. Its
48-layer median projection consumes 69.876% of the entire 250-ms budget, while
the p10/p90 projections span 146.440 to 230.166 ms. The result is substantially
better than serially extrapolating FW-0031 because ten independent projections
share each command transaction.

It does not make FW-0034's 12-GiB design sufficient. That oracle's aggregate
storage interval is 165.318 ms/token. Adding it serially to this experiment's
174.690-ms routed interval yields 340.008 ms/token, or 2.941 TPS, before
attention, shared experts, routing, n-gram lookup, final projection, sampling,
or cache management. Perfect overlap would instead expose the larger
174.690-ms interval and leave 75.310 ms for everything else. Real overlap and
buffer residency therefore decide whether the branch survives.

The clean receipt recorded zero swap growth, zero new throttled pages, 59%
system memory free through completion, and a 125.4-MB maximum observed process
physical footprint. The final release checkpoint occurs after the Rust Metal
objects are dropped; Darwin retained allocator/driver pages, but the footprint
remained far below the 4-GiB post-phase bound.

Raw receipt:
`/Users/chad/Models/firewing/evidence/FW-0035/metal-top10-moe-16903327.json`

Receipt SHA-256:
`93f3704049c9972d7d56b20b11c41a704a8cd5ab41ad3bdc4f4824a3ec5f81bb`

The repository has 47 Rust tests and strict Clippy passes.

## Decision

Retain amortized exact Metal MoE as a validated compute primitive. Do not call
5.724426 routed-only TPS endpoint throughput and do not build the 12-GiB cache
yet. The next cheap falsifier must enclose actual miss transport and Metal
execution with bounded buffers so overlap, installation, and contention are
charged. If that enclosing lower bound cannot retain 4 median / 3 p10 TPS
before fixed non-MoE work, redirect to exact lossless representation or MTP
union reduction rather than increasing residency.
