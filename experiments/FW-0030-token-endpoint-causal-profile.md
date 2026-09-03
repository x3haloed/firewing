# FW-0030 - Token endpoint causal profile

- Status: completed
- Disposition: conditional diagnostic; no runtime default promoted
- Date: 2026-09-03
- Parent experiment: FW-0029
- Exactness: L0 endpoint replay unchanged; measurement fields are observational
- Hardware/runtime: Apple M1 Mac mini (`Macmini9,1`), 16 GiB, internal SSD,
  macOS 26.6.2 (`25G83`)

## Question and hypothesis

Which measured portions of Firewing's first exact tokenizer-to-logits replay
dominate wall time, and therefore deserve the first accelerated implementation?

The hypothesis was that scalar BF16 projections in decoder/MoE and attention
would dominate, while tokenization, embedding lookup, safety telemetry, and
final output would be secondary. This profile is intended to choose engineering
work, not to establish accepted TPS: the endpoint still replays two frozen
teacher-forced positions layer-major, verifies hashes, and cannot feed its own
selected token into a subsequent step.

## Frozen authority and implementation

- Checkpoint revision:
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Model-lock SHA-256:
  `f87399e8659ab3274601fcd455b78b73c600f57e5fc1e91499eec3ac1f4b9444`
- Endpoint fixture SHA-256:
  `2eb3712c7959837a24fe6db5b5d7b1f87c9926b4dd80eae524590e43287331ca`
- Clean implementation commit:
  `138533f980cc15f9b3da21c5f8f72766dd39d0be`
- Rust/Cargo 1.96.0; release LTO; batch one; concurrency one
- Cache state:
  `uncontrolled_mixed_os_cache_no_application_tensor_cache`

The implementation records monotonic setup, embedding, per-layer attention,
per-layer decoder, safety-checkpoint, output-head, and complete wall times in
integer nanoseconds. Darwin process counters add physical read/write bytes to
every fail-closed safety snapshot. The arithmetic, tensor loading, fixture
authority, and exact output gates from FW-0029 are unchanged.

Command:

```shell
cargo run --release -- verify-token-text-endpoint \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  fixtures/tokenizer/qwen3_8_flash_next.json \
  fixtures/ngram/qwen3_8_flash_next.json \
  fixtures/ngram/qwen3_8_flash_next_row_hashes.json \
  fixtures/ple/qwen3_8_flash_next_layer1_decode.json \
  fixtures/endpoint/qwen3_8_flash_next_firewing_two_token.json \
  /Users/chad/Models/firewing/evidence/FW-0030/endpoint-causal-profile-138533f.json
```

Compiler time emitted by `cargo run` is outside the in-process measurement.
There is no discarded warm-up or outlier. The single mixed-cache observation is
adequate to rank large wall shares, but not to estimate cold, warm, median, p10,
or sustained performance.

## Gates and exclusions

- Preserve exact FW-0029 tokenizer, embedding, PLE, cache, routing, all-layer,
  final-mixer, complete-logit, and top-20 results.
- Record every measured interval in the raw receipt; do not derive timing from
  terminal timestamps.
- Record physical process I/O and all host-safety fields at all 51 boundaries.
- Keep cache state explicitly uncontrolled and make no accepted-token or TPS
  claim.
- Stop and preserve failed evidence on any semantic or safety mismatch.

Excluded: a reusable endpoint, self-fed decode, sampling, real prefill,
application tensor caching, controlled cold/warm comparisons, latency
percentiles, accepted TPS, MTP, modalities, and hosted parity.

## Result

All exact FW-0029 gates pass: seven tokenizer cases, two real embedding rows,
48 layers, 960 dynamic expert selections, both complete 248,320-value logit
vectors, and 17,068,332,800 authenticated logical payload bytes.

| Measured interval | Wall time | Complete-wall share |
| --- | ---: | ---: |
| Decoder/MoE across 48 layers | 40,678.125 ms | 52.35% |
| Attention across 48 layers | 29,531.664 ms | 38.01% |
| Final mixer and full LM head | 5,729.418 ms | 7.37% |
| Phase safety checks, including final | 1,392.890 ms | 1.79% |
| Setup | 341.448 ms | 0.44% |
| Embedding lookup | 29.767 ms | 0.04% |
| Complete in-process replay | 77,703.382 ms | 100% |

The 36 linear-attention layers consume 51,372.708 ms in aggregate. The 12
full-attention layers consume 20,198.283 ms. Full-attention layers are slower
per layer, but the more numerous linear layers consume more total time. Layer 1
is the slowest single layer at 1,703.453 ms because it includes PLE; ordinary
full-attention layers cluster near 1,680 ms.

The process physically reads 17,318,928,384 bytes and writes zero bytes during
the measured interval. This is 1.0147 physical bytes per authenticated logical
byte, but it is not a production traffic factor: verification hashes and the
two-position layer-major schedule are part of this implementation. Normalizing
the pair gives 38.852 seconds and 8.659 GB of physical reads per teacher-forced
position, or 0.0257 positions/s. This is **not** accepted TPS. Relative to the
250 ms Firewing-4 decode budget, the current correctness implementation is
155.4x too slow per position; that gap does not prove an optimized runtime
impossible because the verifier includes integrity work and amortizes shared
weights across the pair differently from a token-major generation loop.

All safety gates pass. System-free memory remains at least 59%, physical
footprint peaks at 2,981,894,848 bytes, peak RSS is 2,811,183,104 bytes, swap
does not grow, no new throttled page appears, and every initially resident
protected service survives.

Raw receipt:
`/Users/chad/Models/firewing/evidence/FW-0030/endpoint-causal-profile-138533f.json`

Receipt SHA-256:
`301dae9af10fddd2988603909ea2f015bb269bf5c5f63058f6735621a4a04115`

The repository gate at the implementation commit has 65 Python and 43 Rust
tests; strict Clippy passes.

## Decision

The profile confirms that projection-heavy decoder/MoE plus attention account
for 90.36% of complete wall time. Do not optimize tokenization, embedding, or
safety telemetry. The next useful slice is a production-shaped BF16 Metal GEMV
whose reduction order is designed against Firewing's source-derived PyTorch
aarch64 authority, first on a real selected Qwen expert and then at one complete
routed layer. Prismwing's one-row BF16 Metal specialization is useful scheduling
evidence, but its model-specific FP8 formats and MiMo arithmetic are not Firewing
authority and must not be ported as Qwen semantics.
