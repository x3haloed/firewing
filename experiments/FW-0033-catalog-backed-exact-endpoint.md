# FW-0033 - Catalog-backed exact endpoint profile

- Status: completed
- Disposition: conditional full-path gain; physical-demand branch required
- Date: 2026-09-03
- Parent experiments: FW-0030, FW-0032
- Exactness: L0 checkpoint identity and L1 storage/lifetime substitution with
  exact whole-endpoint captures
- Hardware/runtime: Apple M1 Mac mini (`Macmini9,1`), 16 GiB, internal SSD,
  macOS 26.6.2 (`25G83`)

## Hypothesis and prediction error

FW-0032 showed that one expert can execute from a once-authenticated mapping in
3.362 ms instead of spending 31.064 ms on repeated open/copy/hash work. The
hypothesis was that propagating this source through the complete exact endpoint
would remove most of FW-0030's 77.703-second verifier cost.

The endpoint improves materially but not to the arithmetic-only range. The
smallest corrected model is that once-authentication removes redundant
integrity and header work while the 17-GB source working set still exceeds host
memory and must be physically supplied. That remaining traffic, plus CPU
arithmetic, is the next causal cut.

## Implementation and authority

- Checkpoint revision:
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Model-lock SHA-256:
  `f87399e8659ab3274601fcd455b78b73c600f57e5fc1e91499eec3ac1f4b9444`
- Live-identity manifest SHA-256:
  `9830d5ae87a0586b6c8090b0f05274e958eca062fa04504978fa0041b2b714df`
- Exact endpoint fixture: the unchanged FW-0029 two-token authority
- Clean implementation commit:
  `becc3b3f66eb0216618c58b0f8ea888d867d1573`

The explicit catalog endpoint command installs the FW-0032 catalog once. Every
loader validates the requested tensor name, indexed shard, dtype, shape, and
slice bounds before copying a bounded execution buffer. The original full
checkpoint SHA-256 receipt plus current live identity replaces repeated tensor
payload hashes; fixture and intermediate output hashes remain mandatory.
Consumed mmap ranges receive `MADV_DONTNEED` only after the copy is complete,
so mapped source pages cannot accumulate without bound.

The ordinary verifier remains the control and does not install a catalog. The
same release binary runs candidate/control/candidate. Batch and concurrency
are one. The fixture teacher-forces two positions and accepts no generated
tokens, so `A=0`, `U=0`, and accepted TPS is not applicable.

```shell
target/release/firewing bench-catalog-token-text-endpoint \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  /Users/chad/Models/firewing/evidence/FW-0032/checkpoint-live-identity-b3d7810.json \
  9830d5ae87a0586b6c8090b0f05274e958eca062fa04504978fa0041b2b714df \
  fixtures/tokenizer/qwen3_8_flash_next.json \
  fixtures/ngram/qwen3_8_flash_next.json \
  fixtures/ngram/qwen3_8_flash_next_row_hashes.json \
  fixtures/ple/qwen3_8_flash_next_layer1_decode.json \
  fixtures/endpoint/qwen3_8_flash_next_firewing_two_token.json \
  becc3b3f66eb0216618c58b0f8ea888d867d1573 REPORT_JSON

target/release/firewing verify-token-text-endpoint \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  fixtures/tokenizer/qwen3_8_flash_next.json \
  fixtures/ngram/qwen3_8_flash_next.json \
  fixtures/ngram/qwen3_8_flash_next_row_hashes.json \
  fixtures/ple/qwen3_8_flash_next_layer1_decode.json \
  fixtures/endpoint/qwen3_8_flash_next_firewing_two_token.json REPORT_JSON
```

## Gates

- Preserve all FW-0029 final logits, top-20 token IDs, routes, and intermediate
  capture hashes.
- Execute all 48 layers, both attention types, PLE, 960 dynamic expert
  selections, and both complete 248,320-value logit vectors.
- Account 17,068,332,800 logical source bytes and physical reads separately.
- Enforce the normative host-safety policy at catalog, embedding, every layer,
  endpoint, and release boundaries.
- Require a repeatable full-path gain in a candidate/control/candidate sequence.
- Make no accepted-TPS or sustained-throughput claim.

## Result

All exactness and safety gates pass. Both candidates and the control reproduce
the identical top-20 vectors and route/capture authority.

| Run | Complete wall | Physical reads | Decoder/MoE | Attention | Output |
| --- | ---: | ---: | ---: | ---: | ---: |
| Catalog candidate 1 | 49.947 s | 17.164 GB | 25.728 s | 17.302 s | 4.715 s |
| File/hash control | 81.338 s | 17.316 GB | 42.638 s | 30.720 s | 6.015 s |
| Catalog candidate 2 | 47.949 s | 17.186 GB | 24.122 s | 17.093 s | 4.606 s |

The candidate median is 48.948 seconds, a 1.661711x full-path speedup over the
interleaved control. Median decoder time falls 41.54%, attention 44.02%, and
output 22.52%. This confirms that repeated payload authentication was broadly
distributed across FW-0030's coarse buckets rather than confined to one
loader.

Median candidate physical traffic is 17,175,160,832 bytes, 1.0063 physical
bytes per logical byte, and effective complete-path supply is only 0.351 GB/s.
Normalized per teacher-forced position, the candidate takes 24.474 seconds and
reads 8.588 GB. It is 97.90x slower than the 250-ms Firewing-4 decode budget.
This is not an impossibility proof for caching, recoding, Metal, or MTP, but it
decisively reconfirms that raw all-miss source demand cannot be the final path.

Across the candidate pair, peak RSS is 5.64 GB or less, final physical
footprint is below 46 MB, system-free memory stays at least 55%, and swap and
throttled pages do not grow. The high peak RSS with low post-release physical
footprint is consistent with bounded copied tensors plus explicitly discarded
mapped pages; it is recorded, not treated as resident cache.

Raw receipts and SHA-256 values:

- candidate 1: `catalog-endpoint-becc3b3f.json`,
  `e7e3cf4f9c8b1d3d70451bb7222610239c83498fad4acd177bf92f8412eb367e`;
- control: `control-becc3b3f.json`,
  `461fd34a4271963f255d4e4530b4c979f80c30d3dff9a277515a9d38679fc706`;
- candidate 2: `catalog-endpoint-becc3b3f-candidate2.json`,
  `7bb597b13f9222452d300db85c1be09a865233e9cf2f55eff6b731e42bcb645e`.

All are under `/Users/chad/Models/firewing/evidence/FW-0033/`. The earlier
all-zero-commit exploratory receipt is retained separately and hashes to
`a6695513d2f6ff6a832416086aeda10bdb3e9370f3b31418d8029ba01aaa8bdc`;
it contributes no promoted timing.

The repository gate has 69 Python and 46 Rust tests, and strict Clippy passes.

## Decision

Retain the explicit catalog endpoint as the faster exact profiling path, but
do not call it a production generation default. It lacks generation, an 8K
prefill, MTP, and accepted tokens, and its source demand remains two orders of
magnitude outside the final budget.

The next experiment must reduce physical demand before broad kernel work. Use
the actual Qwen route trace and Firewing byte ledger to test an
impossible-favorable exact residency/cache bound first, following Prismwing
PW-0207/PW-0332's lesson that implemented residency can be correct yet nearly
neutral when the enclosing traffic cut does not move. Only a surviving bound
authorizes a resident cache; otherwise test a distinct lossless representation
or MTP union schedule.
