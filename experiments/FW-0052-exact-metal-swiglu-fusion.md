# FW-0052 - Exact Metal SwiGLU fusion

- Status: completed
- Disposition: promoted component implementation
- Date: 2026-09-03
- Parent experiment: FW-0035
- Exactness: L1 exact BF16 semantics through a real layer-0 top-10 mixture
- Hardware/runtime: Apple M1 Mac mini, 16 GiB, internal SSD, no companion hardware

## Question and hypothesis

Can the exact resident routed-MoE primitive recover meaningful endpoint budget
by keeping gate/up, BF16-staged SwiGLU, and down projection in one Metal command
buffer instead of synchronizing through CPU SwiGLU between two buffers?

Prismwing showed that a 65,536-entry lookup table can preserve a finite BF16
SiLU domain without depending on a device `exp` implementation. Firewing uses
that scheduling insight only: it constructs its table from Firewing's CPU
reference and independently verifies Qwen's real tensors and captures.

## Frozen authority and baseline

- Implementation commit: `c2bac8514c807ff0d2f2f47a0d918cf91c9ec54b`
- Kernel SHA-256: `d65650d6809124118ecea88fbffdc7e6cd6b14fd04a99565bddc828134054349`
- Checkpoint revision: `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Correctness authority: FW-0011's real layer-0 ten-expert mixture fixture
- Control: the existing two-command-buffer Metal path with CPU BF16 SwiGLU
- Candidate: one Metal command buffer with gate, LUT SwiGLU, and down encoders
- Persistent candidate allocation declaration: 98,529,812 bytes, including a
  131,072-byte SiLU table and four-byte shape buffer

The Prismwing scheduling sources are pinned at commit
`c87d0c1aa2c118f71ca5348434be35d02f62f031`: kernel SHA-256
`9bc149eee32ebf28af35929d5fa160edfe9e1767cdcde59a54ec61b7016882ee`
and host code SHA-256
`3eedceed3c4ff6de4e76d15047d91cb16e18919eb46eac021cbcaecfbde9c85a`.
MiMo dimensions, FP8 arithmetic, fixtures, and measurements do not transfer.

## Method and gates

An exhaustive deterministic fixture checks the table against the CPU BF16
boundary for every finite BF16 gate value. The physical probe then runs three
warmups of both implementations and 30 interleaved control/candidate pairs.
Every execution must match all ten route-weighted expert hashes and the final
source-order mixture hash. Batch size and concurrency are one, weights remain
warm and resident, and installation is excluded.

Promote only if exactness and host safety pass and the candidate has a material
median gain. This is a resident routed component test with `accepted_tokens=0`,
not endpoint TPS; storage, cache management, routing, attention, shared experts,
MTP, logits, and sampling are omitted.

```shell
cargo build --release
target/release/firewing bench-metal-top10-moe \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  fixtures/router/qwen3_8_flash_next_real.json \
  fixtures/expert/qwen3_8_flash_next_real.json \
  fixtures/mixture/qwen3_8_flash_next_real.json \
  kernels/bf16_gemv.metal \
  c2bac8514c807ff0d2f2f47a0d918cf91c9ec54b \
  /Users/chad/Models/firewing/evidence/FW-0052/metal-swiglu-c2bac85.json
```

## Result

All 33 candidate executions and all 33 control executions pass the real-weight
hash gates. The candidate wins 21 of 30 paired samples.

| Measurement | Two-buffer control | One-buffer candidate |
| --- | ---: | ---: |
| p10 | 3.041 ms | 2.789 ms |
| median | 3.513 ms | 3.165 ms |
| p90 | 5.063 ms | 4.642 ms |

The ratio of medians is **1.110094x**; the median of paired ratios is 1.115422x.
At the candidate median, 48 routed layers project to 151.906 ms, or 6.583017
routed-only TPS. That consumes 60.76% of the 250-ms Firewing-4 token budget and
leaves 98.094 ms for all omitted work, versus 81.370 ms for the measured
two-buffer control in this run.

Host safety records no swap growth, throttling, or lost protected services,
53% final free memory, and a 125.6-MB final physical footprint.

Raw receipt:
`/Users/chad/Models/firewing/evidence/FW-0052/metal-swiglu-c2bac85.json`

Receipt SHA-256:
`d8561eae477282e59cf1ed32828f993ef0f99bafd6457f053e53f6df3221100b`

## Decision

Promote the one-command-buffer LUT SwiGLU path for the resident exact runner.
It is a repeatable component improvement and creates necessary budget, but it
does not alter FW-0051's offline-cache result or establish endpoint throughput.
Causal cache state and the fixed-work envelope remain unresolved.
