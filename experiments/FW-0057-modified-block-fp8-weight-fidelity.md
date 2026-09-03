# FW-0057 - Modified block-FP8 weight fidelity

- Status: completed
- Disposition: rejected
- Date: 2026-09-03
- Parent experiment: FW-0056
- Mode: `modified_block_fp8_weight_only`
- Hardware: Apple M1 Mac mini, 16 GiB, internal SSD

## Question and hypothesis

Can a Prismwing-derived E4M3 block-FP8 executable form halve routed weight
traffic without already violating a strict real-layer fidelity screen? The
hypothesis was that per-128x128 absmax scaling might keep layer-0 top-10 mixture
relative L2 at or below 1% and every selected expert at or below 2%, making a
Metal kernel worth porting.

This is explicitly modified mode. It does not preserve the checkpoint's BF16
weights and cannot be called target-faithful merely because it passes a local
fixture.

## Frozen authority and method

- Implementation commit:
  `4e86afdd0a615ce36654f610c39cd8ff73bfc957`
- Model lock SHA-256:
  `f87399e8659ab3274601fcd455b78b73c600f57e5fc1e91499eec3ac1f4b9444`
- Real layer-0 mixture fixture SHA-256:
  `975a9982919297d37dd077f774693c782295cba496542c6adf278182e27b4d89`

For each of the ten selected real experts, divide the `[1280,2560]` gate/up
and `[2560,640]` down matrices into 128x128 blocks. Store one E4M3FN byte per
weight and one F32 absmax/448 scale per block. Dequantize weights to BF16, then
run the unchanged official BF16 expert equation, route weighting, and
expert-order BF16 accumulation.

This is favorable to the candidate: activations are not quantized and every
intermediate boundary remains BF16. The exact source baseline must reproduce
the fixture's ten expert hashes and final mixture hash before approximate
metrics are accepted.

```shell
.venv/bin/python tools/analyze_block_fp8_weight_fidelity.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  fixtures/mixture/qwen3_8_flash_next_real.json \
  4e86afdd0a615ce36654f610c39cd8ff73bfc957 \
  /Users/chad/Models/firewing/evidence/FW-0057/block-fp8-weight-fidelity-4e86afd.json
```

## Gates and result

- Continue only if mixture relative L2 is at most 1%.
- Continue only if every expert weighted-output relative L2 is at most 2%.
- Never infer endpoint or hosted-reference fidelity from this one-layer screen.

All eleven exact baseline hashes match. The representation occupies 49,164,000
bytes versus 98,304,000 source bytes, a ratio of **0.500122**.

Fidelity fails decisively:

- Top-10 mixture relative L2: **0.036827**.
- Maximum expert weighted-output relative L2: **0.046847**.
- Best expert relative L2: **0.036966**.
- Mixture BF16 equality: **5.078%**.
- Mixture maximum absolute error: **0.0015869**.

Raw receipt:
`/Users/chad/Models/firewing/evidence/FW-0057/block-fp8-weight-fidelity-4e86afd.json`

Receipt SHA-256:
`77fbbc0374f0af6f6e2e67943edd90447db9fd85090c31627154596869d50a71`

The quantizer has deterministic aligned-block, byte-ledger, and fail-closed
tests; the Python suite now has 89 passing tests.

## Decision

Reject direct block-128 E4M3 weight quantization and do not port Prismwing's
Metal kernel for this form. Weight error alone misses both gates; dynamic-FP8
activation quantization would add another error source and cannot make this
specific weight-only result satisfy its declared test.

This does not reject calibrated FP8, smaller blocks, mixed-precision outliers,
INT8, low-rank recovery, or training. Those forms add capacity and need new
development/holdout evidence. It also does not alter FW-0056's rejection of
the exact materialized-BF16 cache.
