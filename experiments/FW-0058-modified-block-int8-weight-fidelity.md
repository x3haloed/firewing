# FW-0058 - Modified block-INT8 weight fidelity

- Status: completed
- Disposition: rejected
- Date: 2026-09-03
- Parent experiment: FW-0057
- Mode: `modified_block_int8_weight_only`
- Hardware: Apple M1 Mac mini, 16 GiB, internal SSD

## Question and hypothesis

Does symmetric INT8 preserve enough of the real routed mixture to keep the
same approximately half-size executable representation alive after block-FP8
failed FW-0057? The hypothesis was that signed linear codes with one F32 scale
per 128x128 block might keep layer-0 top-10 mixture relative L2 at or below 1%
and every selected expert at or below 2%.

This is explicitly modified mode. It does not preserve the checkpoint's BF16
weights, and passing this local screen would not make it target-faithful.

## Frozen authority and method

- Implementation commit:
  `e91f28569c14c27f6c72a2f6dce20f747ad7c335`
- Model lock SHA-256:
  `f87399e8659ab3274601fcd455b78b73c600f57e5fc1e91499eec3ac1f4b9444`
- Real layer-0 mixture fixture SHA-256:
  `975a9982919297d37dd077f774693c782295cba496542c6adf278182e27b4d89`

For each of the ten selected real experts, divide the `[1280,2560]` gate/up
and `[2560,640]` down matrices into 128x128 blocks. Store one signed INT8 code
per weight and one F32 absmax/127 scale per block. Dequantize weights to BF16,
then run the unchanged official BF16 expert equation, route weighting, and
expert-order BF16 accumulation.

The test remains favorable to the candidate: activations are exact BF16 and
every intermediate boundary remains BF16. The source path must reproduce all
ten expert hashes and the final mixture hash before approximate metrics count.

```shell
.venv/bin/python tools/analyze_block_fp8_weight_fidelity.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  fixtures/mixture/qwen3_8_flash_next_real.json \
  e91f28569c14c27f6c72a2f6dce20f747ad7c335 \
  /Users/chad/Models/firewing/evidence/FW-0058/block-int8-weight-fidelity-e91f285.json \
  --weight-format block_int8
```

## Gates and result

- Continue only if mixture relative L2 is at most 1%.
- Continue only if every expert weighted-output relative L2 is at most 2%.
- Never infer endpoint or hosted-reference fidelity from this one-layer screen.

All eleven exact baseline hashes match. The representation occupies 49,164,000
bytes versus 98,304,000 source bytes, a ratio of **0.500122**.

Fidelity still fails both continuation gates:

- Top-10 mixture relative L2: **0.021704**.
- Maximum expert weighted-output relative L2: **0.030226**.
- Best expert relative L2: **0.019013**.
- Mixture BF16 equality: **7.852%**.
- Mixture maximum absolute error: **0.0010376**.

Raw receipt:
`/Users/chad/Models/firewing/evidence/FW-0058/block-int8-weight-fidelity-e91f285.json`

Receipt SHA-256:
`055ba83f4d251aacb3a0cfc3db35d0d419a3e889aac8169bd6478740a5585509`

The shared format analyzer has deterministic block-INT8 quantization,
aligned-block, byte-ledger, and fail-closed tests. The Python suite has 90
passing tests.

## Decision

Reject naïve symmetric block-128 INT8 weight quantization and do not build a
kernel or deeper-layer fixture campaign for this form. It improves materially
over FW-0057's E4M3 result, but its favorable one-layer weight-only screen still
misses both frozen gates.

This does not reject smaller blocks, channelwise scales, mixed-precision
outliers, calibrated clipping, error recovery, or training. Those formats add
capacity or complexity and require separate byte ledgers and fidelity evidence.
It also does not reopen FW-0056's exact materialized-BF16 cache branch.
