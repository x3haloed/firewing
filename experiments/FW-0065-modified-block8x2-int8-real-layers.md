# FW-0065 - Modified block-8x2 INT8 real-layer screen

- Status: completed
- Disposition: rejected
- Date: 2026-09-03
- Parent experiment: FW-0064
- Mode: `modified_block8x2_int8_weight_only`
- Hardware: Apple M1 Mac mini, 16 GiB, internal SSD

## Question and method

Does FW-0063's remaining equal-byte topology pass the authenticated early,
middle, and late real-layer gate? Regenerate the complete committed two-token
48-layer source fixture and evaluate 8x2 symmetric INT8 with fixed source routes
at layers 4, 24, and 46. Every mixture must remain at or below 1% relative L2
and every expert at or below 2%.

Implementation commit:
`4532962ced82eeb425fc0d645c23396e3d2410b3`

```shell
.venv/bin/python tools/analyze_block4_int8_real_layers.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  4532962ced82eeb425fc0d645c23396e3d2410b3 \
  /Users/chad/Models/firewing/evidence/FW-0065/block8x2-int8-real-layers-4532962.json \
  --block-rows 8 \
  --block-columns 2
```

The regenerated source fixture exactly matches SHA-256
`6ed2e16da1e64fb8001c26608d7972f4910190f74768055b5778dc7891ebf525`.
All candidates occupy 0.625 of BF16 weight bytes.

| Layer | Token state | Mixture relative L2 | Worst expert |
| ---: | ---: | ---: | ---: |
| 4 | 0 | 0.010550 | 0.010218 |
| 4 | 1 | 0.010858 | 0.010223 |
| 24 | 0 | 0.008057 | 0.008980 |
| 24 | 1 | 0.011139 | 0.011335 |
| 46 | 0 | 0.010412 | 0.010677 |
| 46 | 1 | 0.010769 | 0.010786 |

Five mixtures miss 1%; the worst reaches **1.1139%**. Every expert passes 2%.
The raw receipt SHA-256 is
`1558fc5723fe4fb30d392f6d9da41e1a21b2b78db0b4c7d45ab924f1c1f9717b`.
No performance claim is made.

## Decision

Reject 8x2 before bank or kernel work. Together with FW-0058 through FW-0064,
this closes uncalibrated square and rectangular symmetric INT8 at every tested
scale topology that both has a byte advantage and earned a stronger screen.

A successor must change the quantization error itself—calibrated clipping,
affine zero points, outlier preservation, error propagation, recovery, or
training. Do not merely test more uncalibrated factorizations of 16 weights.
