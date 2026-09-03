# FW-0068 - Modified block-8x2 affine-UINT8 real-layer screen

- Status: completed
- Disposition: rejected
- Date: 2026-09-03
- Parent experiment: FW-0067
- Mode: `modified_block8x2_affine_uint8_weight_only`
- Hardware: Apple M1 Mac mini, 16 GiB, internal SSD

## Question and method

Does FW-0067's selected 8x2 affine-UINT8 topology pass the authenticated
early/middle/late real-layer gate? Regenerate the complete committed two-token
48-layer source fixture, hold source routes fixed, and require every mixture at
layers 4, 24, and 46 to remain at or below 1% relative L2 and every expert at
or below 2%.

Implementation commit:
`f5ff9e05e95efa11383f126c6862abaed7827db6`

```shell
.venv/bin/python tools/analyze_block4_int8_real_layers.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  f5ff9e05e95efa11383f126c6862abaed7827db6 \
  /Users/chad/Models/firewing/evidence/FW-0068/block8x2-affine-uint8-real-layers-f5ff9e0.json \
  --block-rows 8 \
  --block-columns 2 \
  --weight-format affine_uint8
```

The complete source regeneration matches fixture SHA-256
`6ed2e16da1e64fb8001c26608d7972f4910190f74768055b5778dc7891ebf525`.
Every candidate uses 0.65625 of BF16 weight bytes.

| Layer | Token state | Mixture relative L2 | Worst expert |
| ---: | ---: | ---: | ---: |
| 4 | 0 | 0.009780 | 0.009784 |
| 4 | 1 | 0.009826 | 0.010309 |
| 24 | 0 | 0.007588 | 0.008503 |
| 24 | 1 | 0.010970 | 0.009830 |
| 46 | 0 | 0.009863 | 0.010332 |
| 46 | 1 | 0.010107 | 0.009486 |

Two mixtures fail 1%; the worst reaches **1.0970%**. Every expert passes 2%.
The raw receipt SHA-256 is
`ce7801f16eef55de08e3c9ae955755093f7fd08484d8a66e44a89f2b1c458e61`.
No performance claim is made.

## Decision

Reject selected 8x2 before bank or kernel work. Do not yet reject affine UINT8
as a family: all four other equal-byte topologies passed FW-0067 and topology
rank on one layer need not generalize. Run those four under the same gate, then
either select a genuine survivor or close the family.
