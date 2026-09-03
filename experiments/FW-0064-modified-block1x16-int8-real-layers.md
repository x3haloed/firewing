# FW-0064 - Modified block-1x16 INT8 real-layer screen

- Status: completed
- Disposition: rejected
- Date: 2026-09-03
- Parent experiment: FW-0063
- Mode: `modified_block1x16_int8_weight_only`
- Hardware: Apple M1 Mac mini, 16 GiB, internal SSD

## Question and method

Does FW-0063's best equal-byte topology pass the authenticated early, middle,
and late real-layer gate? Regenerate the complete committed two-token 48-layer
source fixture, hold source routes fixed, and evaluate 1x16 symmetric INT8 at
layers 4, 24, and 46. Each mixture must remain at or below 1% relative L2 and
each expert at or below 2%.

Implementation commit:
`8c4015aea440bd5243783f637b87205e4e08a9c6`

```shell
.venv/bin/python tools/analyze_block4_int8_real_layers.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  8c4015aea440bd5243783f637b87205e4e08a9c6 \
  /Users/chad/Models/firewing/evidence/FW-0064/block1x16-int8-real-layers-8c4015a.json \
  --block-rows 1 \
  --block-columns 16
```

The regenerated source fixture exactly matches SHA-256
`6ed2e16da1e64fb8001c26608d7972f4910190f74768055b5778dc7891ebf525`.
All candidates occupy 0.625 of BF16 weight bytes.

| Layer | Token state | Mixture relative L2 | Worst expert |
| ---: | ---: | ---: | ---: |
| 4 | 0 | 0.010305 | 0.010460 |
| 4 | 1 | 0.011255 | 0.010859 |
| 24 | 0 | 0.008265 | 0.009079 |
| 24 | 1 | 0.011401 | 0.011475 |
| 46 | 0 | 0.010559 | 0.011514 |
| 46 | 1 | 0.010754 | 0.010516 |

Five mixtures miss 1%; the worst reaches **1.1401%**. Every expert passes 2%.
The raw receipt SHA-256 is
`157f83fb0ae741597df973bf9e7ba5e05de03f66c832c2bfcd83fa8eafd3a3b2`.
No performance claim is made.

## Decision

Reject 1x16 before bank or kernel work. Its layer-0 improvement does not
generalize to the six real-layer cases. FW-0063's 8x2 topology remains the only
untested equal-byte candidate that passed the first rung; run it once under the
same gate before closing rectangular symmetric INT8.
