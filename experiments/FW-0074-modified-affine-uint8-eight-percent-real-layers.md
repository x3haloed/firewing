# FW-0074 - Modified affine-UINT8 8% exact groups across real layers

- Status: completed
- Disposition: conditional
- Date: 2026-09-03
- Parent experiment: FW-0073
- Mode: `modified_block4_affine_uint8_exact_groups_weight_only`
- Hardware: Apple M1 Mac mini, 16 GiB, internal SSD

## Question and hypothesis

Does FW-0072's 8% exact-group representation clear the six-case
early/middle/late real-layer gate after the 4% and 6% points narrowly fail?

## Method and results

Implementation commit:
`642e332351f13a61fddcaa4e5e7730f2d703c8eb`

```shell
.venv/bin/python tools/analyze_block4_int8_real_layers.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  642e332351f13a61fddcaa4e5e7730f2d703c8eb \
  /Users/chad/Models/firewing/evidence/FW-0074/block4-affine-exact-800bps-real-layers-642e332.json \
  --block-rows 4 --block-columns 4 \
  --weight-format affine_uint8_exact_groups \
  --exact-group-bps 800
```

| Layer | State | Artifact/BF16 | Mixture relative L2 | Worst expert |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 0 | 0.746250 | 0.009140 | 0.008654 |
| 4 | 1 | 0.746250 | 0.009407 | 0.009017 |
| 24 | 0 | 0.746250 | 0.007400 | 0.009213 |
| 24 | 1 | 0.746250 | 0.009915 | 0.009127 |
| 46 | 0 | 0.746250 | 0.009220 | 0.008820 |
| 46 | 1 | 0.746250 | 0.009587 | 0.009382 |

All six mixture slices pass the 1% gate and all expert slices pass the 2% gate.
The worst mixture has only 0.008504 percentage points (0.850 basis points) of
headroom. The artifact is 74.625% of BF16 weight and scale bytes.

Raw receipt SHA-256:
`1f660c083c922b19de9ef605041cab647f2ecd966d68177305ce5a310e997246`

No performance claim is made.

## Decision

Promote 8% only to the next correctness rung. Do not build or promote a
performance kernel yet: this experiment holds source routes and source-
accumulated BF16 activations fixed, so it cannot reveal error accumulation,
route changes, or final-logit drift caused by the candidate representation.
The next experiment must propagate candidate outputs across consecutive layers
and compare routes, hidden states, and final logits against the exact path.
