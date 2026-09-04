# FW-0073 - Modified affine-UINT8 6% exact groups across real layers

- Status: completed
- Disposition: rejected
- Date: 2026-09-03
- Parent experiment: FW-0072
- Mode: `modified_block4_affine_uint8_exact_groups_weight_only`
- Hardware: Apple M1 Mac mini, 16 GiB, internal SSD

## Question and hypothesis

Does FW-0072's smallest surviving point, 6% exact groups, clear the six-case
early/middle/late real-layer gate at 72.375% of BF16 bytes?

## Method and results

Implementation commit:
`eaa79f821ee29546b25811f4e878973cfb289837`

```shell
.venv/bin/python tools/analyze_block4_int8_real_layers.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  eaa79f821ee29546b25811f4e878973cfb289837 \
  /Users/chad/Models/firewing/evidence/FW-0073/block4-affine-exact-600bps-real-layers-eaa79f8.json \
  --block-rows 4 --block-columns 4 \
  --weight-format affine_uint8_exact_groups \
  --exact-group-bps 600
```

| Layer | State | Artifact/BF16 | Mixture relative L2 | Worst expert |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 0 | 0.723750 | 0.009393 | 0.008837 |
| 4 | 1 | 0.723750 | 0.009669 | 0.009064 |
| 24 | 0 | 0.723750 | 0.007484 | 0.008870 |
| 24 | 1 | 0.723750 | 0.010204 | 0.009544 |
| 46 | 0 | 0.723750 | 0.009337 | 0.009130 |
| 46 | 1 | 0.723750 | 0.009770 | 0.009561 |

All expert slices pass the 2% gate and five mixture slices pass the 1% gate.
Layer 24 state 1 reaches 1.020352%, 2.035 basis points above the limit, so the
result fails closed. This improves FW-0071's 4% miss by only 0.376 basis points.

Raw receipt SHA-256:
`0e153b2e72682fd9e70f90cf5c030d14a45f7a40a409409eceaac669f47c12a2`

No performance claim is made.

## Decision

Reject 6% before bank or kernel work. Advance FW-0072's next point, 8%, through
the same gate. If it also fails, test the already-screened 10% boundary before
closing this simple weight-error-ranked residual family.

FW-0074 subsequently passes 8% across all six real-layer cases. It advances to
candidate-accumulated validation, not to performance implementation.
