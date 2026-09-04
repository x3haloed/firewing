# FW-0072 - Modified affine-UINT8 larger exact-group screen

- Status: completed
- Disposition: conditional
- Date: 2026-09-03
- Parent experiment: FW-0071
- Mode: `modified_block4_affine_uint8_exact_groups_weight_only`
- Hardware: Apple M1 Mac mini, 16 GiB, internal SSD

## Question and hypothesis

After FW-0071 missed the deeper mixture gate by 2.411 basis points at 4%
exact groups, do modestly larger residual budgets improve the cheap layer-0
screen while remaining materially smaller than BF16? Test 6%, 8%, and 10%
with the same per-matrix squared-weight-error ranking and exact byte ledger.

## Method and results

Implementation commit:
`c560bee686942f7a65f127122164b143aac7c6f7`

```shell
.venv/bin/python tools/analyze_block_fp8_weight_fidelity.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  fixtures/mixture/qwen3_8_flash_next_real.json \
  c560bee686942f7a65f127122164b143aac7c6f7 \
  REPORT_JSON \
  --weight-format block_uint8_affine_exact_groups \
  --block-rows 4 \
  --block-columns 4 \
  --exact-group-bps BPS
```

| Exact groups | Artifact/BF16 | Mixture relative L2 | Worst expert |
| ---: | ---: | ---: | ---: |
| 6% | 0.723750 | 0.008739 | 0.009213 |
| 8% | 0.746250 | 0.008583 | 0.008722 |
| 10% | 0.768750 | 0.008493 | 0.008918 |

All points pass both layer-0 gates. Mixture error decreases monotonically over
this interval, while worst-expert error does not; routed-output validation
therefore remains authoritative over the selection proxy.

Receipt SHA-256 values:

- 6% `b41de6d6fcd92b471b48571fa29e47f608d7e5bb67a0aebc8eaa3ddf1af65e0e`
- 8% `85cf3758c1681d1c66a0edb86964ba34c3482014cbe8cff9db4d6118bfe11883`
- 10% `76069bacf1a09b2271322ffe91376de7bfda5cda2f7c41890e15fa1960c1af43`

Raw receipts are under `/Users/chad/Models/firewing/evidence/FW-0072/`. No
performance claim is made.

## Decision

Advance 6% first to the six-case real-layer gate. It is the smallest surviving
point and retains a 27.625% byte reduction from BF16. Test 8% or 10% only if 6%
fails, so a passing result establishes the tightest measured artifact bound.

FW-0073 subsequently rejected 6% at 1.020352% worst mixture error. Advance 8%
next; do not infer it passes from the layer-0 trend.
