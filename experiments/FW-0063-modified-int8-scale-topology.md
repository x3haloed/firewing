# FW-0063 - Modified INT8 scale topology

- Status: completed
- Disposition: conditional
- Follow-up: FW-0064 and FW-0065 reject both first-rung survivors
- Date: 2026-09-03
- Parent experiment: FW-0062
- Mode: modified rectangular symmetric-INT8 weight-only screens
- Hardware: Apple M1 Mac mini, 16 GiB, internal SSD

## Question and hypothesis

Can scale orientation improve FW-0062's routed-mixture fidelity without changing
its 62.5% byte ratio? Compare every rectangular factorization of 16 weights with
power-of-two sides around the 4x4 control: 1x16, 2x8, 8x2, and 16x1. Here weight
rows are output channels and columns are input channels.

These are explicitly modified one-layer weight-only screens. A pass authorizes
only the existing early/middle/late real-layer gate.

## Frozen method

Implementation commit:
`70d359d3df5933383a2cd6de683cafa399dabaed`

The authenticated layer-0 input, ten experts, source hashes, BF16 activation
grant, expert equation, accumulation order, and 1%/2% gates are unchanged from
FW-0057 through FW-0061. Each shape stores one INT8 code per weight and one F32
absmax/127 scale per 16 weights, so all candidates occupy exactly 0.625 of BF16.

```shell
.venv/bin/python tools/analyze_block_fp8_weight_fidelity.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  fixtures/mixture/qwen3_8_flash_next_real.json \
  70d359d3df5933383a2cd6de683cafa399dabaed \
  REPORT_JSON \
  --weight-format block_int8 \
  --block-rows ROWS \
  --block-columns COLUMNS
```

## Results

| Scale shape | Mixture relative L2 | Worst expert | Gate result |
| --- | ---: | ---: | --- |
| 1x16 | 0.009364 | 0.010337 | pass |
| 2x8 | 0.010050 | 0.010363 | reject |
| 8x2 | 0.009803 | 0.010835 | pass |
| 16x1 | 0.010064 | 0.009983 | reject |

All 44 source hashes reproduce. Receipt hashes:

- 1x16: `f40ac25ee3b33f59e3bb3dcb17cc2f09e9f77a209f38d343917dbb03a50e0a38`
- 2x8: `3f5a675a2437c4f93defe3aeeff1343bc65c23c464bb8b19eed840f16faf20bb`
- 8x2: `d697cd2bcc515f90bad48ef8f0419b859943b9efb44f9ee193de0f6b7bab8c81`
- 16x1: `46d0c8706ff13f7f114793094031619a7489c8ac5ab724c4098af6b60b411cc3`

Raw receipts are under
`/Users/chad/Models/firewing/evidence/FW-0063/`. No performance claim is made.

## Decision

Retain 1x16 as the sole next candidate because it has the best mixture margin;
8x2 is a valid first-rung result but dominated at identical bytes. Reject 2x8
and 16x1 locally. Scale topology materially affects routed output error, so
FW-0062 closes square-block symmetric INT8 rather than every symmetric-INT8
topology.

Run 1x16 through the same six authenticated early/middle/late layer cases
before any packed bank, accumulated candidate path, or Metal kernel.

FW-0064 and FW-0065 ran that gate and rejected 1x16 and 8x2. No survivor from
this topology screen remains active.
