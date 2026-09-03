# FW-0067 - Modified affine-UINT8 topology

- Status: completed
- Disposition: conditional
- Follow-up: FW-0068 and FW-0069 reject all five topologies
- Date: 2026-09-03
- Parent experiment: FW-0066
- Mode: modified block-affine UINT8 weight-only screens
- Hardware: Apple M1 Mac mini, 16 GiB, internal SSD

## Question and hypothesis

Does one stored zero point per 16-weight group reduce the asymmetry error that
remained after symmetric INT8 and clipping failed? Screen all power-of-two
rectangular factorizations of 16 weights: 1x16, 2x8, 4x4, 8x2, and 16x1.

Each group stores 16 UINT8 codes, one F32 min/max scale, and one UINT8 zero
point: 21 bytes versus 32 BF16 bytes, or **0.65625**. These are explicitly
modified one-layer weight-only screens.

## Method and results

Implementation commit:
`16a2fc4ca425ba95a8a006bbcfa881bdbb2af465`

The authenticated layer-0 mixture authority, source hashes, BF16 activation
grant, expert equation, accumulation order, and 1%/2% gates remain unchanged.

```shell
.venv/bin/python tools/analyze_block_fp8_weight_fidelity.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  fixtures/mixture/qwen3_8_flash_next_real.json \
  16a2fc4ca425ba95a8a006bbcfa881bdbb2af465 \
  REPORT_JSON \
  --weight-format block_uint8_affine \
  --block-rows ROWS \
  --block-columns COLUMNS
```

| Shape | Mixture relative L2 | Worst expert | Result |
| --- | ---: | ---: | --- |
| 1x16 | 0.009045 | 0.009362 | pass |
| 2x8 | 0.009101 | 0.008833 | pass |
| 4x4 | 0.009093 | 0.009581 | pass |
| 8x2 | 0.008941 | 0.009374 | pass |
| 16x1 | 0.009128 | 0.009737 | pass |

All 55 source hashes reproduce. Receipt SHA-256 values:

- 1x16 `6b833095112feb7c800b1d002fc2a7e4dac09cfb1e3e2ef07efcccaac213e216`
- 2x8 `984bc84a38c11760c918c54ad065efe43f25daec22dddf02bee38d821ef41896`
- 4x4 `9b65b8e2a884ecf9424532a92378885585a606fcdba24c145bc1d04a5aefbfff`
- 8x2 `b92654b4978e5852365d964489c38b156980283dc269a674fd9c79077c41fa66`
- 16x1 `fd98a6311c3af6779fb11eb7a94fc77fe0bd6e8a21bcdc837eda78f28af99bb0`

Raw receipts are under
`/Users/chad/Models/firewing/evidence/FW-0067/`. No performance claim is made.

## Decision

Retain affine 8x2 as the selected first-rung survivor because it has the best
mixture result at identical bytes. Run it through the six authenticated
early/middle/late real-layer cases before any bank, kernel, or candidate-
accumulated work. The other passing topologies are not rejected globally, but
are dominated for this next test.

FW-0068 rejects 8x2 on two of six real-layer mixtures. Because the other four
topologies also passed this record, they remain unresolved until a common
real-layer frontier is run.

FW-0069 completed that frontier and rejected all four. No affine-UINT8 survivor
from this record remains active.
