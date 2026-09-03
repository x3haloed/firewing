# FW-0066 - Modified INT8 global clipping

- Status: completed
- Disposition: rejected
- Date: 2026-09-03
- Parent experiment: FW-0065
- Mode: `modified_clipped_block1x16_int8_weight_only`
- Hardware: Apple M1 Mac mini, 16 GiB, internal SSD

## Question and hypothesis

Can a frozen global clipping factor reduce 1x16 symmetric-INT8 error enough to
repair FW-0064 without changing its 62.5% byte ratio? Screen factors 0.80, 0.85,
0.90, 0.95, 0.975, 0.99, 0.995, and 0.999 against the same authenticated
layer-0 mixture. The unclipped 1.0 result from FW-0063 is the control.

This is modified weight-only calibration. A local improvement would authorize
a train/validation real-layer experiment, not deeper fidelity or performance.

## Method and result

Implementation commit:
`ab1e4b2ce773afd5fd0ae5a2bfcf288706122183`

Each 1x16 group uses `clip_factor * absmax / 127` as its F32 scale and clamps
codes to signed INT8. Artifact bytes are unchanged.

| Clip factor | Mixture relative L2 | Worst expert | Gate |
| ---: | ---: | ---: | --- |
| 1.000 control | 0.009364 | 0.010337 | pass |
| 0.999 | 0.009579 | 0.010552 | pass, worse than control |
| 0.995 | 0.012250 | 0.014978 | reject |
| 0.990 | 0.015278 | 0.017536 | reject |
| 0.975 | 0.031565 | 0.034525 | reject |
| 0.950 | 0.061545 | 0.068961 | reject |
| 0.900 | 0.123892 | 0.138143 | reject |
| 0.850 | 0.188406 | 0.208748 | reject |
| 0.800 | 0.254197 | 0.278442 | reject |

Every tested clip is worse than the unclipped control, and degradation grows
as clipping increases. The eight raw receipts live under
`/Users/chad/Models/firewing/evidence/FW-0066/`; their SHA-256 values are:

- 0.80 `3e3ef320a65783a797a8b681ac4397addc05d4de22dc14901f8e3005076e370a`
- 0.85 `6cd697bacee9b12f8eb65df799c97a68ecfce052299762295c0f6c060af40602`
- 0.90 `06a0f9b35246aeebbcb895509c9c60c07040cabee431de3bda3f55f3caf3dbf3`
- 0.95 `2136cb4acd24fa8d900902e0371b511ced1b5f019290df41a411d526b50945af`
- 0.975 `d6e2a5a2181dcdd71510d8fbe3475ef39c28d9f7831418d73515bfde23e5ad08`
- 0.99 `c8977036f47326466d8c4498839f2fc5c10e3ffda1421e05ed9cc55e46cc599a`
- 0.995 `8484c167cf50cf6cd5611d935450c7fbf4e1c5cce110b9b51dd5f1064eeab88e`
- 0.999 `b9bfd9eb9c34a4caa4e535e3cdf950b4ae9cf649b6182ad298013d7b5340ee03`

No performance claim is made.

## Decision

Reject fixed global clipping for 1x16 INT8 and do not run a 48-layer replay.
The closest clipped point is already dominated by the unclipped control, which
FW-0064 rejected. This result does not reject activation-aware per-group scales,
affine zero points, outlier exceptions, GPTQ-style error propagation, or
trained recovery; those change more than a global multiplier.
