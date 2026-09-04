# FW-0071 - Modified affine-UINT8 exact groups across real layers

- Status: completed
- Disposition: rejected
- Date: 2026-09-03
- Parent experiment: FW-0070
- Mode: `modified_block4_affine_uint8_exact_groups_weight_only`
- Hardware: Apple M1 Mac mini, 16 GiB, internal SSD

## Question and hypothesis

Does FW-0070's selected 4% exact-group correction survive the six-case
early/middle/late real-layer gate? The hypothesis was that restoring the groups
with the largest affine reconstruction error would repair correlated mixture
error while retaining a substantial byte reduction.

## Method and results

Implementation commit:
`f4cf89e5f29376e939d5bb6a55d7bed0818d95f0`

```shell
.venv/bin/python tools/analyze_block4_int8_real_layers.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  f4cf89e5f29376e939d5bb6a55d7bed0818d95f0 \
  /Users/chad/Models/firewing/evidence/FW-0071/block4-affine-exact-400bps-real-layers-f4cf89e.json \
  --block-rows 4 \
  --block-columns 4 \
  --weight-format affine_uint8_exact_groups \
  --exact-group-bps 400
```

| Layer | State | Artifact/BF16 | Mixture relative L2 | Worst expert |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 0 | 0.701250 | 0.009611 | 0.009034 |
| 4 | 1 | 0.701250 | 0.009814 | 0.009293 |
| 24 | 0 | 0.701250 | 0.007557 | 0.007962 |
| 24 | 1 | 0.701250 | 0.010241 | 0.009184 |
| 46 | 0 | 0.701250 | 0.009403 | 0.009574 |
| 46 | 1 | 0.701250 | 0.009685 | 0.009928 |

All expert slices pass the 2% gate. Five mixture slices pass the 1% gate, but
layer 24 state 1 reaches 1.024111%, 0.024111 percentage points (2.411 basis
points) above the limit. The result therefore fails closed. The 4% artifact is
70.125% of source BF16 weight and scale bytes.

Raw receipt:
`/Users/chad/Models/firewing/evidence/FW-0071/block4-affine-exact-400bps-real-layers-f4cf89e.json`

Receipt SHA-256:
`eb948a4f65d5f3499e4cc4708af5c1f44951f5911d4485934eeca4765e2f991a`

No performance claim is made.

## Decision

Reject the 4% point before an exception bank or kernel. Do not close the
exact-group family yet: the miss is small, every expert slice passes, and the
artifact has room to increase its exact fraction while remaining materially
smaller than BF16. Screen 6%, 8%, and 10% on the cheap layer-0 fixture, then
send the strongest useful point through this same real-layer gate.

FW-0072 subsequently found all three larger points pass layer 0 and selected
6% for the next real-layer gate because it is the smallest surviving artifact.
