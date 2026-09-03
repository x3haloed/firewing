# FW-0062 - Modified block-4 INT8 real-layer screen

- Status: completed
- Disposition: rejected
- Date: 2026-09-03
- Parent experiment: FW-0061
- Mode: `modified_block4_int8_weight_only`
- Hardware: Apple M1 Mac mini, 16 GiB, internal SSD

## Question and hypothesis

Does FW-0061's block-4 INT8 first-rung survivor generalize to authenticated
source-accumulated inputs at early, middle, and late decoder layers? Passing
requires both sequential token states at layers 4, 24, and 46 to keep routed
mixture relative L2 at or below 1% and every selected expert at or below 2%.

This is an explicitly modified, source-route-held, weight-only screen. A pass
would authorize candidate-accumulated validation, not a bank, kernel, endpoint,
or performance claim.

## Authority and method

- Implementation commit:
  `144c1d02c199784a486a6a1b7cd7147af0daecc0`
- Model lock SHA-256:
  `f87399e8659ab3274601fcd455b78b73c600f57e5fc1e91499eec3ac1f4b9444`
- Accumulated layer-4-through-47 fixture SHA-256:
  `6ed2e16da1e64fb8001c26608d7972f4910190f74768055b5778dc7891ebf525`

The analyzer regenerates the complete two-token 48-layer source path and
requires exact equality with the committed hash-only fixture. A narrow observer
then evaluates 4x4 symmetric INT8 weights on the in-memory source MLP inputs and
fixed source routes at the six frozen layer/token pairs. No activation payloads
or weights are committed.

```shell
.venv/bin/python tools/analyze_block4_int8_real_layers.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  144c1d02c199784a486a6a1b7cd7147af0daecc0 \
  /Users/chad/Models/firewing/evidence/FW-0062/block4-int8-real-layers-144c1d0.json
```

## Result

The complete source regeneration passes. Every tested representation remains
exactly 0.625 of BF16 source weight bytes.

| Layer | Token state | Mixture relative L2 | Worst expert |
| ---: | ---: | ---: | ---: |
| 4 | 0 | 0.010604 | 0.010750 |
| 4 | 1 | 0.011112 | 0.010922 |
| 24 | 0 | 0.007946 | 0.008878 |
| 24 | 1 | 0.011174 | 0.011506 |
| 46 | 0 | 0.010258 | 0.010672 |
| 46 | 1 | 0.011313 | 0.010643 |

All experts pass 2%, but five of six routed mixtures fail 1%. The worst mixture
is layer 46 token state 1 at **1.1313%**.

Raw receipt:
`/Users/chad/Models/firewing/evidence/FW-0062/block4-int8-real-layers-144c1d0.json`

Receipt SHA-256:
`41b42109dfe9cf078e2154c80c4f9623d5b35bb7f0fdfe67217f65f0213ead15`

The Python suite has 96 passing tests. No performance claim is made.

## Decision

Reject naïve symmetric block-4 INT8 and supersede FW-0061's conditional
survivor. Do not build a packed bank, Metal kernel, or candidate-accumulated
path for this form. The failure is decisive at the cheaper source-held-route
layer-local rung.

This closes plain square-block symmetric INT8: coarser grids already failed,
while a 2x2 F32-scale grid consumes BF16-equivalent bytes before metadata. A
future compact branch must change the error mechanism—calibration, scale
topology, outlier treatment, recovery, or training—not merely shrink the same
square grid.
