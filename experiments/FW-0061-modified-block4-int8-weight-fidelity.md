# FW-0061 - Modified block-4 INT8 weight fidelity

- Status: completed
- Disposition: conditional
- Superseded by: FW-0062 rejection on six real early/middle/late layer cases
- Date: 2026-09-03
- Parent experiment: FW-0060
- Mode: `modified_block4_int8_weight_only`
- Hardware: Apple M1 Mac mini, 16 GiB, internal SSD

## Question and hypothesis

Can the last square symmetric-INT8 grid with a material byte advantage clear
the frozen real layer-0 fidelity screen? A 4x4 grid stores one signed byte per
weight and one F32 scale per 16 weights, totaling 62.5% of BF16 source bytes.
A 2x2 grid would total 100% before metadata and cannot rescue the traffic
premise.

This is explicitly modified weight-only mode. A local pass does not establish
accumulated, endpoint, hosted-reference, or full-capability fidelity.

## Frozen method

- Implementation commit:
  `3f59dcab1099e8693b071b6d08fe894bdb7b1183`
- Model lock SHA-256:
  `f87399e8659ab3274601fcd455b78b73c600f57e5fc1e91499eec3ac1f4b9444`
- Real layer-0 mixture fixture SHA-256:
  `975a9982919297d37dd077f774693c782295cba496542c6adf278182e27b4d89`

The source-hash precondition, ten real experts, BF16 activation grant, official
expert equation and expert-order accumulation are unchanged from FW-0058
through FW-0060. Only scale granularity changes.

```shell
.venv/bin/python tools/analyze_block_fp8_weight_fidelity.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  fixtures/mixture/qwen3_8_flash_next_real.json \
  3f59dcab1099e8693b071b6d08fe894bdb7b1183 \
  /Users/chad/Models/firewing/evidence/FW-0061/block4-int8-weight-fidelity-3f59dca.json \
  --weight-format block_int8 \
  --block-size 4
```

## Result

All eleven exact source hashes reproduce. The artifact occupies 61,440,000
bytes versus 98,304,000 BF16 source bytes, exactly **0.625**.

- Top-10 mixture relative L2: **0.0097757**, passing the 1% gate by only
  **0.0002243** absolute.
- Maximum expert weighted-output relative L2: **0.010119**, passing 2%.
- Mixture BF16 equality: **20.430%**.
- Mixture maximum absolute error: **0.00048828**.

Raw receipt:
`/Users/chad/Models/firewing/evidence/FW-0061/block4-int8-weight-fidelity-3f59dca.json`

Receipt SHA-256:
`91e55bc412b0b4172fe7b4e5150f266deb3c1916786d97a7a89769e812211103`

The Python suite has 94 passing tests. No performance claim is made.

## Decision

Retain block-4 symmetric INT8 as a **conditional first-rung survivor**. Do not
build a bank or Metal kernel yet. The mixture margin is too narrow and the
screen covers one layer and one routed input only.

The next gate must exercise authenticated real inputs at early, middle, and
late layers, then an accumulated path. It must preserve source routes for the
layer-local comparison, charge the 62.5% representation exactly, and reject
the branch before kernel work if any frozen slice fails.

FW-0062 executed that gate and rejected this form. Five of six routed mixtures
miss the 1% threshold, so the conditional survivor is no longer active.
