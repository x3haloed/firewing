# FW-0060 - Modified fine-grid INT8 frontier

- Status: completed
- Disposition: rejected
- Date: 2026-09-03
- Parent experiment: FW-0059
- Mode: `modified_block16_int8_weight_only` and
  `modified_block8_int8_weight_only`
- Hardware: Apple M1 Mac mini, 16 GiB, internal SSD

## Question and hypothesis

Does plain symmetric INT8 cross the frozen real layer-0 mixture gate at a
16x16 or 8x8 F32 absmax scale grid while remaining materially smaller than
BF16? FW-0059 established that 32x32 is a near miss. This sweep freezes the two
next finer power-of-two grids before considering any kernel.

Both modes are explicitly modified and weight-only. A pass would authorize
deeper fidelity work, not establish target equivalence or TPS.

## Method and commands

Implementation commit:
`929cdf5f5d4d128f77cf9e7a39856b2a3410e786`

The model-lock and real-mixture hashes, ten experts, BF16 activation grant,
exact source-hash precondition, expert equation, accumulation order, and
1%/2% gates are unchanged from FW-0058 and FW-0059. Run the analyzer once with
each `BLOCK` below:

```shell
.venv/bin/python tools/analyze_block_fp8_weight_fidelity.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  fixtures/mixture/qwen3_8_flash_next_real.json \
  929cdf5f5d4d128f77cf9e7a39856b2a3410e786 \
  REPORT_JSON \
  --weight-format block_int8 \
  --block-size BLOCK
```

## Results

| Grid | Artifact/source | Mixture relative L2 | Worst expert | Decision |
| --- | ---: | ---: | ---: | --- |
| 16x16 | 0.507812 | 0.013479 | 0.015033 | reject |
| 8x8 | 0.531250 | 0.011676 | 0.012398 | reject |

All 22 source hashes reproduce, and every expert passes the 2% local gate.
Both mixtures fail the 1% gate. The 8x8 grid improves mixture error by only
13.4% over 16x16 while increasing scale overhead fourfold.

Raw receipts:

- `/Users/chad/Models/firewing/evidence/FW-0060/block16-int8-weight-fidelity-929cdf5.json`
  SHA-256 `d53ecfa1637fedd6a0c20ae5e17242c4aa4ea5d996c7ac2b1df34fec8c58c634`
- `/Users/chad/Models/firewing/evidence/FW-0060/block8-int8-weight-fidelity-929cdf5.json`
  SHA-256 `5d4f7044379ea62782fc7eae6c285602fe2856a6d39caf42a8346c023c861d87`

The Python suite has 93 passing tests. No performance claim is made.

## Decision

Reject both grids before kernel or deeper-fixture work. Scale granularity has
diminishing fidelity returns, but the representation remains close enough to
the gate that one final 4x4 test is warranted: it occupies 62.5% of BF16 bytes
and is the finest square INT8/F32-scale grid with a material byte advantage.
If it fails, close naïve symmetric INT8 rather than testing 2x2, whose codes
and scales consume the same bytes as BF16 before metadata.
