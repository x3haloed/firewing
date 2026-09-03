# FW-0059 - Modified block-32 INT8 weight fidelity

- Status: completed
- Disposition: rejected
- Date: 2026-09-03
- Parent experiment: FW-0058
- Mode: `modified_block32_int8_weight_only`
- Hardware: Apple M1 Mac mini, 16 GiB, internal SSD

## Question and hypothesis

Was FW-0058's failure primarily caused by sharing one INT8 scale across a
128x128 block? The hypothesis was that reducing the symmetric absmax scale grid
to 32x32 would pass the same real layer-0 continuation gates while retaining a
near-half-size executable representation.

This remains an explicitly modified weight-only mode. Passing this screen would
only authorize deeper fidelity work.

## Frozen authority and method

- Implementation commit:
  `b0680d09ecb5e60021fd59cbd789ae14909a2d42`
- Model lock SHA-256:
  `f87399e8659ab3274601fcd455b78b73c600f57e5fc1e91499eec3ac1f4b9444`
- Real layer-0 mixture fixture SHA-256:
  `975a9982919297d37dd077f774693c782295cba496542c6adf278182e27b4d89`

FW-0059 changes only the INT8 scale grid from FW-0058: every 32x32 block gets
one F32 absmax/127 scale and one signed code per BF16 source weight. The screen
dequantizes weights to BF16 and keeps activations and intermediate boundaries
at exact BF16. All ten source expert outputs and the final source mixture must
reproduce their fixture hashes before candidate metrics count.

```shell
.venv/bin/python tools/analyze_block_fp8_weight_fidelity.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  fixtures/mixture/qwen3_8_flash_next_real.json \
  b0680d09ecb5e60021fd59cbd789ae14909a2d42 \
  /Users/chad/Models/firewing/evidence/FW-0059/block32-int8-weight-fidelity-b0680d0.json \
  --weight-format block_int8 \
  --block-size 32
```

## Gates and result

- Continue only if mixture relative L2 is at most 1%.
- Continue only if every expert weighted-output relative L2 is at most 2%.
- Never infer endpoint or hosted-reference fidelity from this one-layer screen.

All eleven exact baseline hashes match. The representation occupies 49,344,000
bytes versus 98,304,000 source bytes, a ratio of **0.501953**. All individual
experts pass their gate, but the routed accumulation does not:

- Top-10 mixture relative L2: **0.015282**.
- Maximum expert weighted-output relative L2: **0.017957**.
- Best expert relative L2: **0.015024**.
- Mixture BF16 equality: **12.656%**.
- Mixture maximum absolute error: **0.00061035**.

Raw receipt:
`/Users/chad/Models/firewing/evidence/FW-0059/block32-int8-weight-fidelity-b0680d0.json`

Receipt SHA-256:
`b56499c7fd958468ee3ee12ffe6616f8d90b17fa8c7e27e3841287b41f5d719c`

The Python suite has 92 passing tests, including block-size and byte-ledger
fixtures.

## Decision

Reject symmetric block-32 INT8 and do not build its kernel or deeper-layer
campaign. The smaller scale grid reduces mixture error by 29.6% relative to
FW-0058 and makes every expert pass locally, so scale granularity matters; it
does not make this representation clear the mixture gate.

The next cheap discriminator is a frozen finer-grid sweep, not a kernel. It
must establish whether symmetric INT8 has a passing scale-granularity frontier
at an artifact size that can still alter the unified-memory traffic bound.
