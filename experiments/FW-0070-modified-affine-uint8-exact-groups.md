# FW-0070 - Modified affine-UINT8 exact groups

- Status: completed
- Disposition: conditional
- Date: 2026-09-03
- Parent experiment: FW-0069
- Mode: `modified_block4_affine_uint8_exact_groups_weight_only`
- Hardware: Apple M1 Mac mini, 16 GiB, internal SSD

## Question and hypothesis

Can a sparse store of exact residual groups reduce correlated affine-UINT8
error enough to reopen the compact branch? Start from FW-0069's best deeper
topology, 4x4, rank groups independently in each expert matrix by squared
weight reconstruction error, and restore the top 0.25%, 0.5%, 1%, 2%, or 4%.

The artifact retains the affine core, plus one 16-value BF16 residual and one
U32 ordinal per selected group. This is explicitly modified weight-only mode.

## Method and results

Implementation commit:
`34f563d31e77924ed5754ace87bd8e5e276acd0b`

```shell
.venv/bin/python tools/analyze_block_fp8_weight_fidelity.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  fixtures/mixture/qwen3_8_flash_next_real.json \
  34f563d31e77924ed5754ace87bd8e5e276acd0b \
  REPORT_JSON \
  --weight-format block_uint8_affine_exact_groups \
  --block-rows 4 \
  --block-columns 4 \
  --exact-group-bps BPS
```

| Exact groups | Artifact/BF16 | Mixture relative L2 | Worst expert |
| ---: | ---: | ---: | ---: |
| 0.25% | 0.659063 | 0.009204 | 0.009495 |
| 0.50% | 0.661875 | 0.009071 | 0.009760 |
| 1.00% | 0.667500 | 0.009121 | 0.009925 |
| 2.00% | 0.678750 | 0.008914 | 0.009336 |
| 4.00% | 0.701250 | 0.008829 | 0.009178 |

All source hashes reproduce and every point passes the one-layer gate. The
non-monotonic output metric is expected evidence that raw weight-error ranking
does not directly rank routed-output contribution.

Receipt SHA-256 values:

- 0.25% `0141b2b2bd93371a4f07354351e72611f5fef52b7b8daeb68e924bf164d3c871`
- 0.50% `ca73c4be94b6a5b61915bea0e339f522dd59be388c55f760a0fe25e7d1800f06`
- 1.00% `d7c2f7328e1beb741fd8496efac5cb0ec390cce12bbe3c98ac4e9ea728ee3ead`
- 2.00% `6ed0ff348527194f82999a73aca77216a8695888752ef9ef2ce334d8e22ead7d`
- 4.00% `29540450b31b4c094faeb038d882cde9720d7dadef950ea40e6aeeb0bda2c10f`

Raw receipts are under `/Users/chad/Models/firewing/evidence/FW-0070/`. No
performance claim is made.

## Decision

Retain 4% only as the strongest tested conditional point and run it through the
six-case real-layer gate. Its 2.9% layer-0 improvement may be insufficient to
repair FW-0069's 4x4 miss, so do not build an exception kernel or bank first.
Smaller points are not promoted; their output error is non-monotonic.

FW-0071 subsequently rejected 4% at 1.024111% worst mixture error. Because the
miss is only 2.411 basis points and every expert slice passes, larger residual
fractions remain a separate bounded follow-up rather than being inferred from
this result.
