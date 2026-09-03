# FW-0069 - Modified affine-UINT8 real-layer frontier

- Status: completed
- Disposition: rejected
- Date: 2026-09-03
- Parent experiment: FW-0068
- Mode: modified block-affine UINT8 weight-only screens
- Hardware: Apple M1 Mac mini, 16 GiB, internal SSD

## Question and method

Do any of FW-0067's remaining 1x16, 2x8, 4x4, or 16x1 affine-UINT8
topologies pass the authenticated real-layer gate after selected 8x2 failed
FW-0068? Run each topology serially through an exact regeneration of the
two-token 48-layer source fixture, holding source routes fixed at layers 4, 24,
and 46. Every mixture must be at most 1% relative L2 and every expert at most
2%.

Implementation commit:
`c0042d8b3523b35537f89676ea5b31b15415cd79`

The command is FW-0068's with the respective `--block-rows` and
`--block-columns`. Each artifact remains 0.65625 of BF16 weight bytes.

## Results

| Shape | Maximum mixture relative L2 | Maximum expert | Result |
| --- | ---: | ---: | --- |
| 1x16 | 0.010762 | 0.011118 | reject |
| 2x8 | 0.010678 | 0.010430 | reject |
| 4x4 | 0.010505 | 0.009963 | reject |
| 16x1 | 0.010864 | 0.011059 | reject |

Every expert passes 2%, but every topology has at least one mixture above 1%.
For all four, layer 24 token state 1 is the worst mixture. The four complete
source regenerations each match fixture SHA-256
`6ed2e16da1e64fb8001c26608d7972f4910190f74768055b5778dc7891ebf525`.

Raw receipts are under `/Users/chad/Models/firewing/evidence/FW-0069/`:

- 1x16 `49e4f570bb0e07c1408a00e7cc883f5e7cd57e63e5f63d30e3ae5b500ffe7afd`
- 2x8 `df4d76fd61387dfce95cd7bc74fa7f1b9a8ce9094eb0c116634d2415e961c137`
- 4x4 `5c9f295e215fd6152205436939132cd651cda5366c2f23dd8a4c6818b89623ec`
- 16x1 `d8095732535760ec65b26fb2344bd6bad4ccbc3b51177ac4aafa6c2337b3dd46`

No performance claim is made.

## Decision

Reject the entire tested affine-UINT8 family before bank, kernel, or candidate-
accumulated work. FW-0068 and FW-0069 cover every 16-weight topology that
passed layer 0; none passes the stronger gate.

Future compact work must add a mechanism that specifically reduces correlated
routed-mixture error, such as activation-aware group selection, sparse exact
exceptions, error propagation, or learned recovery. Another zero-point
topology sweep is not justified.
