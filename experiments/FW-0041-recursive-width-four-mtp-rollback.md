# FW-0041 - Recursive width-four MTP rollback

- Status: completed
- Disposition: correctness milestone; width four rejected at this prompt location
- Date: 2026-09-03
- Parent experiment: FW-0040
- Exactness: checkpoint-exact recursive greedy proposal, target verification,
  correction, rollback, and route union
- Hardware: Apple M1 Mac mini, 16 GiB, internal SSD, no companion hardware

## Question

Does source-faithful recursion beyond FW-0040's first draft token increase exact
accepted output enough to justify the extra routed-expert payload, and does a
native joined replay reproduce the first mismatch, correction, and rollback?

This experiment covers one width-four greedy transaction at the same prompt
location as FW-0040. It is not a decode-throughput measurement, an acceptance
distribution, or a runtime default.

## Frozen authority and method

- Implementation commit: `1dc0a7edf6d109f238df2cd99e3f90ac9b87ab6a`
- SGLang source commit: `78c5024e9d9f589dcb4deb7f4ba4fb23f7e85385`
- Checkpoint revision: `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Recursive EAGLE source-lock SHA-256:
  `f13565f96a9334c7a69b551465538b8c83870af889fb5fbd4f2ea886835eb12f`
- Greedy acceptance source-lock SHA-256:
  `fb542f23be3114aaace7da5bcd2202229c411687f19e469ec12c4d41e5c75976`
- Six-token target fixture SHA-256:
  `9d74bff8f1ef829f20dd9300702d2f2c9e9937e00b1f93b26dd99645855c14a4`
- Recursive seed/attention/decoder/logits fixture SHA-256:
  `3e3e86e494ccc673d46e03f7309781cc75db7a9985197b636056403545723ef1`,
  `0cc496aefc21d09ec11b2c4968a0147aa842d185e9b5cc3db33356b2268c5f89`,
  `00348e303bc9308ee26c43fb590372456d25208985d13a66fb78e70d6a377820`,
  and `228ee1d53fb3c1f474e4ccc5ef24048ff4060ad551d89ed9496c39f79c1d8104`
- Recursive transaction fixture SHA-256:
  `c31d1e1ed5be8b437b3d212770d7fa708005c119b7ef177818218778f9a23e24`
- Batch size: 1
- Concurrency: 1
- Sampling: greedy
- Verification width `q=4`
- Target routed layers: 48
- Top-k experts per routed layer and position: 10
- BF16 payload bytes per selected expert: 9,830,400
- `performance_claim=null`

The pinned recursive worker semantics carry the preceding MTP decoder hidden
state into the next four-stream fusion, advance position and cache state, and
run the one-layer draft model again. The generator follows those transitions;
the native verifier does not trust their outputs. It independently verifies the
original causal prefix and then verifies prefix lengths two, three, and four so
each recurrent hidden input is tied to an independently reproduced preceding
decoder output.

The resulting proposal is `[369,264,220,17]`. A separate six-position target
fixture evaluates `[16207,22856,369,264,220,17]`; its four verification rows
produce posterior tokens `[264,2526,16,15]`. The joined verifier replays both
full paths before computing greedy acceptance and retaining only the proposal
rows authorized by the target.

As in corrected FW-0040, `U` uses one-token-equivalent routed-expert bytes. The
denominator is the explicit one-token target baseline `48*top_k=480` expert
rows. The target unions its routes over four verification positions; the draft
namespace contains the three live recursive rows that propose tokens after the
already-authorized anchor. Dense, attention, PLE, LM-head, synchronization, and
physical SSD work are additional costs and are not hidden in `U`.

## Result

The first comparison succeeds (`264 == 264`) and the second fails
(`220 != 2526`). Greedy verification therefore emits `[264,2526]`, makes 2526
the next target anchor, retains two proposal rows, and rolls back the remaining
two:

- one correct draft plus correction gives `A=2`;
- target expert-union rows: 1,183;
- distinct live MTP expert rows: 30;
- combined expert-union rows: 1,213;
- `U=1213/480=2.527083333333333`;
- `A/U=0.7914262159934048`;
- logical routed-expert payload: 11,924,275,200 bytes; and
- total logically verified target-plus-draft payload: 44,564,417,536 bytes.

The recursive-proposal-only native receipt independently verifies four fusion
steps, two recurrent-hidden links, 216 BF16 capture hashes, four F32 capture
hashes, 12 i64 capture hashes, 26 dense tensors, and 40 unique MTP experts over
18,914,104,576 logical bytes:

`/Users/chad/Models/firewing/evidence/FW-0041/recursive-mtp-0488e26.json`

Receipt SHA-256:
`3e8d0351a29a17becae4fd898e5d1491034aea2191b9802206c7879740f23bef`

Joined transaction receipt:
`/Users/chad/Models/firewing/evidence/FW-0041/recursive-transaction-1dc0a7e.json`

Receipt SHA-256:
`f27efff525a63a8dea676e0ce54ad02e7d2c313fd68f05e62d6e9034604a2f03`

The clean joined correctness replay took 268.314 seconds. These scalar,
hash-heavy replays repeatedly authenticate checkpoint-derived values. Their
wall times are diagnostics, not decode latency or accepted TPS.

Compared with FW-0040's width two, width four moves 1.7403x as many unique
expert rows and equal-sized expert payload bytes (`1213/697`) while producing
the same `A=2`. Routed-byte leverage falls from 1.377331 to 0.791426.

## Decision, limitations, and follow-up

Promote the recursive proposal, first-mismatch correction, and two-row rollback
as exact authorities. Reject increasing this transaction from width two to
width four as a routed-byte-leverage win.

Do not infer that width four is globally inferior, or that either width reaches
the target endpoint rate. This is one prompt location, and `U` omits substantial
fixed and physical costs. The next cheap falsification step is to continue from
the exact correction anchor 2526 and collect sequential width-two/width-four
transactions before building a timed loop. That will test whether this early
mismatch is representative enough to guide the runtime width.
