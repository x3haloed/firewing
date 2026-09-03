# FW-0043 - Sequential width-two comparison

- Status: completed
- Disposition: correctness and routed-byte-economics milestone; width two
  preferred over width four on the observed two-transaction prefix
- Date: 2026-09-03
- Parent experiments: FW-0040 and FW-0042
- Exactness: L2 target-distribution-preserving greedy draft with exact target
  verification, commit, bonus, and rollback
- Hardware: Apple M1 Mac mini, 16 GiB, internal SSD, no companion hardware

## Question and hypothesis

Does the favorable width-two result from FW-0040 persist after its exact bonus
is carried into a second transaction, and does width two move less routed-expert
payload than width four while emitting the same exact target tokens?

Because both width-four transactions mismatch on their second comparison, the
hypothesis is that width two will emit the same tokens without paying for the
two later target rows and two recursive draft rows that width four rolls back.
This experiment measures exact routed-byte work, not endpoint time or TPS.

## Frozen authority and method

- Implementation commit: `fb5265f3081c4b2400ae16b9ef7c811a6f2c4bb6`
- SGLang source commit: `78c5024e9d9f589dcb4deb7f4ba4fb23f7e85385`
- Checkpoint revision: `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Greedy acceptance source-lock SHA-256:
  `fb542f23be3114aaace7da5bcd2202229c411687f19e469ec12c4d41e5c75976`
- Second width-two target fixture SHA-256:
  `114dae658be2edf772d3f8b3e4ef7c9ac669387d42c35a26cf739323733ee130`
- Second width-two transaction fixture SHA-256:
  `897cf8ad8278847a41e645cc97db79acc5fe1c7b19e0fa34975bbb896b78d573`
- Batch size: 1
- Concurrency: 1
- Sampling: greedy
- Verification width `q=2`
- Target routed layers: 48
- Top-k experts per routed layer and position: 10
- BF16 payload bytes per selected expert: 9,830,400
- `performance_claim=null`

FW-0040's full match retains target history `[16207,22856,369,264]` and
carries bonus 2526 as the next anchor. The already independently verified
causal MTP path consumes shifted inputs `[22856,369,264,2526]` and proposes
token 11, producing proposal `[2526,11]`. A separate six-token target fixture
evaluates `[16207,22856,369,264,2526,11]`; its final two posterior tokens are
`[11,45815]`.

The generalized width-two transaction generator reproduces FW-0040's original
transaction fixture byte-for-byte before accepting the longer retained
history. The native joined verifier independently replays the four-position
target history, causal MTP proposal, and six-position target branch, then
recomputes acceptance and route unions.

`U` remains normalized by the exact one-token target baseline of
`48*top_k=480` equal-sized expert rows. Integer accepted tokens, expert rows,
and payload bytes are authoritative. Derived JSON F64 ratios are accepted only
when identical or adjacent by one ULP: this is required because Python's
shortest round-tripping decimal for the second `A/U` is parsed by Rust's JSON
stack to the adjacent F64. A two-ULP difference fails the committed regression
test.

## Result

The proposal fully converges. Exact target verification emits `[11,45815]`,
makes 45815 the next anchor, retains both proposal rows, and rolls back none:

- `A=2`;
- target expert-union rows: 731;
- distinct live MTP expert rows: 10;
- combined expert-union rows: 741;
- `U=741/480=1.54375`;
- `A/U=1.2955465587044535`;
- logical routed-expert payload: 7,284,326,400 bytes; and
- total logically verified target-plus-draft payload: 48,083,721,216 bytes.

Raw joined receipt:
`/Users/chad/Models/firewing/evidence/FW-0043/second-q2-transaction-fb5265f.json`

Receipt SHA-256:
`64de0d45e4e7ce87dc4bc473445034555ea48fa54b1b42d338c55764306435b3`

The scalar/hash-heavy replay took 297.009 seconds. It repeatedly authenticates
checkpoint-derived values and is not decode latency or accepted TPS.

For transaction two, width two and width four emit identical `[11,45815]` and
both have `A=2`. Width two uses 741 combined rows versus width four's 1,066,
30.49% fewer equal-sized expert payload rows.

Across both transactions:

| Width | Emitted tokens | Sum of rows | Sum-equivalent `U` | `A/U` |
| --- | ---: | ---: | ---: | ---: |
| `q=2` | 4 | 1,438 | 2.995833 | 1.335188 |
| `q=4` | 4 | 2,279 | 4.747917 | 0.842475 |

Width four therefore moves 1.58484x the routed-expert payload for no additional
accepted output on this prefix. Width two uses 36.90% fewer rows in aggregate.

## Decision, limitations, and follow-up

Prefer width two over width four for the next target-faithful runtime tranche.
This is a scoped architecture decision, not a production default: two
transactions cannot establish the model's acceptance distribution, cache hit
rate, physical SSD traffic, complete latency, or sustained TPS.

The next useful step is no longer deeper scalar fixture expansion. Construct a
repeated width-two transaction path that retains target/MTP state and measures
the complete physical storage, Metal compute, synchronization, correction, and
sampling path. Preserve acceptance and union per transaction so a larger prompt
panel can later challenge the width choice.
