# FW-0042 - Sequential MTP transaction after correction

- Status: completed
- Disposition: sequential correctness milestone; width four remains rejected on
  the observed two-transaction prefix
- Date: 2026-09-03
- Parent experiment: FW-0041
- Exactness: checkpoint-exact recursive greedy proposal, target verification,
  correction, rollback, and route union
- Hardware: Apple M1 Mac mini, 16 GiB, internal SSD, no companion hardware

## Question

Does FW-0041's early width-four mismatch repeat after the exact target
correction is carried into a second transaction, and how do sequential accepted
length and routed-expert union compare with width two?

## Frozen method and evidence

- Native verifier commit: `bd049e993f1d302f08d9de3d4df5e2c37381bd9e`
- Fixture-generation commit: `e4ace5b`

Prismwing's repeated-transaction implementation at commit
`c87d0c1aa2c118f71ca5348434be35d02f62f031` supplied a structural cross-check:
retain target history only through verifier-authorized proposal rows, append the
correction as the next anchor, then form shifted MTP inputs from retained target
tokens and that anchor. Firewing independently applies its pinned SGLang locks
and Qwen checkpoint semantics.

FW-0041 retains target inputs `[16207,22856,369,264]` and makes correction 2526
the next anchor. The resulting shifted MTP inputs are
`[22856,369,264,2526]`. Generator changes must reproduce all four FW-0041
recursive fixture hashes byte-for-byte before producing this extension; that
gate passed.

Current fixture hashes:

- eight-token target:
  `f4712a3493355b2f99400888e969ecbe0c8db052028417345732e211d1801976`;
- recursive seed:
  `2c767550acaa114272c06ef9c6e2b7d607b846db09263807dbc0efbdc27d679b`;
- recursive attention:
  `cb1de62045210b6a01834bc9b66bc236fe687232efafa3f1a0d9837b482bc823`;
- recursive decoder:
  `6679addcb2d86388468304bd68217485dce62977b7a8f83f4edbd5b6c3aa88f6`;
- recursive logits:
  `bf1df63cca1e770c6b240d4b51c72fa088804d329d623a7868cb3707cf6cb341`;
- transaction:
  `04ff09ccbe5b21b0cc89a66433c96a03fb000973df65ba99e8e1f9a25a641130`.

The source generator proposes `[2526,11,8581,11]`. The separate target path
returns `[11,45815,11,321]`, so its ledger records the same second
comparison mismatch as FW-0041: emitted `[11,45815]`, `A=2`, two retained rows,
two rolled-back rows, 1,041 target union rows, 25 live draft rows, and
`U=1066/480=2.220833` (`A/U=0.900563`). The native recursive verifier
independently reproduces six fusion steps, two recurrent-hidden links, 324 BF16,
six F32, and 18 i64 capture hashes, 26 dense tensors, and 47 unique MTP experts.
The joined verifier then reproduces every target capture and recomputes the
proposal, posterior, commit, rollback, and union. Total logically verified
target-plus-draft payload is 50,384,065,536 bytes.

Recursive receipt:
`/Users/chad/Models/firewing/evidence/FW-0042/second-recursive-bd049e9.json`

SHA-256:
`1299839e08be8ed13719e7859d715b45ae4a41faeafc33677ad972479bfa6405`

Joined receipt:
`/Users/chad/Models/firewing/evidence/FW-0042/second-transaction-bd049e9.json`

SHA-256:
`21064b940cb92e35a2674007d14c23ebbd1565c095eb1e6c1dac718846382dae`

The joined scalar/hash replay took 355.708 seconds. This is diagnostic replay
time, not decode latency or accepted TPS.

One prediction error was resolved during the native replay. At exactly eight
attention positions, PyTorch's BF16 matmul reduces products in a contiguous
pairwise tree; Firewing had reused the expert GEMV reducer's cross-lane tree.
The trees first produced different hashes at target layer 27. A real-tensor
diagnostic matched the contiguous tree to PyTorch, and an adversarial unit
fixture now proves the two trees are observably different. The special case is
limited to eight-term attention-value products; prior authorities retain their
original path.

`performance_claim=null`. No fixture replay time is endpoint latency or TPS.

## Decision and next gate

Promote the second recursive proposal and rollback as exact authorities. Across
the two observed transactions, width four emits four tokens total while using
2,279 combined expert rows: aggregate `A=4`, `U=2279/480=4.747917`, and
aggregate `A/U=0.842475`. Both transactions mismatch at the second comparison
and roll back two rows.

Reject width four on this two-transaction prefix as a routed-byte-leverage win,
but do not infer a production acceptance distribution or TPS. Build the
matching sequential width-two authority next. Only repeated full-path
measurements may select a runtime width.
