# FW-0042 - Sequential MTP transaction after correction

- Status: in progress
- Date: 2026-09-03
- Parent experiment: FW-0041
- Exactness: target-faithful greedy chain; native joined replay pending
- Hardware: Apple M1 Mac mini, 16 GiB, internal SSD, no companion hardware

## Question

Does FW-0041's early width-four mismatch repeat after the exact target
correction is carried into a second transaction, and how do sequential accepted
length and routed-expert union compare with width two?

## Frozen method and partial evidence

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
returns `[11,45815,11,321]`, so its provisional ledger records the same second
comparison mismatch as FW-0041: emitted `[11,45815]`, `A=2`, two retained rows,
two rolled-back rows, 1,041 target union rows, 25 live draft rows, and
`U=1066/480=2.220833` (`A/U=0.900563`). This is not yet promoted: the native
recursive and target replays must independently reproduce every committed hash
and recompute the transaction.

`performance_claim=null`. No fixture replay time is endpoint latency or TPS.

## Next gate

Generalize the native recursive verifier without weakening FW-0041's exact
identity checks, replay the eight-token target and second transaction, then
build the matching sequential width-two authority. Preserve either outcome;
only repeated full-path measurements may select a runtime width.
