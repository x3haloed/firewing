# FW-0040 - First complete target-verified MTP transaction

- Status: completed
- Disposition: correctness and routed-byte-economics milestone
- Date: 2026-09-03
- Parent experiment: FW-0039
- Exactness: checkpoint-exact greedy proposal, target verification, commit, bonus, rollback, and route union
- Hardware: Apple M1 Mac mini, 16 GiB, internal SSD, no companion hardware

## Question

Does FW-0039's first live MTP proposal survive exact target verification, and
what are the resulting accepted length `A`, combined target-plus-draft expert
union `U`, commit vector, and rollback decision?

This experiment covers one width-two greedy transaction. It is not a decode
throughput measurement and cannot establish sustained TPS or a runtime default.

## Frozen authority and method

- Implementation commit: `9c002e8411e0d13a6aac34d8a61d42f4e5680a6c`
- SGLang source commit: `78c5024e9d9f589dcb4deb7f4ba4fb23f7e85385`
- Checkpoint revision: `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Greedy EAGLE acceptance source-lock SHA-256:
  `fb542f23be3114aaace7da5bcd2202229c411687f19e469ec12c4d41e5c75976`
- Four-token target fixture SHA-256:
  `e2ccf01a37cc5cb2cf44a30185850b8910b06233bc32d7ddaaeb537204daa899`
- Transaction fixture SHA-256:
  `9a497d60b75f0b0ade7e65dac53dbfa9b06979e76671051c45cd6e0142bb9da7`
- Batch size: 1
- Concurrency: 1
- Sampling: greedy
- Verification width `q=2`
- Target routed layers: 48
- Top-k experts per routed layer and position: 10
- BF16 payload bytes per selected expert: 9,830,400
- `performance_claim=null`

The committed SGLang lock independently pins three relevant semantics: greedy
target argmax is compared with the draft candidate; correct-draft count excludes
the trailing target token; and returned accepted length includes the target
bonus or correction. Firewing applies Prismwing's independently developed
transaction structure: the proposal vector retains its already-authorized
anchor, target posterior row `i` verifies proposal row `i+1`, and a full match
commits the proposal suffix plus the last target posterior as a bonus.

The target fixture evaluates `[16207, 22856, 369, 264]`, corresponding to the
prompt `Firewing`, target anchor 369 (`" is"`), and MTP proposal 264 (`" a"`).
The joined native verifier independently replays FW-0039's checkpoint-derived
MTP path and the four-position, 48-layer target path before recomputing the
decision and route union.

`U` is byte-weighted in one target-layer-position expert-row units. For each of
the 48 target layers, Firewing unions its exact top-10 routes across both target
verification rows. It adds the live MTP row's ten distinct experts as a
separate weight namespace, then divides the 697 equally sized expert payload
rows by `48 * q = 96`. Dense, attention, PLE, LM-head, synchronization, and
physical SSD costs are not hidden in `U`; they remain additional endpoint work.

```shell
.venv/bin/python tools/generate_token_text_endpoint_fixture.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  --model-lock spec/model.lock.json \
  --continuation-token 369 \
  --continuation-token 264 \
  --output fixtures/endpoint/qwen3_8_flash_next_firewing_four_token.json

.venv/bin/python tools/generate_mtp_transaction_fixture.py \
  --target fixtures/endpoint/qwen3_8_flash_next_firewing_four_token.json \
  --mtp-seed fixtures/mtp/qwen3_8_flash_next_causal_prefill_seed.json \
  --mtp-decoder fixtures/mtp/qwen3_8_flash_next_causal_prefill_decoder.json \
  --mtp-output fixtures/mtp/qwen3_8_flash_next_causal_prefill_logits.json \
  --acceptance-lock spec/sglang-eagle-acceptance.lock.json \
  --output fixtures/mtp/qwen3_8_flash_next_first_transaction.json

target/release/firewing verify-mtp-transaction \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  spec/sglang-qwen4-exp-mtp.lock.json \
  spec/sglang-eagle-prefill.lock.json \
  spec/sglang-eagle-acceptance.lock.json \
  fixtures/tokenizer/qwen3_8_flash_next.json \
  fixtures/ngram/qwen3_8_flash_next.json \
  fixtures/ngram/qwen3_8_flash_next_row_hashes.json \
  fixtures/ple/qwen3_8_flash_next_layer1_decode.json \
  fixtures/endpoint/qwen3_8_flash_next_firewing_two_token.json \
  fixtures/mtp/qwen3_8_flash_next_input_fusion.json \
  fixtures/mtp/qwen3_8_flash_next_causal_prefill_seed.json \
  fixtures/mtp/qwen3_8_flash_next_causal_prefill_attention.json \
  fixtures/mtp/qwen3_8_flash_next_causal_prefill_decoder.json \
  fixtures/mtp/qwen3_8_flash_next_causal_prefill_logits.json \
  fixtures/endpoint/qwen3_8_flash_next_firewing_four_token.json \
  fixtures/mtp/qwen3_8_flash_next_first_transaction.json \
  /Users/chad/Models/firewing/evidence/FW-0040/first-transaction-9c002e8.json
```

## Result

The MTP proposal vector is `[369, 264]`. Exact target posterior tokens are
`[264, 2526]`, so proposal token 264 matches and token 2526 is the exact target
bonus. The verifier emits `[264, 2526]`, retains both proposal rows, rolls back
zero rows, and records one correct draft plus the bonus:

- `A=2` accepted/emitted tokens;
- target expert-union rows: 687;
- distinct live MTP expert rows: 10;
- combined expert-union rows: 697;
- `U=697/96=7.260416666666667`;
- `A/U=0.27546628407460544`;
- logical routed-expert payload: 6,851,788,800 bytes; and
- total logically verified target-plus-draft payload: 39,973,590,016 bytes.

Raw joined receipt:
`/Users/chad/Models/firewing/evidence/FW-0040/first-transaction-9c002e8.json`

Receipt SHA-256:
`6cde3394248d60c9735a8e09e1569b513610f517d2aa449e873c63c00c659772`

The joined correctness replay took 205.976 seconds. It deliberately uses
scalar, hash-heavy verification and repeated checkpoint authentication. This is
not decode latency or accepted TPS. The separately preserved four-token target
receipt has SHA-256
`199adc49313ed992d61f822a43c131935ec1e7eb9e281ddaad36664e53104e6f`.

Two bounded implementation prediction errors were resolved before acceptance:
the generalized PLE generator initially labeled its third step with stale
two-step context metadata, and the embedded output verifier assumed exactly two
rows. The fixes preserve the original two-token fixture byte-for-byte, derive
PLE metadata from live context, and require embedded output rows to match the
already identity-bounded parent output count.

The repository has 55 passing Rust tests and strict Clippy passes.

## Decision, limitations, and follow-up

Promote the exact greedy transaction semantics, target bonus, zero-rollback
decision, and route-union calculation as correctness authorities. Retire
FW-0039's `A=0`, `U=0` limitation only for this one transaction.

Do not claim that `A/U=0.275466` predicts endpoint TPS. It omits all fixed work,
uses one favorable prompt location, and includes no physical SSD measurement.
The next cheap falsification step is to continue from anchor 2526 for several
transactions, preserving mismatches and rollback, to learn whether acceptance
and expert reuse persist before building a timed full-path loop.
