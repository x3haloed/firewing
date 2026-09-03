# FW-0039 - Target-derived MTP prefill and first live proposal

- Status: completed
- Disposition: correctness milestone
- Date: 2026-09-03
- Parent experiment: FW-0038
- Exactness: source-pinned EAGLE prefill alignment and real target/MTP checkpoint computation
- Hardware: Apple M1 Mac mini, 16 GiB, internal SSD, no companion hardware

## Question

Can Firewing replace FW-0038's deterministic layer-local roots with exact
pre-final-mixer hidden states from a real target prompt, apply SGLang's actual
EAGLE prefill rotation, and reproduce the resulting live MTP proposal from
checkpoint bytes?

This is the boundary immediately before target verification. A live proposal
is still not an accepted token, so this experiment cannot establish `A`, `U`,
or TPS.

## Frozen authority and method

- Clean implementation commit:
  `042b72bd46aad8c0e01713136caac61aa7dcfaea`
- SGLang source commit:
  `78c5024e9d9f589dcb4deb7f4ba4fb23f7e85385`
- Checkpoint revision:
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- MTP source-lock SHA-256:
  `8160eed0480d8a5bacad0803569f2031626dde26a87eff2b79e62058f7699282`
- EAGLE prefill source-lock SHA-256:
  `8ae9341b655e4310644b2e7442c60305b7b8907f18800407e523b3480b6aac68`
- Target endpoint fixture SHA-256:
  `2eb3712c7959837a24fe6db5b5d7b1f87c9926b4dd80eae524590e43287331ca`
- MTP input-fusion authority SHA-256:
  `5575367e78c0726399f8c6c4b8c495cc0d18721277dd2fe4059120ba5f44dc3e`
- Causal seed fixture SHA-256:
  `31682dacbd346a2529bb40d06a57056dd6cffb1f89f63eb8cf5f30618d1664d3`
- Causal attention fixture SHA-256:
  `7ff9c88f5b8d528cd8ea69e811374384dbe29412b5b11e7c93077ab3a6b46945`
- Causal decoder fixture SHA-256:
  `12a7adfbe2323864bb33da91f4f4ab05b4fbe404bf58a2b80b51e2c86dbfad5c`
- Causal logits fixture SHA-256:
  `a0e08a57d7a2d995af3f34a1d98a713c6830b5b8ddc86cea2015d733302182d4`
- Batch size: 1
- Concurrency: 1
- Boundary dtype: BF16
- Environment: macOS 26.6.2 (`25G83`), Rust 1.96.0, Python 3.11.9,
  PyTorch 2.14.0, Transformers 5.16.1
- Accepted tokens: 0
- `A=0`, `U=0`, and `performance_claim=null`

The separately pinned SGLang EAGLE worker rotates every non-chunked target
prefill left by one token and appends the target's sampled next token. Thus the
target prompt `Firewing`, tokenized as `[16207, 22856]`, first produces target
token `369` (`" is"`), then becomes the two-row MTP prefill `[22856, 369]`.
Those rows pair position-for-position with the target's exact pre-final-mixer
four-stream hiddens after tokens 16207 and 22856.

The Python generator first regenerates the complete committed 48-layer target
endpoint and refuses to continue unless its fixture is byte-for-byte equal to
the committed authority. It then performs the real MTP fusion, sequential MTP
QSA/cache update, top-10 routed plus shared-expert decoder, dedicated MTP final
mixer, and shared target head. The native Rust verifier independently replays
the same target and draft chain from checkpoint bytes.

```shell
.venv/bin/python tools/generate_mtp_causal_prefill_fixture.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  --model-lock spec/model.lock.json \
  --mtp-source-lock spec/sglang-qwen4-exp-mtp.lock.json \
  --scheduler-lock spec/sglang-eagle-prefill.lock.json \
  --endpoint-fixture fixtures/endpoint/qwen3_8_flash_next_firewing_two_token.json \
  --fusion-fixture fixtures/mtp/qwen3_8_flash_next_input_fusion.json \
  --seed-output fixtures/mtp/qwen3_8_flash_next_causal_prefill_seed.json \
  --attention-output fixtures/mtp/qwen3_8_flash_next_causal_prefill_attention.json \
  --decoder-output fixtures/mtp/qwen3_8_flash_next_causal_prefill_decoder.json \
  --logits-output fixtures/mtp/qwen3_8_flash_next_causal_prefill_logits.json

target/release/firewing verify-mtp-causal-prefill \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  spec/sglang-qwen4-exp-mtp.lock.json \
  spec/sglang-eagle-prefill.lock.json \
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
  /Users/chad/Models/firewing/evidence/FW-0039/causal-prefill-042b72b.json
```

## Result

The native path verifies 14 exact fusion captures, 108 exact BF16 captures
through the MTP decoder, two F32 captures, six int64 captures, 26 dense tensor
payloads, 20 distinct selected experts, both complete 248,320-wide logit
vectors, and 18,717,486,336 bounded logical target-plus-draft payload bytes.

The two MTP rows route to:

1. `[117,55,324,136,280,41,140,225,405,353]`; and
2. `[141,397,90,60,104,131,250,135,194,185]`.

The first MTP prefill row has top-1 token 290 but is used to populate draft
state, not emitted as the live proposal. The last row proposes token 264
(`" a"`) after the already target-authorized anchor 369 (`" is"`).

Raw receipt:
`/Users/chad/Models/firewing/evidence/FW-0039/causal-prefill-042b72b.json`

Receipt SHA-256:
`5ecb16ec9b0ffbbd8a47951f7df04e9b7be7637388bafd5b5b72831299f879ba`

The receipt's target replay took 84.999 seconds in uncontrolled mixed OS-cache
state. That timing includes correctness-oriented scalar work and repeated
authentication, excludes a complete acceptance transaction, and is not a TPS
claim. The repository has 53 Rust tests and strict Clippy passes.

## Decision, limitations, and follow-up

Promote the target-derived EAGLE prefill and first live proposal as the next
correctness authority. FW-0038's sequential MTP cache semantics remain valid;
the separately implemented `FROZEN_KV_MTP` worker is not the Qwen4 EAGLE path
used here.

Do not count token 264 as accepted. The exact target posterior after input 369
is not present yet, so no proposal comparison, correction, rollback, combined
target-plus-draft expert union, or accepted-throughput measurement exists.

Prismwing's reusable transaction rule applies directly: preserve the anchor in
the proposal vector, compare target posterior row `i` with proposal row `i+1`,
commit the matching prefix plus the target correction/bonus, and retain only
cache rows authorized by that comparison. The next experiment must extend the
target endpoint through token 369, compare its exact posterior with proposal
264, and record the first truthful acceptance and rollback decision.
