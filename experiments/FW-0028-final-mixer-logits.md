# FW-0028 - Accumulated final mixer and logits

- Status: completed
- Disposition: correctness-repair
- Date: 2026-09-03
- Parent experiment: FW-0027
- Exactness: L0 bit-identical component semantics
- Hardware/runtime: Apple M1 Mac mini (`Macmini9,1`), 16 GiB, internal SSD

## Question and hypothesis

Can the exact two-step decoder result from FW-0027 continue through Qwen's
model-level four-stream collapse and complete untied vocabulary projection,
with native BF16 output identical to the pinned source implementation?

The initial hypothesis also expected the top-20 diagnostic to have a unique
BF16 cutoff. That secondary prediction was falsified before accepting the
fixture.

## Frozen authority and baseline

- Checkpoint revision:
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Model lock SHA-256:
  `f87399e8659ab3274601fcd455b78b73c600f57e5fc1e91499eec3ac1f4b9444`
- Baseline commit: `5c023ac`
- Framework reference: Transformers 5.16.1 and PyTorch 2.14.0
- Reused method: Prismwing's hash-only full-logit authority and bounded ranked
  diagnostic at commit `c87d0c1aa2c118f71ca5348434be35d02f62f031`;
  no MiMo tensor, equation, or result was reused.

## Method and gates

The source generator regenerates the complete FW-0027 parent and rejects any
disagreement with its committed fixture. It loads the three real
`model.language_model.hyper_connection_mixer` tensors, executes the
non-injecting `Qwen4ExpTextGatedResidual` expression, and projects each mixed
2,560-wide state through all 248,320 rows of the untied `lm_head.weight`.

The committed fixture contains tensor identities, payload hashes, every mixer
boundary hash, and the complete BF16 logit-vector hash. It contains no weights,
activation payloads, or full logits. A top-20 diagnostic records PyTorch's
selection plus all tokens strictly above and tied at its rank-20 cutoff.

The native verifier must independently replay all 48 decoder layers, load and
authenticate the output weights, reproduce every mixer boundary and complete
logit hash, and reproduce pinned PyTorch CPU `topk`. Unknown shapes, dtypes,
checkpoint identities, schemas, or parent hashes fail closed.

The source and native commands are:

```shell
.venv/bin/python tools/generate_text_output_fixture.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  --model-lock spec/model.lock.json \
  --output fixtures/accumulated/qwen3_8_flash_next_final_mixer_logits.json

cargo run --release -- verify-text-output \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  fixtures/ngram/qwen3_8_flash_next.json \
  fixtures/ngram/qwen3_8_flash_next_row_hashes.json \
  fixtures/hyper_connection/qwen3_8_flash_next_layer0.json \
  fixtures/deltanet/qwen3_8_flash_next_layer0_decode.json \
  fixtures/attention_residual/qwen3_8_flash_next_layer0.json \
  fixtures/sparse_moe/qwen3_8_flash_next_layer0.json \
  fixtures/decoder_layer/qwen3_8_flash_next_layer0.json \
  fixtures/ple/qwen3_8_flash_next_layer1_decode.json \
  fixtures/attention_residual/qwen3_8_flash_next_layer1_ple.json \
  fixtures/decoder_layer/qwen3_8_flash_next_layer1_ple.json \
  fixtures/accumulated/qwen3_8_flash_next_layers0_1.json \
  fixtures/accumulated/qwen3_8_flash_next_layer2.json \
  fixtures/accumulated/qwen3_8_flash_next_layer3.json \
  fixtures/accumulated/qwen3_8_flash_next_layers4_47.json \
  fixtures/accumulated/qwen3_8_flash_next_final_mixer_logits.json \
  /Users/chad/Models/firewing/evidence/FW-0028/text-output.json
```

Batch size and concurrency are one. Accepted tokens, `A`, `U`, and measured TPS
are zero because this is accumulated correctness evidence. It is not an
endpoint benchmark.

## Prediction error

Step 0 has 19 logits strictly above the BF16 rank-20 cutoff and five tokens
tied at the cutoff: `445`, `67506`, `73206`, `74377`, and `114834`. PyTorch
selects token `73206` for the displayed rank 20. A generic stable sort would
silently choose a different member. The authority now preserves the entire
cutoff partition and uses the existing libc++ implementation shim that was
pinned during FW-0027. Step 1 has a unique cutoff.

This does not affect the authoritative full-vector hashes; it corrects the
interpretation of a lossy ranked diagnostic. The prediction-error record is
`/Users/chad/Models/firewing/evidence/FW-0028/prediction-errors.json`, SHA-256
`21da4061fa8c45d41c55fd55d4c243e5689397c8a47d194957fd3e03ea724842`.

## Result

Native verification passes both complete vocabulary projections exactly. It
matches 18 final-mixer capture hashes, two 248,320-value BF16 logit hashes, and
40 pinned ranked entries. The output tensors authenticate 1,284,526,080
logical payload bytes; including the replayed decoder parent, the chain
authenticates 17,795,772,160 logical payload bytes.

The 6,965-byte fixture has SHA-256
`c9092ea80171a1072869efd3c27e6ddcddee1bf3a1eece45f860f1d0c54f07be`.
The native receipt is
`/Users/chad/Models/firewing/evidence/FW-0028/text-output.json`, SHA-256
`575517152ac73c5e520ba79ebee30cd9b14cd2a8421f0c766219d2ac2234619c`.
All 62 Python and 40 Rust tests pass. Clippy passes for every target and feature
with warnings denied.

## Decision

Pass as a correctness repair. The entire output side after layer 47 is now
exact for both accumulated steps. The next correctness slice should replace
the synthetic layer-0 roots with real token-embedding rows and connect them to
this proven destination, producing the first token-derived text logits.

This result does not establish token-derived execution, an input-to-output text
endpoint, MTP, prefill, modality behavior, latency, or TPS.
