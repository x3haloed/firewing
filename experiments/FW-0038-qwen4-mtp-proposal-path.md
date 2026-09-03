# FW-0038 - Complete Qwen4-Exp MTP proposal path

- Status: completed
- Disposition: correctness milestone
- Date: 2026-09-03
- Parent experiment: FW-0037
- Exactness: source-derived Qwen4-Exp MTP proposal computation with real BF16 checkpoint weights
- Hardware: Apple M1 Mac mini, 16 GiB, internal SSD, no companion hardware

## Question

Can Firewing continue the source-pinned four-stream MTP input through the
checkpoint's complete draft decoder, dedicated final mixer, and shared target
LM head while independently matching every bounded intermediate?

This is the next correctness rung before recursive proposals or acceptance can
be measured. A proposal token is not an accepted token, and a layer-local
fixture cannot establish `A`, `U`, or TPS.

## Frozen authority and method

- Clean implementation commit:
  `d4e20d3b105a5d142084fbfdf5aa9a986d4e7364`
- SGLang source commit:
  `78c5024e9d9f589dcb4deb7f4ba4fb23f7e85385`
- Checkpoint revision:
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Source-lock SHA-256:
  `8160eed0480d8a5bacad0803569f2031626dde26a87eff2b79e62058f7699282`
- Input-fusion fixture SHA-256:
  `5575367e78c0726399f8c6c4b8c495cc0d18721277dd2fe4059120ba5f44dc3e`
- Attention-residual fixture SHA-256:
  `2d92fe9ef0ab3ba50389c17808163e553f4a956434734e6be2c04d840f3f2eed`
- Decoder fixture SHA-256:
  `d0f323de0cf7f75101ddbc4c503432b542d3221eaa62a1e5222ec785b5f2a52e`
- Logit fixture SHA-256:
  `2c05d5a5e7bb99ed566f47d26a53d8f655f72235639d195e5f9a639dd0d5b60f`
- Batch size: 1
- Sequential positions: 0 and 1
- Boundary dtype: BF16
- Environment: macOS 26.6.2 (`25G83`), Rust 1.96.0, Python 3.11.9,
  PyTorch 2.14.0, Transformers 5.16.1
- Accepted tokens: 0
- `A=0`, `U=0`, and `performance_claim=null`

The generator starts from two independently specified deterministic embedding
and four-stream target-hidden inputs. Both pass the real FW-0037 MTP fusion.
It then evaluates:

1. the MTP attention hyper-connection;
2. full attention with the draft's real QSA indexer and sequential key/value/
   indexer cache;
3. the MTP MLP hyper-connection;
4. the real top-10 router, selected routed experts, shared expert, and gated
   residual update;
5. `mtp.hyper_connection_mixer`; and
6. the target's shared `lm_head.weight` over all 248,320 logits.

The two inputs are intentionally different. An earlier uncommitted probe reused
one identical fused state, causing both positions to produce identical outputs
and making the cache transition observationally weak. The committed fixture
requires distinct routes, decoder outputs, full-logit hashes, and top-1 tokens.

The native verifier generalizes the already verified attention, decoder, and
output components over an explicit checkpoint tensor prefix. Existing target
fixtures remain bound to `model.language_model.layers.N`; this fixture is bound
to `mtp.layers.0`, `mtp.hyper_connection_mixer`, and the shared
`lm_head.weight`. Every component hash links back to the source lock and parent
fixture.

```shell
.venv/bin/python tools/generate_mtp_decoder_fixture.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  --model-lock spec/model.lock.json \
  --source-lock spec/sglang-qwen4-exp-mtp.lock.json \
  --fusion-fixture fixtures/mtp/qwen3_8_flash_next_input_fusion.json \
  --attention-output fixtures/mtp/qwen3_8_flash_next_attention_residual.json \
  --decoder-output fixtures/mtp/qwen3_8_flash_next_decoder.json \
  --output-fixture fixtures/mtp/qwen3_8_flash_next_logits.json

target/release/firewing verify-mtp-proposal \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  spec/sglang-qwen4-exp-mtp.lock.json \
  fixtures/mtp/qwen3_8_flash_next_input_fusion.json \
  fixtures/mtp/qwen3_8_flash_next_attention_residual.json \
  fixtures/mtp/qwen3_8_flash_next_decoder.json \
  fixtures/mtp/qwen3_8_flash_next_logits.json \
  /Users/chad/Models/firewing/evidence/FW-0038/mtp-proposal-d4e20d3b.json
```

## Result

The independent native path verifies:

- 108 exact BF16 captures through the decoder, including 14 input-fusion
  captures, plus two F32 captures and six int64 captures;
- 18 exact final-mixer captures and two complete 248,320-value logit hashes;
- 40 ranked logit entries with exact cutoff-tie partitions;
- 26 dense tensor payloads plus 20 distinct selected experts; and
- 1,649,143,296 total bounded logical payload bytes.

Position 0 routes to experts
`[476,234,308,184,26,504,492,203,408,373]` and has top-1 proposal token
`5649`. Position 1 routes to
`[43,198,441,358,74,403,187,14,417,196]` and has top-1 token `14208`.
The sets are disjoint, so the fixture cannot accidentally pass by reusing the
first position's routed payloads.

Raw receipt:
`/Users/chad/Models/firewing/evidence/FW-0038/mtp-proposal-d4e20d3b.json`

Receipt SHA-256:
`a1780ff7bdc3e9e0a8baa69546f39780e83e9cdf255d4130753f32994dedef4f`

The repository has 52 Rust tests and strict Clippy passes.

## Decision, limitations, and follow-up

Promote the complete layer-local MTP proposal path as a correctness authority.
This is still not an acceptance result: its inputs are deterministic
production-shaped fixtures, not hidden states from a real target generation,
and no speculative scheduler or exact verifier commits its proposals. At two
tokens, QSA's 2,048-token budget includes the entire history, so SGLang's
runtime index-sharing optimization cannot change the selected attention set;
long-context MTP index sharing remains separately unverified.

The next experiment must connect the proposal path to real target-derived
pre-final-mixer hidden states and next-token embeddings, recursively draft a
bounded causal window, and compare proposals with exact target verification.
That experiment—not this one—will produce the first truthful `A` and combined
target-plus-draft expert union `U`. Prismwing's reusable lesson is to preserve
proposal, posterior, committed-token, route-union, and rollback evidence in the
same transaction; its MiMo weights and acceptance numbers do not transfer.
