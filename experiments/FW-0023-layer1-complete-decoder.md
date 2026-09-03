# FW-0023 - Complete layer-1 PLE-bearing decoder

- Status: completed
- Disposition: correctness-repair
- Date: 2026-09-03
- Parent experiments: FW-0012, FW-0017, FW-0022
- Exactness: L0 bit-identical component semantics
- Hardware/runtime: Apple M1 Mac mini (`Macmini9,1`), 16 GiB, internal SSD

## Question and hypothesis

Can FW-0022's exact PLE-bearing attention residual compose through layer 1's
separately parameterized MLP hyper-connection, dynamically routed top-10 MoE,
shared expert, and final four-stream residual?

The hypothesis is that the initial token 42 and cached token 43 produce
nontrivial real layer-1 routes and expose any wrapper, expert-slice, execution
order, or BF16 accumulation error. No route or activation from layer 0 is
reused as output authority.

## Frozen authority and baseline

- Checkpoint revision:
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Model lock SHA-256:
  `f87399e8659ab3274601fcd455b78b73c600f57e5fc1e91499eec3ac1f4b9444`
- Baseline commit: `ea629c1`
- Framework reference: Transformers 5.16.1
  `Qwen4ExpTextGatedResidual.forward`, `Qwen4ExpTextSparseMoeBlock.forward`,
  and `Qwen4ExpTextDecoderLayer.forward`

## Method and commands

Regenerate and require byte-identical FW-0022 parent evidence. For each of its
two post-attention states, load layer 1's real MLP hyper-connection, router,
shared expert, and shared-expert gate tensors. Compute the dynamic top-10 route,
read only those experts from the source banks, execute active experts in source
order, and freeze every MoE and final residual boundary plus selected expert
payload and weighted-output hashes.

```shell
.venv/bin/python tools/generate_full_decoder_layer1_fixture.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  --model-lock spec/model.lock.json \
  --ngram-fixture fixtures/ngram/qwen3_8_flash_next.json \
  --ngram-row-fixture fixtures/ngram/qwen3_8_flash_next_row_hashes.json \
  --ple-fixture fixtures/ple/qwen3_8_flash_next_layer1_decode.json \
  --attention-residual-fixture fixtures/attention_residual/qwen3_8_flash_next_layer1_ple.json \
  --output fixtures/decoder_layer/qwen3_8_flash_next_layer1_ple.json

cargo run --release -- verify-decoder-layer1 \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  fixtures/ngram/qwen3_8_flash_next.json \
  fixtures/ngram/qwen3_8_flash_next_row_hashes.json \
  fixtures/ple/qwen3_8_flash_next_layer1_decode.json \
  fixtures/attention_residual/qwen3_8_flash_next_layer1_ple.json \
  fixtures/decoder_layer/qwen3_8_flash_next_layer1_ple.json \
  /Users/chad/Models/firewing/evidence/FW-0023/decoder-layer1.json
```

Batch size and concurrency are one. Accepted tokens, `A`, `U`, and measured
TPS are zero because this is a complete-layer correctness fixture.

## Gates

- Fixture: exact model/config/index/lock identity; exact FW-0022 parent hash;
  deterministic regeneration; nine dense tensor identities; both expert-bank
  identities; selected expert payload hashes; and no payload bytes committed.
- Correctness: every BF16 MLP hyper-connection, router, routed mixture, shared
  expert, injection, and final residual capture must match independently.
- Routing: top-10 selection is recomputed from the actual post-attention state;
  route weights and source-ordered BF16 accumulation must match.
- State: the two calls retain FW-0022's independent PLE and DeltaNet cache
  evolution and must not reuse output authority from another layer.
- Safety: only selected expert slices and bounded ordinary tensors are read;
  generated evidence remains outside Git.
- Continuation: exact parity completes every decoder-layer wrapper variant and
  unlocks accumulated multi-layer execution.
- Kill/repair: stop at the earliest parent, hyper, route, expert, shared,
  injection, or residual mismatch and preserve the discrepancy.

Excluded claims: accumulated layers, embedding or final normalization, logits,
real prefill, endpoint behavior, modality processing, latency, and TPS.

## Result

The reference fixture passes and regenerates byte-identically without changing
FW-0021's layer-3 fixture. The initial route is
`[495, 40, 7, 110, 113, 450, 241, 252, 236, 503]`; the cached route is
`[469, 60, 456, 259, 80, 202, 453, 245, 176, 186]`. The routes are disjoint and
each source execution order is ascending.
The fixture binds nine dense tensors, both expert banks, 16 captures per step,
and twenty selected expert records. Its SHA-256 is
`5a6ba892475dcb73f986d7e5afaacf6705c36b60a574e97ca3255822c2a1b6f0`.
All 44 Python tests pass.

The first native attempt failed closed at the router tensor identity boundary.
That exposed one remaining layer-3 constant in the generalized fixture
generator: all other MLP tensors were from layer 1, but the router and routes
were incorrectly from layer 3. The generator now parameterizes that identity,
the fixture was regenerated, and a regression test requires every dense tensor
to carry the layer-1 prefix. No result from the mixed-layer fixture is retained
as valid evidence.

At commit `bb76827`, the release-mode native verifier exactly matched all 32
BF16 layer captures and twenty weighted-expert hashes for the corrected
fixture. The two routes select twenty unique experts. The verifier
authenticated 194,816,448 bytes through the FW-0022 parent, 25,666,560 bytes
of layer-1 MLP hyper/router/shared tensors, and 196,608,000 bytes of selected
expert payloads, or 417,091,008 verified payload bytes in total. The external
receipt is `/Users/chad/Models/firewing/evidence/FW-0023/decoder-layer1.json`,
SHA-256
`79c065433ed76f3f23b3334f61bb864a0644b6bd69a5d041d4e2147ab0945531`.
The final suite has 44 Python and 34 Rust tests; Clippy passes with warnings
denied.

## Decision

Pass as a correctness repair. The only PLE-bearing decoder wrapper now has
complete exact layer-local parity across initial and cached decode, and the
independent verifier prevented a mixed-layer reference from being accepted.
Together FW-0017, FW-0021, and FW-0023 cover the linear-attention,
full-attention/QSA, and PLE-bearing decoder variants. Proceed to accumulated
multi-layer execution. No endpoint or performance claim follows from this
layer-local result.
