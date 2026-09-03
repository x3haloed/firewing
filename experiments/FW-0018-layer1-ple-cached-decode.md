# FW-0018 - Real layer-1 PLE cached decode

- Status: completed
- Disposition: correctness-repair
- Date: 2026-09-03
- Parent experiments: FW-0005, FW-0006, FW-0017
- Exactness: L0 bit-identical component semantics
- Hardware/runtime: Apple M1 Mac mini (`Macmini9,1`), 16 GiB, internal SSD

## Question and hypothesis

Can a bounded native implementation exactly reproduce the layer-1 PLE
injection for an initial token and a cached second token, including sparse
n-gram row retrieval, token-context state, grouped query/key normalization,
signed-square-root gating, and the nine-token dilated-convolution state?

The hypothesis is that only 16 real 160-wide n-gram rows are needed per token.
The 102.4 GB logical embedding table therefore need not be materialized: the
same address and sparse-row authority established by FW-0005/FW-0006 can feed
the six ordinary PLE tensors. Following Prismwing's incremental-cache fixture
pattern, both cache objects must be captured after each call and the cached
second output must be checked independently of the first output.

## Frozen authority and baseline

- Checkpoint revision:
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Model lock SHA-256:
  `f87399e8659ab3274601fcd455b78b73c600f57e5fc1e91499eec3ac1f4b9444`
- Baseline commit: `2bc409eeb860f975028b85b35241a5fdf8c02063`
- Fixture commit: `3f75bdd65a4e42da729e681878ffe35836bbb523`
- Candidate commit: `6fd0a0afb6b722638fb80427ec230d1b410ef086`
- Frozen fixture SHA-256:
  `7836ba5d02cdfb27d19d3dab4ee672a5ba38e7b375fae2157ddd747da294a924`
- Raw evidence SHA-256:
  `b9a10ced89d7bc6b2a85e9fbd6c3e9b01279cc16685aaf3d0ba80c44716a6874`
- N-gram address fixture SHA-256:
  `cdfd44ad62dc8fe60219b1f97e966faf776e49f30e7f46fb11f07d7e913a1430`
- N-gram row fixture SHA-256:
  `8896518e313ff0cb9d847fe5f6170b8f56ec168196c50d18a527ef89e3e2ffce`
- Framework reference: Transformers 5.16.1
  `Qwen4ExpTextNGramEmbedding.forward`, `Qwen4ExpTextPLELayer.forward`,
  `Qwen4ExpTextPLELayer._short_conv`, and
  `LinearAttentionAndFullAttentionLayer.update_conv_state`

## Method and commands

Use deterministic BF16 four-stream inputs of shape `[1,1,10240]` with token
IDs 42 and 43. The first call begins with the implicit two-EOS n-gram context;
the second must consume the cached `[EOS, 42]` context. Derive all 16 global
row addresses per call from the pinned int64 buffers, read only those 320-byte
rows from the 128-part table, and record their payload hashes and physical
locations.

Execute the source equations with the real key/value projections, three
grouped RMS weights, and dilated depthwise convolution. Freeze exact BF16
captures for embedding, projections, normalized keys/queries, raw and
transformed gates, gated values, normalized convolution input, convolution
result, and final PLE output. Freeze exact int64 token-context state and BF16
short-convolution state after each call. The generator may instantiate small
official submodules for independent projection/norm checks, but must not
instantiate the full 102.4 GB embedding module.

```shell
.venv/bin/python tools/generate_ple_fixture.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  --model-lock spec/model.lock.json \
  --ngram-fixture fixtures/ngram/qwen3_8_flash_next.json \
  --ngram-row-fixture fixtures/ngram/qwen3_8_flash_next_row_hashes.json \
  --output fixtures/ple/qwen3_8_flash_next_layer1_decode.json

cargo run --release -- verify-ple \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  fixtures/ngram/qwen3_8_flash_next.json \
  fixtures/ngram/qwen3_8_flash_next_row_hashes.json \
  fixtures/ple/qwen3_8_flash_next_layer1_decode.json \
  /Users/chad/Models/firewing/evidence/FW-0018/ple.json
```

Batch size and concurrency are one. Accepted tokens, `A`, `U`, and measured
TPS are zero because this is a stateful component correctness fixture.

## Gates

- Fixture: exact model/config/index/lock identity; both n-gram parent fixture
  hashes; six dense tensor identities; 32 selected row identities; deterministic
  regeneration; and no checkpoint-derived payload committed.
- Correctness: all declared BF16 and int64 captures match at both steps,
  including the PLE output added before layer 1's attention hyper-connection.
- State: token context advances from `[EOS, EOS]` to `[EOS, 42]` to `[42, 43]`;
  the nine-position convolution state persists independently.
- Safety: only 5,120 useful embedding bytes per token and the bounded ordinary
  tensors are loaded; no full table mapping or companion hardware is used.
- Continuation: exact parity unlocks complete layer-1 composition and then the
  full-attention gate at layer 3.
- Kill/repair: fail at the earliest address, row, projection, normalization,
  gate, cache, convolution, or output boundary and preserve the mismatch.

Excluded claims: complete layer 1, full attention, multi-layer accumulated
parity, endpoint behavior, modality behavior, latency, physical I/O, and TPS.

## Result

The source-derived fixture regenerated byte-identically. It authenticated 16
sparse n-gram rows for token 42 against the implicit `[EOS, EOS]` context and
16 more for token 43 against cached context `[EOS, 42]`; all 32 global rows
were distinct. The native verifier independently validated each live
safetensors tensor descriptor, physical part/local-row mapping, model-lock
entry, 320-byte row payload, and frozen row hash before using the embedding.

Across both calls it exactly matched 15 BF16 captures per step: hidden input,
the 2,560-wide sparse embedding, key/value projections, grouped key/query
normalization, BF16 key-query products, reduced gate, signed-square-root gate,
sigmoid, gated value, normalized convolution input, nine-position convolution
state, dilated convolution result, and final 10,240-wide PLE output. Both int64
token-context-state captures also matched exactly. The second-token result
therefore exercises both caches rather than repeating an initial-state call.

The verifier authenticated six dense tensors totaling 65,679,360 bytes and
requested exactly 10,240 embedding bytes for the 32 selected rows. Persistent
state is 184,320 BF16 convolution bytes plus 16 int64 context bytes. The full
suites passed with 37 Python and 25 Rust tests, and Clippy passed with warnings
denied.

No physical-I/O timing, accepted tokens, or TPS was measured. The sparse reads
are correctness accesses and do not supersede FW-0008's transport result.

## Decision

Pass as a correctness repair. The only PLE-bearing text layer now has an exact,
bounded native cached-decode primitive. Complete layer-1 composition is
unlocked; full attention at layer 3 remains the other missing decoder semantic.
No performance default follows from component parity.
