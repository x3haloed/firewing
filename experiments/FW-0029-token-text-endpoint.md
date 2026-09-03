# FW-0029 - Token-derived two-step text endpoint

- Status: completed
- Disposition: correctness-repair
- Date: 2026-09-03
- Parent experiment: FW-0028
- Exactness: L0 bit-identical source-derived semantics
- Hardware/runtime: Apple M1 Mac mini (`Macmini9,1`), 16 GiB, internal SSD

## Question and hypothesis

Can Firewing replace the synthetic decoder roots used through FW-0028 with
real token embeddings, preserve token-dependent PLE and cache state through
both steps, and reach the exact complete vocabulary logits after all 48 layers?

The hypothesis is that the existing layer-local and accumulated semantics are
sufficient once the actual `Qwen4ExpTextModel.forward` initialization is added:
look up a 2,560-wide BF16 embedding row and repeat it four times along the
feature dimension.

## Frozen authority and baseline

- Checkpoint revision:
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Model lock SHA-256:
  `f87399e8659ab3274601fcd455b78b73c600f57e5fc1e91499eec3ac1f4b9444`
- Baseline commit: `7b5492d`
- Source stack: Transformers 5.16.1 and PyTorch 2.14.0
- Tokenizer authority: committed raw case `Firewing` -> `[16207, 22856]`

## Method

The source generator first verifies that the committed tokenizer fixture still
maps `Firewing` to exactly two token IDs. It reads those two rows from
`model.language_model.embed_tokens.weight`, records row hashes and locked shard
identity, and reproduces the official four-stream `repeat` initialization.

It then executes two sequential single-token calls through every layer. Each
linear-attention layer owns a fresh convolution and recurrent state across the
two calls; each full-attention layer owns a fresh indexer/K/V cache. Layer 1
uses the actual token IDs and prior-token context for all 16 sparse PLE row
lookups per step. Every layer computes a fresh top-10 route from its real
upstream activation. The two final states pass through the exact model mixer
and all 248,320 LM-head rows.

The native verifier independently tokenizes the frozen text, range-reads only
the two selected embedding rows, reconstructs their four-stream roots, and
replays the embedded hash-only authority. It does not accept fixture-provided
activations, routes, cache values, expert outputs, or logits.

Commands:

```shell
.venv/bin/python tools/generate_token_text_endpoint_fixture.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  --model-lock spec/model.lock.json \
  --output fixtures/endpoint/qwen3_8_flash_next_firewing_two_token.json

cargo run --release -- verify-token-text-endpoint \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  fixtures/tokenizer/qwen3_8_flash_next.json \
  fixtures/ngram/qwen3_8_flash_next.json \
  fixtures/ngram/qwen3_8_flash_next_row_hashes.json \
  fixtures/ple/qwen3_8_flash_next_layer1_decode.json \
  fixtures/endpoint/qwen3_8_flash_next_firewing_two_token.json \
  /Users/chad/Models/firewing/evidence/FW-0029/token-text-endpoint.json
```

Batch size and concurrency are one. Accepted tokens, speculative acceptance
`A`, and speculative union `U` are zero/not applicable: the input tokens are
teacher-forced and this is an untimed correctness run. No TPS is claimed.

## Gates

- Identity: exact checkpoint, model lock, tensor index, tokenizer fixture,
  configuration, tensor shapes, dtypes, and shard identities.
- Token boundary: native tokenization must return `[16207, 22856]`; both sparse
  embedding-row hashes and four-stream root hashes must match.
- PLE: recompute addresses from the actual tokens and prior-token context,
  range-read all selected n-gram rows, and preserve independent PLE state.
- Accumulation: every attention input must equal the preceding real layer
  output; all 48 layers retain private recurrent or KV state.
- Routing: recompute and execute every dynamic top-10 route and authenticate
  all selected expert payloads.
- Output: match all final-mixer boundaries, both complete BF16 vocabulary
  vectors, exact pinned top-20 selections, and cutoff partitions.
- Representation: commit hashes and metadata only; keep checkpoint-derived
  payloads and raw receipts outside Git.
- Host safety: enforce a 10 GiB process cap, a 4 GiB post-release cap, the 10%
  system-free floor, zero swap growth, zero new throttled pages, and protected
  service liveness at process start, after embedding lookup, after every layer,
  and after final buffer release. Predeclare bounded persistent objects and
  record release, allocator-relief, eviction, and pressure-event state.
- Kill/repair: stop at the first mismatched token, row, state, route, residual,
  logit, or host-safety gate. Preserve a failed JSON receipt when an output path
  was requested. Do not relax a downstream comparison to hide accumulated
  drift.

Excluded claims: reusable generation, accepted tokens, MTP, real multi-token
prefill, sampling, modality processing, hosted parity, latency, physical storage
traffic, and TPS.

## Result

The source fixture and independent native replay pass without a semantic
repair. Native execution verifies all seven tokenizer fixtures, both 5,120-byte
embedding rows, four-stream roots, 36 linear layers, 12 full-attention layers,
960 dynamic expert selections, both final mixers, and both complete vocabulary
projections. The two steps use 859 distinct layer-scoped expert payloads.

The final BF16 logit hashes are
`fbf954da57588b78638bf71f45d6a186294a2c4a9fadd4854c30d4b84ad10eff`
and
`3d70117bd0120f1c8394f31accd621b511f98bfed9ee667c496552d209f38a4d`.
After the complete input text, token `369` is top-ranked and decodes to ` is`;
this is a diagnostic, not a quality claim.

The native decoder portion authenticates 15,783,796,480 logical payload bytes,
the output side authenticates 1,284,526,080, and sparse embedding lookup adds
10,240, for 17,068,332,800 total. The 2,166,508-byte committed fixture has
SHA-256
`2eb3712c7959837a24fe6db5b5d7b1f87c9926b4dd80eae524590e43287331ca`.
The native receipt is
`/Users/chad/Models/firewing/evidence/FW-0029/token-text-endpoint.json`, SHA-256
`f608d72c8640fc55fe030f06caa3c291b610e811bbb7b1ef3f857d0fca00dc55`.

All 51 fail-closed safety snapshots pass. Across process start, embedding
completion, 48 layer boundaries, and final release, system-free memory never
falls below 59%, physical footprint never exceeds 2,993,232,704 bytes, peak RSS
never exceeds 3,004,596,224 bytes, and neither swap use nor throttled pages grow.
Fifty release boundaries record phase-buffer eviction and allocator-relief
state. No pressure event is observed; the verifier's 10 GiB cap means it never
enters the target's above-10-GiB pressure-observer mode. Protected services
present at process start remain present through final release.

The final repository gate has 65 Python and 43 Rust tests. Clippy passes for
all targets and features with warnings denied. A second complete source
generation is byte-identical at the fixture hash above, and the generalized
PLE generator still reproduces its original committed fixture byte-for-byte.

## Decision

Pass as a correctness repair and treat this as the first bounded native text
endpoint. The next slice should turn the fixture-shaped two-token execution
into a reusable incremental runner that can accept its own selected token,
preserve all layer caches across an arbitrary number of steps, and measure its
first complete-token wall time. This result does not yet satisfy M2's usable
decode endpoint or any performance gate.
