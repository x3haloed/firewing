# FW-0032 - Once-authenticated checkpoint catalog

- Status: completed
- Disposition: production infrastructure; endpoint propagation required
- Date: 2026-09-03
- Parent experiment: FW-0031
- Exactness: L0 identity/storage preserving and L1 exact tested expert execution
- Hardware/runtime: Apple M1 Mac mini (`Macmini9,1`), 16 GiB, internal SSD,
  macOS 26.6.2 (`25G83`)

## Hypothesis

FW-0031 measured 31.064 ms to reopen, copy, and hash one warm 9.830 MB expert,
versus 3.546 ms for its exact CPU arithmetic. A catalog that authenticates the
checkpoint once, verifies its unchanged live filesystem identity at startup,
and retains read-only mappings should remove that repeated acquisition and
integrity tax without weakening the source checkpoint contract.

This experiment tests only that mechanism and one exact real expert. It does
not measure a full decoder layer or endpoint TPS.

## Authority and implementation

- Checkpoint revision:
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Model-lock SHA-256:
  `f87399e8659ab3274601fcd455b78b73c600f57e5fc1e91499eec3ac1f4b9444`
- Full checkpoint verification receipt SHA-256:
  `b6a0a6f5590ec4a4455f3f19aeb59edd722f7e854a170feaa0f10e35354ac45d`
- Live-identity manifest SHA-256:
  `9830d5ae87a0586b6c8090b0f05274e958eca062fa04504978fa0041b2b714df`
- Real expert fixture SHA-256:
  `10315f99986464e85e186cc32d55488d9c68f7db0979f5cef1411c6b7e8a4752`
- Clean implementation commit:
  `73e7da9451ede69fdb034c0499e2b067a3626573`

The catalog first hashes and parses the small, pinned identity manifest. It
verifies the model lock and the original full-checkpoint verification receipt,
then checks every one of the 144 current files against the bound device, inode,
size, modification time, and change time. Symlinks and unknown paths fail
closed. It maps all 131 safetensors shards read-only, validates each dtype,
shape, byte range, and non-overlap constraint, and reconciles all 1,658 tensors
bidirectionally with the checkpoint index. Inference views then borrow bytes
directly from those mappings without reopening, copying, or hashing payloads.

The benchmark runs the complete FW-0010 scalar authority before timing, opens
the catalog, and executes real layer-0 expert 376 thirty times from catalog
views. Every iteration checks the gate/up, SwiGLU, down, and route-weighted BF16
hashes after the timed arithmetic. Batch and concurrency are one. Cache state
is explicitly warm OS/application expert pages. Accepted tokens, `A`, `U`, and
endpoint TPS are zero/not applicable.

```shell
target/release/firewing bench-checkpoint-catalog \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  /Users/chad/Models/firewing/evidence/FW-0032/checkpoint-live-identity-b3d7810.json \
  9830d5ae87a0586b6c8090b0f05274e958eca062fa04504978fa0041b2b714df \
  fixtures/router/qwen3_8_flash_next_real.json \
  fixtures/expert/qwen3_8_flash_next_real.json \
  73e7da9451ede69fdb034c0499e2b067a3626573 \
  /Users/chad/Models/firewing/evidence/FW-0032/checkpoint-catalog-73e7da94.json
```

## Gates

- Require the exact pinned model, revision, lock, identity-manifest hash, and
  original full-verification receipt hash.
- Reject live file identity drift, symlinks, unsafe paths, unknown dtypes,
  malformed or overlapping tensor ranges, and index/header disagreement.
- Reconcile 131 shards, 1,658 tensors, and 360,023,351,514 bound checkpoint
  bytes before exposing a tensor view.
- Match all four BF16 expert captures on each of 30 measured executions.
- Predeclare 128 MiB for catalog metadata and the one-expert working set, and
  enforce every normative host-safety threshold.
- Report catalog and component timing only; never infer accepted TPS.

## Result

All gates pass. The clean run reconciles 360,000,192,888 mapped shard bytes and
229,760 safetensors header bytes. Catalog startup takes 6.363 ms in the warm
run. Exact expert arithmetic measures 3.361 ms p10, 3.362 ms median, and
3.373 ms p90 across 30 executions.

No physical disk reads occur during the clean warm run, including catalog open
and the measured loop. The separate exploratory process observed 3,801,088
physical read bytes while opening the catalog, then zero additional bytes
between `catalog_open_complete` and `measurements_complete`. Its 23.987 ms open
time and 3.364 ms expert median are diagnostic corroboration, not part of the
clean commit-bound result.

Compared across the separate FW-0031 and FW-0032 warm runs, the catalog's
3.362-ms expert median is 9.24x faster than FW-0031's 31.064-ms repeated
load-plus-hash interval and agrees within 5.2% of FW-0031's 3.546-ms CPU
arithmetic control. This comparison isolates the removed verifier overhead;
it is not an interleaved endpoint speedup claim.

All safety gates pass: system-free memory remains 63%, physical footprint stays
at or below 27,420,224 bytes before release, peak RSS stays at or below
42,942,464 bytes, and swap, throttled pages, and process writes do not grow.
Protected services remain live.

Raw clean receipt:
`/Users/chad/Models/firewing/evidence/FW-0032/checkpoint-catalog-73e7da94.json`

Receipt SHA-256:
`7a6cc4f0da9e5543990c8122ce228ec5348a742b1fa935339f4e412ccf3ea21d`

The all-zero-commit exploratory receipt is preserved at
`/Users/chad/Models/firewing/evidence/FW-0032/catalog-exploratory.json` and
hashes to
`c11cd9b0fb6324ec2ef534db1f7ec0be6a2dbf3c669eedf943afeae930e70737`.
It contributes only the explicitly labeled diagnostic observations above.

The repository gate has 69 Python and 46 Rust tests, and strict Clippy passes.
The `block` transitive dependency emits a future-incompatibility notice under
Rust 1.96.0; it is not a current build or test failure.

## Decision

Promote the authenticated catalog as storage/identity infrastructure, not as a
performance endpoint default. Propagate borrowed catalog views through the
unchanged exact endpoint loaders and reprofile the complete two-position path.
That measurement will determine how much of FW-0030's 77.703 seconds was
repeated acquisition/integrity work and where unavoidable source traffic and
arithmetic remain.

The catalog does not make 360 GB resident and does not solve Firewing 4's
all-miss expert traffic. Route reuse, bounded residency, lossless layout work,
Metal integration, and MTP expert union remain conditional on the next exact
full-path profile.
