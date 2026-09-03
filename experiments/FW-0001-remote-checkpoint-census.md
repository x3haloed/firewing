# FW-0001 - Remote checkpoint census and lock

- Status: complete
- Disposition: production — checkpoint lock, census, and verifier retained
- Date: 2026-09-02
- Owner: project owner with Codex implementation support
- Parent experiments: none
- Exactness: L0 artifact census
- Hardware/runtime: Apple M1 Mac mini (`Macmini9,1`), 16 GiB, macOS 26.6.2;
  source download staged on `/Volumes/Elements`, final verified copy on the
  internal SSD

## Question and hypothesis

Can every required Qwen3.8-Flash-Next artifact and tensor be identified,
revision-pinned, byte-accounted, and licensed before downloading the roughly
checkpoint-sized payload? The hypothesis is that repository metadata,
safetensors indices, and bounded header reads provide enough evidence for a
complete fail-closed census and storage decision.

## Frozen authority and baseline

The target repository is `Qwen/Qwen3.8-Flash-Next`. The run must resolve and
record the revision rather than inheriting a moving branch name. There is no
prior model lock, tensor census, runtime, or performance baseline.

## Method and commands

The source download was already in progress, so the experiment reused its local
Hugging Face tree manifest and completed files. It performed no network access
and read only eight-byte prefixes and JSON headers from completed safetensors
shards:

```shell
python3 tools/checkpoint_census.py \
  /Volumes/Elements/Models/Qwen3.8-Flash-Next \
  --output /Users/chad/Models/firewing/evidence/FW-0001/partial-census-20260902T201259Z.json \
  --model-lock /Users/chad/Models/firewing/evidence/FW-0001/source-lock-preliminary-20260902T201259Z.json
```

Command exit code: 0. Repository base commit: `e6399902e63047febf794a3c5b475ce1ef151dab`;
the census and verifier implementation were uncommitted at capture time.

## Gates

All repository files and tensor payload bytes must reconcile with the pinned
index, configuration, and published architecture. Required tokenizer,
processor, template, vision, n-gram, QSA, Gated DeltaNet, gated-residual, routed
expert, shared-expert, and MTP assets must be present and understood. Unknown
revision, layout, dtype, shape, offset, shard extent, license, or internal-SSD
capacity fails closed before full acquisition.

This experiment accepts zero tokens and makes no runtime, fidelity, executable
memory, latency, or TPS claim.

## Partial result

The local tree pins revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540`. Its tree-manifest SHA-256 is
`6042846bc80da9b7946c9b5814d791e899ac162c8cf4ae5a35985dcbee180542`.
It declares 144 files and 360,023,351,514 total bytes; 131 safetensors shards
account for 360,000,192,888 bytes.

At `2026-09-02T20:13:07Z`, 39 files were complete, including 32 weight shards
and 99,339,882,328 shard bytes. Their 499 headers described 49,669,909,427
parameters and 99,339,818,872 tensor bytes with no shape/offset/extent, file
size, or metadata Git-blob mismatch. The largest observed category was
88,000,422,400 bytes of BF16
n-gram embeddings. This is a partial placement fact, not per-token traffic.

The census SHA-256 is
`3475a5f19ff7d7f1f6b7634e80d1ee9eaa6da6ab3996d1a1c46af4dfa748b498`.
The preliminary source-lock SHA-256 is
`70625c5a9c20876395fd211361e8b6cc7511270d04b4188a41663d27777b23b0`.
Both artifacts are outside Git under
`/Users/chad/Models/firewing/evidence/FW-0001/`.

The empty intended internal destination was also checked with a zero-byte
safety reserve:

```shell
python3 tools/checkpoint_capacity.py \
  /Users/chad/Models/firewing/checkpoints \
  --model-lock /Users/chad/Models/firewing/evidence/FW-0001/source-lock-preliminary-20260902T201259Z.json \
  --reserve-bytes 0 \
  --output /Users/chad/Models/firewing/evidence/FW-0001/internal-capacity-20260902T201502Z.json
```

It failed closed with exit code 3. The filesystem had 353,947,160,576 bytes
available against 360,023,351,514 required, a deficit of 6,076,190,938 bytes
before a real safety reserve. The capacity evidence SHA-256 is
`805b33e1b033fa1a336567651e536cc65f77694259efc7b3985ff60677f0fe2e`.

An independent local-only preflight with Python 3.11.9 and Transformers 5.14.1
failed at `AutoConfig.from_pretrained(..., local_files_only=True)` because the
installed package does not recognize `qwen4_exp`. The checkpoint config itself
declares `transformers_version: 5.8.0.dev0`; version ordering therefore does not
establish support for this experimental architecture.

Twelve deterministic unit tests cover partial and complete census behavior,
shape/offset and metadata-integrity rejection, exact-copy verification, hash
mismatch reporting, capacity pass/fail behavior, and rejection of a preliminary
lock. All pass. Full local SHA-256 verification is intentionally deferred until
the download and internal-SSD copy finish because it will read roughly 360 GB.

## Partial decision (superseded by the final result)

Conditionally retain the census and copy-verification tooling. The revision and
expected inventory are pinned, but FW-0001 has not passed: 99 shard headers,
all local payload SHA-256 checks, the final model lock, internal-SSD capacity,
and required-asset reconciliation remain. Do not promote a runtime or TPS
claim. The internal-copy branch is currently blocked even at zero reserve;
freeing more than the measured 6.08 GB deficit plus a declared safety reserve
is required. In parallel, the next cheap independent branch is to resolve an
executable `qwen4_exp` reference while the existing download continues.

## Final result

After the checkpoint was copied to the internal SSD as
`/Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d`, the
complete census reconciled all 144 files, 131 weight shards, 1,658 tensor
headers, and 179,999,981,459 stored values. The census found
359,999,963,128 tensor bytes with no missing file, size mismatch, content-hash
mismatch, duplicate tensor, unknown dtype, invalid shape, offset gap, overlap,
or shard-extent discrepancy.

The first complete run exposed a validator prediction error: `tokenizer.json`
is an LFS-backed non-weight artifact, so its Git blob identifies the LFS
pointer rather than downloaded content. The implementation now distinguishes
ordinary Git objects, LFS artifacts, and safetensors shards; a deterministic
fixture covers the distinction. No checkpoint file was changed.

The full verifier then read all 360,023,351,514 local bytes and matched all 144
content identities. Evidence:

- final census SHA-256:
  `043b5b45edd1f4aeb628a66b00fec60c035204c30a48d34d0a95f3e10d0bd937`;
- repository model-lock SHA-256:
  `f87399e8659ab3274601fcd455b78b73c600f57e5fc1e91499eec3ac1f4b9444`;
- full verification receipt SHA-256:
  `b6a0a6f5590ec4a4455f3f19aeb59edd722f7e854a170feaa0f10e35354ac45d`.

## Final decision

FW-0001 passes its artifact census and integrity gates. It makes no runtime,
fidelity, memory, latency, or TPS claim. Storage performance remains FW-0003;
tokenizer/template semantics begin in FW-0004.
