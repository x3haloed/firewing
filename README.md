# Firewing

Firewing is a consumer-hardware runtime research project for the full-capability
open-weight Qwen3.8-Flash-Next model. The qualifying machine is a 16 GB Apple M1
Mac mini using its internal SSD, with no companion hardware.

The project is not finished. Its final gate remains a near-equivalent native
multimodal runtime sustaining at least **4 accepted tokens/s** for one
interactive request. A bounded exact two-token text-to-logits endpoint now
exists, but no usable generation endpoint, accepted-TPS result, hosted-parity
result, or native multimodal result has been established yet.

## Mission and definition of done

Firewing is complete only when every gate in [TARGET.md](TARGET.md) passes from
a clean checkout. In condensed form:

- exact, auditable model/tokenizer/processor and hosted-reference locks;
- native local text, image, multi-image, video, mixed-modality, tool,
  multi-turn, and long-context execution;
- near-equivalent distributions over at least 92,500 scored tokens, plus
  capability non-inferiority;
- median batch-one decode of at least 4 accepted TPS after an 8K prefill,
  with the required tail, latency, safety, and sustained-run gates;
- three cold reproductions, a warm run, raw content-addressed evidence, and an
  independent reproduction.

The 8-TPS result is a stretch goal. Proposed tokens, aggregate multi-user TPS,
kernel-only timing, decompression-only timing, or modified-model output do not
satisfy the primary target. See [RED_LINES.md](RED_LINES.md).

## Repository map

- [TARGET.md](TARGET.md) — normative completion and stopping conditions.
- [RED_LINES.md](RED_LINES.md) — shortcuts that do not count.
- [LEARNINGS.md](LEARNINGS.md) — durable evidence, reversals, and deductions.
- [docs/WORKFLOW.md](docs/WORKFLOW.md) — experiment and promotion discipline.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — checkpoint-derived component
  and traffic ledger.
- [docs/VALIDATION_PROTOCOL.md](docs/VALIDATION_PROTOCOL.md) — fidelity and
  performance methodology.
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) — active staged research plan.
- [docs/SOURCES.md](docs/SOURCES.md) — external authority and provenance ledger.
- [experiments/](experiments/) — immutable records for executed, rejected, and
  reversed experiments.
- [spec/throughput-model.json](spec/throughput-model.json) — machine-readable
  measured constants and provenance.

## Checkpoint workflow

The pinned checkpoint is installed and SHA-256 verified at
`/Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d`. A
metadata-only census does not contact the network or read tensor payload bytes:

```shell
python3 tools/checkpoint_census.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  --output /Users/chad/Models/firewing/evidence/FW-0001/source-census.json \
  --model-lock /Users/chad/Models/firewing/evidence/FW-0001/model-lock.json
```

Before a future installation, rerun that command with `--require-complete` and
check the destination parent with a declared reserve:

```shell
python3 tools/checkpoint_capacity.py \
  /Users/chad/Models/firewing/checkpoints \
  --model-lock /Users/chad/Models/firewing/evidence/FW-0001/model-lock.json \
  --reserve-bytes 0 \
  --output /Users/chad/Models/firewing/evidence/FW-0001/capacity.json
```

Verify every copied byte against the final lock. This deliberately reads the
full checkpoint and should not run concurrently with a download:

```shell
python3 tools/checkpoint_verify.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  --model-lock /Users/chad/Models/firewing/evidence/FW-0001/model-lock.json \
  --output /Users/chad/Models/firewing/evidence/FW-0001/copy-verification.json
```

After a complete verification, bind its hashes to the current files' live
device, inode, size, modification time, and change time. Runtime startup can
then fail closed on filesystem identity drift without hashing 360 GB again:

```shell
.venv/bin/python tools/bind_checkpoint_identity.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  --model-lock spec/model.lock.json \
  --verification /Users/chad/Models/firewing/evidence/FW-0001/checkpoint-verification-de4b8e4d.json \
  --output /Users/chad/Models/firewing/evidence/FW-0032/checkpoint-live-identity.json
```

The census is not a payload-integrity or performance result. The verifier is
not an endpoint benchmark.

## Development

The initial executable reference, native tokenizer slice, and checkpoint-backed
n-gram addressing slice are reproducible with:

```shell
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-reference.txt
brew install sleef
.venv/bin/python tools/generate_tokenizer_fixtures.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  --output fixtures/tokenizer/qwen3_8_flash_next.json
.venv/bin/python tools/generate_ngram_address_fixture.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  --model-lock spec/model.lock.json \
  --output fixtures/ngram/qwen3_8_flash_next.json
.venv/bin/python tools/generate_ngram_row_hash_fixture.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  fixtures/ngram/qwen3_8_flash_next.json \
  --output fixtures/ngram/qwen3_8_flash_next_row_hashes.json
.venv/bin/python tools/generate_expert_fixture.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  --model-lock spec/model.lock.json \
  --output fixtures/expert/qwen3_8_flash_next_real.json
.venv/bin/python tools/generate_mixture_fixture.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  --model-lock spec/model.lock.json \
  --output fixtures/mixture/qwen3_8_flash_next_real.json
.venv/bin/python tools/generate_sparse_moe_block_fixture.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  --model-lock spec/model.lock.json \
  --output fixtures/sparse_moe/qwen3_8_flash_next_layer0.json
.venv/bin/python tools/generate_mtp_input_fusion_fixture.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  --model-lock spec/model.lock.json \
  --source-lock spec/sglang-qwen4-exp-mtp.lock.json \
  --output fixtures/mtp/qwen3_8_flash_next_input_fusion.json
.venv/bin/python tools/generate_mtp_decoder_fixture.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  --model-lock spec/model.lock.json \
  --source-lock spec/sglang-qwen4-exp-mtp.lock.json \
  --fusion-fixture fixtures/mtp/qwen3_8_flash_next_input_fusion.json \
  --attention-output fixtures/mtp/qwen3_8_flash_next_attention_residual.json \
  --decoder-output fixtures/mtp/qwen3_8_flash_next_decoder.json \
  --output-fixture fixtures/mtp/qwen3_8_flash_next_logits.json

cargo test
cargo run --release -- verify-tokenizer \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  fixtures/tokenizer/qwen3_8_flash_next.json
cargo run --release -- verify-ngram \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json fixtures/ngram/qwen3_8_flash_next.json
cargo run --release -- verify-ngram-rows \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json fixtures/ngram/qwen3_8_flash_next.json \
  fixtures/ngram/qwen3_8_flash_next_row_hashes.json
cargo run --release -- verify-router \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json fixtures/router/qwen3_8_flash_next_real.json
cargo run --release -- verify-expert \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json fixtures/router/qwen3_8_flash_next_real.json \
  fixtures/expert/qwen3_8_flash_next_real.json
cargo run --release -- verify-mixture \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json fixtures/router/qwen3_8_flash_next_real.json \
  fixtures/expert/qwen3_8_flash_next_real.json \
  fixtures/mixture/qwen3_8_flash_next_real.json
cargo run --release -- verify-sparse-moe \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json fixtures/router/qwen3_8_flash_next_real.json \
  fixtures/expert/qwen3_8_flash_next_real.json \
  fixtures/mixture/qwen3_8_flash_next_real.json \
  fixtures/sparse_moe/qwen3_8_flash_next_layer0.json

# Source-pinned real-weight MTP input fusion (component verification, not TPS)
cargo run --release -- verify-mtp-input-fusion \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  spec/sglang-qwen4-exp-mtp.lock.json \
  fixtures/mtp/qwen3_8_flash_next_input_fusion.json

# Complete source-pinned MTP proposal path through the shared target LM head
cargo run --release -- verify-mtp-proposal \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  spec/sglang-qwen4-exp-mtp.lock.json \
  fixtures/mtp/qwen3_8_flash_next_input_fusion.json \
  fixtures/mtp/qwen3_8_flash_next_attention_residual.json \
  fixtures/mtp/qwen3_8_flash_next_decoder.json \
  fixtures/mtp/qwen3_8_flash_next_logits.json

# Target-derived EAGLE prefill through the first live MTP proposal (slow)
cargo run --release -- verify-mtp-causal-prefill \
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
  fixtures/mtp/qwen3_8_flash_next_causal_prefill_logits.json

# Bounded exact tokenizer-to-logits replay (slow; reads selected experts across all 48 layers)
cargo run --release -- verify-token-text-endpoint \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  fixtures/tokenizer/qwen3_8_flash_next.json \
  fixtures/ngram/qwen3_8_flash_next.json \
  fixtures/ngram/qwen3_8_flash_next_row_hashes.json \
  fixtures/ple/qwen3_8_flash_next_layer1_decode.json \
  fixtures/endpoint/qwen3_8_flash_next_firewing_two_token.json

# Exact real-expert Metal BF16 GEMV probe (component timing, not TPS)
cargo run --release -- bench-metal-bf16-gemv \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  fixtures/router/qwen3_8_flash_next_real.json \
  fixtures/expert/qwen3_8_flash_next_real.json \
  kernels/bf16_gemv.metal \
  IMPLEMENTATION_COMMIT

# Once-authenticated read-only tensor catalog (component timing, not TPS)
target/release/firewing bench-checkpoint-catalog \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  /Users/chad/Models/firewing/evidence/FW-0032/checkpoint-live-identity-b3d7810.json \
  IDENTITY_MANIFEST_SHA256 \
  fixtures/router/qwen3_8_flash_next_real.json \
  fixtures/expert/qwen3_8_flash_next_real.json \
  IMPLEMENTATION_COMMIT

# Exact resident top-10 routed MoE fused-Metal probe (component timing, not TPS)
target/release/firewing bench-metal-top10-moe \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  fixtures/router/qwen3_8_flash_next_real.json \
  fixtures/expert/qwen3_8_flash_next_real.json \
  fixtures/mixture/qwen3_8_flash_next_real.json \
  kernels/bf16_gemv.metal \
  IMPLEMENTATION_COMMIT

# Exact two-position endpoint through the authenticated catalog (not TPS)
target/release/firewing bench-catalog-token-text-endpoint \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  /Users/chad/Models/firewing/evidence/FW-0032/checkpoint-live-identity-b3d7810.json \
  IDENTITY_MANIFEST_SHA256 \
  fixtures/tokenizer/qwen3_8_flash_next.json \
  fixtures/ngram/qwen3_8_flash_next.json \
  fixtures/ngram/qwen3_8_flash_next_row_hashes.json \
  fixtures/ple/qwen3_8_flash_next_layer1_decode.json \
  fixtures/endpoint/qwen3_8_flash_next_firewing_two_token.json \
  IMPLEMENTATION_COMMIT

# Impossible-favorable exact residency/cache screen (not endpoint TPS)
.venv/bin/python tools/analyze_exact_residency_oracle.py \
  --endpoint fixtures/endpoint/qwen3_8_flash_next_firewing_two_token.json \
  --census /Users/chad/Models/firewing/evidence/FW-0001/final-census-20260903T063239Z.json \
  --acquisition /Users/chad/Models/firewing/evidence/FW-0013/expert-acquisition.json \
  --implementation-commit IMPLEMENTATION_COMMIT \
  --output REPORT_JSON

# Favorable exact 12-GiB miss-transport/Metal-overlap bound (not endpoint TPS)
target/release/firewing bench-exact-overlap-bound \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  fixtures/router/qwen3_8_flash_next_real.json \
  fixtures/expert/qwen3_8_flash_next_real.json \
  fixtures/mixture/qwen3_8_flash_next_real.json \
  fixtures/endpoint/qwen3_8_flash_next_firewing_two_token.json \
  kernels/bf16_gemv.metal \
  IMPLEMENTATION_COMMIT \
  REPORT_JSON

# Favorable exact q2 accepted-throughput bound (not endpoint TPS)
target/release/firewing bench-q2-exact-overlap-bound \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  fixtures/router/qwen3_8_flash_next_real.json \
  fixtures/expert/qwen3_8_flash_next_real.json \
  fixtures/mixture/qwen3_8_flash_next_real.json \
  fixtures/endpoint/qwen3_8_flash_next_firewing_four_token.json \
  fixtures/mtp/qwen3_8_flash_next_first_transaction.json \
  kernels/bf16_gemv.metal \
  IMPLEMENTATION_COMMIT \
  REPORT_JSON

# Exact page-aligned zstd q2 physical read/decode/Metal bound (not endpoint TPS)
target/release/firewing bench-parallel-zstd-overlap \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  fixtures/mixture/qwen3_8_flash_next_real.json \
  kernels/bf16_gemv.metal \
  /Users/chad/Models/firewing/evidence/FW-0046/q2-zstd1-manifest-a782e77.json \
  /Users/chad/Models/firewing/evidence/FW-0046/q2-zstd1.fwz \
  IMPLEMENTATION_COMMIT \
  REPORT_JSON

# Exact page-aligned two-transaction BF16-shuffle/zstd physical overlap bound
# (not endpoint TPS)
target/release/firewing bench-sequential-shuffle-overlap \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  fixtures/mixture/qwen3_8_flash_next_real.json \
  kernels/bf16_gemv.metal \
  /Users/chad/Models/firewing/evidence/FW-0049/q2-sequential-bf16-shuffle-zstd1-manifest-6271f3d.json \
  /Users/chad/Models/firewing/evidence/FW-0049/q2-sequential-bf16-shuffle-zstd1.fwz \
  IMPLEMENTATION_COMMIT \
  REPORT_JSON

# Exact FW-0050 capacity-respecting offline cache physical replay
# (not endpoint TPS or a causal cache)
target/release/firewing bench-capacity-cache-overlap \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  fixtures/mixture/qwen3_8_flash_next_real.json \
  kernels/bf16_gemv.metal \
  /Users/chad/Models/firewing/evidence/FW-0049/q2-sequential-bf16-shuffle-zstd1-manifest-6271f3d.json \
  /Users/chad/Models/firewing/evidence/FW-0049/q2-sequential-bf16-shuffle-zstd1.fwz \
  /Users/chad/Models/firewing/evidence/FW-0050/capacity-cache-milp-4fa77fd.json \
  IMPLEMENTATION_COMMIT \
  REPORT_JSON

# Mixed compressed/decoded capacity oracle (offline bound, not endpoint TPS)
.venv/bin/python tools/analyze_executable_cache_milp.py \
  /Users/chad/Models/firewing/evidence/FW-0049/q2-sequential-bf16-shuffle-zstd1-manifest-6271f3d.json \
  /Users/chad/Models/firewing/evidence/FW-0051/capacity-cache-overlap-d671bd1.json \
  /Users/chad/Models/firewing/evidence/FW-0052/metal-swiglu-c2bac85.json \
  IMPLEMENTATION_COMMIT \
  REPORT_JSON \
  --capacity-bytes CAPACITY_BYTES

# Physical FW-0053 mixed-cache replay (favorable bound, not endpoint TPS;
# the recorded FW-0054 attempt stops on host-safety swap growth)
target/release/firewing bench-executable-cache-overlap \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  fixtures/mixture/qwen3_8_flash_next_real.json \
  kernels/bf16_gemv.metal \
  /Users/chad/Models/firewing/evidence/FW-0049/q2-sequential-bf16-shuffle-zstd1-manifest-6271f3d.json \
  /Users/chad/Models/firewing/evidence/FW-0049/q2-sequential-bf16-shuffle-zstd1.fwz \
  /Users/chad/Models/firewing/evidence/FW-0053/executable-cache-5f69eee.json \
  IMPLEMENTATION_COMMIT \
  REPORT_JSON

# Favorable unified-memory byte floor for materialized mixed caching
# (analytical rejection, not endpoint TPS)
.venv/bin/python tools/analyze_materialized_memory_floor.py \
  /Users/chad/Models/firewing/evidence/FW-0049/q2-sequential-bf16-shuffle-zstd1-manifest-6271f3d.json \
  /Users/chad/Models/firewing/evidence/FW-0034/exact-residency-oracle-2fd14bc5.json \
  /Users/chad/Models/firewing/evidence/FW-0053/executable-cache-5f69eee.json \
  /Users/chad/Models/firewing/evidence/FW-0055/executable-cache-4000000000-6bae8dc.json \
  IMPLEMENTATION_COMMIT \
  REPORT_JSON

# Modified block-FP8 weight-only real-mixture fidelity screen (not TPS)
.venv/bin/python tools/analyze_block_fp8_weight_fidelity.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  fixtures/mixture/qwen3_8_flash_next_real.json \
  IMPLEMENTATION_COMMIT \
  REPORT_JSON

# Modified block-INT8 weight-only real-mixture fidelity screen (not TPS)
.venv/bin/python tools/analyze_block_fp8_weight_fidelity.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  fixtures/mixture/qwen3_8_flash_next_real.json \
  IMPLEMENTATION_COMMIT \
  REPORT_JSON \
  --weight-format block_int8

# Finer scale-grid variant (BLOCK may be 8, 16, or 32)
.venv/bin/python tools/analyze_block_fp8_weight_fidelity.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  spec/model.lock.json \
  fixtures/mixture/qwen3_8_flash_next_real.json \
  IMPLEMENTATION_COMMIT \
  REPORT_JSON \
  --weight-format block_int8 \
  --block-size BLOCK
```

The native DeltaNet verifier currently targets Apple silicon and requires
SLEEF for bit-identical transcendental functions. `build.rs` discovers the
Homebrew prefix at `/opt/homebrew/opt/sleef`; set `SLEEF_ROOT` when using a
different installation prefix.

Transformers is a fixture authority, not the qualifying runtime. The native
implementation remains responsible for every Qwen4-Exp semantic and all final
performance accounting.

## Licensing

Firewing's original source code and documentation are licensed under Apache
License 2.0. Qwen3.8-Flash-Next is a separate upstream work distributed under
the Qwen Community License 1.0; model weights are not part of this repository.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Terms

- **Accepted TPS:** verifier-committed output tokens divided by the declared
  complete timed interval, including drafting, verification, misses, transfers,
  synchronization, and rollback.
- **Target-faithful:** original weights, routing, model distribution, and named
  source semantics apart from documented finite-precision effects.
- **Modified mode:** any changed weights, routing, topology, expert count, or
  accepted surrogate output; it remains named separately even when useful.
- **Component result:** a kernel, layer, storage, or verifier measurement that
  diagnoses a cut but is not complete endpoint throughput.
