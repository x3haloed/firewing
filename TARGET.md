# Normative target and stopping conditions

This file defines when Firewing is allowed to say **done**. Thresholds are
also mirrored in `spec/acceptance.yaml`. If the two disagree, this document is
normative until the discrepancy is corrected in the same commit.

## 1. Target system

The starting computer is a 16 GB Apple M1 Mac mini.

Results on other machines are useful research results but do not satisfy the
primary completion target.

## 2. Model identity

Before a release candidate is evaluated, the repository must contain a model
lock recording:

- The exact Qwen/Qwen3.8-Flash-Next checkpoint revision.
- SHA-256 hashes for the source index, configuration, tokenizer, processor
  configuration, custom modeling code, MTP weights, and every downloaded shard.
- The exact chat template and generation defaults.
- Vision encoder revisions and preprocessing configuration.
- Every repacking, quantization, permutation, factorization, or training step,
  including deterministic seeds and output hashes where reproducible.

A smaller student model, or a
workload-specific overlay is not the target model. Such artifacts may be used
as experiments only when named distinctly.

## 3. Two references, for two questions

### 3.1 Canonical source-derived component reference

The pinned open checkpoint, Alibaba's published modeling and processing code,
and source-derived deterministic oracles are the numerical and architectural
reference for components. The project does not require or claim access to a
whole-model official-framework run. Reference fixtures must instead climb from
seeded tiny tensors to sampled real tensors loaded from the pinned checkpoint,
using readable scalar or CPU implementations whose derivation from the
published semantics is auditable. They answer:

- Were tensors repacked correctly?
- Do routers select the same experts?
- Do encoders and projectors produce equivalent embeddings?
- Do individual layers and sampled local logits agree within their declared
  numerical mode?

This component reference remains required because a hosted endpoint may use
undocumented quantization, kernels, templates, or serving revisions. It does
not prove accumulated whole-model parity with an official-framework execution;
that unavailable evidence must remain explicit in release reports. The frozen
hosted reference in Section 3.2 is the only external whole-model reference.

### 3.2 Frozen OpenRouter behavioral reference

OpenRouter model `qwen/qwen3.8-flash` is the externally hosted behavioral
reference requested by the project owner. A reference epoch must:

- Record the canonical model slug, available endpoint metadata, provider slug,
  quantization metadata when exposed, UTC time, request parameters, and raw
  response.
- Pin one provider with fallbacks disabled.
- Require support for every requested parameter, particularly `logprobs` and
  `top_logprobs`.
- Request routing metadata when the API exposes it.
- Freeze every prompt and media asset by SHA-256.
- Use one fixed reasoning mode and chat template policy.
- Store raw, immutable JSON responses before calculating metrics.

If the hosted endpoint cannot return logprobs for a modality, that modality's
distributional gate remains **not proven**; behavioral tests alone do not waive
it. A provider or model change creates a new reference epoch rather than
silently replacing the old one.

## 4. Full-capability gate

All preprocessing and inference below must occur locally, through the native
Qwen components from the pinned checkpoint:

- Text-only conversation and raw completion.
- Single-image understanding.
- Multi-image comparison and spatial reasoning.
- Video temporal reasoning, event ordering, and scene understanding.
- Mixed inputs containing text plus image and video.
- Structured output and tool-call generation supported by the reference model.
- Multi-turn conversations with cached and uncached prefixes.
- Context tests at 8K, 64K, 256K, and one 1M-token smoke case, subject to the
  pinned checkpoint's actual advertised limit.

It does not count to replace native inputs with OCR, captions, transcripts,
sampled text summaries, or a separate remote encoder. Diagnostic versions of
those pipelines may be compared but cannot satisfy this gate.

The final frozen evaluation set must contain at least:

| Slice | Cases | Scored output tokens |
| --- | ---: | ---: |
| Text and reasoning | 500 | 40,000 |
| Images and multi-image | 200 | 15,000 |
| Video | 100 | 7,500 |
| Mixed modality | 100 | 7,500 |
| Tools and structured output | 100 | 7,500 |
| Long context | 30 | 15,000 |
| **Total minimum** | **1,130** | **100,000** |

Fixtures must cover common, adversarial, low-signal, multilingual, safety,
rare-domain, and modality-conflict cases. At least 20% of each modality slice
is held out until thresholds and implementation choices are frozen.

## 5. “Almost-exact” distributional gate

The primary comparison teacher-forces each frozen OpenRouter completion through
the local runtime at the identical token prefix. The hosted response requests
the top 20 logprobs at every generated position. Local logits are projected
onto those same 20 tokens plus one `OTHER` bucket containing all remaining
probability mass.

Across at least 100,000 scored positions, all of these must pass:

1. **Tokenizer identity:** 100% agreement on token IDs for every text prompt,
   serialized chat, hosted completion, and textualized modality boundary.
2. **Reference-token regret:**
   `mean(log p_reference(y) - log p_local(y)) <= 0.03 nats/token`, with no
   modality slice above 0.05.
3. **Chosen-token logprob error:** mean absolute error at most 0.08 nats and
   p99 at most 0.50 nats.
4. **Projected distribution:** mean Jensen-Shannon divergence at most 0.01
   nats and p99 at most 0.08 nats over reference-top-20 plus `OTHER`.
5. **Top-1 agreement:** at least 98.0% over all positions.
6. **Stable top-1 agreement:** at least 99.5% where the hosted top-1/top-2
   margin is at least 0.10 nats.
7. **Greedy sequence agreement:** over at least 300 prompts with 64 generated
   tokens, at least 90% of complete token sequences are identical and at least
   98% share the first 16 tokens.

Metrics are reported overall and separately for every modality, context band,
language band, and rarity band. A strong text average cannot conceal a failed
video or tail-capability slice.

These thresholds define behavioral near-equivalence, not bit identity. Any
changed model still must be labeled modified even when it passes.

## 6. Capability non-inferiority gate

Logprob similarity is necessary but not sufficient. On deterministic or
programmatically scored tasks:

- The lower bound of a paired 95% confidence interval for
  `local score - hosted-reference score` must be at least **-1.0 percentage
  point overall**.
- No required modality, long-context, tool-use, multilingual, safety, or
  rare-domain slice may have a lower bound below **-2.0 percentage points**.
- Catastrophic failures—crashes, empty outputs, malformed tool calls, missing
  media, or context truncation—must not occur in the final three runs.

Open-ended comparisons use blinded, order-randomized judging with a published
rubric and a human audit of at least 10% of disagreements. LLM judging is
secondary evidence and cannot override deterministic failures.

## 7. Performance gate

Performance is measured on the complete declared system, using the same
near-equivalent runtime configuration that passes Sections 4–6.

### 7.1 Primary completion target

For batch size one, after an 8K text prefill, across at least 30 generations of
512 accepted tokens:

- Median decode throughput is at least **50 accepted TPS**.
- The 10th-percentile run is at least **40 accepted TPS**.
- Peak resident memory, SSD traffic, network traffic, power, cache hit rate,
  speculative acceptance `A`, expert-set union `U`, and rollback are recorded.
- The runtime completes a 60-minute sustained test without OOM, thermal
  collapse, data corruption, or more than 10% throughput decay.

Decode timing includes drafting, verification, expert misses, transfers,
decompression, synchronization, sampling, and rollback. It excludes model
installation and prefill, but both are reported separately. Aggregate TPS from
multiple users is a separate metric and cannot satisfy this gate.

### 7.2 Full-capability latency

- 8K text time-to-first-token: at most 150 seconds.
- One 1080p image plus text: at most 450 seconds.
- Thirty seconds of 720p video plus text: at most 900 seconds.
- A mixed image/video fixture: at most 1200 seconds.
- The 1M-token smoke case must complete prefill without truncation or OOM,
  begin generation within 300 minutes, and sustain at least 1 accepted TPS.

These are generous staging limits, not the 4-TPS decode target.

### 7.3 Stretch target

The same conditions at median 8 TPS and 10th-percentile 6 TPS constitute the
stretch result **Firewing 8**. The project is done at the 4-TPS gate.

## 8. Reproducibility gate

A passing release candidate requires:

- Three complete runs from a clean checkout and empty runtime cache.
- One additional warm-cache run reported separately.
- Exact source commit, compiler versions, OS build, firmware, hardware BOM,
  model lock, configuration, and random seeds.
- Raw reference responses, local logits or sufficient projected probabilities,
  timings, traces, system counters, and test outputs in a content-addressed
  evidence manifest.
- A single command that recalculates every acceptance metric from frozen raw
  evidence.
- Independent reproduction by a second environment or operator before a final
  “done” claim.

Every full model walk and acceptance run on the shared 16 GiB host must also
enforce, at phase boundaries, a fail-closed host-safety policy. The run stops
and remains preserved as failed evidence if any of these conditions occurs:

- System free memory falls below 10%. This is an independent emergency floor;
  it does not enlarge the process allowance below.
- Current process physical footprint or peak resident memory exceeds 13 GiB,
  reserving at least 3 GiB of the 16 GiB host outside the measured process for
  the OS and protected services.
- Process physical footprint remains above 12 GiB after a phase that declares
  its phase-scoped model buffers released. Up to 12 GiB may remain only when
  the run predeclares the resident objects, their byte bounds, lifetime, and
  eviction order; undeclared residency is a leak, not a cache.
- Swap use grows at all from the run's baseline, or any new throttled page is
  observed.
- A protected service that was resident at run start disappears. At minimum,
  protect the user-facing application, WindowServer, the active inference
  service, and the checkpoint synchronization service when present.

Any mode that intentionally exceeds 10 GiB must additionally observe Darwin
memory-pressure events: a warning triggers immediate eviction in the declared
order and a critical event stops the run. It may not disable OS pressure
notifications, infer safety from file-backed or purgeable bytes alone, or
allocate the last free pages speculatively.

Each phase records current footprint, peak RSS, system-free percentage, swap
growth, throttled pages, buffer-release/allocator-relief state, declared
persistent residency and evictions, memory-pressure events, and protected
service identities. A successful run must demonstrate that phase-scoped model
buffers are released and that the resident inference service remains healthy;
recording unsafe pressure without stopping does not satisfy this gate. Records
created under an older target retain their original safety contract and must
not be reinterpreted under these relaxed limits.

## 9. Milestones that are valuable but not done

- **F0 — Correct skeleton:** official tiny/reference fixtures pass.
- **F1 — Full modalities:** every native modality runs locally at any speed.
- **F2 — Target-faithful stream:** unmodified routing and weights generate
  coherent output from streamed experts.
- **F3 — Firewing 1:** near-equivalent batch-one decode reaches 1 TPS.
- **F4 — Firewing 2:** near-equivalent batch-one decode reaches 2 TPS.
- **F5 — Firewing 4:** every required gate passes; project is done.
- **F6 — Firewing 8:** stretch target.
