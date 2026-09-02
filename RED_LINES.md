# Red lines

These boundaries protect the meaning of the project. Crossing one is allowed
only for a clearly labeled experiment; the resulting system cannot be called a
passing Firewing runtime.

## Model identity and capability

- Do not substitute a smaller model or a workload-specific student while
  continuing to call it Qwen3.8-Flash-Next.
- Do not silently change the checkpoint, tokenizer, chat template, processor,
  modality encoders, MTP weights, or hosted endpoint.
- Do not replace native image or video processing with OCR, captions, sampled
  summaries, or another model for the final capability gate.
- Do not drop rare experts, global-attention history, long-context positions,
  modality tokens, or tool behavior merely because common text benchmarks are
  unaffected.
- Do not describe changed weights, top-k, routers, topology, accepted surrogate
  outputs, or distilled components as exact. Report their exactness class and
  training history.

## Evidence

- Do not use a few visually plausible generations as correctness evidence.
- Do not report only aggregate quality. Every modality, context band, language
  band, and rare-capability slice must remain visible.
- Do not tune on the held-out evaluation split, cache its outputs, or use
  fixture-specific routing, prefetch, or answer tables.
- Do not discard failed, divergent, cold-cache, throttled, or unfavorable runs.
- Do not move thresholds after seeing final results without versioning the
  target and rerunning on a new untouched holdout.
- Do not compare against an unpinned OpenRouter route. Disable fallbacks, record
  provider metadata, and create a new reference epoch on drift.
- Do not claim parity when logprobs are unavailable. Mark the distributional
  gate unproven.

## Performance accounting

- Do not report proposed speculative tokens as output tokens. Only committed,
  accepted tokens count.
- Do not report aggregate multi-request TPS as single-request interactive TPS.
- Do not omit drafter, verification, rollback, transfer, decompression, cache
  misses, synchronization, or sampling time from decode measurements.
- Do not present theoretical SSD, DRAM, GPU, ANE, or network bandwidth as an
  achieved model rate.
- Do not call latency hidden “bytes avoided.” Maintain separate storage,
  executable-memory, compute, and barrier ledgers.
- Do not warm caches secretly. Cold and warm runs are distinct named results.
- Do not use remote or rented compute on the final inference critical path.
- Do not use companion compute, memory, storage, or networking hardware and
  still claim the primary M1-and-internal-SSD target.

## Engineering and research conduct

- Do not start expensive training before the
  corresponding cheap kill test passes.
- Do not expand compressed weights into a large temporary representation and
  count only their on-disk size. Measure the representation consumed by the
  compute engine.
- Do not trust parameter-space compression alone. Validate on routed
  activations and measure downstream route and logit divergence.
- Do not let a cache or prefetcher issue unbounded speculative I/O that starves
  demand reads or damages the SSD without accounting for it.
- Do not treat the relaxed 13 GiB process ceiling as permission to consume the
  host blindly.
- Do not count file-backed, purgeable, compressed, or reclaimable pages as
  released merely because the runtime expects the OS to recover them. Measure
  physical footprint after allocator relief and record actual evictions.
- Do not bypass checkpoint integrity, tensor-shape checks, bounds checks, or
  fail-closed validation for speed.

## Safety, privacy, and legal constraints

- Never commit API keys, provider credentials, private prompts, unlicensed
  media, or model weights to the repository.
- Reference fixtures must be licensed for redistribution or represented by
  hashes and private acquisition instructions.
- Respect checkpoint, tokenizer, dependency, and dataset licenses and preserve
  required notices.
- Do not upload private user media to OpenRouter. Reference acquisition uses
  project-owned or redistributable fixtures and the selected provider's
  documented privacy controls.
- Do not perform mains-voltage modifications, defeat thermal protections, run
  uncertified battery/power assemblies, or operate used server hardware without
  appropriate cooling and electrical capacity.
- Do not trade silent data corruption for throughput. Every packed artifact and
  evidence bundle is checksummed.

## Exactness vocabulary

Use these labels consistently:

- **L0 — Bit-identical:** identical checkpoint representation and deterministic
  arithmetic result.
- **L1 — Function-preserving transform:** layout, permutation, or lossless
  recoding with equivalent computation.
- **L2 — Target-distribution-preserving:** approximations exist only in a draft
  path and an exact correction/verification procedure preserves the target
  distribution.
- **L3 — Bounded approximation:** a documented error knob can reduce to an
  exact path, but accepted outputs may differ.
- **L4 — Architecture-modified:** routing, expert count, weights, or topology
  changed and calibrated or distilled.
- **L5 — New student:** a separately trained model inspired by
  Qwen3.8-Flash-Next.

Passing the near-equivalence gates does not promote L3–L5 systems to L0–L2; it
demonstrates controlled quality, not identity.
