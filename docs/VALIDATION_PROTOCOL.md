# Validation protocol

This protocol turns “looks like the same model” into reproducible numerical and
behavioral evidence. It is designed to detect subtle text degradation, lost
tail capabilities, and modality-specific failures.

## 1. Evidence layers

Use four layers of evidence; none substitutes for the preceding layer.

1. **Artifact integrity:** source hashes, tensor shapes, repack audit, tokenizer,
   processor, and template identity.
2. **Component parity:** encoder embeddings, projectors, routers, selected
   experts, layer states, KV updates, sampled local logits, and incremental
   decode compared with deterministic source-derived oracles built from the
   pinned checkpoint and Qwen's published implementation semantics.
3. **Distributional parity:** local next-token probabilities compared with a
   frozen hosted reference under identical prefixes.
4. **Capability non-inferiority:** deterministic multimodal tasks, long context,
   tools, safety, and audited open-ended comparisons.

Performance is evaluated only after one configuration passes quality gates.

A bounded whole-model official-framework execution may be acquired on
independent reference hardware for FW-0002. It is not assumed to remain
available as a continuing oracle for development or final evaluation. Reports
must state that limitation explicitly; component evidence and a qualified
hosted behavioral reference must not be described as proving continuing
checkpoint-matched whole-model parity.

## 2. Reference epochs

A hosted reference may be frozen only after FW-0002 qualifies the relationship
between the selected `qwen/qwen3.8-flash` provider and the pinned open
Qwen3.8-Flash-Next checkpoint. Similar names, shared architecture, or plausible
outputs are not identity evidence. If the relationship remains unavailable or
the comparison fails, preserve the result and do not acquire a final epoch
under that provider.

A reference epoch is immutable and receives an ID such as
`or-qwen38-flash-YYYY-MM-DD-a`. Its manifest contains:

- Canonical OpenRouter model slug: `qwen/qwen3.8-flash`.
- Endpoint/provider slug and endpoint metadata.
- Model architecture metadata returned by the Models/Endpoints APIs.
- Provider routing request: one provider only, fallbacks disabled,
  `require_parameters: true`.
- Reasoning mode, template policy, temperature, top-p, stop conditions, output
  limit, seed when supported, `logprobs: true`, and `top_logprobs: 20`.
- Opt-in routing metadata when available.
- UTC timestamps and complete raw HTTP request/response bodies with secrets
  removed.
- SHA-256 for every JSON fixture and media asset.

OpenRouter normally load-balances among endpoints, so default routing is not an
acceptable reference. Its documentation provides provider pinning,
`allow_fallbacks`, `require_parameters`, and model endpoint discovery:

- <https://openrouter.ai/docs/guides/routing/provider-selection>
- <https://openrouter.ai/docs/api/api-reference/endpoints/list-endpoints>
- <https://openrouter.ai/qwen/qwen3.8-flash>

OpenRouter currently advertises `logprobs` and `top_logprobs` for
`qwen/qwen3.8-flash`, not for an endpoint explicitly identified as the open
Qwen3.8-Flash-Next checkpoint. Parameter support must be proven for the selected
endpoint and each modality, and FW-0002 must establish whether the hosted model
is a valid behavioral authority for the target checkpoint before a final epoch
is captured.

## 3. Fixture representation

Each fixture has a stable ID and manifest fields for:

```json
{
  "id": "image/spatial/0001",
  "slice": "image",
  "messages_sha256": "...",
  "asset_sha256": ["..."],
  "license": "...",
  "reference_epoch": "or-qwen38-flash-2026-08-04-a",
  "scorer": "exact|numeric|schema|program|rubric",
  "max_output_tokens": 128,
  "tags": ["spatial", "stable", "english"]
}
```

Messages and media remain separate content-addressed objects. Secrets and
private user data are forbidden.

## 4. Teacher-forced logprob comparison

For each hosted completion token `y_i`:

1. Reconstruct exactly the hosted prefix through `y_(i-1)` using the pinned
   tokenizer and chat serialization.
2. Run the local model for the next-token logits at that prefix.
3. Record local `log p(y_i)`, local argmax, and local probabilities assigned to
   the hosted top-20 token IDs.
4. Add an `OTHER` probability equal to one minus the sum of those probabilities
   for both distributions.
5. Calculate chosen-token error, signed regret, top-1 agreement, hosted margin,
   and Jensen-Shannon divergence over the 21 buckets.

The projection deliberately uses the hosted token set rather than local top-20;
otherwise unavailable hosted probabilities would be guessed. Store the hosted
top-20 mass so low-coverage positions can be analyzed separately.

Definitions:

```text
regret_i = log p_ref(y_i) - log p_local(y_i)
abs_error_i = |log p_ref(y_i) - log p_local(y_i)|
stable_i = log p_ref(top1) - log p_ref(top2) >= 0.10
```

Jensen-Shannon divergence uses natural logarithms. Thresholds are in
`TARGET.md` and `spec/acceptance.yaml`.

## 5. Autoregressive comparisons

Teacher forcing prevents an early mismatch from changing every later prefix.
It must be complemented by greedy end-to-end comparisons:

- Temperature zero or the endpoint's documented deterministic mode.
- Identical serialization, media, max tokens, and stopping conditions.
- Exact token sequence, common-prefix length, first-token agreement, stop reason,
  and structured-output validity.
- Repeat hosted canaries to measure its own nondeterminism. Positions unstable
  within the hosted endpoint are reported separately, never deleted.

For sampled decoding, use paired fixed seeds only when both endpoints support
the same sampling semantics. Otherwise compare distributions and capabilities,
not literal sampled strings.

## 6. Capability scoring

Prefer deterministic scorers:

- Exact and normalized text answers.
- Numeric tolerance.
- JSON Schema and tool-call argument validation.
- OCR/spatial coordinates with published tolerance.
- Video temporal order, action labels, counts, and timestamp intervals.
- Retrieval accuracy and needle position for long context.
- Program execution or unit tests for code.

For open-ended responses:

- Blind model identity and randomize order.
- Publish the rubric before running the holdout.
- Use at least two independent judges when practical.
- Human-audit at least 10% of disagreements and every severe regression.
- Report win, tie, and loss—not a single opaque score.

Bootstrap paired confidence intervals by fixture, not token, so long responses
do not dominate capability conclusions.

## 7. Modality requirements

Every slice contains easy, hard, adversarial, multilingual, low-signal, and
conflicting-evidence fixtures. Required mixed tests include:

- Text asks about a detail present only in video.
- Video requires temporal order rather than single-frame recognition.
- Multiple images require cross-image comparison.
- Long context includes modality tokens near the beginning and questions near
  the end.

Record encoder/projector parity separately from language-backbone parity. A
modality failure should be localizable rather than hidden inside final text.

## 8. Hosted-reference drift

Before comparing a new local build:

1. Replay a small canary set against the hosted endpoint.
2. Compare it with the frozen epoch using the same logprob metrics.
3. If canary drift exceeds half of any final threshold, stop. Investigate the
   provider and create a new epoch if necessary.
4. Never merge epochs into one average.

The frozen raw epoch remains usable even when the live service changes.

## 9. Performance measurement

Each measurement records:

- Cold/warm state and cache contents.
- Prompt modality, token counts, context length, output length, batch size, and
  concurrency.
- Proposed, verified, accepted, rejected, and committed tokens.
- Verification width `q`, accepted length `A`, expert union `U`, and `A/U`.
- SSD demand/speculative bytes, unified-memory bytes when measurable, network
  bytes, cache hits/misses, and decompression bytes.
- TTFT, prefill TPS, decode TPS, per-token latency distribution, and wall time.
- CPU/GPU/ANE utilization, memory footprint, power, temperature, and throttling.

The timer for accepted decode TPS starts immediately before the first draft or
target decode action after prefill and stops after the final accepted token is
sampled and available to the caller.

Run order is randomized across compared configurations. Report median, p10,
p90, and every individual run. A claim includes the exact command and evidence
manifest.

## 10. Pass/fail calculation

A single offline command must:

1. Verify hashes for the reference epoch, fixtures, local results, model lock,
   and hardware manifest.
2. Recompute every metric without network access.
3. Emit a machine-readable report containing threshold, observed value,
   confidence interval where applicable, and pass/fail status.
4. Fail nonzero if any required slice is absent, undersized, corrupted, or
   outside threshold.

No dashboard-only or manually edited result can satisfy the project gate.
