# Experiment program

The program is ordered by information gained per dollar and engineering week.
Later stages are conditional on earlier kill gates. This is a prospective queue,
not a results log. Executed work receives a stable `FW-NNNN` record under
`experiments/` and follows the promotion and reversal rules in
[WORKFLOW.md](WORKFLOW.md).

## E0 / FW-0001 - Remote checkpoint census and lock

**Question:** What exactly must reside, stream, execute, and remain available for
each modality?

Reuse an existing `hf download --local-dir` tree when present; never start a
duplicate acquisition merely to inspect it. Pin the Hugging Face revision and
read repository metadata, configuration, tokenizer/processor manifests, chat
template, generation configuration, license, local tree metadata, and each
locally completed shard header. If no download exists, bounded remote header
reads are permitted. Produce a tensor census grouped into:

- routed experts by layer and projection;
- attention, routers, norms, embeddings, LM head, MTP, and gated residual;
- n-gram tables, lookup row size, addressing semantics, and prefetch position;
- vision encoder and projectors;
- tokenizer, processor, template, generation defaults, and implementation;
- recurrent-state and KV bytes by context length and attention type; and
- source, active-per-token, active-per-verification, and modality-specific bytes.

**Pass:** All files and tensors are accounted for exactly; tensor names, dtypes,
shapes, offsets, shard extents, and total bytes reconcile with the pinned index
and published configuration. The model lock and source ledger contain immutable
revision and content identities. Unknown layouts or unavailable required files
fail closed.

**Kill/redirect:** Stop before full checkpoint acquisition if the checkpoint
cannot be pinned, the index and headers do not reconcile, required native
multimodal/MTP assets are absent, licensing blocks the intended local use, or
the minimum source storage cannot fit the internal SSD with a declared safety
margin.

**Claims excluded:** Runtime correctness, model fidelity, executable memory,
prefill latency, and accepted TPS.

## E1 / FW-0002 - Independent hosted-reference equivalence falsifier

**Question:** Can OpenRouter `qwen/qwen3.8-flash` legitimately serve as the
external behavioral authority for the pinned open Qwen3.8-Flash-Next checkpoint?

Freeze this experiment before capturing a final hosted epoch. It has two gated
stages:

1. **Cheap metadata and serialization audit.** Pin one provider with fallbacks
   disabled. Record its model and endpoint metadata, context limit,
   quantization, supported parameters, tokenizer behavior, reasoning controls,
   tool schema, image/video serialization, chat template behavior, stop tokens,
   and generation defaults. Compare these with the FW-0001 lock and Qwen's
   published distinction between Flash and Flash-Next.
2. **Independent output comparison, only if Stage 1 survives.** On a small panel
   frozen in advance, compare provider top-20 logprobs and greedy outputs with a
   pinned official Transformers execution of the open checkpoint on independent
   reference hardware. Remote or non-M1 compute is allowed only for this
   reference acquisition and never contributes to qualifying local inference.
   Include raw completion, tool, image, and video cases; preserve exact prompts,
   templates, request parameters, provider identity, outputs, and hashes.

Stage 1 passes only when no material semantic mismatch is found and the selected
provider exposes every required parameter. Before Stage 2, freeze at least 20
fixtures—eight text/raw-completion, four tool/structured-output, four image, and
four video—with at least 64 scored output positions each and three repeated
hosted canaries. Stage 2 requires 100% tokenizer-ID agreement and applies the
aggregate and per-modality distributional thresholds from `TARGET.md` without
deleting unstable or unfavorable positions. The experiment record must state
that this small panel qualifies reference use; it does not prove the final
92,500-position gate.

**Kill/redirect:** Any tokenizer, template, processor, tool, modality, checkpoint
relationship, or distributional mismatch rejects this provider as the
near-equivalence authority. Preserve it as a capability benchmark if useful.
Do not tune the local runtime toward a failed hosted target or silently replace
the provider.

**Claims excluded:** Final hosted parity, local correctness, capability
non-inferiority, and accepted TPS.

## E2 / FW-0003 - Acquisition feasibility and storage baseline

After FW-0001 passes, verify internal-SSD capacity and acquire the pinned source
checkpoint into a content-addressed store. Measure bounded sequential and
production-shaped random reads for source expert, n-gram row, common-weight, and
MTP extents under declared cold, warm-OS, and warm-application states.

**Pass:** Every acquired byte verifies against the lock; the host-safety gate
passes; measurements report physical bytes and establish storage-only bounds for
ordinary decode and four-step MTP without calling either endpoint TPS.

**Kill/redirect:** Stop runtime construction if even an impossible-best byte
ledger is below Firewing 1. Revisit representation hypotheses before kernels.

## E3 / FW-0004 onward - Correctness ladder

Implement deterministic tiny fixtures and slow references for tokenizer and
template serialization, n-gram addressing, routing, routed/shared experts,
Gated DeltaNet, QSA, gated residual state, MTP, vision processing, and cache
updates. Promote each semantic only after its fixture passes. Then advance to
sampled real tensors, accelerated parity, a slow whole-model text endpoint, and
native modality endpoints in that order.

No performance default may be promoted in this stage. A mismatch stops the
ladder at the lowest failing semantic.

## E4 - First complete baseline and causal profile

Run a source-derived complete request on the M1 with bounded streaming and all
safety telemetry. Record prefill separately, then time complete decode including
I/O, execution, synchronization, sampling, and any MTP correction. Profile the
largest measured wall-time shares and update `spec/throughput-model.json`.

Only this complete path may establish baseline accepted TPS. Short diagnostic
runs remain lower milestones until the 30-by-512 and 60-minute protocols run.

## E5 - Representation and expert-cache branches

Test source-preserving layout/transport changes before approximate weight
formats. Any INT8/INT4 or mixed-precision branch is named `modified`, receives
real routed-activation and downstream-logit gates, and cannot become the
near-equivalent default until every required slice passes. Cache experiments
must charge misses, refills, executable-memory traffic, and eviction overhead.

Kill a branch when its impossible-best capacity/traffic bound misses its target
or its cheapest representative fidelity fixture fails.

## E6 - Native MTP and accepted-throughput branch

Measure the checkpoint's trained multi-step MTP against the exact target path.
Record proposal cost, verification width `q`, committed `A`, byte-weighted expert
union `U`, `A/U`, rollback, cache effects, and complete wall time on development
and untouched text slices. Reusing QSA indices or expert work counts only when
the complete transaction preserves the target distribution.

Promote only a repeatable full-path gain. Proposed tokens and perfect-proposer
or fixed-union oracles remain analytical bounds.

## E7 - Full capability, holdouts, and sustained gates

After one configuration passes the complete correctness ladder, freeze the
1,030-case, 92,500-token evaluation set and run every modality, context,
distributional, capability, latency, safety, reproducibility, and performance
gate in `TARGET.md`. Keep cold and warm results distinct. Firewing is complete
only when the same near-equivalent configuration passes 4 median/3 p10 accepted
TPS and the 60-minute sustained test; 8/6 is the stretch result.
