# FW-0002 - Hosted-reference equivalence falsifier

- Status: planned; frozen before execution
- Disposition: unexecuted
- Date: 2026-09-02
- Owner: project owner with Codex implementation support
- Parent experiments: FW-0001
- Exactness: reference qualification; no model transform
- Hardware/runtime: provider-pinned OpenRouter endpoint plus independent
  reference hardware; neither contributes to qualifying M1 inference

## Question and hypothesis

Can OpenRouter `qwen/qwen3.8-flash` legitimately serve as the external
behavioral authority for the pinned open Qwen3.8-Flash-Next checkpoint? The
null position is that similar naming and shared architecture are insufficient;
the provider remains unqualified until both gated stages pass.

## Frozen authority and baseline

FW-0001 must first provide the pinned checkpoint, tokenizer, processor,
template, generation, and implementation identities. Select exactly one
OpenRouter provider, disable fallbacks, and preserve its endpoint metadata and
supported parameters. There is no existing hosted reference epoch.

## Method and commands

Stage 1 performs the metadata and serialization audit in
`docs/EXPERIMENTS.md`. Stage 2 runs only if Stage 1 passes. Before Stage 2,
freeze at least 20 fixtures—eight text/raw-completion, four tool/structured,
four image, and four video—with at least 64 scored positions each, plus three
hosted canary repeats. Compare provider top-20 logprobs and greedy outputs with
a pinned official Transformers execution of the open checkpoint on independent
reference hardware.

The execution commit must add exact commands, prompt and asset hashes,
provider/request identities, source revision, seeds and generation controls,
raw-response manifests, and secret-redaction verification.

## Gates

Stage 1 requires no material checkpoint-relationship, tokenizer, template,
processor, reasoning, tool, modality, stop-token, or generation-default
mismatch and requires every requested provider parameter. Stage 2 requires
100% tokenizer-ID agreement and the aggregate and per-modality distributional
thresholds in `TARGET.md`, with no unstable or unfavorable position removed.

Failure rejects this provider as the near-equivalence authority. It may remain
a separately labeled capability benchmark. This experiment makes no final
hosted-parity, local-correctness, capability, or TPS claim.

## Result

Not executed.

## Decision

Unexecuted. Passing qualifies capture of a final provider-pinned reference
epoch while preserving its limitations. Failure leaves hosted parity unproven
until an explicit target change selects a checkpoint-matched authority.
