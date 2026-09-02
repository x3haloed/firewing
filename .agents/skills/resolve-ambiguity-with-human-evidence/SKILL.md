---
name: resolve-ambiguity-with-human-evidence
description: Use when plausible answers would change the next decision and automated evidence, repository inspection, or an existing user decision cannot resolve the question; especially for UX, usefulness, comprehension, workflow fit, trust, taste, accessibility experience, or developer ergonomics. Do not use human review to defer a specified decision, validate ordinary correctness, or substitute prototypes for finished-grade implementation.
---

# Resolve Ambiguity With Human Evidence

Use human attention only to answer a decision-relevant question. Optimize the entire path from ambiguity to observation, then retain the lesson rather than the experimental artifact.

## Pass the ambiguity gate

Proceed only when all are true:

1. State one unresolved question precisely.
2. Identify at least two plausible answers that would cause meaningfully different decisions.
3. Confirm that existing evidence, direct inspection, automated checks, or a cheaper safe probe cannot resolve it.
4. Explain why human experience or judgment is the necessary observation surface.
5. Define what observation would discriminate between the plausible answers.

If the user has already specified the desired behavior, implement it. If the uncertainty is ordinary correctness, test it. If the answers would not change a decision, continue without an experiment.

## Design the smallest faithful evaluation

Work backward from the discriminating observation:

- Preserve realism on every dimension that could change the answer.
- Fake, seed, isolate, or bypass infrastructure irrelevant to the question.
- Present the artifact in the exact state needed for evaluation; remove avoidable installation, authentication, configuration, deployment, population, and navigation steps.
- Give the human one bounded task and one clear question tied to the pending decision.
- Use parallel variants only when relative comparison is more discriminating than absolute judgment. Keep each variant tied to a distinct hypothesis and avoid fatigue, anchoring, and simultaneous unrelated questions.
- Prefer a local, reversible, disposable evaluation surface. Do not deploy, contact other people, or mutate production merely to reduce review friction unless separately authorized.

Do not optimize for the smallest prototype. Optimize for the lowest total cost of a credible answer, including agent effort, reviewer activation energy, reviewer attention, fidelity, confounds, and likelihood of response.

## Stage the observation

Bring the evaluation directly to a ready-to-use state when the available tools and authorization permit it. Tell the human:

1. What bounded action to perform.
2. What to attend to without revealing a preferred answer.
3. How to move between variants, when applicable.
4. The specific question whose answer will change the next decision.

Observe actual behavior and response rather than converting politeness, completion, or lack of complaint into validation. Preserve uncertainty when the result does not discriminate between the alternatives.

## Assimilate the lesson and discard the experiment

After the observation:

1. State the smallest decision-relevant lesson supported by the observation.
2. Update the work frontier only when the lesson changes an outcome invariant or leaves a material prediction error unresolved. Otherwise use it directly for the next decision without creating frontier residue.
3. Remove prototypes, variants, seeded data, review harnesses, screenshots, and other disposable scaffolding created for the evaluation. Do not delete pre-existing or user-owned artifacts. Preserve an experimental artifact only when the user explicitly requests it or it independently meets project standards and is deliberately promoted through ordinary implementation review.
4. Implement the chosen direction as finished-grade project work. Do not copy experimental shortcuts into production merely because the experiment succeeded.
5. Verify the integrated result independently in the real system.

Completing the evaluation resolves an ambiguity; it does not complete the corresponding product work. Only the lesson should normally survive the experiment.
