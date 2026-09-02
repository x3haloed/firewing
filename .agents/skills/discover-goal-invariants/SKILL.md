---
name: discover-goal-invariants
description: Use after live behavior, user feedback, performance evidence, integration constraints, maintenance experience, or a resolved prediction error shows that some property may be constitutive of success even though it was absent from the original brief.
---

# Discover Goal Invariants

Determine whether the evidence refines what success means rather than merely how the current implementation works.

## Admission test

Add a goal invariant only when all are true:

1. Concrete evidence supports it.
2. Losing the property would materially degrade the requested outcome.
3. The statement can guide more than one immediate implementation decision.
4. It describes an observable or experiential property, not a chosen mechanism.

Good: `Perceived immediacy is part of the desired editing experience.`

Bad: `Use incremental rendering.`

The bad example is an architectural hypothesis that may satisfy the invariant, not the invariant itself.

Ongoing pliability can be a goal invariant when evidence shows that safe continued evolution is constitutive of the requested outcome. State the observable project property, not a generic virtue or preferred architecture.

Good: `Routine policy changes must remain safely implementable without reconstructing behavior across multiple systems.`

Bad: `Keep the code maintainable.`

The good example constrains the future experience of changing the project. The bad example is too vague to test or guide a tradeoff.

## Record the learning

Write the smallest durable statement that future work should optimize or preserve. Add a brief evidence pointer only when the claim would otherwise become unsupported or ambiguous.

Avoid vague virtues such as “good UX” or “maintainability,” requirements already explicit in the outcome, implementation constraints already captured by code or tests, and values inferred without contact with reality. Do not promote every instance of development friction into an invariant; first determine whether it reveals a recurring property required for success.

Treat goal invariants as learned beliefs, not immutable laws. Merge duplicates, narrow overbroad claims, and revise or remove an invariant when later evidence contradicts it.
