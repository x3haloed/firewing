---
name: resolve-prediction-errors
description: Use when a test, measurement, runtime observation, integration response, implementation result, attempted change, or consequential development friction contradicts Codex's working model and the discrepancy could affect future decisions. This includes prior design choices that unexpectedly make the system harder to understand, change, test, or verify. Do not trigger for ordinary command mistakes or already-understood failures with no decision value.
---

# Resolve Prediction Errors

Treat disagreement with reality as evidence.

Difficulty making progress is a valid observation when it exposes a falsified expectation about the system. Ground it in a concrete discrepancy: repeated rediscovery, hidden coordination, duplicated authorities, unexpectedly broad change impact, or disproportionate verification cost. `This code is difficult` alone is not a prediction error.

1. State the expected result precisely enough to be falsifiable.
2. State the observed result and its evidence.
3. Identify the smallest assumption or model now in doubt. Do not force a cause before evidence supports one.
4. Decide whether the discrepancy can change future work. If not, correct the ordinary error and continue without frontier residue.
5. When competing explanations matter, run the cheapest safe experiment that distinguishes them.
6. Keep the prediction error open while the discrepancy remains materially unexplained.

Record an open item compactly:

```text
Expected: ...
Observed: ...
Uncertain: the assumption or relationship now in doubt
Evidence: the reproducible test, measurement, or observation
```

Do not record attempt history, blame, speculative conclusions, or implementation facts already evident in the repository.

Example:

```text
Expected: The shared abstraction would let one behavior change in one place.
Observed: The change required coordinating three representations and two verification paths.
Uncertain: Whether the abstraction boundary or duplicated authorities are responsible.
Evidence: The affected change and the locations that had to move together.
```

When understood:

- Remove the prediction error.
- Put implementation knowledge into code, tests, tooling, or documentation where it belongs. When authorized and useful, remove the evidenced source of recurring friction rather than preserving a workaround.
- Invoke `$discover-goal-invariants` only if the evidence reveals a property constitutive of the desired outcome.
- Otherwise retain no frontier residue unless the learning can still change a future decision.
