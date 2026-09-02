---
name: iterate-toward-outcomes
description: Advance a large, uncertain, or long-running software goal through repeated useful work without requiring a complete up-front architecture or project-sized plan. Use when Codex must build or substantially change a system over many iterations, when the requested end shape is underdetermined, or when plans should adapt to evidence while remaining anchored to user outcomes. Do not use for small, fully specified one-step edits.
---

# Iterate Toward Outcomes

Work persistently toward the stated outcome. Treat architecture and plans as revisable hypotheses, not as the target.

## Maintain the frontier

Use the repository as the authoritative implementation record. Maintain only this additional decision state when it is useful:

- **Outcome:** What must eventually be true from the user's or ecosystem's perspective.
- **Goal invariants:** Evidence-backed properties discovered to be constitutive of success.
- **Prediction errors:** Unresolved cases where an expected result materially differed from observation.

For a fresh project, read [references/work-frontier-template.md](references/work-frontier-template.md) and copy it to a durable project-local location before adding frontier state. Prefer `<repo>/.codex/work-frontier.md` when project-local agent state belongs in the repository. Otherwise keep `work-frontier.md` immediately beside the repository or in the nearest established project-notes directory. Do not edit the bundled template, create competing frontier files, or hide the file in a distant global notes location.

Keep transient state in the task context when persistence across sessions is unnecessary. When a persistent frontier already exists, update that file in place and preserve the template's section order. Leave a section empty rather than inventing entries. Do not copy implementation status, plans, task lists, attempt history, or facts already authoritative in code, tests, issues, or documentation into the frontier.

## Run one work loop

1. Reorient from the outcome, current goal invariants, unresolved prediction errors, and repository reality.
2. Choose one useful work unit that advances the artifact, tests an important assumption, or reduces uncertainty blocking useful progress. Price uncertainty-reducing work by the total cost of obtaining decisive evidence, not merely the cost of changing code.
3. Prefer work likely to remain valuable across multiple plausible architectures, including work that removes evidenced friction from future understanding, change, or verification.
4. Act. Make the change or run the probe instead of extending the plan unnecessarily.
5. Observe reality through the strongest practical evidence: live behavior, tests, measurements, compiler output, logs, or direct inspection.
6. Incorporate only decision-relevant learning into the frontier.
7. Choose again. Continue until the outcome is satisfied or a concrete external blocker is reached.

Use plans to coordinate immediate work, but replace them freely when evidence changes the best direction. Do not substitute completion of a task list for satisfaction of the outcome.

## Buy useful evidence cheaply

Include all activation and observation costs when comparing experiments: implementation effort, setup demanded from the observer, time to first observation, cognitive load, fidelity on decision-relevant dimensions, confounds, reversibility, and the likelihood that the result will arrive and distinguish the alternatives. Do not call an experiment cheap merely because its code change is small while a human must configure, deploy, populate, navigate, or infer what to assess.

When a consequential ambiguity requires human experience or judgment, invoke `$resolve-ambiguity-with-human-evidence`. Use experimental artifacts only to answer the ambiguity. Their validation does not make them finished-grade work: retain the lesson, remove disposable scaffolding, implement the chosen direction to project standards, and verify it in the real system before treating that part of the outcome as satisfied.

## Protect the ability to progress

Treat the agent's ability to make safe, informed progress as part of project reality. Unexpectedly expensive understanding, changes that require coordinating hidden or duplicated authorities, repeated rediscovery, and verification that cannot isolate affected behavior are evidence about the system—not merely agent inconvenience.

When concrete friction materially impedes the outcome or is likely to recur, choose work that improves the project's pliability: make relevant behavior easier to locate, understand, change, test, and verify. Prefer removing the demonstrated source of friction over documenting a workaround. Do not use pliability as blanket permission for speculative cleanup, unrelated refactoring, or replacing a working design with a preferred style.

Treat each design choice as a prediction about future work. If a prior choice expected to simplify progress instead makes consequential work harder, route the discrepancy through `$resolve-prediction-errors`. If sustained safe changeability proves constitutive of the outcome, route that learning through `$discover-goal-invariants` rather than assuming maintainability is automatically a goal invariant.

## Route exceptional conditions

- Invoke `$resolve-prediction-errors` when observation materially contradicts expectation.
- Invoke `$resolve-ambiguity-with-human-evidence` when a consequential choice requires bounded human judgment and ordinary review would impose avoidable friction.
- Invoke `$discover-goal-invariants` when evidence may reveal a previously implicit dimension of success.
- Invoke `$reorient-from-outcomes` when momentum, a milestone, growing uncertainty, or a substantial next chunk makes drift plausible.
- Invoke `$compact-work-frontier` when frontier state becomes repetitive, stale, large, or milestone-bound.

These are conditional transition handlers, not mandatory phases. Continue ordinary work when no condition is present.

## Preserve epistemic discipline

- Record disagreement with reality, not a diary of failed attempts.
- Prefer a discriminating experiment over speculation when it is cheap and safe.
- Preserve uncertainty when evidence does not resolve it.
- Distinguish outcome properties from implementation choices.
- Periodically derive direction anew from the original outcome instead of recent momentum.
- Stop only for genuine completion, a concrete external blocker, exhausted authorized scope, or a required user decision that would materially change the result.
