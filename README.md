# Firewing

Firewing is a consumer-hardware runtime research project for the full-capability
open-weight Qwen3.8-Flash-Next model. The starting machine is a 16 GB Apple M1 Mac
mini.

The project is not finished. Its final gate remains a near-equivalent native
multimodal runtime sustaining at least **4 accepted tokens/s** for one
interactive request. The current complete text path is still orders of
magnitude slower, hosted accumulated parity is incomplete, and native image,
video, and mixed-modality delivery are not yet proven.

## Mission and definition of done

Firewing is complete only when every gate in [TARGET.md](TARGET.md) passes from
a clean checkout. In condensed form:

- exact, auditable model/tokenizer/processor and hosted-reference locks;
- native local text, image, multi-image, video, mixed-modality, tool,
  multi-turn, and long-context execution;
- near-equivalent distributions over at least 100,000 scored tokens, plus
  capability non-inferiority;
- median batch-one decode of at least 4 accepted TPS after an 8K prefill,
  with the required tail, latency, power, and sustained-run gates;
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
- [docs/VALIDATION_PROTOCOL.md](docs/VALIDATION_PROTOCOL.md) — fidelity and
  performance methodology.
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) — active staged research plan.

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
