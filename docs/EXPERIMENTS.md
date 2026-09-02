# Experiment program

The program is ordered by information gained per dollar and engineering week.
Later stages are conditional on earlier kill gates.

This file is a prospective queue, not a results log. Executed work receives a
stable `FW-NNNN` record under `experiments/` and follows the promotion and
reversal rules in [WORKFLOW.md](WORKFLOW.md).

## E0 — Checkpoint census and lock

**Question:** What exactly must reside, stream, execute, and remain available for
each modality?

Produce a tensor census grouped into:

- Routed experts by layer and projection.
- Attention, routers, norms, embeddings, LM head, MTP, and dense layer zero.
- Vision encoders and all projectors.
- Tokenizer/processor/custom code.
- KV bytes by context length and attention type.