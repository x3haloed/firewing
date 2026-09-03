# Source ledger

This ledger records external authorities before they inform implementation or
experimental decisions. Pin exact revisions and content hashes in FW-0001;
branch names and moving URLs below are discovery pointers only.

| Source | Current role | Authority limitation |
| --- | --- | --- |
| [Qwen3.8-Flash-Next checkpoint](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | Model, tokenizer, processor, configuration, template, and expected weight authority at revision `de4b8e4d43b917e7706784d8bb445c9af86a3540` | FW-0001 pinned the tree manifest and verified every local payload; upstream identity does not prove runtime correctness |
| [Qwen3.8-Flash-Next technical report](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf) | Architecture and published MTP/QSA/n-gram rationale | Paper results are not M1 measurements or endpoint TPS |
| [Qwen3.8-Flash on OpenRouter](https://openrouter.ai/qwen/qwen3.8-flash) | Candidate hosted behavioral reference | Different published model name; requires FW-0002 qualification |
| [Transformers v5.16.1](https://github.com/huggingface/transformers/tree/v5.16.1/src/transformers/models/qwen4_exp) | Initial executable configuration, tokenizer/processor, and tiny-fixture semantic reference | A framework implementation is not an independent oracle or a qualifying 16 GiB runtime; pin exact source files used by each fixture |
| Prismwing PW-0003 at commit `c87d0c1aa2c118f71ca5348434be35d02f62f031` | Fixture methodology reused by FW-0005: readable oracle, deterministic cases, independent scalar path, and fail-closed identity. Record SHA-256 `8baa9ef6b4641e0b7f39910bbc87b802429d6c8de924ee2e8173f291862fccca` | Process evidence only; MiMo equations, tensor layouts, fixture values, and performance results are not Qwen authorities |
| Prismwing uncached transport and Darwin counters at commit `c87d0c1aa2c118f71ca5348434be35d02f62f031` | FW-0007 transport mechanics: 16 KiB widening, aligned `F_NOCACHE`/`F_RDAHEAD=0` positional reads, and `proc_pid_rusage` disk-byte accounting. Source SHA-256: `b02fd8a6bcfd790e84139ff56c78e7b2d3966dc9d21fb3764576592634ec3280` and `3ea037d2e588b7af7e840cf0b3acdd09cb446b57aa9ea82a7ee0d772d3b49044` | Generic Darwin I/O evidence only; Firewing uses independently verified Qwen row schedules and must establish its own measurements |
| Prismwing PW-0073 and `text_endpoint.rs` at commit `c87d0c1aa2c118f71ca5348434be35d02f62f031` | FW-0010 source-derived PyTorch aarch64 BF16 GEMV reduction topology. Record/source SHA-256: `b64e4c4e97b12a7e76915546054477db34c7838592bbde0b8c2909166f681448` and `3ea037d2e588b7af7e840cf0b3acdd09cb446b57aa9ea82a7ee0d772d3b49044` | Numerical implementation insight only; Qwen tensors, expert equation, capture hashes, and verification are independently bound to the Firewing checkpoint |
| Prismwing PyTorch aarch64 F32 RMS cascade at commit `c87d0c1aa2c118f71ca5348434be35d02f62f031` | FW-0014 independent grouped-RMS reduction topology. `src/lib.rs` SHA-256: `2f2de84115cc99bcf2bca8714682fd374582a671df3254841f6adaa64b3d6717` | Reduction-order insight and test provenance only; Firewing binds the equation, group width, weights, inputs, and captures independently to Qwen |

Add every codebase, paper, API document, fixture source, and benchmark used by
an experiment. Record the exact commit or immutable content hash and the
decision it informed in that experiment's `FW-NNNN` record.
