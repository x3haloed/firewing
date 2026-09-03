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

Add every codebase, paper, API document, fixture source, and benchmark used by
an experiment. Record the exact commit or immutable content hash and the
decision it informed in that experiment's `FW-NNNN` record.
