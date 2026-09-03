# Source ledger

This ledger records external authorities before they inform implementation or
experimental decisions. Pin exact revisions and content hashes in FW-0001;
branch names and moving URLs below are discovery pointers only.

| Source | Current role | Authority limitation |
| --- | --- | --- |
| [Qwen3.8-Flash-Next checkpoint](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | Model, tokenizer, processor, configuration, template, and expected weight authority at revision `de4b8e4d43b917e7706784d8bb445c9af86a3540` | FW-0001 has pinned the tree manifest; local payload hashes remain unverified until acquisition completes |
| [Qwen3.8-Flash-Next technical report](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf) | Architecture and published MTP/QSA/n-gram rationale | Paper results are not M1 measurements or endpoint TPS |
| [Qwen3.8-Flash on OpenRouter](https://openrouter.ai/qwen/qwen3.8-flash) | Candidate hosted behavioral reference | Different published model name; requires FW-0002 qualification |
| [Transformers v5.16.1](https://github.com/huggingface/transformers/tree/v5.16.1/src/transformers/models/qwen4_exp) | Initial executable configuration, tokenizer/processor, and tiny-fixture semantic reference | A framework implementation is not an independent oracle or a qualifying 16 GiB runtime; pin exact source files used by each fixture |

Add every codebase, paper, API document, fixture source, and benchmark used by
an experiment. Record the exact commit or immutable content hash and the
decision it informed in that experiment's `FW-NNNN` record.
