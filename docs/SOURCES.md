# Source ledger

This ledger records external authorities before they inform implementation or
experimental decisions. Pin exact revisions and content hashes in FW-0001;
branch names and moving URLs below are discovery pointers only.

| Source | Current role | Authority limitation |
| --- | --- | --- |
| [Qwen3.8-Flash-Next checkpoint](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | Candidate model, tokenizer, processor, configuration, template, and weight authority | Must be revision-pinned and content-hashed by FW-0001 |
| [Qwen3.8-Flash-Next technical report](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf) | Architecture and published MTP/QSA/n-gram rationale | Paper results are not M1 measurements or endpoint TPS |
| [Qwen3.8-Flash on OpenRouter](https://openrouter.ai/qwen/qwen3.8-flash) | Candidate hosted behavioral reference | Different published model name; requires FW-0002 qualification |
| [Transformers](https://github.com/huggingface/transformers) | Candidate official-format executable reference | Exact supported revision and Qwen semantics must be pinned |

Add every codebase, paper, API document, fixture source, and benchmark used by
an experiment. Record the exact commit or immutable content hash and the
decision it informed in that experiment's `FW-NNNN` record.
