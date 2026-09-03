# Correctness fixtures

Small deterministic fixtures are committed here. Large or licensed artifacts
remain outside Git and are referenced by hashes from experiment records.

`tokenizer/qwen3_8_flash_next.json` is generated from the pinned checkpoint by
Transformers 5.16.1 using `tools/generate_tokenizer_fixtures.py`. It freezes raw
tokenization and representative chat-template serialization before a native
implementation is introduced.
