# FW-0004 - Tokenizer and chat-template reference slice

- Status: complete
- Disposition: production — fixture generator and native verifier retained
- Date: 2026-09-03
- Parent experiments: FW-0001, FW-0003
- Exactness: L0 token IDs and serialized UTF-8 fixture
- Reference: Transformers 5.16.1

## Question and hypothesis

Can a small native Rust path reproduce pinned Qwen raw tokenization and the
tokens of reference-rendered chat prompts exactly? The hypothesis is that the
Hugging Face tokenizer JSON is sufficient once chat rendering is frozen by the
official executable template reference.

## Method

Transformers 5.16.1 loaded the local checkpoint with network access disabled.
`tools/generate_tokenizer_fixtures.py` froze four raw strings and three chat
cases, including thinking enabled/disabled and a completed assistant turn. The
fixture binds the tokenizer, tokenizer configuration, and chat-template hashes.

The release-mode Rust executable loaded the same `tokenizer.json`, rechecked
all three content identities, and tokenized every raw and rendered-chat case:

```shell
cargo run --release -- verify-tokenizer \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  fixtures/tokenizer/qwen3_8_flash_next.json \
  /Users/chad/Models/firewing/evidence/FW-0004/tokenizer-verification-de4b8e4d.json
```

## Result and decision

All four raw and three chat cases matched exact token IDs. Fixture SHA-256:
`681859c481dc229add5173e6e210f43914b18f09a4e4639f7a20cc33c206e6b9`.
Verification receipt SHA-256:
`bf21bf8142e6bf4f0ea8ff35b09c6fe649e2c68aff31bfef8b48dce4ea9134a8`.

Retain Transformers 5.16.1 as the initial fixture generator and the Rust
tokenizer verifier as the first native vertical slice. Chat rendering itself
is still reference-produced; native template rendering, tools, multimodal
boundaries, and hosted tokenizer identity remain open. This result accepts zero
model output tokens and makes no inference or TPS claim.
