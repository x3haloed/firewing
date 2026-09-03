# FW-0003 - Internal checkpoint acquisition

- Status: in progress
- Disposition: conditional — exact acquisition passed; storage performance pending
- Date: 2026-09-03
- Parent experiment: FW-0001
- Exactness: L0 artifact identity
- Hardware: Apple M1 Mac mini, 16 GiB, internal APFS SSD

## Question and hypothesis

Can the pinned source checkpoint reside on the qualifying internal SSD with
every byte intact? The acquisition hypothesis is that an ordinary local copy
can preserve all pinned identities. Storage-throughput hypotheses remain
unexecuted until production-shaped access traces exist.

## Method and result

The owner copied the completed Hugging Face directory to
`/Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d`. Firewing
generated a final lock and sequentially hashed every expected file:

```shell
python3 tools/checkpoint_verify.py \
  /Users/chad/Models/firewing/checkpoints/Qwen3.8-Flash-Next-de4b8e4d \
  --model-lock /Users/chad/Models/firewing/evidence/FW-0001/model-lock-de4b8e4d.json \
  --output /Users/chad/Models/firewing/evidence/FW-0001/checkpoint-verification-de4b8e4d.json
```

Exit code 0. All 144 files and 360,023,351,514 bytes matched. Receipt SHA-256:
`b6a0a6f5590ec4a4455f3f19aeb59edd722f7e854a170feaa0f10e35354ac45d`.

## Decision

Promote the internal copy as the only qualifying source installation. Keep
FW-0003 open for cold/warm sequential and production-shaped sparse-read
measurements. This integrity result accepts zero tokens and is not storage or
endpoint TPS.
