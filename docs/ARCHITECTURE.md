# Architecture and byte ledger

This document records checkpoint-derived architecture facts and separates them
from runtime hypotheses. The authoritative raw census is FW-0001; values below
do not constitute endpoint throughput.

## Model identity

The pinned checkpoint is `Qwen/Qwen3.8-Flash-Next` revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540`. “3.8” is the Qwen release
number, not the number of active parameters. Qwen describes the model as a
125B-parameter main model plus 51B n-gram embeddings, with 6B parameters active
per token. The complete local safetensors census contains 179,999,981,459
stored scalar values, including vision and MTP tensors.

The text stack has 48 layers: 36 Gated DeltaNet layers and 12 Qwen Sparse
Attention layers at every fourth position. Every layer has 512 routed experts,
selects 10, and also evaluates one shared expert. The hidden size is 2,560 and
each routed or shared expert has intermediate width 640. Gated Residual carries
four 2,560-wide streams. The checkpoint includes one MTP layer and a 27-layer
vision encoder.

## Source checkpoint bytes

| Component | Stored parameters | Tensor bytes |
| --- | ---: | ---: |
| Routed experts | 120,795,955,200 | 241,591,910,400 |
| N-gram embeddings | 51,200,245,760 | 102,400,491,520 |
| MTP | 2,607,150,848 | 5,214,301,696 |
| Gated DeltaNet | 2,086,510,464 | 4,173,020,928 |
| Gated Residual | 640,624,640 | 1,281,249,280 |
| LM head | 635,699,200 | 1,271,398,400 |
| Token embeddings | 635,699,200 | 1,271,398,400 |
| Qwen Sparse Attention | 617,358,336 | 1,234,716,672 |
| Vision | 448,931,056 | 897,862,112 |
| Shared experts | 235,929,600 | 471,859,200 |
| Routers and expert gates | 63,037,440 | 126,074,880 |
| N-gram projection and state | 32,839,715 | 65,679,640 |
| **Total tensors** | **179,999,981,459** | **359,999,963,128** |

The complete repository occupies 360,023,351,514 bytes across 144 files. All
1,658 tensors are BF16 except three I64 n-gram metadata tensors.

## Decode traffic hypotheses

One routed expert contains 4,915,200 BF16 values, or 9,830,400 bytes. Reading
exactly ten selected experts across all 48 layers therefore moves
4,718,592,000 source bytes per ordinary token before cache hits, expert-set
union across speculative positions, filesystem amplification, or repacking.
This is approximately half MiMo-V2.5's measured 9,464,659,968 routed bytes per
token, but it is not the whole Qwen path.

Reading every ordinary fixed-weight matrix once adds approximately
8,623,999,000 bytes, including the LM head. The naïve combined source-weight
ledger is therefore about 13.34 GB per token. A viable runtime must keep a
large fraction of common weights resident or otherwise avoid rereading them;
the routed-byte comparison alone cannot predict endpoint TPS.

The n-gram table contains 16 independently hashed heads—eight bigram and eight
trigram heads—with 160 BF16 values selected from each head per position. The
useful tensor payload is only 5,120 bytes per token, but the 16 sparse lookups
can amplify to at least 16 storage pages when cold. Page locality, prefetch
timing, and cache behavior must be measured before treating the table as free.

FW-0005 verifies the addressing bridge rather than estimating it. The logical
table has 320,001,536 padded rows of width 160 and is concatenated in numeric
order from 128 checkpoint tensors, each containing 2,500,012 rows. A global row
`r` therefore maps to table part `r / 2,500,012` and local row
`r % 2,500,012`. Five reference cases cover initial EOS padding, ordinary
sequences, an EOS boundary, incremental cached context, and vocabulary-edge
IDs. An independent scalar Rust path reproduced all 224 head addresses while
validating every physical table descriptor and the three actual int64 metadata
payloads. This establishes what to read, not how many filesystem bytes the SSD
will actually move.

FW-0006 adds the corresponding bounded reader. It seeks directly to
`8 + safetensors_header_bytes + tensor_data_offset + local_row * 320`, reads
exactly 320 bytes, and rejects invalid rows or arithmetic overflow. It matched
SHA-256 identities independently captured for all 224 FW-0005 addresses while
requesting 71,680 payload bytes. Only hashes and invented synthetic bytes are
committed; sampled Qwen weight bytes remain outside Git. Physical SSD traffic,
page amplification, and useful throughput remain unmeasured.

FW-0008 supplies the first scoped physical measurement. For FW-0005's fixed
14-position trace, explicit range invalidation followed by page-aligned
`F_NOCACHE` reads moved exactly 3,719,168 bytes and took 22.079 ms median. That
is 265,654.9 bytes and 1.577 ms per token, a 51.886x amplification over 5,120
useful row bytes. The serialized reader missed its frozen 1 ms/token
continuation threshold, so it is evidence rather than a promoted transport.
Real decode traces, coalescing, parallelism, BF16 conversion, and PLE compute
remain outside this result.

MTP can reduce routed SSD demand only when committed-token acceptance grows
faster than the byte-weighted union of experts required for proposal and
verification. Firewing records those quantities as `A`, `U`, and `A/U` rather
than counting proposed tokens as throughput.

## Current implementation boundary

Transformers 5.16.1 recognizes the pinned `qwen4_exp` configuration, tokenizer,
and Qwen3-VL processor. It is the initial executable semantic reference for
tiny fixtures, not a qualifying runtime and not evidence that the 180B
checkpoint can execute within 16 GiB. The native tokenizer, n-gram address
verifier, and bounded sparse row reader are the first target-specific Rust
slices. The runtime reuses
Prismwing's reference/oracle/independent-fixture discipline, while Qwen4-Exp
semantics and checkpoint layouts remain independently derived and fail closed.
