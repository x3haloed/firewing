# FW-0001 - Remote checkpoint census and lock

- Status: planned; frozen before execution
- Disposition: unexecuted
- Date: 2026-09-02
- Owner: project owner with Codex implementation support
- Parent experiments: none
- Exactness: L0 artifact census
- Hardware/runtime: Apple M1 Mac mini, 16 GiB, internal SSD; remote metadata
  and bounded range reads only

## Question and hypothesis

Can every required Qwen3.8-Flash-Next artifact and tensor be identified,
revision-pinned, byte-accounted, and licensed before downloading the roughly
checkpoint-sized payload? The hypothesis is that repository metadata,
safetensors indices, and bounded header reads provide enough evidence for a
complete fail-closed census and storage decision.

## Frozen authority and baseline

The target repository is `Qwen/Qwen3.8-Flash-Next`. The run must resolve and
record the revision rather than inheriting a moving branch name. There is no
prior model lock, tensor census, runtime, or performance baseline.

## Method and commands

The implementation and exact commands remain to be added in the execution
commit. They must fetch metadata and bounded headers without tensor payloads,
reconcile all tensor extents, produce a source ledger and model lock, and record
network bytes, commands, exit codes, environment, and content hashes.

## Gates

All repository files and tensor payload bytes must reconcile with the pinned
index, configuration, and published architecture. Required tokenizer,
processor, template, vision, n-gram, QSA, Gated DeltaNet, gated-residual, routed
expert, shared-expert, and MTP assets must be present and understood. Unknown
revision, layout, dtype, shape, offset, shard extent, license, or internal-SSD
capacity fails closed before full acquisition.

This experiment accepts zero tokens and makes no runtime, fidelity, executable
memory, latency, or TPS claim.

## Result

Not executed.

## Decision

Unexecuted. Passing authorizes FW-0002 and bounded source acquisition; failure
redirects the project before checkpoint download or runtime construction.
