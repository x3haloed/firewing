#!/usr/bin/env python3
"""Bind exact target and MTP authorities into one greedy width-two transaction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


MODEL = "Qwen/Qwen3.8-Flash-Next"
REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
SGLANG_COMMIT = "78c5024e9d9f589dcb4deb7f4ba4fb23f7e85385"
EXPERT_BYTES = 9_830_400
TARGET_LAYERS = 48


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def top1s(value: dict[str, Any]) -> list[int]:
    return [step["top20_token_ids"][0] for step in value["output"]["steps"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--mtp-seed", required=True, type=Path)
    parser.add_argument("--mtp-decoder", required=True, type=Path)
    parser.add_argument("--mtp-output", required=True, type=Path)
    parser.add_argument("--acceptance-lock", required=True, type=Path)
    parser.add_argument("--transaction-index", type=int, choices=(1, 2), default=1)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    target = load(args.target)
    seed = load(args.mtp_seed)
    draft_decoder = load(args.mtp_decoder)
    draft_output = load(args.mtp_output)
    if (
        target.get("model") != MODEL
        or target.get("revision") != REVISION
        or target.get("semantic")
        not in {
            "qwen3_8_flash_next_firewing_four_token_cached_text_logits",
            "qwen3_8_flash_next_firewing_six_token_cached_text_logits",
        }
        or seed.get("semantic") != "qwen3_8_flash_next_target_derived_mtp_prefill_fusion"
        or draft_decoder.get("semantic") != "qwen3_8_flash_next_target_derived_mtp_prefill_decoder"
        or draft_output.get("semantic") != "qwen3_8_flash_next_target_derived_mtp_prefill_logits"
    ):
        raise ValueError("transaction input authority mismatch")

    target_top1s = top1s(target)
    draft_top1s = [step["top20_token_ids"][0] for step in draft_output["steps"]]
    proposal = [seed["configuration"]["target_next_token_id"], draft_top1s[-1]]
    target_history = seed["configuration"]["target_input_token_ids"]
    if (
        len(target_history) not in (2, 4)
        or target["configuration"]["token_ids"] != target_history + proposal
        or args.transaction_index != len(target_history) // 2
    ):
        raise ValueError("transaction target history or index mismatch")
    posterior = target_top1s[-2:]
    mismatch = next(
        (index for index in range(len(proposal) - 1) if posterior[index] != proposal[index + 1]),
        None,
    )
    if mismatch is None:
        correct_drafts = len(proposal) - 1
        accepted = len(proposal)
        retained = len(proposal)
        emitted = proposal[1:] + [posterior[-1]]
        next_anchor = posterior[-1]
        converged = True
    else:
        correct_drafts = mismatch
        accepted = mismatch + 1
        retained = mismatch + 1
        emitted = proposal[1 : mismatch + 1] + [posterior[mismatch]]
        next_anchor = posterior[mismatch]
        converged = False

    target_unions = []
    for layer in target["layers"]:
        routes = layer["decoder"]["steps"][-2:]
        target_unions.append(len(set(routes[0]["selected_experts"]) | set(routes[1]["selected_experts"])))
    draft_route = draft_decoder["steps"][-1]["selected_experts"]
    draft_unique = len(set(draft_route))
    target_union_rows = sum(target_unions)
    combined_union_rows = target_union_rows + draft_unique
    one_token_expert_rows = TARGET_LAYERS * 10
    union_u = combined_union_rows / one_token_expert_rows

    fixture = {
        "schema_version": 1,
        "semantic": f"qwen3_8_flash_next_{'first' if args.transaction_index == 1 else 'second'}_greedy_mtp_transaction",
        "model": MODEL,
        "revision": REVISION,
        "reference": {
            "implementation": "source_derived_sglang_greedy_eagle_and_firewing_exact_native_authorities",
            "source_commit": SGLANG_COMMIT,
            "acceptance_source_lock_sha256": sha256_file(args.acceptance_lock),
            "target_fixture_sha256": sha256_file(args.target),
            "mtp_seed_fixture_sha256": sha256_file(args.mtp_seed),
            "mtp_decoder_fixture_sha256": sha256_file(args.mtp_decoder),
            "mtp_output_fixture_sha256": sha256_file(args.mtp_output),
        },
        "configuration": {
            "sampling": "greedy",
            "batch_size": 1,
            "concurrency": 1,
            "q": len(posterior),
            "target_layers": TARGET_LAYERS,
            "top_k_experts": 10,
            "expert_payload_bytes": EXPERT_BYTES,
        },
        "decision": {
            "proposal_token_ids": proposal,
            "target_posterior_token_ids": posterior,
            "correct_draft_tokens": correct_drafts,
            "accepted_tokens": accepted,
            "retained_proposal_rows": retained,
            "rolled_back_proposal_rows": len(proposal) - retained,
            "emitted_token_ids": emitted,
            "next_anchor_token_id": next_anchor,
            "proposal_converged": converged,
        },
        "expert_union": {
            "target_unique_experts_by_layer": target_unions,
            "target_union_expert_rows": target_union_rows,
            "draft_unique_expert_rows": draft_unique,
            "combined_union_expert_rows": combined_union_rows,
            "one_token_expert_rows": one_token_expert_rows,
            "U": union_u,
            "A_over_U": accepted / union_u,
            "logical_expert_payload_bytes": combined_union_rows * EXPERT_BYTES,
        },
        "claims": {
            "accepted_tokens": accepted,
            "performance_claim": None,
            "scope": "one exact greedy width-two transaction; no timing, sustained TPS, or endpoint promotion claim",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(fixture, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"output": os.fspath(args.output), "A": accepted, "U": union_u}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
