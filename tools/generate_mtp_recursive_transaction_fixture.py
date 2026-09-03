#!/usr/bin/env python3
"""Bind the recursive MTP and exact target authorities into a width-four transaction."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from generate_mtp_transaction_fixture import EXPERT_BYTES, MODEL, REVISION, SGLANG_COMMIT, load, sha256_file, top1s


TARGET_LAYERS = 48
Q = 4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--mtp-seed", required=True, type=Path)
    parser.add_argument("--mtp-decoder", required=True, type=Path)
    parser.add_argument("--mtp-output", required=True, type=Path)
    parser.add_argument("--acceptance-lock", required=True, type=Path)
    parser.add_argument("--recursive-lock", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    target = load(args.target)
    seed = load(args.mtp_seed)
    draft_decoder = load(args.mtp_decoder)
    draft_output = load(args.mtp_output)
    if (
        target.get("model") != MODEL
        or target.get("revision") != REVISION
        or target.get("semantic") != "qwen3_8_flash_next_firewing_six_token_cached_text_logits"
        or target["configuration"]["token_ids"][:4] != [16_207, 22_856, 369, 264]
        or seed.get("semantic") != "qwen3_8_flash_next_recursive_mtp_fusion"
        or draft_decoder.get("semantic") != "qwen3_8_flash_next_recursive_mtp_decoder"
        or draft_output.get("semantic") != "qwen3_8_flash_next_recursive_mtp_logits"
    ):
        raise ValueError("recursive transaction input authority mismatch")

    draft_top1s = [step["top20_token_ids"][0] for step in draft_output["steps"]]
    proposal = [seed["configuration"]["target_next_token_id"]] + draft_top1s[1:]
    if target["configuration"]["token_ids"] != [16_207, 22_856] + proposal:
        raise ValueError("target branch does not follow the recursive proposal vector")
    posterior = top1s(target)[-Q:]
    mismatch = next(
        (index for index in range(Q - 1) if posterior[index] != proposal[index + 1]),
        None,
    )
    if mismatch is None:
        correct_drafts = Q - 1
        accepted = Q
        retained = Q
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
        routes = layer["decoder"]["steps"][-Q:]
        target_unions.append(len(set().union(*(set(step["selected_experts"]) for step in routes))))
    draft_routes = draft_decoder["steps"][1:]
    draft_unique = len(set().union(*(set(step["selected_experts"]) for step in draft_routes)))
    target_union_rows = sum(target_unions)
    combined_union_rows = target_union_rows + draft_unique
    one_token_expert_rows = TARGET_LAYERS * 10
    union_u = combined_union_rows / one_token_expert_rows

    fixture = {
        "schema_version": 1,
        "semantic": "qwen3_8_flash_next_first_recursive_greedy_mtp_transaction",
        "model": MODEL,
        "revision": REVISION,
        "reference": {
            "implementation": "source_derived_sglang_recursive_greedy_eagle_and_firewing_exact_native_authorities",
            "source_commit": SGLANG_COMMIT,
            "acceptance_source_lock_sha256": sha256_file(args.acceptance_lock),
            "recursive_source_lock_sha256": sha256_file(args.recursive_lock),
            "target_fixture_sha256": sha256_file(args.target),
            "mtp_seed_fixture_sha256": sha256_file(args.mtp_seed),
            "mtp_decoder_fixture_sha256": sha256_file(args.mtp_decoder),
            "mtp_output_fixture_sha256": sha256_file(args.mtp_output),
        },
        "configuration": {
            "sampling": "greedy",
            "batch_size": 1,
            "concurrency": 1,
            "q": Q,
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
            "rolled_back_proposal_rows": Q - retained,
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
            "scope": "one exact recursive greedy width-four transaction; no timing, sustained TPS, or endpoint promotion claim",
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
