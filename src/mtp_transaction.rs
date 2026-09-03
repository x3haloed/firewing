use crate::token_text_endpoint::verify_token_text_endpoint_fixture_with_expected_outputs;
use crate::{
    verify_mtp_causal_prefill_fixture, verify_mtp_recursive_fixture,
    verify_token_text_transaction_fixture,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
use std::time::Instant;

const MODEL: &str = "Qwen/Qwen3.8-Flash-Next";
const REVISION: &str = "de4b8e4d43b917e7706784d8bb445c9af86a3540";
const SGLANG_COMMIT: &str = "78c5024e9d9f589dcb4deb7f4ba4fb23f7e85385";
const EAGLE_UTILS_SHA256: &str = "87e9dc749e94f5899140457393389397840a2258978c021fd3ac490e9da4c053";
const EAGLE_WORKER_SHA256: &str =
    "9a66d31868385646b9fb9f78053730f55d2e885e72382a8c8dc6db9f07709271";
const TARGET_LAYERS: usize = 48;
const EXPERT_BYTES: usize = 9_830_400;

#[derive(Deserialize)]
struct Fixture {
    schema_version: u32,
    semantic: String,
    model: String,
    revision: String,
    reference: Reference,
    configuration: Configuration,
    decision: Decision,
    expert_union: ExpertUnion,
    claims: Claims,
}

#[derive(Deserialize)]
struct Reference {
    implementation: String,
    source_commit: String,
    acceptance_source_lock_sha256: String,
    #[serde(default)]
    recursive_source_lock_sha256: Option<String>,
    target_fixture_sha256: String,
    mtp_seed_fixture_sha256: String,
    mtp_decoder_fixture_sha256: String,
    mtp_output_fixture_sha256: String,
}

#[derive(Deserialize)]
struct Configuration {
    sampling: String,
    batch_size: usize,
    concurrency: usize,
    q: usize,
    target_layers: usize,
    top_k_experts: usize,
    expert_payload_bytes: usize,
}

#[derive(Deserialize)]
struct Decision {
    proposal_token_ids: Vec<usize>,
    target_posterior_token_ids: Vec<usize>,
    correct_draft_tokens: usize,
    accepted_tokens: usize,
    retained_proposal_rows: usize,
    rolled_back_proposal_rows: usize,
    emitted_token_ids: Vec<usize>,
    next_anchor_token_id: usize,
    proposal_converged: bool,
}

#[derive(Deserialize)]
struct ExpertUnion {
    target_unique_experts_by_layer: Vec<usize>,
    target_union_expert_rows: usize,
    draft_unique_expert_rows: usize,
    combined_union_expert_rows: usize,
    one_token_expert_rows: usize,
    #[serde(rename = "U")]
    u: f64,
    #[serde(rename = "A_over_U")]
    a_over_u: f64,
    logical_expert_payload_bytes: usize,
}

#[derive(Deserialize)]
struct Claims {
    accepted_tokens: usize,
    performance_claim: Option<String>,
    scope: String,
}

#[derive(Deserialize)]
struct SourceLock {
    schema_version: u32,
    repository: String,
    pull_request: String,
    commit: String,
    files: Vec<SourceFile>,
}

#[derive(Deserialize)]
struct SourceFile {
    path: String,
    git_blob: String,
    sha256: String,
}

#[derive(Debug, Serialize)]
pub struct MtpTransactionVerificationReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub model: String,
    pub revision: String,
    pub source_commit: String,
    pub sampling: &'static str,
    pub batch_size: usize,
    pub concurrency: usize,
    pub q: usize,
    pub proposal_token_ids: Vec<usize>,
    pub target_posterior_token_ids: Vec<usize>,
    pub correct_draft_tokens: usize,
    pub emitted_token_ids: Vec<usize>,
    pub next_anchor_token_id: usize,
    pub retained_proposal_rows: usize,
    pub rolled_back_proposal_rows: usize,
    pub proposal_converged: bool,
    pub target_unique_experts_by_layer: Vec<usize>,
    pub target_union_expert_rows: usize,
    pub draft_unique_expert_rows: usize,
    pub combined_union_expert_rows: usize,
    pub one_token_expert_rows: usize,
    #[serde(rename = "A")]
    pub accepted_tokens: usize,
    #[serde(rename = "U")]
    pub expert_union: f64,
    #[serde(rename = "A_over_U")]
    pub accepted_over_union: f64,
    pub logical_expert_payload_bytes: usize,
    pub total_verified_payload_bytes: usize,
    pub complete_wall_time_ns: u128,
    pub performance_claim: Option<String>,
}

fn sha256_file(path: &Path) -> Result<String, String> {
    fs::read(path)
        .map(|bytes| format!("{:x}", Sha256::digest(bytes)))
        .map_err(|error| format!("cannot read {}: {error}", path.display()))
}

fn validate_source_lock(path: &Path) -> Result<(), String> {
    let lock: SourceLock = serde_json::from_slice(
        &fs::read(path).map_err(|error| format!("cannot read acceptance lock: {error}"))?,
    )
    .map_err(|error| format!("malformed acceptance lock: {error}"))?;
    let find = |name: &str| lock.files.iter().find(|file| file.path == name);
    let utils = find("python/sglang/srt/speculative/eagle_utils.py")
        .ok_or("acceptance lock lacks eagle_utils.py")?;
    let worker = find("python/sglang/srt/speculative/eagle_worker_v2.py")
        .ok_or("acceptance lock lacks eagle_worker_v2.py")?;
    if lock.schema_version != 1
        || lock.repository != "https://github.com/sgl-project/sglang"
        || lock.pull_request != "https://github.com/sgl-project/sglang/pull/36497"
        || lock.commit != SGLANG_COMMIT
        || lock.files.len() != 2
        || utils.git_blob != "18c9f0cbdd849667f9b743f704e79ff08d2e5827"
        || utils.sha256 != EAGLE_UTILS_SHA256
        || worker.git_blob != "93fdd61761c4f976305d7af4e4aecd65430e0539"
        || worker.sha256 != EAGLE_WORKER_SHA256
    {
        return Err("unsupported greedy EAGLE acceptance source authority".to_owned());
    }
    Ok(())
}

fn selected_routes(value: &Value, step_start: usize) -> Result<Vec<Vec<Vec<usize>>>, String> {
    value
        .get("layers")
        .and_then(Value::as_array)
        .ok_or("target transaction layers missing")?
        .iter()
        .map(|layer| {
            layer
                .pointer("/decoder/steps")
                .and_then(Value::as_array)
                .ok_or("target transaction decoder steps missing")?
                .iter()
                .skip(step_start)
                .map(|step| {
                    step.get("selected_experts")
                        .and_then(Value::as_array)
                        .ok_or("target transaction route missing")?
                        .iter()
                        .map(|expert| {
                            expert
                                .as_u64()
                                .and_then(|value| usize::try_from(value).ok())
                                .ok_or_else(|| "invalid target transaction expert".to_owned())
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub fn verify_mtp_transaction_fixture(
    checkpoint_dir: &Path,
    model_lock_path: &Path,
    mtp_source_lock_path: &Path,
    scheduler_lock_path: &Path,
    acceptance_lock_path: &Path,
    tokenizer_fixture_path: &Path,
    ngram_fixture_path: &Path,
    ngram_row_fixture_path: &Path,
    ple_fixture_path: &Path,
    endpoint_fixture_path: &Path,
    fusion_fixture_path: &Path,
    seed_fixture_path: &Path,
    mtp_attention_fixture_path: &Path,
    mtp_decoder_fixture_path: &Path,
    mtp_output_fixture_path: &Path,
    target_transaction_fixture_path: &Path,
    transaction_fixture_path: &Path,
) -> Result<MtpTransactionVerificationReport, String> {
    let started = Instant::now();
    validate_source_lock(acceptance_lock_path)?;
    let fixture: Fixture = serde_json::from_slice(
        &fs::read(transaction_fixture_path)
            .map_err(|error| format!("cannot read MTP transaction fixture: {error}"))?,
    )
    .map_err(|error| format!("malformed MTP transaction fixture: {error}"))?;
    let config = &fixture.configuration;
    if fixture.schema_version != 1
        || fixture.semantic != "qwen3_8_flash_next_first_greedy_mtp_transaction"
        || fixture.model != MODEL
        || fixture.revision != REVISION
        || fixture.reference.implementation
            != "source_derived_sglang_greedy_eagle_and_firewing_exact_native_authorities"
        || fixture.reference.source_commit != SGLANG_COMMIT
        || fixture.reference.acceptance_source_lock_sha256 != sha256_file(acceptance_lock_path)?
        || fixture.reference.target_fixture_sha256 != sha256_file(target_transaction_fixture_path)?
        || fixture.reference.mtp_seed_fixture_sha256 != sha256_file(seed_fixture_path)?
        || fixture.reference.mtp_decoder_fixture_sha256 != sha256_file(mtp_decoder_fixture_path)?
        || fixture.reference.mtp_output_fixture_sha256 != sha256_file(mtp_output_fixture_path)?
        || config.sampling != "greedy"
        || config.batch_size != 1
        || config.concurrency != 1
        || config.q != 2
        || config.target_layers != TARGET_LAYERS
        || config.top_k_experts != 10
        || config.expert_payload_bytes != EXPERT_BYTES
        || fixture.claims.performance_claim.is_some()
        || fixture.claims.scope
            != "one exact greedy width-two transaction; no timing, sustained TPS, or endpoint promotion claim"
    {
        return Err("MTP transaction identity or configuration mismatch".to_owned());
    }

    let draft = verify_mtp_causal_prefill_fixture(
        checkpoint_dir,
        model_lock_path,
        mtp_source_lock_path,
        scheduler_lock_path,
        tokenizer_fixture_path,
        ngram_fixture_path,
        ngram_row_fixture_path,
        ple_fixture_path,
        endpoint_fixture_path,
        fusion_fixture_path,
        seed_fixture_path,
        mtp_attention_fixture_path,
        mtp_decoder_fixture_path,
        mtp_output_fixture_path,
    )?;
    let target = verify_token_text_transaction_fixture(
        checkpoint_dir,
        model_lock_path,
        tokenizer_fixture_path,
        ngram_fixture_path,
        ngram_row_fixture_path,
        ple_fixture_path,
        target_transaction_fixture_path,
    )?;
    let proposal = vec![draft.target_next_token_id, draft.proposal_token_id];
    let posterior = target
        .top20_token_ids_by_step
        .iter()
        .skip(2)
        .map(|tokens| tokens.first().copied().ok_or("target posterior is empty"))
        .collect::<Result<Vec<_>, _>>()?;
    if proposal.len() != 2 || posterior.len() != 2 {
        return Err("width-two transaction vectors are incomplete".to_owned());
    }
    let mismatch = (0..proposal.len() - 1).find(|index| posterior[*index] != proposal[*index + 1]);
    let (correct_drafts, accepted, retained, emitted, next_anchor, converged) =
        if let Some(index) = mismatch {
            (
                index,
                index + 1,
                index + 1,
                proposal[1..=index]
                    .iter()
                    .copied()
                    .chain(std::iter::once(posterior[index]))
                    .collect(),
                posterior[index],
                false,
            )
        } else {
            (
                proposal.len() - 1,
                proposal.len(),
                proposal.len(),
                proposal[1..]
                    .iter()
                    .copied()
                    .chain(std::iter::once(posterior[posterior.len() - 1]))
                    .collect(),
                posterior[posterior.len() - 1],
                true,
            )
        };

    let target_value: Value = serde_json::from_slice(
        &fs::read(target_transaction_fixture_path).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let target_routes = selected_routes(&target_value, 2)?;
    if target_routes.len() != TARGET_LAYERS
        || target_routes.iter().any(|steps| {
            steps.len() != config.q
                || steps
                    .iter()
                    .any(|route| route.len() != config.top_k_experts)
        })
    {
        return Err("target transaction route shape mismatch".to_owned());
    }
    let target_unions = target_routes
        .iter()
        .map(|steps| {
            steps
                .iter()
                .flatten()
                .copied()
                .collect::<BTreeSet<_>>()
                .len()
        })
        .collect::<Vec<_>>();
    let draft_route = draft
        .selected_experts_by_step
        .last()
        .ok_or("live MTP proposal route missing")?;
    let draft_unique = draft_route.iter().copied().collect::<BTreeSet<_>>().len();
    let target_union_rows = target_unions.iter().sum::<usize>();
    let combined_union_rows = target_union_rows
        .checked_add(draft_unique)
        .ok_or("combined expert union overflow")?;
    let one_token_expert_rows = TARGET_LAYERS * config.top_k_experts;
    let union_u = combined_union_rows as f64 / one_token_expert_rows as f64;
    let a_over_u = accepted as f64 / union_u;
    let logical_expert_payload_bytes = combined_union_rows
        .checked_mul(EXPERT_BYTES)
        .ok_or("logical expert byte count overflow")?;
    let decision = &fixture.decision;
    let union = &fixture.expert_union;
    if decision.proposal_token_ids != proposal
        || decision.target_posterior_token_ids != posterior
        || decision.correct_draft_tokens != correct_drafts
        || decision.accepted_tokens != accepted
        || decision.retained_proposal_rows != retained
        || decision.rolled_back_proposal_rows != proposal.len() - retained
        || decision.emitted_token_ids != emitted
        || decision.next_anchor_token_id != next_anchor
        || decision.proposal_converged != converged
        || fixture.claims.accepted_tokens != accepted
        || union.target_unique_experts_by_layer != target_unions
        || union.target_union_expert_rows != target_union_rows
        || union.draft_unique_expert_rows != draft_unique
        || union.combined_union_expert_rows != combined_union_rows
        || union.one_token_expert_rows != one_token_expert_rows
        || union.u.to_bits() != union_u.to_bits()
        || union.a_over_u.to_bits() != a_over_u.to_bits()
        || union.logical_expert_payload_bytes != logical_expert_payload_bytes
    {
        return Err("MTP transaction decision or expert union mismatch".to_owned());
    }
    let total_verified_payload_bytes = draft
        .total_verified_payload_bytes
        .checked_add(target.total_verified_payload_bytes)
        .ok_or("transaction verified byte count overflow")?;
    Ok(MtpTransactionVerificationReport {
        schema_version: 1,
        semantic: "qwen3_8_flash_next_first_greedy_mtp_transaction_verification",
        model: fixture.model,
        revision: fixture.revision,
        source_commit: fixture.reference.source_commit,
        sampling: "greedy",
        batch_size: config.batch_size,
        concurrency: config.concurrency,
        q: config.q,
        proposal_token_ids: proposal,
        target_posterior_token_ids: posterior,
        correct_draft_tokens: correct_drafts,
        emitted_token_ids: emitted,
        next_anchor_token_id: next_anchor,
        retained_proposal_rows: retained,
        rolled_back_proposal_rows: decision.rolled_back_proposal_rows,
        proposal_converged: converged,
        target_unique_experts_by_layer: target_unions,
        target_union_expert_rows: target_union_rows,
        draft_unique_expert_rows: draft_unique,
        combined_union_expert_rows: combined_union_rows,
        one_token_expert_rows,
        accepted_tokens: accepted,
        expert_union: union_u,
        accepted_over_union: a_over_u,
        logical_expert_payload_bytes,
        total_verified_payload_bytes,
        complete_wall_time_ns: started.elapsed().as_nanos(),
        performance_claim: None,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn verify_mtp_recursive_transaction_fixture(
    checkpoint_dir: &Path,
    model_lock_path: &Path,
    mtp_source_lock_path: &Path,
    scheduler_lock_path: &Path,
    recursive_lock_path: &Path,
    acceptance_lock_path: &Path,
    tokenizer_fixture_path: &Path,
    ngram_fixture_path: &Path,
    ngram_row_fixture_path: &Path,
    ple_fixture_path: &Path,
    endpoint_fixture_path: &Path,
    fusion_fixture_path: &Path,
    causal_seed_fixture_path: &Path,
    causal_attention_fixture_path: &Path,
    causal_decoder_fixture_path: &Path,
    causal_output_fixture_path: &Path,
    recursive_seed_fixture_path: &Path,
    recursive_attention_fixture_path: &Path,
    recursive_decoder_fixture_path: &Path,
    recursive_output_fixture_path: &Path,
    target_transaction_fixture_path: &Path,
    transaction_fixture_path: &Path,
) -> Result<MtpTransactionVerificationReport, String> {
    let started = Instant::now();
    validate_source_lock(acceptance_lock_path)?;
    let fixture: Fixture = serde_json::from_slice(
        &fs::read(transaction_fixture_path).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let config = &fixture.configuration;
    let recursive_lock_sha256 = sha256_file(recursive_lock_path)?;
    if fixture.schema_version != 1
        || fixture.semantic != "qwen3_8_flash_next_first_recursive_greedy_mtp_transaction"
        || fixture.model != MODEL
        || fixture.revision != REVISION
        || fixture.reference.implementation
            != "source_derived_sglang_recursive_greedy_eagle_and_firewing_exact_native_authorities"
        || fixture.reference.source_commit != SGLANG_COMMIT
        || fixture.reference.acceptance_source_lock_sha256 != sha256_file(acceptance_lock_path)?
        || fixture.reference.recursive_source_lock_sha256.as_deref()
            != Some(recursive_lock_sha256.as_str())
        || fixture.reference.target_fixture_sha256 != sha256_file(target_transaction_fixture_path)?
        || fixture.reference.mtp_seed_fixture_sha256 != sha256_file(recursive_seed_fixture_path)?
        || fixture.reference.mtp_decoder_fixture_sha256
            != sha256_file(recursive_decoder_fixture_path)?
        || fixture.reference.mtp_output_fixture_sha256
            != sha256_file(recursive_output_fixture_path)?
        || config.sampling != "greedy"
        || config.batch_size != 1
        || config.concurrency != 1
        || config.q != 4
        || config.target_layers != TARGET_LAYERS
        || config.top_k_experts != 10
        || config.expert_payload_bytes != EXPERT_BYTES
        || fixture.claims.performance_claim.is_some()
        || fixture.claims.scope
            != "one exact recursive greedy width-four transaction; no timing, sustained TPS, or endpoint promotion claim"
    {
        return Err("recursive MTP transaction identity or configuration mismatch".to_owned());
    }

    let draft = verify_mtp_recursive_fixture(
        checkpoint_dir,
        model_lock_path,
        mtp_source_lock_path,
        scheduler_lock_path,
        recursive_lock_path,
        tokenizer_fixture_path,
        ngram_fixture_path,
        ngram_row_fixture_path,
        ple_fixture_path,
        endpoint_fixture_path,
        fusion_fixture_path,
        causal_seed_fixture_path,
        causal_attention_fixture_path,
        causal_decoder_fixture_path,
        causal_output_fixture_path,
        recursive_seed_fixture_path,
        recursive_attention_fixture_path,
        recursive_decoder_fixture_path,
        recursive_output_fixture_path,
    )?;
    let expected_token_ids = [16_207, 22_856]
        .into_iter()
        .chain(draft.proposal_token_ids.iter().copied())
        .collect::<Vec<_>>();
    let (target, _) = verify_token_text_endpoint_fixture_with_expected_outputs(
        checkpoint_dir,
        model_lock_path,
        tokenizer_fixture_path,
        ngram_fixture_path,
        ngram_row_fixture_path,
        ple_fixture_path,
        target_transaction_fixture_path,
        "qwen3_8_flash_next_firewing_six_token_cached_text_logits",
        "qwen3_8_flash_next_firewing_six_token_cached_text_logits_verification",
        &expected_token_ids,
    )?;
    let proposal = draft.proposal_token_ids;
    let posterior = target
        .top20_token_ids_by_step
        .iter()
        .skip(2)
        .map(|tokens| tokens.first().copied().ok_or("target posterior is empty"))
        .collect::<Result<Vec<_>, _>>()?;
    if proposal.len() != config.q || posterior.len() != config.q {
        return Err("recursive width-four transaction vectors are incomplete".to_owned());
    }
    let mismatch = (0..proposal.len() - 1).find(|index| posterior[*index] != proposal[*index + 1]);
    let (correct_drafts, accepted, retained, emitted, next_anchor, converged) =
        if let Some(index) = mismatch {
            (
                index,
                index + 1,
                index + 1,
                proposal[1..=index]
                    .iter()
                    .copied()
                    .chain(std::iter::once(posterior[index]))
                    .collect(),
                posterior[index],
                false,
            )
        } else {
            (
                proposal.len() - 1,
                proposal.len(),
                proposal.len(),
                proposal[1..]
                    .iter()
                    .copied()
                    .chain(std::iter::once(posterior[posterior.len() - 1]))
                    .collect(),
                posterior[posterior.len() - 1],
                true,
            )
        };

    let target_value: Value = serde_json::from_slice(
        &fs::read(target_transaction_fixture_path).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let target_routes = selected_routes(&target_value, 2)?;
    if target_routes.len() != TARGET_LAYERS
        || target_routes.iter().any(|steps| {
            steps.len() != config.q
                || steps
                    .iter()
                    .any(|route| route.len() != config.top_k_experts)
        })
    {
        return Err("recursive target route shape mismatch".to_owned());
    }
    let target_unions = target_routes
        .iter()
        .map(|steps| {
            steps
                .iter()
                .flatten()
                .copied()
                .collect::<BTreeSet<_>>()
                .len()
        })
        .collect::<Vec<_>>();
    let draft_unique = draft
        .selected_experts_by_step
        .iter()
        .skip(1)
        .flatten()
        .copied()
        .collect::<BTreeSet<_>>()
        .len();
    let target_union_rows = target_unions.iter().sum::<usize>();
    let combined_union_rows = target_union_rows
        .checked_add(draft_unique)
        .ok_or("recursive combined expert union overflow")?;
    let one_token_expert_rows = TARGET_LAYERS * config.top_k_experts;
    let union_u = combined_union_rows as f64 / one_token_expert_rows as f64;
    let a_over_u = accepted as f64 / union_u;
    let logical_expert_payload_bytes = combined_union_rows
        .checked_mul(EXPERT_BYTES)
        .ok_or("recursive logical expert byte count overflow")?;
    let decision = &fixture.decision;
    let union = &fixture.expert_union;
    if decision.proposal_token_ids != proposal
        || decision.target_posterior_token_ids != posterior
        || decision.correct_draft_tokens != correct_drafts
        || decision.accepted_tokens != accepted
        || decision.retained_proposal_rows != retained
        || decision.rolled_back_proposal_rows != proposal.len() - retained
        || decision.emitted_token_ids != emitted
        || decision.next_anchor_token_id != next_anchor
        || decision.proposal_converged != converged
        || fixture.claims.accepted_tokens != accepted
        || union.target_unique_experts_by_layer != target_unions
        || union.target_union_expert_rows != target_union_rows
        || union.draft_unique_expert_rows != draft_unique
        || union.combined_union_expert_rows != combined_union_rows
        || union.one_token_expert_rows != one_token_expert_rows
        || union.u.to_bits() != union_u.to_bits()
        || union.a_over_u.to_bits() != a_over_u.to_bits()
        || union.logical_expert_payload_bytes != logical_expert_payload_bytes
    {
        return Err("recursive MTP decision or expert union mismatch".to_owned());
    }
    let total_verified_payload_bytes = draft
        .total_verified_payload_bytes
        .checked_add(target.total_verified_payload_bytes)
        .ok_or("recursive transaction verified byte count overflow")?;
    Ok(MtpTransactionVerificationReport {
        schema_version: 1,
        semantic: "qwen3_8_flash_next_first_recursive_greedy_mtp_transaction_verification",
        model: fixture.model,
        revision: fixture.revision,
        source_commit: fixture.reference.source_commit,
        sampling: "greedy",
        batch_size: config.batch_size,
        concurrency: config.concurrency,
        q: config.q,
        proposal_token_ids: proposal,
        target_posterior_token_ids: posterior,
        correct_draft_tokens: correct_drafts,
        emitted_token_ids: emitted,
        next_anchor_token_id: next_anchor,
        retained_proposal_rows: retained,
        rolled_back_proposal_rows: decision.rolled_back_proposal_rows,
        proposal_converged: converged,
        target_unique_experts_by_layer: target_unions,
        target_union_expert_rows: target_union_rows,
        draft_unique_expert_rows: draft_unique,
        combined_union_expert_rows: combined_union_rows,
        one_token_expert_rows,
        accepted_tokens: accepted,
        expert_union: union_u,
        accepted_over_union: a_over_u,
        logical_expert_payload_bytes,
        total_verified_payload_bytes,
        complete_wall_time_ns: started.elapsed().as_nanos(),
        performance_claim: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn committed_transaction_exercises_full_match_bonus_branch() {
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../fixtures/mtp/qwen3_8_flash_next_first_transaction.json"
        ))
        .unwrap();
        assert_eq!(fixture.decision.proposal_token_ids, [369, 264]);
        assert_eq!(fixture.decision.target_posterior_token_ids, [264, 2526]);
        assert_eq!(fixture.decision.emitted_token_ids, [264, 2526]);
        assert_eq!(fixture.decision.accepted_tokens, 2);
        assert!(fixture.decision.proposal_converged);
        assert_eq!(fixture.expert_union.combined_union_expert_rows, 697);
        assert_eq!(fixture.expert_union.one_token_expert_rows, 480);
    }

    #[test]
    fn committed_recursive_transaction_exercises_rollback_branch() {
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../fixtures/mtp/qwen3_8_flash_next_first_recursive_transaction.json"
        ))
        .unwrap();
        assert_eq!(fixture.configuration.q, 4);
        assert_eq!(fixture.decision.proposal_token_ids, [369, 264, 220, 17]);
        assert_eq!(
            fixture.decision.target_posterior_token_ids,
            [264, 2526, 16, 15]
        );
        assert_eq!(fixture.decision.emitted_token_ids, [264, 2526]);
        assert_eq!(fixture.decision.retained_proposal_rows, 2);
        assert_eq!(fixture.decision.rolled_back_proposal_rows, 2);
        assert!(!fixture.decision.proposal_converged);
        assert_eq!(fixture.expert_union.combined_union_expert_rows, 1213);
        assert_eq!(fixture.expert_union.one_token_expert_rows, 480);
    }

    #[test]
    fn committed_second_recursive_transaction_repeats_early_rollback() {
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../fixtures/mtp/qwen3_8_flash_next_second_recursive_transaction.json"
        ))
        .unwrap();
        assert_eq!(fixture.configuration.q, 4);
        assert_eq!(fixture.decision.proposal_token_ids, [2526, 11, 8581, 11]);
        assert_eq!(
            fixture.decision.target_posterior_token_ids,
            [11, 45_815, 11, 321]
        );
        assert_eq!(fixture.decision.emitted_token_ids, [11, 45_815]);
        assert_eq!(fixture.decision.retained_proposal_rows, 2);
        assert_eq!(fixture.decision.rolled_back_proposal_rows, 2);
        assert!(!fixture.decision.proposal_converged);
        assert_eq!(fixture.expert_union.combined_union_expert_rows, 1066);
        assert_eq!(fixture.expert_union.one_token_expert_rows, 480);
    }
}
