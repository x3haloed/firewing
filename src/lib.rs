use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;
use tokenizers::Tokenizer;

mod accumulated_layer2;
mod accumulated_layer3;
mod accumulated_layers01;
mod accumulated_layers4_47;
mod attention_residual;
mod checkpoint_catalog;
mod compressed_overlap;
mod decoder_layer;
mod decoder_layer1;
mod decoder_layer3;
mod deltanet;
mod expert;
mod expert_acquisition;
pub mod full_attention;
mod full_attention_residual;
mod host_safety;
mod hyper_connection;
mod metal_bf16;
mod metal_moe;
mod mtp;
mod mtp_transaction;
mod ngram;
mod overlap_bound;
mod ple;
mod ple_attention_residual;
mod router;
mod text_output;
mod token_text_endpoint;

pub use accumulated_layer2::{
    AccumulatedLayer2VerificationReport, verify_accumulated_layer2_fixture,
};
pub use accumulated_layer3::{
    AccumulatedLayer3VerificationReport, verify_accumulated_layer3_fixture,
};
pub use accumulated_layers01::{
    AccumulatedLayers01VerificationReport, verify_accumulated_layers01_fixture,
};
pub use accumulated_layers4_47::{
    AccumulatedLayers4Through47VerificationReport, verify_accumulated_layers4_through47_fixture,
};
pub use attention_residual::{
    AttentionResidualVerificationReport, verify_attention_residual_fixture,
};
pub use checkpoint_catalog::{CheckpointCatalogReport, benchmark_checkpoint_catalog};
pub use compressed_overlap::{
    CompressedOverlapTrial, ParallelZstdOverlapReport, SequentialShuffleOverlapReport,
    benchmark_capacity_cache_overlap, benchmark_parallel_zstd_overlap,
    benchmark_sequential_shuffle_overlap,
};
pub use decoder_layer::{DecoderLayerVerificationReport, verify_decoder_layer_fixture};
pub use decoder_layer1::{DecoderLayer1VerificationReport, verify_decoder_layer1_fixture};
pub use decoder_layer3::{DecoderLayer3VerificationReport, verify_decoder_layer3_fixture};
pub use deltanet::{DeltaNetVerificationReport, verify_deltanet_fixture};
pub use expert::{
    ExpertVerificationReport, MixtureVerificationReport, SparseMoeVerificationReport,
    verify_expert_fixture, verify_mixture_fixture, verify_sparse_moe_fixture,
};
pub use expert_acquisition::{
    ExpertAcquisitionBenchmarkReport, ExpertAcquisitionSummary, ExpertAcquisitionTrial,
    benchmark_expert_acquisition,
};
pub use full_attention::{FullAttentionVerificationReport, verify_full_attention_fixture};
pub use full_attention_residual::{
    FullAttentionResidualVerificationReport, verify_full_attention_residual_fixture,
};
pub use host_safety::{HostSafetyPolicy, HostSafetySnapshot, PersistentResidencyDeclaration};
pub use hyper_connection::{HyperConnectionVerificationReport, verify_hyper_connection_fixture};
pub use metal_bf16::{MetalBf16GemvReport, benchmark_metal_bf16_gemv};
pub use metal_moe::{MetalTop10MoeReport, benchmark_metal_top10_moe};
pub use mtp::{
    MtpCausalPrefillVerificationReport, MtpInputFusionVerificationReport,
    MtpProposalVerificationReport, MtpRecursiveVerificationReport,
    verify_mtp_causal_prefill_fixture, verify_mtp_input_fusion_fixture,
    verify_mtp_proposal_fixture, verify_mtp_recursive_fixture,
};
pub use mtp_transaction::{
    MtpTransactionVerificationReport, verify_mtp_recursive_transaction_fixture,
    verify_mtp_transaction_fixture,
};
pub use ngram::{
    NGramRowVerificationReport, NGramTransportBenchmarkReport, NGramTransportSummary,
    NGramTransportTrial, NGramVerificationReport, benchmark_ngram_transport, verify_ngram_fixture,
    verify_ngram_rows,
};
pub use overlap_bound::{
    ExactOverlapBoundReport, OverlapSummary, OverlapTrial, Q2ExactOverlapBoundReport,
    benchmark_exact_overlap_bound, benchmark_q2_exact_overlap_bound,
};
pub use ple::{PleVerificationReport, verify_ple_fixture};
pub use ple_attention_residual::{
    PleAttentionResidualVerificationReport, verify_ple_attention_residual_fixture,
};
pub use router::{RouterCaseReport, RouterVerificationReport, verify_router_fixture};
pub use text_output::{TextOutputVerificationReport, verify_text_output_fixture};
pub use token_text_endpoint::{
    CatalogTokenTextEndpointReport, EndpointLayerTiming, TokenTextEndpointVerificationReport,
    benchmark_catalog_token_text_endpoint, verify_token_text_continuation_fixture,
    verify_token_text_endpoint_fixture, verify_token_text_transaction_fixture,
};

#[derive(Debug, Deserialize)]
struct TokenizerFixture {
    schema_version: u32,
    semantic: String,
    model: String,
    revision: String,
    reference: TokenizerReference,
    raw_cases: Vec<RawTokenizerCase>,
    chat_cases: Vec<ChatTokenizerCase>,
}

#[derive(Debug, Deserialize)]
struct TokenizerReference {
    tokenizer_json_sha256: String,
    tokenizer_config_sha256: String,
    chat_template_sha256: String,
}

#[derive(Debug, Deserialize)]
struct RawTokenizerCase {
    name: String,
    text: String,
    add_special_tokens: bool,
    token_ids: Vec<u32>,
}

#[derive(Debug, Deserialize)]
struct ChatTokenizerCase {
    name: String,
    rendered: String,
    token_ids: Vec<u32>,
}

#[derive(Debug, Eq, PartialEq, Serialize)]
pub struct TokenizerVerificationReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub model: String,
    pub revision: String,
    pub raw_cases_verified: usize,
    pub chat_cases_verified: usize,
    pub tokenizer_json_sha256: String,
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let bytes =
        fs::read(path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn require_hash(path: &Path, expected: &str) -> Result<String, String> {
    let actual = sha256_file(path)?;
    if actual != expected {
        return Err(format!(
            "content identity mismatch for {}: expected {expected}, got {actual}",
            path.display()
        ));
    }
    Ok(actual)
}

fn validate_fixture_identity(fixture: &TokenizerFixture) -> Result<(), String> {
    if fixture.schema_version != 1
        || fixture.semantic != "qwen3_8_flash_next_tokenizer_and_chat_template"
        || fixture.model != "Qwen/Qwen3.8-Flash-Next"
        || fixture.revision.len() != 40
        || !fixture
            .revision
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        return Err("tokenizer fixture identity or schema is unsupported".to_owned());
    }
    Ok(())
}

pub fn verify_tokenizer_fixture(
    checkpoint_dir: &Path,
    fixture_path: &Path,
) -> Result<TokenizerVerificationReport, String> {
    let fixture_bytes = fs::read(fixture_path)
        .map_err(|error| format!("cannot read fixture {}: {error}", fixture_path.display()))?;
    let fixture: TokenizerFixture = serde_json::from_slice(&fixture_bytes)
        .map_err(|error| format!("malformed tokenizer fixture: {error}"))?;
    validate_fixture_identity(&fixture)?;

    let tokenizer_path = checkpoint_dir.join("tokenizer.json");
    let tokenizer_hash = require_hash(&tokenizer_path, &fixture.reference.tokenizer_json_sha256)?;
    require_hash(
        &checkpoint_dir.join("tokenizer_config.json"),
        &fixture.reference.tokenizer_config_sha256,
    )?;
    require_hash(
        &checkpoint_dir.join("chat_template.jinja"),
        &fixture.reference.chat_template_sha256,
    )?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|error| {
        format!(
            "cannot load tokenizer {}: {error}",
            tokenizer_path.display()
        )
    })?;

    for case in &fixture.raw_cases {
        let encoded = tokenizer
            .encode(case.text.as_str(), case.add_special_tokens)
            .map_err(|error| format!("raw case {} failed to encode: {error}", case.name))?;
        if encoded.get_ids() != case.token_ids {
            return Err(format!(
                "raw case {} token mismatch: expected {:?}, got {:?}",
                case.name,
                case.token_ids,
                encoded.get_ids()
            ));
        }
    }
    for case in &fixture.chat_cases {
        let encoded = tokenizer
            .encode(case.rendered.as_str(), false)
            .map_err(|error| format!("chat case {} failed to encode: {error}", case.name))?;
        if encoded.get_ids() != case.token_ids {
            return Err(format!(
                "chat case {} token mismatch: expected {:?}, got {:?}",
                case.name,
                case.token_ids,
                encoded.get_ids()
            ));
        }
    }

    Ok(TokenizerVerificationReport {
        schema_version: 1,
        semantic: "qwen3_8_flash_next_tokenizer_fixture_verification",
        model: fixture.model,
        revision: fixture.revision,
        raw_cases_verified: fixture.raw_cases.len(),
        chat_cases_verified: fixture.chat_cases.len(),
        tokenizer_json_sha256: tokenizer_hash,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_identity_requires_exact_model() {
        let invalid = TokenizerFixture {
            schema_version: 1,
            semantic: "qwen3_8_flash_next_tokenizer_and_chat_template".to_owned(),
            model: "not-the-target".to_owned(),
            revision: "d".repeat(40),
            reference: TokenizerReference {
                tokenizer_json_sha256: "0".repeat(64),
                tokenizer_config_sha256: "0".repeat(64),
                chat_template_sha256: "0".repeat(64),
            },
            raw_cases: vec![],
            chat_cases: vec![],
        };
        assert_eq!(
            validate_fixture_identity(&invalid),
            Err("tokenizer fixture identity or schema is unsupported".to_owned())
        );
    }
}
