use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;
use tokenizers::Tokenizer;

mod attention_residual;
mod decoder_layer;
mod deltanet;
mod expert;
mod expert_acquisition;
mod hyper_connection;
mod ngram;
mod ple;
mod router;

pub use attention_residual::{
    AttentionResidualVerificationReport, verify_attention_residual_fixture,
};
pub use decoder_layer::{DecoderLayerVerificationReport, verify_decoder_layer_fixture};
pub use deltanet::{DeltaNetVerificationReport, verify_deltanet_fixture};
pub use expert::{
    ExpertVerificationReport, MixtureVerificationReport, SparseMoeVerificationReport,
    verify_expert_fixture, verify_mixture_fixture, verify_sparse_moe_fixture,
};
pub use expert_acquisition::{
    ExpertAcquisitionBenchmarkReport, ExpertAcquisitionSummary, ExpertAcquisitionTrial,
    benchmark_expert_acquisition,
};
pub use hyper_connection::{HyperConnectionVerificationReport, verify_hyper_connection_fixture};
pub use ngram::{
    NGramRowVerificationReport, NGramTransportBenchmarkReport, NGramTransportSummary,
    NGramTransportTrial, NGramVerificationReport, benchmark_ngram_transport, verify_ngram_fixture,
    verify_ngram_rows,
};
pub use ple::{PleVerificationReport, verify_ple_fixture};
pub use router::{RouterCaseReport, RouterVerificationReport, verify_router_fixture};

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
