use crate::accumulated_layer2::verify_accumulated_layer2_fixture_with_outputs;
use crate::decoder_layer3::verify_decoder_mlp_fixture_bytes_with_outputs;
use crate::expert::bf16_hash;
use crate::full_attention_residual::verify_full_attention_residual_fixture_bytes_with_outputs;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

const MODEL: &str = "Qwen/Qwen3.8-Flash-Next";
const HC_HIDDEN: usize = 10_240;

#[derive(Deserialize)]
struct Fixture {
    schema_version: u32,
    semantic: String,
    model: String,
    revision: String,
    reference: Reference,
    configuration: Configuration,
    attention: Value,
    decoder: Value,
    steps: Vec<Step>,
}

#[derive(Deserialize)]
struct Reference {
    implementation: String,
    transformers_version: String,
    source: String,
    model_lock_sha256: String,
    layer2_fixture_sha256: String,
}

#[derive(Deserialize)]
struct Configuration {
    layer: usize,
    layer_type: String,
    ple_applied: bool,
    hidden_size: usize,
    hc_count: usize,
    boundary_dtype: String,
    cache_lengths: Vec<usize>,
}

#[derive(Deserialize)]
struct Step {
    ordinal: usize,
    mode: String,
    position: usize,
    past_length: usize,
    selected_experts: Vec<usize>,
    captures: Captures,
}

#[derive(Deserialize)]
struct Captures {
    layer2_output: Capture,
    post_attention: Capture,
    layer3_output: Capture,
}

#[derive(Deserialize)]
struct Capture {
    dtype: String,
    shape: Vec<usize>,
    sha256: String,
}

#[derive(Debug, Eq, PartialEq, Serialize)]
pub struct AccumulatedLayer3VerificationReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub model: String,
    pub revision: String,
    pub accumulated_parent_layers: Vec<usize>,
    pub layer: usize,
    pub steps_verified: usize,
    pub boundary_links_verified: usize,
    pub exact_attention_bf16_capture_hashes: usize,
    pub exact_attention_f32_capture_hashes: usize,
    pub exact_attention_i64_capture_hashes: usize,
    pub exact_attention_bool_capture_hashes: usize,
    pub exact_decoder_bf16_capture_hashes: usize,
    pub exact_weighted_expert_hashes: usize,
    pub unique_experts_verified: usize,
    pub parent_verified_payload_bytes: usize,
    pub layer3_verified_payload_bytes: usize,
    pub total_verified_payload_bytes: usize,
    pub selected_experts_by_step: Vec<Vec<usize>>,
    pub accepted_tokens: usize,
    pub performance_claim: Option<String>,
}

fn sha256_file(path: &Path) -> Result<String, String> {
    fs::read(path)
        .map(|bytes| format!("{:x}", Sha256::digest(bytes)))
        .map_err(|error| format!("cannot read {}: {error}", path.display()))
}

fn require_capture(capture: &Capture, values: &[u16], name: &str) -> Result<(), String> {
    if capture.dtype != "BF16"
        || capture.shape != [1, 1, HC_HIDDEN]
        || capture.sha256.len() != 64
        || !capture.sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
        || capture.sha256 != bf16_hash(values)
    {
        return Err(format!("accumulated layer-3 capture mismatch for {name}"));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_accumulated_layer3_fixture_with_outputs(
    checkpoint_dir: &Path,
    model_lock_path: &Path,
    ngram_fixture_path: &Path,
    ngram_row_fixture_path: &Path,
    layer0_hyper_fixture_path: &Path,
    layer0_deltanet_fixture_path: &Path,
    layer0_attention_fixture_path: &Path,
    layer0_sparse_moe_fixture_path: &Path,
    layer0_fixture_path: &Path,
    ple_fixture_path: &Path,
    layer1_attention_fixture_path: &Path,
    layer1_fixture_path: &Path,
    layers01_fixture_path: &Path,
    layer2_fixture_path: &Path,
    fixture_path: &Path,
) -> Result<(AccumulatedLayer3VerificationReport, Vec<Vec<u16>>), String> {
    let fixture: Fixture = serde_json::from_slice(
        &fs::read(fixture_path).map_err(|error| format!("cannot read fixture: {error}"))?,
    )
    .map_err(|error| format!("malformed accumulated layer-3 fixture: {error}"))?;
    let config = &fixture.configuration;
    if fixture.schema_version != 1
        || fixture.semantic != "qwen3_8_flash_next_accumulated_layer3_cached_decode"
        || fixture.model != MODEL
        || fixture.reference.implementation
            != "source_derived_and_official_huggingface_transformers_qwen4_exp"
        || fixture.reference.transformers_version != "5.16.1"
        || fixture.reference.source
            != "transformers.models.qwen4_exp.modeling_qwen4_exp.Qwen4ExpTextDecoderLayer.forward"
        || config.layer != 3
        || config.layer_type != "full_attention"
        || config.ple_applied
        || config.hidden_size != 2560
        || config.hc_count != 4
        || config.boundary_dtype != "BF16"
        || config.cache_lengths != [0, 1]
        || fixture.steps.len() != 2
        || sha256_file(model_lock_path)? != fixture.reference.model_lock_sha256
        || sha256_file(layer2_fixture_path)? != fixture.reference.layer2_fixture_sha256
    {
        return Err("accumulated layer-3 identity or configuration is unsupported".to_owned());
    }
    let (parent_report, layer2_outputs) = verify_accumulated_layer2_fixture_with_outputs(
        checkpoint_dir,
        model_lock_path,
        ngram_fixture_path,
        ngram_row_fixture_path,
        layer0_hyper_fixture_path,
        layer0_deltanet_fixture_path,
        layer0_attention_fixture_path,
        layer0_sparse_moe_fixture_path,
        layer0_fixture_path,
        ple_fixture_path,
        layer1_attention_fixture_path,
        layer1_fixture_path,
        layers01_fixture_path,
        layer2_fixture_path,
    )?;
    let attention_bytes =
        serde_json::to_vec(&fixture.attention).map_err(|error| error.to_string())?;
    let (attention_report, post_attention) =
        verify_full_attention_residual_fixture_bytes_with_outputs(
            checkpoint_dir,
            model_lock_path,
            &attention_bytes,
            "qwen3_8_flash_next_layer3_attention_accumulated_from_layer2",
            "qwen3_8_flash_next_layer3_attention_accumulated_verification",
            3,
            [0, 1],
            ["initial", "cached_incremental"],
            true,
            None,
            Some(&layer2_outputs),
        )?;
    let decoder_bytes = serde_json::to_vec(&fixture.decoder).map_err(|error| error.to_string())?;
    let (decoder_report, layer3_outputs) = verify_decoder_mlp_fixture_bytes_with_outputs(
        checkpoint_dir,
        model_lock_path,
        &decoder_bytes,
        3,
        "full_attention",
        "qwen3_8_flash_next_layer3_complete_accumulated_from_layer2",
        "qwen3_8_flash_next_layer3_complete_accumulated_verification",
        ["initial", "cached_incremental"],
        attention_report.total_verified_payload_bytes,
        post_attention.clone(),
    )?;
    for ordinal in 0..2 {
        let step = &fixture.steps[ordinal];
        if step.ordinal != ordinal
            || step.mode
                != if ordinal == 0 {
                    "initial"
                } else {
                    "cached_incremental"
                }
            || step.position != ordinal
            || step.past_length != ordinal
            || step.selected_experts != decoder_report.selected_experts_by_step[ordinal]
        {
            return Err(format!(
                "accumulated layer-3 step {ordinal} metadata mismatch"
            ));
        }
        require_capture(
            &step.captures.layer2_output,
            &layer2_outputs[ordinal],
            "layer2_output",
        )?;
        require_capture(
            &step.captures.post_attention,
            &post_attention[ordinal],
            "post_attention",
        )?;
        require_capture(
            &step.captures.layer3_output,
            &layer3_outputs[ordinal],
            "layer3_output",
        )?;
    }
    let parent_bytes = parent_report.total_verified_payload_bytes;
    let layer3_bytes = decoder_report.total_verified_payload_bytes;
    Ok((
        AccumulatedLayer3VerificationReport {
            schema_version: 1,
            semantic: "qwen3_8_flash_next_accumulated_layer3_verification",
            model: fixture.model,
            revision: fixture.revision,
            accumulated_parent_layers: vec![0, 1, 2],
            layer: 3,
            steps_verified: 2,
            boundary_links_verified: 6,
            exact_attention_bf16_capture_hashes: attention_report.exact_bf16_capture_hashes,
            exact_attention_f32_capture_hashes: attention_report.exact_f32_capture_hashes,
            exact_attention_i64_capture_hashes: attention_report.exact_i64_capture_hashes,
            exact_attention_bool_capture_hashes: attention_report.exact_bool_capture_hashes,
            exact_decoder_bf16_capture_hashes: decoder_report.exact_bf16_capture_hashes,
            exact_weighted_expert_hashes: decoder_report.exact_weighted_expert_hashes,
            unique_experts_verified: decoder_report.unique_experts_verified,
            parent_verified_payload_bytes: parent_bytes,
            layer3_verified_payload_bytes: layer3_bytes,
            total_verified_payload_bytes: parent_bytes + layer3_bytes,
            selected_experts_by_step: decoder_report.selected_experts_by_step,
            accepted_tokens: 0,
            performance_claim: None,
        },
        layer3_outputs,
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn verify_accumulated_layer3_fixture(
    checkpoint_dir: &Path,
    model_lock_path: &Path,
    ngram_fixture_path: &Path,
    ngram_row_fixture_path: &Path,
    layer0_hyper_fixture_path: &Path,
    layer0_deltanet_fixture_path: &Path,
    layer0_attention_fixture_path: &Path,
    layer0_sparse_moe_fixture_path: &Path,
    layer0_fixture_path: &Path,
    ple_fixture_path: &Path,
    layer1_attention_fixture_path: &Path,
    layer1_fixture_path: &Path,
    layers01_fixture_path: &Path,
    layer2_fixture_path: &Path,
    fixture_path: &Path,
) -> Result<AccumulatedLayer3VerificationReport, String> {
    verify_accumulated_layer3_fixture_with_outputs(
        checkpoint_dir,
        model_lock_path,
        ngram_fixture_path,
        ngram_row_fixture_path,
        layer0_hyper_fixture_path,
        layer0_deltanet_fixture_path,
        layer0_attention_fixture_path,
        layer0_sparse_moe_fixture_path,
        layer0_fixture_path,
        ple_fixture_path,
        layer1_attention_fixture_path,
        layer1_fixture_path,
        layers01_fixture_path,
        layer2_fixture_path,
        fixture_path,
    )
    .map(|(report, _)| report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer3_fixture_links_parent_output_to_attention_input() {
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../fixtures/accumulated/qwen3_8_flash_next_layer3.json"
        ))
        .unwrap();
        for ordinal in 0..2 {
            assert_eq!(
                fixture.steps[ordinal].captures.layer2_output.sha256,
                fixture.attention["cases"][ordinal]["captures"]["hyper_input"]["sha256"]
                    .as_str()
                    .unwrap()
            );
        }
    }
}
