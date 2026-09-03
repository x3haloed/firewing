use crate::decoder_layer::verify_decoder_layer_fixture_with_outputs;
use crate::decoder_layer1::verify_decoder_layer1_fixture_bytes_with_outputs;
use crate::expert::bf16_hash;
use crate::ple::verify_ple_fixture_bytes_with_outputs;
use crate::ple_attention_residual::verify_ple_attention_residual_fixture_bytes_with_outputs;
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
    layer1_ple: Value,
    layer1_attention: Value,
    layer1_decoder: Value,
    steps: Vec<Step>,
}

#[derive(Deserialize)]
struct Reference {
    implementation: String,
    transformers_version: String,
    source: String,
    model_lock_sha256: String,
    ngram_fixture_sha256: String,
    ngram_row_fixture_sha256: String,
    layer0_attention_fixture_sha256: String,
    layer0_sparse_moe_fixture_sha256: String,
    layer0_fixture_sha256: String,
    source_ple_fixture_sha256: String,
    source_layer1_attention_fixture_sha256: String,
    source_layer1_fixture_sha256: String,
}

#[derive(Deserialize)]
struct Configuration {
    first_layer: usize,
    last_layer: usize,
    tokens: Vec<i64>,
    hidden_size: usize,
    hc_count: usize,
    boundary_dtype: String,
    layer_types: Vec<String>,
    ple_layers: Vec<usize>,
}

#[derive(Deserialize)]
struct Step {
    ordinal: usize,
    mode: String,
    token_id: i64,
    input_spec: InputSpec,
    layer0_selected_experts: Vec<usize>,
    layer1_selected_experts: Vec<usize>,
    captures: Captures,
}

#[derive(Deserialize)]
struct InputSpec {
    multiplier: i64,
    add: i64,
    modulus: i64,
    center: i64,
    divisor: i64,
    sparse_stride: usize,
}

#[derive(Deserialize)]
struct Captures {
    layer0_output: Capture,
    layer1_ple_output: Capture,
    layer1_post_attention: Capture,
    layer1_output: Capture,
}

#[derive(Deserialize)]
struct Capture {
    dtype: String,
    shape: Vec<usize>,
    sha256: String,
}

#[derive(Debug, Eq, PartialEq, Serialize)]
pub struct AccumulatedLayers01VerificationReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub model: String,
    pub revision: String,
    pub layers_verified: Vec<usize>,
    pub steps_verified: usize,
    pub boundary_links_verified: usize,
    pub layer0_exact_bf16_capture_hashes: usize,
    pub layer0_exact_weighted_expert_hashes: usize,
    pub layer1_ple_attention_exact_bf16_capture_hashes: usize,
    pub layer1_ple_attention_exact_f32_capture_hashes: usize,
    pub layer1_ple_exact_i64_capture_hashes: usize,
    pub layer1_decoder_exact_bf16_capture_hashes: usize,
    pub layer1_decoder_exact_weighted_expert_hashes: usize,
    pub layer0_verified_payload_bytes: usize,
    pub layer1_verified_payload_bytes: usize,
    pub total_verified_payload_bytes: usize,
    pub layer0_selected_experts_by_step: Vec<Vec<usize>>,
    pub layer1_selected_experts_by_step: Vec<Vec<usize>>,
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
        return Err(format!("accumulated capture mismatch for {name}"));
    }
    Ok(())
}

fn validate_step_metadata(step: &Step, ordinal: usize) -> Result<(), String> {
    let expected_specs = [(43, 17, 263, 131, 128), (61, 29, 277, 138, 128)];
    if step.ordinal != ordinal
        || step.mode
            != if ordinal == 0 {
                "initial_chunk"
            } else {
                "cached_recurrent"
            }
        || step.token_id != [42, 43][ordinal]
        || (
            step.input_spec.multiplier,
            step.input_spec.add,
            step.input_spec.modulus,
            step.input_spec.center,
            step.input_spec.divisor,
        ) != expected_specs[ordinal]
        || step.input_spec.sparse_stride != 1
        || step.layer0_selected_experts.len() != 10
        || step.layer1_selected_experts.len() != 10
    {
        return Err(format!("accumulated step {ordinal} metadata mismatch"));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_accumulated_layers01_fixture_with_outputs(
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
    fixture_path: &Path,
) -> Result<(AccumulatedLayers01VerificationReport, Vec<Vec<u16>>), String> {
    let fixture: Fixture = serde_json::from_slice(
        &fs::read(fixture_path).map_err(|error| format!("cannot read fixture: {error}"))?,
    )
    .map_err(|error| format!("malformed accumulated fixture: {error}"))?;
    let config = &fixture.configuration;
    if fixture.schema_version != 1
        || fixture.semantic != "qwen3_8_flash_next_accumulated_layers0_1_cached_decode"
        || fixture.model != MODEL
        || fixture.reference.implementation != "source_derived_huggingface_transformers_qwen4_exp"
        || fixture.reference.transformers_version != "5.16.1"
        || fixture.reference.source
            != "transformers.models.qwen4_exp.modeling_qwen4_exp.Qwen4ExpTextDecoderLayer.forward"
        || config.first_layer != 0
        || config.last_layer != 1
        || config.tokens != [42, 43]
        || config.hidden_size != 2560
        || config.hc_count != 4
        || config.boundary_dtype != "BF16"
        || config.layer_types != ["linear_attention", "linear_attention"]
        || config.ple_layers != [1]
        || fixture.steps.len() != 2
    {
        return Err("accumulated fixture identity or configuration is unsupported".to_owned());
    }
    let references = [
        (model_lock_path, &fixture.reference.model_lock_sha256),
        (ngram_fixture_path, &fixture.reference.ngram_fixture_sha256),
        (
            ngram_row_fixture_path,
            &fixture.reference.ngram_row_fixture_sha256,
        ),
        (
            layer0_attention_fixture_path,
            &fixture.reference.layer0_attention_fixture_sha256,
        ),
        (
            layer0_sparse_moe_fixture_path,
            &fixture.reference.layer0_sparse_moe_fixture_sha256,
        ),
        (
            layer0_fixture_path,
            &fixture.reference.layer0_fixture_sha256,
        ),
        (
            ple_fixture_path,
            &fixture.reference.source_ple_fixture_sha256,
        ),
        (
            layer1_attention_fixture_path,
            &fixture.reference.source_layer1_attention_fixture_sha256,
        ),
        (
            layer1_fixture_path,
            &fixture.reference.source_layer1_fixture_sha256,
        ),
    ];
    for (path, expected) in references {
        if sha256_file(path)? != *expected {
            return Err(format!(
                "accumulated reference mismatch for {}",
                path.display()
            ));
        }
    }

    let (layer0_report, layer0_outputs) = verify_decoder_layer_fixture_with_outputs(
        checkpoint_dir,
        model_lock_path,
        layer0_hyper_fixture_path,
        layer0_deltanet_fixture_path,
        layer0_attention_fixture_path,
        layer0_sparse_moe_fixture_path,
        layer0_fixture_path,
    )?;
    let ple_bytes = serde_json::to_vec(&fixture.layer1_ple).map_err(|error| error.to_string())?;
    let ple_execution = verify_ple_fixture_bytes_with_outputs(
        checkpoint_dir,
        model_lock_path,
        ngram_fixture_path,
        ngram_row_fixture_path,
        &ple_bytes,
        "qwen3_8_flash_next_layer1_ple_accumulated_from_layer0",
        Some(&layer0_outputs),
    )?;
    let ple_outputs = ple_execution.1.clone();
    let attention_bytes =
        serde_json::to_vec(&fixture.layer1_attention).map_err(|error| error.to_string())?;
    let attention_execution = verify_ple_attention_residual_fixture_bytes_with_outputs(
        checkpoint_dir,
        model_lock_path,
        ngram_fixture_path,
        ngram_row_fixture_path,
        ple_fixture_path,
        &attention_bytes,
        "qwen3_8_flash_next_layer1_ple_attention_residual_accumulated_from_layer0",
        Some(&layer0_outputs),
        Some(ple_execution),
    )?;
    let attention_outputs = attention_execution.1.clone();
    let attention_bf16_hashes = attention_execution.0.exact_bf16_capture_hashes;
    let attention_f32_hashes = attention_execution.0.exact_f32_capture_hashes;
    let ple_i64_hashes = attention_execution.0.exact_i64_capture_hashes;
    let decoder_bytes =
        serde_json::to_vec(&fixture.layer1_decoder).map_err(|error| error.to_string())?;
    let (layer1_report, layer1_outputs) = verify_decoder_layer1_fixture_bytes_with_outputs(
        checkpoint_dir,
        model_lock_path,
        ngram_fixture_path,
        ngram_row_fixture_path,
        ple_fixture_path,
        layer1_attention_fixture_path,
        &decoder_bytes,
        "qwen3_8_flash_next_layer1_complete_accumulated_from_layer0",
        Some(attention_execution),
    )?;

    for ordinal in 0..2 {
        let step = &fixture.steps[ordinal];
        validate_step_metadata(step, ordinal)?;
        require_capture(
            &step.captures.layer0_output,
            &layer0_outputs[ordinal],
            "layer0_output",
        )?;
        require_capture(
            &step.captures.layer1_ple_output,
            &ple_outputs[ordinal],
            "layer1_ple_output",
        )?;
        require_capture(
            &step.captures.layer1_post_attention,
            &attention_outputs[ordinal],
            "layer1_post_attention",
        )?;
        require_capture(
            &step.captures.layer1_output,
            &layer1_outputs[ordinal],
            "layer1_output",
        )?;
        if step.layer0_selected_experts != layer0_report.selected_experts_by_step[ordinal]
            || step.layer1_selected_experts != layer1_report.selected_experts_by_step[ordinal]
        {
            return Err(format!("accumulated route link mismatch at step {ordinal}"));
        }
    }

    Ok((
        AccumulatedLayers01VerificationReport {
            schema_version: 1,
            semantic: "qwen3_8_flash_next_accumulated_layers0_1_verification",
            model: fixture.model,
            revision: fixture.revision,
            layers_verified: vec![0, 1],
            steps_verified: 2,
            boundary_links_verified: 8,
            layer0_exact_bf16_capture_hashes: layer0_report.exact_bf16_capture_hashes,
            layer0_exact_weighted_expert_hashes: layer0_report.exact_weighted_expert_hashes,
            layer1_ple_attention_exact_bf16_capture_hashes: attention_bf16_hashes,
            layer1_ple_attention_exact_f32_capture_hashes: attention_f32_hashes,
            layer1_ple_exact_i64_capture_hashes: ple_i64_hashes,
            layer1_decoder_exact_bf16_capture_hashes: layer1_report.exact_bf16_capture_hashes,
            layer1_decoder_exact_weighted_expert_hashes: layer1_report.exact_weighted_expert_hashes,
            layer0_verified_payload_bytes: layer0_report.total_verified_payload_bytes,
            layer1_verified_payload_bytes: layer1_report.total_verified_payload_bytes,
            total_verified_payload_bytes: layer0_report.total_verified_payload_bytes
                + layer1_report.total_verified_payload_bytes,
            layer0_selected_experts_by_step: layer0_report.selected_experts_by_step,
            layer1_selected_experts_by_step: layer1_report.selected_experts_by_step,
            accepted_tokens: 0,
            performance_claim: None,
        },
        layer1_outputs,
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn verify_accumulated_layers01_fixture(
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
    fixture_path: &Path,
) -> Result<AccumulatedLayers01VerificationReport, String> {
    verify_accumulated_layers01_fixture_with_outputs(
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
        fixture_path,
    )
    .map(|(report, _)| report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulated_fixture_links_layer0_to_ple_input() {
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../fixtures/accumulated/qwen3_8_flash_next_layers0_1.json"
        ))
        .unwrap();
        for ordinal in 0..2 {
            let ple_hash =
                fixture.layer1_ple["case"]["steps"][ordinal]["captures"]["hidden_states"]["sha256"]
                    .as_str()
                    .unwrap();
            assert_eq!(
                ple_hash,
                fixture.steps[ordinal].captures.layer0_output.sha256
            );
        }
    }
}
