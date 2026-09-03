use crate::accumulated_layer3::verify_accumulated_layer3_fixture_with_outputs;
use crate::attention_residual::verify_attention_residual_fixture_bytes_with_outputs;
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
    layers: Vec<Layer>,
    final_outputs: Vec<Capture>,
}

#[derive(Deserialize)]
struct Reference {
    implementation: String,
    transformers_version: String,
    model_lock_sha256: String,
    layer3_fixture_sha256: String,
}

#[derive(Deserialize)]
struct Configuration {
    first_layer: usize,
    last_layer: usize,
    layer_types: Vec<String>,
    ple_layer_ids: Vec<usize>,
    hidden_size: usize,
    hc_count: usize,
    boundary_dtype: String,
}

#[derive(Deserialize)]
struct Layer {
    layer: usize,
    layer_type: String,
    attention: Value,
    decoder: Value,
    steps: Vec<Step>,
}

#[derive(Deserialize)]
struct Step {
    ordinal: usize,
    mode: String,
    selected_experts: Vec<usize>,
    captures: Captures,
}

#[derive(Deserialize)]
struct Captures {
    layer_input: Capture,
    post_attention: Capture,
    layer_output: Capture,
}

#[derive(Clone, Deserialize)]
struct Capture {
    dtype: String,
    shape: Vec<usize>,
    sha256: String,
}

#[derive(Debug, Eq, PartialEq, Serialize)]
pub struct AccumulatedLayers4Through47VerificationReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub model: String,
    pub revision: String,
    pub parent_layers_verified: Vec<usize>,
    pub layers_verified: Vec<usize>,
    pub linear_layers_verified: usize,
    pub full_attention_layers_verified: usize,
    pub steps_per_layer: usize,
    pub boundary_links_verified: usize,
    pub exact_attention_bf16_capture_hashes: usize,
    pub exact_attention_f32_capture_hashes: usize,
    pub exact_attention_i64_capture_hashes: usize,
    pub exact_attention_bool_capture_hashes: usize,
    pub exact_decoder_bf16_capture_hashes: usize,
    pub exact_weighted_expert_hashes: usize,
    pub layer_scoped_unique_experts_verified: usize,
    pub parent_verified_payload_bytes: usize,
    pub remaining_layers_verified_payload_bytes: usize,
    pub total_verified_payload_bytes: usize,
    pub selected_experts_by_layer_and_step: Vec<Vec<Vec<usize>>>,
    pub final_output_hashes: Vec<String>,
    pub accepted_tokens: usize,
    pub performance_claim: Option<String>,
}

fn sha256_file(path: &Path) -> Result<String, String> {
    fs::read(path)
        .map(|bytes| format!("{:x}", Sha256::digest(bytes)))
        .map_err(|error| format!("cannot read {}: {error}", path.display()))
}

fn require_capture(capture: &Capture, values: &[u16], label: &str) -> Result<(), String> {
    if capture.dtype != "BF16"
        || capture.shape != [1, 1, HC_HIDDEN]
        || capture.sha256.len() != 64
        || !capture.sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
        || capture.sha256 != bf16_hash(values)
    {
        return Err(format!("remaining decoder capture mismatch for {label}"));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn verify_accumulated_layers4_through47_fixture(
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
    layer3_fixture_path: &Path,
    fixture_path: &Path,
) -> Result<AccumulatedLayers4Through47VerificationReport, String> {
    let fixture: Fixture = serde_json::from_slice(
        &fs::read(fixture_path).map_err(|error| format!("cannot read fixture: {error}"))?,
    )
    .map_err(|error| format!("malformed remaining decoder fixture: {error}"))?;
    let config = &fixture.configuration;
    let expected_types = (0..48)
        .map(|layer| {
            if layer % 4 == 3 {
                "full_attention"
            } else {
                "linear_attention"
            }
            .to_owned()
        })
        .collect::<Vec<_>>();
    if fixture.schema_version != 1
        || fixture.semantic != "qwen3_8_flash_next_accumulated_layers4_47_cached_decode"
        || fixture.model != MODEL
        || fixture.reference.implementation
            != "source_derived_and_official_huggingface_transformers_qwen4_exp"
        || fixture.reference.transformers_version != "5.16.1"
        || config.first_layer != 4
        || config.last_layer != 47
        || config.layer_types != expected_types
        || config.ple_layer_ids != [2]
        || config.hidden_size != 2560
        || config.hc_count != 4
        || config.boundary_dtype != "BF16"
        || fixture.layers.len() != 44
        || fixture.final_outputs.len() != 2
        || sha256_file(model_lock_path)? != fixture.reference.model_lock_sha256
        || sha256_file(layer3_fixture_path)? != fixture.reference.layer3_fixture_sha256
    {
        return Err(
            "remaining decoder fixture identity or configuration is unsupported".to_owned(),
        );
    }
    let (parent_report, mut current_outputs) = verify_accumulated_layer3_fixture_with_outputs(
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
        layer3_fixture_path,
    )?;

    let mut linear_layers = 0;
    let mut full_layers = 0;
    let mut attention_bf16 = 0;
    let mut attention_f32 = 0;
    let mut attention_i64 = 0;
    let mut attention_bool = 0;
    let mut decoder_bf16 = 0;
    let mut weighted_experts = 0;
    let mut unique_experts = 0;
    let mut remaining_bytes = 0;
    let mut routes = Vec::with_capacity(44);

    for (offset, layer_fixture) in fixture.layers.iter().enumerate() {
        let layer = offset + 4;
        let expected_type = &expected_types[layer];
        if layer_fixture.layer != layer
            || &layer_fixture.layer_type != expected_type
            || layer_fixture.steps.len() != 2
        {
            return Err(format!("remaining decoder layer {layer} metadata mismatch"));
        }
        let attention_bytes =
            serde_json::to_vec(&layer_fixture.attention).map_err(|error| error.to_string())?;
        let attention_semantic = format!("qwen3_8_flash_next_layer{layer}_attention_accumulated");
        let (post_attention, attention_payload_bytes) = if expected_type == "linear_attention" {
            let (report, outputs) = verify_attention_residual_fixture_bytes_with_outputs(
                checkpoint_dir,
                model_lock_path,
                &attention_bytes,
                layer,
                &attention_semantic,
                &format!("layer_{layer}_two_token_attention_residual"),
                "qwen3_8_flash_next_remaining_linear_attention_verification",
                Some(&current_outputs),
            )?;
            linear_layers += 1;
            attention_bf16 += report.exact_bf16_capture_hashes;
            attention_f32 += report.exact_f32_capture_hashes;
            (outputs, report.tensor_payload_bytes)
        } else {
            let (report, outputs) = verify_full_attention_residual_fixture_bytes_with_outputs(
                checkpoint_dir,
                model_lock_path,
                &attention_bytes,
                &attention_semantic,
                "qwen3_8_flash_next_remaining_full_attention_verification",
                layer,
                [0, 1],
                ["initial", "cached_incremental"],
                true,
                None,
                Some(&current_outputs),
            )?;
            full_layers += 1;
            attention_bf16 += report.exact_bf16_capture_hashes;
            attention_f32 += report.exact_f32_capture_hashes;
            attention_i64 += report.exact_i64_capture_hashes;
            attention_bool += report.exact_bool_capture_hashes;
            (outputs, report.total_verified_payload_bytes)
        };
        let modes = if expected_type == "linear_attention" {
            ["initial_chunk", "cached_recurrent"]
        } else {
            ["initial", "cached_incremental"]
        };
        let decoder_bytes =
            serde_json::to_vec(&layer_fixture.decoder).map_err(|error| error.to_string())?;
        let decoder_semantic = format!("qwen3_8_flash_next_layer{layer}_complete_accumulated");
        let (decoder_report, layer_outputs) = verify_decoder_mlp_fixture_bytes_with_outputs(
            checkpoint_dir,
            model_lock_path,
            &decoder_bytes,
            layer,
            expected_type,
            &decoder_semantic,
            "qwen3_8_flash_next_remaining_decoder_verification",
            modes,
            attention_payload_bytes,
            post_attention.clone(),
        )
        .map_err(|error| format!("layer {layer} decoder verification failed: {error}"))?;
        for ordinal in 0..2 {
            let step = &layer_fixture.steps[ordinal];
            if step.ordinal != ordinal
                || step.mode != modes[ordinal]
                || step.selected_experts != decoder_report.selected_experts_by_step[ordinal]
            {
                return Err(format!(
                    "remaining decoder layer {layer} step {ordinal} mismatch"
                ));
            }
            require_capture(
                &step.captures.layer_input,
                &current_outputs[ordinal],
                &format!("layer {layer} step {ordinal} input"),
            )?;
            require_capture(
                &step.captures.post_attention,
                &post_attention[ordinal],
                &format!("layer {layer} step {ordinal} attention"),
            )?;
            require_capture(
                &step.captures.layer_output,
                &layer_outputs[ordinal],
                &format!("layer {layer} step {ordinal} output"),
            )?;
        }
        decoder_bf16 += decoder_report.exact_bf16_capture_hashes;
        weighted_experts += decoder_report.exact_weighted_expert_hashes;
        unique_experts += decoder_report.unique_experts_verified;
        remaining_bytes += decoder_report.total_verified_payload_bytes;
        routes.push(decoder_report.selected_experts_by_step);
        current_outputs = layer_outputs;
    }
    for (ordinal, output) in current_outputs.iter().enumerate() {
        require_capture(
            &fixture.final_outputs[ordinal],
            output,
            &format!("final step {ordinal}"),
        )?;
    }
    let parent_bytes = parent_report.total_verified_payload_bytes;
    Ok(AccumulatedLayers4Through47VerificationReport {
        schema_version: 1,
        semantic: "qwen3_8_flash_next_accumulated_layers4_47_verification",
        model: fixture.model,
        revision: fixture.revision,
        parent_layers_verified: vec![0, 1, 2, 3],
        layers_verified: (4..48).collect(),
        linear_layers_verified: linear_layers,
        full_attention_layers_verified: full_layers,
        steps_per_layer: 2,
        boundary_links_verified: 44 * 6 + 2,
        exact_attention_bf16_capture_hashes: attention_bf16,
        exact_attention_f32_capture_hashes: attention_f32,
        exact_attention_i64_capture_hashes: attention_i64,
        exact_attention_bool_capture_hashes: attention_bool,
        exact_decoder_bf16_capture_hashes: decoder_bf16,
        exact_weighted_expert_hashes: weighted_experts,
        layer_scoped_unique_experts_verified: unique_experts,
        parent_verified_payload_bytes: parent_bytes,
        remaining_layers_verified_payload_bytes: remaining_bytes,
        total_verified_payload_bytes: parent_bytes + remaining_bytes,
        selected_experts_by_layer_and_step: routes,
        final_output_hashes: current_outputs
            .iter()
            .map(|values| bf16_hash(values))
            .collect(),
        accepted_tokens: 0,
        performance_claim: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_has_exact_periodic_suffix_schedule() {
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../fixtures/accumulated/qwen3_8_flash_next_layers4_47.json"
        ))
        .unwrap();
        assert_eq!(fixture.layers.len(), 44);
        for (offset, layer) in fixture.layers.iter().enumerate() {
            let number = offset + 4;
            assert_eq!(layer.layer, number);
            assert_eq!(layer.layer_type == "full_attention", number % 4 == 3);
        }
    }
}
