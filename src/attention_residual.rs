use crate::deltanet::{read_tensor, run_deltanet_step, zero_deltanet_state};
use crate::expert::{bf16_hash, from_bf16, to_bf16};
use crate::hyper_connection::run_hyper_connection;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

const MODEL: &str = "Qwen/Qwen3.8-Flash-Next";
const HIDDEN: usize = 2560;
const HC_COUNT: usize = 4;
const HC_HIDDEN: usize = HIDDEN * HC_COUNT;
const CONV_DIM: usize = 10_240;
const CONV_KERNEL: usize = 4;
const V_HEADS: usize = 48;
const HEAD_DIM: usize = 128;

#[derive(Deserialize)]
struct Fixture {
    schema_version: u32,
    semantic: String,
    model: String,
    revision: String,
    reference: Reference,
    configuration: Configuration,
    case: Case,
}

#[derive(Deserialize)]
struct Reference {
    implementation: String,
    transformers_version: String,
    source: String,
    config_sha256: String,
    tensor_index_sha256: String,
    model_lock_sha256: String,
    hyper_fixture_sha256: Option<String>,
    deltanet_fixture_sha256: Option<String>,
}

#[derive(Deserialize)]
struct Configuration {
    layer: usize,
    layer_type: String,
    ple_applied: bool,
    hidden_size: usize,
    hc_count: usize,
    boundary_dtype: String,
    recurrent_state_dtype: String,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    tensors: BTreeMap<String, Tensor>,
    steps: Vec<Step>,
}

#[derive(Deserialize)]
struct Tensor {
    tensor: String,
    dtype: String,
    shape: Vec<usize>,
    shard: String,
    shard_bytes: u64,
    shard_sha256: String,
    payload_sha256: String,
}

#[derive(Deserialize)]
struct Step {
    ordinal: usize,
    mode: String,
    input_spec: InputSpec,
    captures: BTreeMap<String, Capture>,
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
struct Capture {
    dtype: String,
    shape: Vec<usize>,
    sha256: String,
}

#[derive(Deserialize)]
struct ModelLock {
    model: String,
    revision: String,
    files: Vec<LockedFile>,
}

#[derive(Deserialize)]
struct LockedFile {
    path: String,
    size: u64,
    lfs_sha256: Option<String>,
}

#[derive(Debug, Eq, PartialEq, Serialize)]
pub struct AttentionResidualVerificationReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub model: String,
    pub revision: String,
    pub layer: usize,
    pub steps_verified: usize,
    pub tensors_verified: usize,
    pub exact_bf16_capture_hashes: usize,
    pub exact_f32_capture_hashes: usize,
    pub tensor_payload_bytes: usize,
    pub accepted_tokens: usize,
    pub performance_claim: Option<String>,
}

fn is_hash(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let bytes =
        fs::read(path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn f32_hash(values: &[f32]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn make_input(spec: &InputSpec, ordinal: usize) -> Result<Vec<u16>, String> {
    let expected = [(43, 17, 263, 131, 128), (61, 29, 277, 138, 128)];
    if ordinal >= expected.len()
        || (
            spec.multiplier,
            spec.add,
            spec.modulus,
            spec.center,
            spec.divisor,
        ) != expected[ordinal]
        || spec.sparse_stride != 1
    {
        return Err(format!(
            "unsupported attention-residual input specification at step {ordinal}"
        ));
    }
    Ok((0..HC_HIDDEN)
        .map(|index| {
            let raw =
                (index as i64 * spec.multiplier + spec.add).rem_euclid(spec.modulus) - spec.center;
            to_bf16(raw as f32 / spec.divisor as f32)
        })
        .collect())
}

fn require_bf16(step: &Step, name: &str, shape: &[usize], values: &[u16]) -> Result<(), String> {
    let capture = step
        .captures
        .get(name)
        .ok_or_else(|| format!("missing attention-residual capture {name}"))?;
    let actual_hash = bf16_hash(values);
    if capture.dtype != "BF16"
        || capture.shape != shape
        || !is_hash(&capture.sha256)
        || actual_hash != capture.sha256
    {
        return Err(format!(
            "attention-residual capture mismatch at step {} {name}: expected {}, got {actual_hash}",
            step.ordinal, capture.sha256
        ));
    }
    Ok(())
}

fn require_f32(step: &Step, name: &str, shape: &[usize], values: &[f32]) -> Result<(), String> {
    let capture = step
        .captures
        .get(name)
        .ok_or_else(|| format!("missing attention-residual capture {name}"))?;
    let actual_hash = f32_hash(values);
    if capture.dtype != "F32"
        || capture.shape != shape
        || !is_hash(&capture.sha256)
        || actual_hash != capture.sha256
    {
        return Err(format!(
            "attention-residual capture mismatch at step {} {name}: expected {}, got {actual_hash}",
            step.ordinal, capture.sha256
        ));
    }
    Ok(())
}

fn expected_tensors(layer: usize) -> Vec<(String, Vec<usize>, String, String)> {
    let mut expected = vec![
        (
            "attn_hyper_connection.hc_norm",
            vec![HC_HIDDEN],
            "attn_hyper_connection.hc_norm.weight",
        ),
        (
            "attn_hyper_connection.input_mix_weight_down",
            vec![320, HC_HIDDEN],
            "attn_hyper_connection.input_mix_weight_down.weight",
        ),
        (
            "attn_hyper_connection.input_mix_weight_up",
            vec![HC_HIDDEN, 320],
            "attn_hyper_connection.input_mix_weight_up.weight",
        ),
        (
            "attn_hyper_connection.block_inject_weight",
            vec![HC_COUNT, HC_HIDDEN],
            "attn_hyper_connection.block_inject_weight.weight",
        ),
        ("linear_attn.A_log", vec![V_HEADS], "linear_attn.A_log"),
        (
            "linear_attn.conv1d.weight",
            vec![CONV_DIM, 1, CONV_KERNEL],
            "linear_attn.conv1d.weight",
        ),
        ("linear_attn.dt_bias", vec![V_HEADS], "linear_attn.dt_bias"),
        (
            "linear_attn.in_proj_a.weight",
            vec![V_HEADS, HIDDEN],
            "linear_attn.in_proj_a.weight",
        ),
        (
            "linear_attn.in_proj_b.weight",
            vec![V_HEADS, HIDDEN],
            "linear_attn.in_proj_b.weight",
        ),
        (
            "linear_attn.in_proj_qkv.weight",
            vec![CONV_DIM, HIDDEN],
            "linear_attn.in_proj_qkv.weight",
        ),
        (
            "linear_attn.in_proj_z.weight",
            vec![V_HEADS * HEAD_DIM, HIDDEN],
            "linear_attn.in_proj_z.weight",
        ),
        (
            "linear_attn.norm.weight",
            vec![HEAD_DIM],
            "linear_attn.norm.weight",
        ),
        (
            "linear_attn.out_proj.weight",
            vec![HIDDEN, V_HEADS * HEAD_DIM],
            "linear_attn.out_proj.weight",
        ),
    ];
    expected
        .drain(..)
        .map(|(key, shape, suffix)| {
            (
                key.to_owned(),
                shape,
                format!("model.language_model.layers.{layer}.{suffix}"),
                key.split_once('.').unwrap().1.to_owned(),
            )
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_attention_residual_fixture_bytes_with_outputs(
    checkpoint_dir: &Path,
    model_lock_path: &Path,
    fixture_bytes: &[u8],
    layer: usize,
    expected_semantic: &str,
    expected_case_name: &str,
    verification_semantic: &'static str,
    hidden_overrides: Option<&[Vec<u16>]>,
) -> Result<(AttentionResidualVerificationReport, Vec<Vec<u16>>), String> {
    let fixture: Fixture = serde_json::from_slice(fixture_bytes)
        .map_err(|error| format!("malformed attention-residual fixture: {error}"))?;
    let config = &fixture.configuration;
    let case = &fixture.case;
    if fixture.schema_version != 1
        || fixture.semantic != expected_semantic
        || fixture.model != MODEL
        || fixture.reference.implementation != "huggingface_transformers_qwen4_exp"
        || fixture.reference.transformers_version != "5.16.1"
        || fixture.reference.source
            != "transformers.models.qwen4_exp.modeling_qwen4_exp.Qwen4ExpTextDecoderLayer.forward"
        || config.layer != layer
        || config.layer_type != "linear_attention"
        || config.ple_applied
        || config.hidden_size != HIDDEN
        || config.hc_count != HC_COUNT
        || config.boundary_dtype != "BF16"
        || config.recurrent_state_dtype != "F32"
        || case.name != expected_case_name
        || case.tensors.len() != 13
        || case.steps.len() != 2
        || hidden_overrides.is_some_and(|values| values.len() != case.steps.len())
    {
        return Err(
            "attention-residual fixture identity or configuration is unsupported".to_owned(),
        );
    }
    if sha256_file(model_lock_path)? != fixture.reference.model_lock_sha256
        || sha256_file(&checkpoint_dir.join("config.json"))? != fixture.reference.config_sha256
        || sha256_file(&checkpoint_dir.join("model.safetensors.index.json"))?
            != fixture.reference.tensor_index_sha256
    {
        return Err("attention-residual reference identity mismatch".to_owned());
    }
    let lock: ModelLock =
        serde_json::from_slice(&fs::read(model_lock_path).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    if lock.model != fixture.model || lock.revision != fixture.revision {
        return Err("attention-residual model lock mismatch".to_owned());
    }

    let mut hyper_weights = BTreeMap::new();
    let mut deltanet_weights = BTreeMap::new();
    let mut tensor_payload_bytes = 0;
    for (key, shape, tensor_name, local_name) in expected_tensors(layer) {
        let tensor = case
            .tensors
            .get(&key)
            .ok_or_else(|| format!("missing attention-residual tensor {key}"))?;
        let matches: Vec<_> = lock
            .files
            .iter()
            .filter(|entry| entry.path == tensor.shard)
            .collect();
        if tensor.tensor != tensor_name
            || tensor.dtype != "BF16"
            || tensor.shape != shape
            || !is_hash(&tensor.shard_sha256)
            || !is_hash(&tensor.payload_sha256)
            || matches.len() != 1
            || matches[0].size != tensor.shard_bytes
            || matches[0].lfs_sha256.as_deref() != Some(tensor.shard_sha256.as_str())
            || fs::metadata(checkpoint_dir.join(&tensor.shard))
                .map_err(|error| error.to_string())?
                .len()
                != tensor.shard_bytes
        {
            return Err(format!(
                "attention-residual tensor identity mismatch for {key}"
            ));
        }
        let payload = read_tensor(&checkpoint_dir.join(&tensor.shard), &tensor.tensor, &shape)?;
        if bf16_hash(&payload) != tensor.payload_sha256 {
            return Err(format!(
                "attention-residual tensor payload mismatch for {key}"
            ));
        }
        tensor_payload_bytes += payload.len() * 2;
        if key.starts_with("attn_hyper_connection.") {
            hyper_weights.insert(local_name, payload);
        } else {
            deltanet_weights.insert(local_name, payload);
        }
    }

    let mut state = zero_deltanet_state();
    let mut composed_outputs = Vec::with_capacity(case.steps.len());
    for (ordinal, step) in case.steps.iter().enumerate() {
        if step.ordinal != ordinal
            || step.mode
                != if ordinal == 0 {
                    "initial_chunk"
                } else {
                    "cached_recurrent"
                }
            || step.captures.len() != 8
        {
            return Err(format!(
                "attention-residual step {ordinal} metadata mismatch"
            ));
        }
        let generated_input = make_input(&step.input_spec, ordinal)?;
        let hyper_input = hidden_overrides
            .map(|values| values[ordinal].clone())
            .unwrap_or(generated_input);
        if hyper_input.len() != HC_HIDDEN {
            return Err(format!(
                "attention-residual hidden override shape mismatch at step {ordinal}"
            ));
        }
        require_bf16(step, "hyper_input", &[1, 1, HC_HIDDEN], &hyper_input)?;
        let hyper = run_hyper_connection(&hyper_input, &hyper_weights)?;
        require_bf16(step, "mixed_input", &[1, 1, HIDDEN], &hyper.mixed)?;
        require_bf16(
            step,
            "injection_weights",
            &[1, 1, HC_COUNT],
            &hyper.injection_weights,
        )?;
        let attention =
            run_deltanet_step(&hyper.mixed, &deltanet_weights, &mut state, ordinal == 0)?;
        require_bf16(step, "attention_output", &[1, 1, HIDDEN], &attention.output)?;
        require_bf16(
            step,
            "convolution_state",
            &[1, CONV_DIM, CONV_KERNEL],
            &state.convolution,
        )?;
        require_f32(
            step,
            "recurrent_state",
            &[1, V_HEADS, HEAD_DIM, HEAD_DIM],
            &state.recurrent,
        )?;
        let injection_products: Vec<_> = hyper
            .injection_weights
            .iter()
            .flat_map(|weight| {
                attention
                    .output
                    .iter()
                    .map(|value| to_bf16(from_bf16(*value) * from_bf16(*weight)))
            })
            .collect();
        require_bf16(
            step,
            "injection_products",
            &[1, 1, HC_COUNT, HIDDEN],
            &injection_products,
        )?;
        let composed: Vec<_> = hyper_input
            .iter()
            .zip(&injection_products)
            .map(|(residual, injection)| to_bf16(from_bf16(*residual) + from_bf16(*injection)))
            .collect();
        require_bf16(step, "composed_output", &[1, 1, HC_HIDDEN], &composed)?;
        composed_outputs.push(composed);
    }

    Ok((
        AttentionResidualVerificationReport {
            schema_version: 1,
            semantic: verification_semantic,
            model: fixture.model,
            revision: fixture.revision,
            layer,
            steps_verified: 2,
            tensors_verified: 13,
            exact_bf16_capture_hashes: 14,
            exact_f32_capture_hashes: 2,
            tensor_payload_bytes,
            accepted_tokens: 0,
            performance_claim: None,
        },
        composed_outputs,
    ))
}

pub(crate) fn verify_attention_residual_fixture_with_outputs(
    checkpoint_dir: &Path,
    model_lock_path: &Path,
    hyper_fixture_path: &Path,
    deltanet_fixture_path: &Path,
    fixture_path: &Path,
) -> Result<(AttentionResidualVerificationReport, Vec<Vec<u16>>), String> {
    let bytes = fs::read(fixture_path).map_err(|error| error.to_string())?;
    let fixture: Fixture = serde_json::from_slice(&bytes)
        .map_err(|error| format!("malformed attention-residual fixture: {error}"))?;
    if fixture.reference.hyper_fixture_sha256.as_deref()
        != Some(sha256_file(hyper_fixture_path)?.as_str())
        || fixture.reference.deltanet_fixture_sha256.as_deref()
            != Some(sha256_file(deltanet_fixture_path)?.as_str())
    {
        return Err("attention-residual component authority mismatch".to_owned());
    }
    verify_attention_residual_fixture_bytes_with_outputs(
        checkpoint_dir,
        model_lock_path,
        &bytes,
        0,
        "qwen3_8_flash_next_layer0_attention_residual_cached_decode",
        "layer_0_two_token_attention_residual",
        "qwen3_8_flash_next_layer0_attention_residual_verification",
        None,
    )
}

pub fn verify_attention_residual_fixture(
    checkpoint_dir: &Path,
    model_lock_path: &Path,
    hyper_fixture_path: &Path,
    deltanet_fixture_path: &Path,
    fixture_path: &Path,
) -> Result<AttentionResidualVerificationReport, String> {
    verify_attention_residual_fixture_with_outputs(
        checkpoint_dir,
        model_lock_path,
        hyper_fixture_path,
        deltanet_fixture_path,
        fixture_path,
    )
    .map(|(report, _)| report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn injection_layout_repeats_attention_with_one_weight_per_stream() {
        let attention = [to_bf16(1.0), to_bf16(2.0)];
        let weights = [to_bf16(0.5), to_bf16(1.5)];
        let products: Vec<_> = weights
            .iter()
            .flat_map(|weight| {
                attention
                    .iter()
                    .map(|value| to_bf16(from_bf16(*value) * from_bf16(*weight)))
            })
            .collect();
        assert_eq!(
            products,
            vec![to_bf16(0.5), to_bf16(1.0), to_bf16(1.5), to_bf16(3.0)]
        );
    }
}
