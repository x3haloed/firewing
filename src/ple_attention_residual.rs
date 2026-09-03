use crate::deltanet::{read_tensor, run_deltanet_step, zero_deltanet_state};
use crate::expert::{bf16_hash, from_bf16, to_bf16};
use crate::hyper_connection::run_hyper_connection;
use crate::ple::verify_ple_fixture_with_outputs;
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
    ple_fixture_sha256: String,
    ngram_fixture_sha256: String,
    ngram_row_fixture_sha256: String,
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
    token_id: i64,
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
pub struct PleAttentionResidualVerificationReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub model: String,
    pub revision: String,
    pub layer: usize,
    pub steps_verified: usize,
    pub ple_rows_verified: usize,
    pub tensors_verified: usize,
    pub exact_bf16_capture_hashes: usize,
    pub exact_f32_capture_hashes: usize,
    pub exact_i64_capture_hashes: usize,
    pub ple_verified_payload_bytes: usize,
    pub attention_tensor_payload_bytes: usize,
    pub total_verified_payload_bytes: usize,
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
        return Err(format!("unsupported PLE-attention input at step {ordinal}"));
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
        .ok_or_else(|| format!("missing PLE-attention capture {name}"))?;
    let actual = bf16_hash(values);
    if capture.dtype != "BF16"
        || capture.shape != shape
        || !is_hash(&capture.sha256)
        || capture.sha256 != actual
    {
        return Err(format!(
            "PLE-attention capture mismatch at step {} {name}: expected {}, got {actual}",
            step.ordinal, capture.sha256
        ));
    }
    Ok(())
}

fn require_f32(step: &Step, name: &str, shape: &[usize], values: &[f32]) -> Result<(), String> {
    let capture = step
        .captures
        .get(name)
        .ok_or_else(|| format!("missing PLE-attention capture {name}"))?;
    let actual = f32_hash(values);
    if capture.dtype != "F32"
        || capture.shape != shape
        || !is_hash(&capture.sha256)
        || capture.sha256 != actual
    {
        return Err(format!(
            "PLE-attention capture mismatch at step {} {name}: expected {}, got {actual}",
            step.ordinal, capture.sha256
        ));
    }
    Ok(())
}

fn expected_tensors() -> Vec<(String, Vec<usize>, String, String)> {
    let base = vec![
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
    base.into_iter()
        .map(|(key, shape, suffix)| {
            (
                key.to_owned(),
                shape,
                format!("model.language_model.layers.1.{suffix}"),
                key.split_once('.').unwrap().1.to_owned(),
            )
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_ple_attention_residual_fixture_with_outputs(
    checkpoint_dir: &Path,
    model_lock_path: &Path,
    ngram_fixture_path: &Path,
    ngram_row_fixture_path: &Path,
    ple_fixture_path: &Path,
    fixture_path: &Path,
) -> Result<(PleAttentionResidualVerificationReport, Vec<Vec<u16>>), String> {
    let fixture: Fixture =
        serde_json::from_slice(&fs::read(fixture_path).map_err(|error| error.to_string())?)
            .map_err(|error| format!("malformed PLE-attention fixture: {error}"))?;
    let config = &fixture.configuration;
    let case = &fixture.case;
    if fixture.schema_version != 1
        || fixture.semantic != "qwen3_8_flash_next_layer1_ple_attention_residual_cached_decode"
        || fixture.model != MODEL
        || fixture.reference.implementation != "huggingface_transformers_qwen4_exp"
        || fixture.reference.transformers_version != "5.16.1"
        || fixture.reference.source
            != "transformers.models.qwen4_exp.modeling_qwen4_exp.Qwen4ExpTextDecoderLayer.forward"
        || config.layer != 1
        || config.layer_type != "linear_attention"
        || !config.ple_applied
        || config.hidden_size != HIDDEN
        || config.hc_count != HC_COUNT
        || config.boundary_dtype != "BF16"
        || config.recurrent_state_dtype != "F32"
        || case.name != "layer_1_two_token_ple_attention_residual"
        || case.tensors.len() != 13
        || case.steps.len() != 2
    {
        return Err("PLE-attention fixture identity or configuration is unsupported".to_owned());
    }
    if sha256_file(model_lock_path)? != fixture.reference.model_lock_sha256
        || sha256_file(&checkpoint_dir.join("config.json"))? != fixture.reference.config_sha256
        || sha256_file(&checkpoint_dir.join("model.safetensors.index.json"))?
            != fixture.reference.tensor_index_sha256
        || sha256_file(ple_fixture_path)? != fixture.reference.ple_fixture_sha256
        || sha256_file(ngram_fixture_path)? != fixture.reference.ngram_fixture_sha256
        || sha256_file(ngram_row_fixture_path)? != fixture.reference.ngram_row_fixture_sha256
    {
        return Err("PLE-attention reference identity mismatch".to_owned());
    }
    let (ple_report, ple_outputs) = verify_ple_fixture_with_outputs(
        checkpoint_dir,
        model_lock_path,
        ngram_fixture_path,
        ngram_row_fixture_path,
        ple_fixture_path,
    )?;
    let lock: ModelLock =
        serde_json::from_slice(&fs::read(model_lock_path).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    if lock.model != fixture.model || lock.revision != fixture.revision {
        return Err("PLE-attention model lock mismatch".to_owned());
    }

    let mut hyper_weights = BTreeMap::new();
    let mut deltanet_weights = BTreeMap::new();
    let mut attention_tensor_payload_bytes = 0;
    for (key, shape, tensor_name, local_name) in expected_tensors() {
        let tensor = case
            .tensors
            .get(&key)
            .ok_or_else(|| format!("missing PLE-attention tensor {key}"))?;
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
            return Err(format!("PLE-attention tensor identity mismatch for {key}"));
        }
        let payload = read_tensor(&checkpoint_dir.join(&tensor.shard), &tensor.tensor, &shape)?;
        if bf16_hash(&payload) != tensor.payload_sha256 {
            return Err(format!("PLE-attention tensor payload mismatch for {key}"));
        }
        attention_tensor_payload_bytes += payload.len() * 2;
        if key.starts_with("attn_hyper_connection.") {
            hyper_weights.insert(local_name, payload);
        } else {
            deltanet_weights.insert(local_name, payload);
        }
    }

    let mut state = zero_deltanet_state();
    let mut composed_outputs = Vec::with_capacity(case.steps.len());
    for (ordinal, (step, ple_output)) in case.steps.iter().zip(&ple_outputs).enumerate() {
        if step.ordinal != ordinal
            || step.mode
                != if ordinal == 0 {
                    "initial_chunk"
                } else {
                    "cached_recurrent"
                }
            || step.token_id != [42, 43][ordinal]
            || step.captures.len() != 10
        {
            return Err(format!("PLE-attention step {ordinal} metadata mismatch"));
        }
        let hidden = make_input(&step.input_spec, ordinal)?;
        require_bf16(step, "hidden_states", &[1, 1, HC_HIDDEN], &hidden)?;
        require_bf16(step, "ple_output", &[1, 1, HC_HIDDEN], ple_output)?;
        let post_ple: Vec<_> = hidden
            .iter()
            .zip(ple_output)
            .map(|(left, right)| to_bf16(from_bf16(*left) + from_bf16(*right)))
            .collect();
        require_bf16(step, "post_ple", &[1, 1, HC_HIDDEN], &post_ple)?;
        let hyper = run_hyper_connection(&post_ple, &hyper_weights)?;
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
        let composed: Vec<_> = post_ple
            .iter()
            .zip(&injection_products)
            .map(|(residual, injection)| to_bf16(from_bf16(*residual) + from_bf16(*injection)))
            .collect();
        require_bf16(step, "composed_output", &[1, 1, HC_HIDDEN], &composed)?;
        composed_outputs.push(composed);
    }

    let ple_verified_payload_bytes = ple_report.total_verified_payload_bytes;
    Ok((
        PleAttentionResidualVerificationReport {
            schema_version: 1,
            semantic: "qwen3_8_flash_next_layer1_ple_attention_residual_verification",
            model: fixture.model,
            revision: fixture.revision,
            layer: 1,
            steps_verified: 2,
            ple_rows_verified: ple_report.rows_verified,
            tensors_verified: ple_report.dense_tensors_verified + 13,
            exact_bf16_capture_hashes: ple_report.exact_bf16_capture_hashes + 18,
            exact_f32_capture_hashes: 2,
            exact_i64_capture_hashes: ple_report.exact_i64_capture_hashes,
            ple_verified_payload_bytes,
            attention_tensor_payload_bytes,
            total_verified_payload_bytes: ple_verified_payload_bytes
                + attention_tensor_payload_bytes,
            accepted_tokens: 0,
            performance_claim: None,
        },
        composed_outputs,
    ))
}

pub fn verify_ple_attention_residual_fixture(
    checkpoint_dir: &Path,
    model_lock_path: &Path,
    ngram_fixture_path: &Path,
    ngram_row_fixture_path: &Path,
    ple_fixture_path: &Path,
    fixture_path: &Path,
) -> Result<PleAttentionResidualVerificationReport, String> {
    verify_ple_attention_residual_fixture_with_outputs(
        checkpoint_dir,
        model_lock_path,
        ngram_fixture_path,
        ngram_row_fixture_path,
        ple_fixture_path,
        fixture_path,
    )
    .map(|(report, _)| report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn post_ple_addition_rounds_each_value_to_bf16() {
        let hidden = [to_bf16(1.0), to_bf16(2.0)];
        let ple = [to_bf16(0.00390625), to_bf16(-0.0078125)];
        let result: Vec<_> = hidden
            .iter()
            .zip(&ple)
            .map(|(left, right)| to_bf16(from_bf16(*left) + from_bf16(*right)))
            .collect();
        assert_eq!(result, vec![to_bf16(1.0), to_bf16(1.9921875)]);
    }
}
