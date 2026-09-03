use crate::deltanet::read_tensor;
use crate::expert::{add_bf16, bf16_hash, from_bf16, to_bf16};
use crate::full_attention::{
    Capture as AttentionCapture, verify_full_attention_fixture_bytes_with_overrides,
};
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

#[derive(Deserialize)]
struct Fixture {
    schema_version: u32,
    semantic: String,
    model: String,
    revision: String,
    reference: Reference,
    configuration: Configuration,
    tensors: BTreeMap<String, Tensor>,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Reference {
    implementation: String,
    transformers_version: String,
    source: Vec<String>,
    config_sha256: String,
    tensor_index_sha256: String,
    model_lock_sha256: String,
    full_attention_fixture_sha256: Option<String>,
}

#[derive(Deserialize)]
struct Configuration {
    layer: usize,
    layer_type: String,
    hidden_size: usize,
    hc_count: usize,
    boundary_dtype: String,
    active_qsa_past_length: usize,
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
struct Case {
    ordinal: usize,
    mode: String,
    position: usize,
    past_length: usize,
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

#[derive(Clone, Deserialize)]
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

#[derive(Debug, Serialize)]
pub struct FullAttentionResidualVerificationReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub model: String,
    pub revision: String,
    pub layer: usize,
    pub cases_verified: usize,
    pub exact_bf16_capture_hashes: usize,
    pub exact_f32_capture_hashes: usize,
    pub exact_i64_capture_hashes: usize,
    pub exact_bool_capture_hashes: usize,
    pub dense_tensors_verified: usize,
    pub attention_tensor_payload_bytes: usize,
    pub hyper_tensor_payload_bytes: usize,
    pub total_verified_payload_bytes: usize,
    pub synthetic_cache_bytes: usize,
    pub accepted_tokens: usize,
    pub performance_claim: Option<String>,
}

fn sha256_file(path: &Path) -> Result<String, String> {
    fs::read(path)
        .map(|bytes| format!("{:x}", Sha256::digest(bytes)))
        .map_err(|error| format!("cannot read {}: {error}", path.display()))
}

fn is_hash(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn require_bf16(case: &Case, name: &str, shape: &[usize], values: &[u16]) -> Result<(), String> {
    let capture = case
        .captures
        .get(name)
        .ok_or_else(|| format!("missing full-attention residual capture {name}"))?;
    let actual = bf16_hash(values);
    if capture.dtype != "BF16"
        || capture.shape != shape
        || !is_hash(&capture.sha256)
        || capture.sha256 != actual
    {
        return Err(format!(
            "full-attention residual mismatch at case {} {name}: expected {}, got {actual}",
            case.ordinal, capture.sha256
        ));
    }
    Ok(())
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
            "unsupported full-attention residual input specification at case {ordinal}"
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

fn expected_tensors(layer: usize) -> Vec<(String, String, Vec<usize>, Option<&'static str>)> {
    let hyper = [
        ("hc_norm", vec![HC_HIDDEN]),
        ("input_mix_weight_down", vec![320, HC_HIDDEN]),
        ("input_mix_weight_up", vec![HC_HIDDEN, 320]),
        ("block_inject_weight", vec![HC_COUNT, HC_HIDDEN]),
    ];
    let mut expected = hyper
        .into_iter()
        .map(|(local, shape)| {
            (
                format!("attn_hyper_connection.{local}"),
                format!("model.language_model.layers.{layer}.attn_hyper_connection.{local}.weight"),
                shape,
                Some(local),
            )
        })
        .collect::<Vec<_>>();
    let attention = [
        ("q_proj.weight", vec![12288, HIDDEN]),
        ("k_proj.weight", vec![512, HIDDEN]),
        ("v_proj.weight", vec![512, HIDDEN]),
        ("o_proj.weight", vec![HIDDEN, 6144]),
        ("q_norm.weight", vec![256]),
        ("k_norm.weight", vec![256]),
        ("indexer.index_qk_proj.weight", vec![640, HIDDEN]),
        ("indexer.q_layernorm.weight", vec![128]),
        ("indexer.k_layernorm.weight", vec![128]),
    ];
    expected.extend(attention.into_iter().map(|(local, shape)| {
        (
            format!("self_attn.{local}"),
            format!("model.language_model.layers.{layer}.self_attn.{local}"),
            shape,
            None,
        )
    }));
    expected
}

pub fn verify_full_attention_residual_fixture(
    checkpoint_dir: &Path,
    model_lock_path: &Path,
    full_attention_fixture_path: &Path,
    fixture_path: &Path,
) -> Result<FullAttentionResidualVerificationReport, String> {
    verify_full_attention_residual_fixture_with_outputs(
        checkpoint_dir,
        model_lock_path,
        full_attention_fixture_path,
        fixture_path,
    )
    .map(|(report, _)| report)
}

pub(crate) fn verify_full_attention_residual_fixture_with_outputs(
    checkpoint_dir: &Path,
    model_lock_path: &Path,
    full_attention_fixture_path: &Path,
    fixture_path: &Path,
) -> Result<(FullAttentionResidualVerificationReport, Vec<Vec<u16>>), String> {
    let bytes = fs::read(fixture_path).map_err(|error| format!("cannot read fixture: {error}"))?;
    let fixture: Fixture = serde_json::from_slice(&bytes)
        .map_err(|error| format!("malformed full-attention residual fixture: {error}"))?;
    if fixture.reference.full_attention_fixture_sha256.as_deref()
        != Some(sha256_file(full_attention_fixture_path)?.as_str())
    {
        return Err("full-attention residual component authority mismatch".to_owned());
    }
    let full_attention_bytes = fs::read(full_attention_fixture_path)
        .map_err(|error| format!("cannot read full-attention fixture: {error}"))?;
    verify_full_attention_residual_fixture_bytes_with_outputs(
        checkpoint_dir,
        model_lock_path,
        &bytes,
        "qwen3_8_flash_next_layer3_full_attention_residual",
        "qwen3_8_flash_next_layer3_full_attention_residual_verification",
        3,
        [0, 2080],
        ["initial", "active_qsa_pruning"],
        false,
        Some(&full_attention_bytes),
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_full_attention_residual_fixture_bytes_with_outputs(
    checkpoint_dir: &Path,
    model_lock_path: &Path,
    fixture_bytes: &[u8],
    expected_semantic: &str,
    verification_semantic: &'static str,
    layer: usize,
    past_lengths: [usize; 2],
    modes: [&str; 2],
    sequential_cache: bool,
    full_attention_fixture_bytes: Option<&[u8]>,
    hidden_overrides: Option<&[Vec<u16>]>,
) -> Result<(FullAttentionResidualVerificationReport, Vec<Vec<u16>>), String> {
    let fixture: Fixture = serde_json::from_slice(fixture_bytes)
        .map_err(|error| format!("malformed full-attention residual fixture: {error}"))?;
    let config = &fixture.configuration;
    if fixture.schema_version != 1
        || fixture.semantic != expected_semantic
        || fixture.model != MODEL
        || fixture.reference.implementation
            != "source_derived_and_official_huggingface_transformers_qwen4_exp"
        || fixture.reference.transformers_version != "5.16.1"
        || fixture.reference.source.len() != 3
        || config.layer != layer
        || config.layer_type != "full_attention"
        || config.hidden_size != HIDDEN
        || config.hc_count != HC_COUNT
        || config.boundary_dtype != "BF16"
        || config.active_qsa_past_length != past_lengths[1]
        || fixture.tensors.len() != 13
        || fixture.cases.len() != 2
        || hidden_overrides.is_some_and(|values| {
            values.len() != fixture.cases.len()
                || values.iter().any(|value| value.len() != HC_HIDDEN)
        })
    {
        return Err("full-attention residual identity or configuration is unsupported".to_owned());
    }
    if sha256_file(model_lock_path)? != fixture.reference.model_lock_sha256
        || sha256_file(&checkpoint_dir.join("config.json"))? != fixture.reference.config_sha256
        || sha256_file(&checkpoint_dir.join("model.safetensors.index.json"))?
            != fixture.reference.tensor_index_sha256
    {
        return Err("full-attention residual reference identity mismatch".to_owned());
    }
    let lock: ModelLock = serde_json::from_slice(
        &fs::read(model_lock_path).map_err(|error| format!("cannot read model lock: {error}"))?,
    )
    .map_err(|error| format!("malformed model lock: {error}"))?;
    if lock.model != fixture.model || lock.revision != fixture.revision {
        return Err("full-attention residual model lock mismatch".to_owned());
    }

    let mut hyper_weights = BTreeMap::new();
    let mut hyper_bytes = 0;
    for (key, name, shape, hyper_local) in expected_tensors(layer) {
        let tensor = fixture
            .tensors
            .get(&key)
            .ok_or_else(|| format!("missing full-attention residual tensor {key}"))?;
        let records = lock
            .files
            .iter()
            .filter(|entry| entry.path == tensor.shard)
            .collect::<Vec<_>>();
        if tensor.tensor != name
            || tensor.dtype != "BF16"
            || tensor.shape != shape
            || !is_hash(&tensor.shard_sha256)
            || !is_hash(&tensor.payload_sha256)
            || records.len() != 1
            || records[0].size != tensor.shard_bytes
            || records[0].lfs_sha256.as_deref() != Some(tensor.shard_sha256.as_str())
            || fs::metadata(checkpoint_dir.join(&tensor.shard))
                .map_err(|error| error.to_string())?
                .len()
                != tensor.shard_bytes
        {
            return Err(format!("full-attention residual tensor mismatch for {key}"));
        }
        let payload = read_tensor(&checkpoint_dir.join(&tensor.shard), &name, &shape)?;
        if bf16_hash(&payload) != tensor.payload_sha256 {
            return Err(format!(
                "full-attention residual tensor payload mismatch for {key}"
            ));
        }
        if let Some(local) = hyper_local {
            hyper_bytes += payload.len() * 2;
            hyper_weights.insert(local.to_owned(), payload);
        }
    }

    let mut attention_hidden_overrides = Vec::with_capacity(2);
    let mut attention_captures = Vec::with_capacity(2);
    let mut hyper_inputs = Vec::with_capacity(2);
    let mut injection_weights = Vec::with_capacity(2);
    for (ordinal, case) in fixture.cases.iter().enumerate() {
        let past = past_lengths[ordinal];
        if case.ordinal != ordinal
            || case.mode != modes[ordinal]
            || case.position != past
            || case.past_length != past
            || case.captures.len() != 36
        {
            return Err(format!(
                "full-attention residual case {ordinal} metadata mismatch"
            ));
        }
        let generated_input = make_input(&case.input_spec, ordinal)?;
        let input = hidden_overrides
            .map(|values| values[ordinal].clone())
            .unwrap_or(generated_input);
        require_bf16(case, "hyper_input", &[1, 1, HC_HIDDEN], &input)?;
        let hyper = run_hyper_connection(&input, &hyper_weights)?;
        require_bf16(case, "mixed_input", &[1, 1, HIDDEN], &hyper.mixed)?;
        require_bf16(
            case,
            "injection_weights",
            &[1, 1, HC_COUNT],
            &hyper.injection_weights,
        )?;
        let captures = case
            .captures
            .iter()
            .filter_map(|(name, capture)| {
                name.strip_prefix("attention.").map(|name| {
                    (
                        name.to_owned(),
                        AttentionCapture {
                            dtype: capture.dtype.clone(),
                            shape: capture.shape.clone(),
                            sha256: capture.sha256.clone(),
                        },
                    )
                })
            })
            .collect::<BTreeMap<_, _>>();
        if captures.len() != 31 {
            return Err(format!(
                "full-attention residual case {ordinal} attention captures missing"
            ));
        }
        attention_hidden_overrides.push(hyper.mixed);
        attention_captures.push(captures);
        hyper_inputs.push(input);
        injection_weights.push(hyper.injection_weights);
    }

    let synthesized_attention_bytes;
    let attention_fixture_bytes = if let Some(bytes) = full_attention_fixture_bytes {
        bytes
    } else {
        let raw: serde_json::Value = serde_json::from_slice(fixture_bytes)
            .map_err(|error| format!("malformed embedded attention fixture: {error}"))?;
        let mut tensors = serde_json::Map::new();
        for (name, record) in raw["tensors"]
            .as_object()
            .ok_or("embedded attention tensors are missing")?
        {
            if let Some(local) = name.strip_prefix("self_attn.") {
                tensors.insert(local.to_owned(), record.clone());
            }
        }
        let cases = raw["cases"]
            .as_array()
            .ok_or("embedded attention cases are missing")?
            .iter()
            .map(|case| {
                let mut captures = serde_json::Map::new();
                for (name, capture) in case["captures"].as_object().unwrap() {
                    if let Some(local) = name.strip_prefix("attention.") {
                        captures.insert(local.to_owned(), capture.clone());
                    }
                }
                serde_json::json!({
                    "ordinal": case["ordinal"],
                    "mode": case["mode"],
                    "position": case["position"],
                    "past_length": case["past_length"],
                    "input_spec": case["input_spec"],
                    "state_specs": {},
                    "captures": captures,
                })
            })
            .collect::<Vec<_>>();
        synthesized_attention_bytes = serde_json::to_vec(&serde_json::json!({
            "schema_version": 1,
            "semantic": "qwen3_8_flash_next_layer3_full_attention_embedded",
            "model": raw["model"],
            "revision": raw["revision"],
            "reference": {
                "implementation": "source_derived_and_official_huggingface_transformers_qwen4_exp",
                "transformers_version": raw["reference"]["transformers_version"],
                "source": [
                    "transformers.models.qwen4_exp.modeling_qwen4_exp.Qwen4ExpTextAttention.forward",
                    "transformers.models.qwen4_exp.modeling_qwen4_exp.Qwen4ExpTextQSAIndexer.forward"
                ],
                "config_sha256": raw["reference"]["config_sha256"],
                "tensor_index_sha256": raw["reference"]["tensor_index_sha256"],
                "model_lock_sha256": raw["reference"]["model_lock_sha256"]
            },
            "configuration": {
                "layer": layer,
                "hidden_size": 2560,
                "attention_heads": 24,
                "kv_heads": 2,
                "head_dim": 256,
                "rotary_dim": 64,
                "rope_theta": 10000000,
                "mrope_section": [11, 11, 10],
                "indexer_heads": 4,
                "indexer_kv_heads": 1,
                "indexer_head_dim": 128,
                "indexer_budget": 2048,
                "indexer_compress_ratio": 4,
                "boundary_dtype": "BF16"
            },
            "tensors": tensors,
            "cases": cases
        }))
        .map_err(|error| error.to_string())?;
        &synthesized_attention_bytes
    };
    let (attention_report, attention_outputs) = verify_full_attention_fixture_bytes_with_overrides(
        checkpoint_dir,
        model_lock_path,
        attention_fixture_bytes,
        if full_attention_fixture_bytes.is_some() {
            "qwen3_8_flash_next_layer3_full_attention_qsa"
        } else {
            "qwen3_8_flash_next_layer3_full_attention_embedded"
        },
        "qwen3_8_flash_next_layer3_full_attention_embedded_verification",
        layer,
        past_lengths,
        modes,
        sequential_cache,
        Some(&attention_captures),
        Some(&attention_hidden_overrides),
    )?;
    let mut composed_outputs = Vec::with_capacity(2);
    for ordinal in 0..2 {
        let case = &fixture.cases[ordinal];
        let injection = injection_weights[ordinal]
            .iter()
            .flat_map(|weight| {
                attention_outputs[ordinal]
                    .iter()
                    .map(|value| to_bf16(from_bf16(*value) * from_bf16(*weight)))
            })
            .collect::<Vec<_>>();
        require_bf16(
            case,
            "injection_products",
            &[1, 1, HC_COUNT, HIDDEN],
            &injection,
        )?;
        let composed = hyper_inputs[ordinal]
            .iter()
            .zip(&injection)
            .map(|(preserved, injected)| add_bf16(*preserved, *injected))
            .collect::<Vec<_>>();
        require_bf16(case, "composed_output", &[1, 1, HC_HIDDEN], &composed)?;
        composed_outputs.push(composed);
    }

    let report = FullAttentionResidualVerificationReport {
        schema_version: 1,
        semantic: verification_semantic,
        model: fixture.model,
        revision: fixture.revision,
        layer,
        cases_verified: 2,
        exact_bf16_capture_hashes: attention_report.exact_bf16_capture_hashes + 10,
        exact_f32_capture_hashes: attention_report.exact_f32_capture_hashes,
        exact_i64_capture_hashes: attention_report.exact_i64_capture_hashes,
        exact_bool_capture_hashes: attention_report.exact_bool_capture_hashes,
        dense_tensors_verified: 13,
        attention_tensor_payload_bytes: attention_report.dense_tensor_payload_bytes,
        hyper_tensor_payload_bytes: hyper_bytes,
        total_verified_payload_bytes: attention_report.dense_tensor_payload_bytes + hyper_bytes,
        synthetic_cache_bytes: attention_report.synthetic_cache_bytes,
        accepted_tokens: 0,
        performance_claim: None,
    };
    Ok((report, composed_outputs))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn residual_injection_repeats_attention_once_per_stream() {
        let attention = [to_bf16(2.0), to_bf16(-3.0)];
        let weights = [to_bf16(0.5), to_bf16(-1.0)];
        let injection = weights
            .iter()
            .flat_map(|weight| {
                attention
                    .iter()
                    .map(|value| to_bf16(from_bf16(*value) * from_bf16(*weight)))
            })
            .collect::<Vec<_>>();
        assert_eq!(
            injection,
            [to_bf16(1.0), to_bf16(-1.5), to_bf16(-2.0), to_bf16(3.0)]
        );
    }
}
