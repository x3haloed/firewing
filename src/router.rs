use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

const MODEL: &str = "Qwen/Qwen3.8-Flash-Next";

#[derive(Deserialize)]
struct Fixture {
    schema_version: u32,
    semantic: String,
    model: String,
    revision: String,
    reference: Reference,
    configuration: Configuration,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Reference {
    implementation: String,
    transformers_version: String,
    source: String,
    config_sha256: String,
    tensor_index_sha256: String,
    model_lock_sha256: String,
}

#[derive(Deserialize)]
struct Configuration {
    hidden_size: usize,
    num_experts: usize,
    top_k: usize,
    norm_topk_prob: bool,
    input_dtype: String,
    weight_dtype: String,
    router_logits_dtype: String,
    softmax_dtype: String,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    layer: usize,
    tensor: String,
    shard: String,
    shard_bytes: u64,
    shard_sha256: String,
    weight_payload_sha256: String,
    input_spec: InputSpec,
    input_bf16_sha256: String,
    selected_experts: Vec<usize>,
    selected_logits_bf16: Vec<f32>,
    normalized_scores_bf16: Vec<f32>,
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
pub struct RouterCaseReport {
    pub name: String,
    pub layer: usize,
    pub selected_experts: Vec<usize>,
    pub maximum_selected_logit_absolute_error: f32,
    pub maximum_normalized_score_absolute_error: f32,
    pub weight_payload_bytes: usize,
}

#[derive(Debug, Serialize)]
pub struct RouterVerificationReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub model: String,
    pub revision: String,
    pub cases_verified: usize,
    pub exact_selected_expert_lists: usize,
    pub maximum_selected_logit_absolute_error: f32,
    pub maximum_normalized_score_absolute_error: f32,
    pub cases: Vec<RouterCaseReport>,
    pub accepted_tokens: usize,
    pub performance_claim: Option<String>,
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = File::open(path).map_err(|error| error.to_string())?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn is_hash(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn from_bf16(value: u16) -> f32 {
    f32::from_bits((value as u32) << 16)
}

fn as_bytes(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn make_hidden(size: usize, spec: &InputSpec) -> Result<Vec<u16>, String> {
    if spec.modulus <= 0 || spec.divisor <= 0 || spec.sparse_stride == 0 {
        return Err("invalid router input specification".to_owned());
    }
    (0..size)
        .map(|index| {
            let affine = (index as i64)
                .checked_mul(spec.multiplier)
                .and_then(|value| value.checked_add(spec.add))
                .ok_or("router input overflow")?;
            let raw = affine.rem_euclid(spec.modulus) - spec.center;
            let value = if index % spec.sparse_stride == 0 {
                raw as f32 / spec.divisor as f32
            } else {
                0.0
            };
            Ok(to_bf16(value))
        })
        .collect()
}

fn read_weight(
    path: &Path,
    tensor: &str,
    experts: usize,
    width: usize,
) -> Result<Vec<u16>, String> {
    let mut file = File::open(path).map_err(|error| error.to_string())?;
    let mut raw = [0_u8; 8];
    file.read_exact(&mut raw)
        .map_err(|error| error.to_string())?;
    let header_len = u64::from_le_bytes(raw);
    if header_len == 0 || header_len > 16 * 1024 * 1024 {
        return Err("invalid safetensors header length".to_owned());
    }
    let mut header = vec![0; header_len as usize];
    file.read_exact(&mut header)
        .map_err(|error| error.to_string())?;
    let header: Value = serde_json::from_slice(&header).map_err(|error| error.to_string())?;
    let item = header.get(tensor).ok_or("router tensor is missing")?;
    let shape = item
        .get("shape")
        .and_then(Value::as_array)
        .ok_or("router shape is missing")?;
    if item.get("dtype").and_then(Value::as_str) != Some("BF16")
        || shape.len() != 2
        || shape[0].as_u64() != Some(experts as u64)
        || shape[1].as_u64() != Some(width as u64)
    {
        return Err("unsupported router tensor dtype or shape".to_owned());
    }
    let offsets = item
        .get("data_offsets")
        .and_then(Value::as_array)
        .ok_or("router offsets are missing")?;
    let start = offsets
        .first()
        .and_then(Value::as_u64)
        .ok_or("invalid router offset")?;
    let end = offsets
        .get(1)
        .and_then(Value::as_u64)
        .ok_or("invalid router offset")?;
    let count = experts.checked_mul(width).ok_or("router size overflow")?;
    if end.checked_sub(start) != Some((count * 2) as u64) {
        return Err("router byte count mismatch".to_owned());
    }
    file.seek(SeekFrom::Start(8 + header_len + start))
        .map_err(|error| error.to_string())?;
    let mut payload = vec![0; count * 2];
    file.read_exact(&mut payload)
        .map_err(|error| error.to_string())?;
    Ok(payload
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect())
}

fn route(
    weight: &[u16],
    hidden: &[u16],
    experts: usize,
    width: usize,
    top_k: usize,
) -> (Vec<usize>, Vec<f32>, Vec<f32>) {
    let hidden: Vec<_> = hidden.iter().map(|value| from_bf16(*value)).collect();
    let logits: Vec<_> = weight
        .chunks_exact(width)
        .take(experts)
        .map(|row| {
            let sum = row
                .iter()
                .zip(&hidden)
                .fold(0.0_f32, |sum, (weight, input)| {
                    sum + from_bf16(*weight) * input
                });
            from_bf16(to_bf16(sum))
        })
        .collect();
    let mut indices: Vec<_> = (0..experts).collect();
    indices.sort_by(|left, right| {
        logits[*right]
            .total_cmp(&logits[*left])
            .then(left.cmp(right))
    });
    indices.truncate(top_k);
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exponentials: Vec<_> = logits
        .iter()
        .map(|value| (*value - maximum).exp())
        .collect();
    let denominator: f32 = exponentials.iter().sum();
    let probabilities: Vec<_> = exponentials
        .iter()
        .map(|value| value / denominator)
        .collect();
    let selected_sum: f32 = indices.iter().map(|index| probabilities[*index]).sum();
    let scores = indices
        .iter()
        .map(|index| from_bf16(to_bf16(probabilities[*index] / selected_sum)))
        .collect();
    let selected_logits = indices.iter().map(|index| logits[*index]).collect();
    (indices, selected_logits, scores)
}

pub fn verify_router_fixture(
    checkpoint_dir: &Path,
    model_lock_path: &Path,
    fixture_path: &Path,
) -> Result<RouterVerificationReport, String> {
    let fixture: Fixture =
        serde_json::from_slice(&fs::read(fixture_path).map_err(|error| error.to_string())?)
            .map_err(|error| format!("malformed router fixture: {error}"))?;
    let config = &fixture.configuration;
    if fixture.schema_version != 1
        || fixture.semantic != "qwen3_8_flash_next_real_top10_router"
        || fixture.model != MODEL
        || fixture.reference.implementation != "huggingface_transformers_qwen4_exp"
        || fixture.reference.transformers_version != "5.16.1"
        || fixture.reference.source
            != "transformers.models.qwen4_exp.modeling_qwen4_exp.Qwen4ExpTextTopKRouter"
        || config.hidden_size != 2560
        || config.num_experts != 512
        || config.top_k != 10
        || !config.norm_topk_prob
        || [
            &config.input_dtype,
            &config.weight_dtype,
            &config.router_logits_dtype,
        ] != ["BF16", "BF16", "BF16"]
        || config.softmax_dtype != "F32"
        || fixture.cases.len() != 3
    {
        return Err("router fixture identity or configuration is unsupported".to_owned());
    }
    if sha256_file(model_lock_path)? != fixture.reference.model_lock_sha256
        || sha256_file(&checkpoint_dir.join("config.json"))? != fixture.reference.config_sha256
        || sha256_file(&checkpoint_dir.join("model.safetensors.index.json"))?
            != fixture.reference.tensor_index_sha256
    {
        return Err("router reference identity mismatch".to_owned());
    }
    let lock: ModelLock =
        serde_json::from_slice(&fs::read(model_lock_path).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    if lock.model != fixture.model || lock.revision != fixture.revision {
        return Err("router model lock mismatch".to_owned());
    }
    let mut reports = Vec::new();
    for case in &fixture.cases {
        let locked: Vec<_> = lock
            .files
            .iter()
            .filter(|file| file.path == case.shard)
            .collect();
        if case.name != format!("layer_{}_affine_mod", case.layer)
            || case.tensor != format!("model.language_model.layers.{}.mlp.gate.weight", case.layer)
            || ![
                &case.shard_sha256,
                &case.weight_payload_sha256,
                &case.input_bf16_sha256,
            ]
            .iter()
            .all(|value| is_hash(value))
            || locked.len() != 1
            || locked[0].size != case.shard_bytes
            || locked[0].lfs_sha256.as_deref() != Some(case.shard_sha256.as_str())
            || fs::metadata(checkpoint_dir.join(&case.shard))
                .map_err(|error| error.to_string())?
                .len()
                != case.shard_bytes
            || case.selected_experts.len() != config.top_k
            || case.selected_logits_bf16.len() != config.top_k
            || case.normalized_scores_bf16.len() != config.top_k
        {
            return Err(format!("router case {} metadata mismatch", case.name));
        }
        let weight = read_weight(
            &checkpoint_dir.join(&case.shard),
            &case.tensor,
            config.num_experts,
            config.hidden_size,
        )?;
        if format!("{:x}", Sha256::digest(as_bytes(&weight))) != case.weight_payload_sha256 {
            return Err(format!("router case {} weight hash mismatch", case.name));
        }
        let hidden = make_hidden(config.hidden_size, &case.input_spec)?;
        if format!("{:x}", Sha256::digest(as_bytes(&hidden))) != case.input_bf16_sha256 {
            return Err(format!("router case {} input hash mismatch", case.name));
        }
        let (indices, logits, scores) = route(
            &weight,
            &hidden,
            config.num_experts,
            config.hidden_size,
            config.top_k,
        );
        if indices != case.selected_experts {
            return Err(format!(
                "router case {} expert mismatch: expected {:?}, got {:?}",
                case.name, case.selected_experts, indices
            ));
        }
        let logit_error = logits
            .iter()
            .zip(&case.selected_logits_bf16)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        let score_error = scores
            .iter()
            .zip(&case.normalized_scores_bf16)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        if logit_error > 0.00390625 || score_error > 0.001953125 {
            return Err(format!(
                "router case {} tolerance exceeded: logits={logit_error}, scores={score_error}",
                case.name
            ));
        }
        reports.push(RouterCaseReport {
            name: case.name.clone(),
            layer: case.layer,
            selected_experts: indices,
            maximum_selected_logit_absolute_error: logit_error,
            maximum_normalized_score_absolute_error: score_error,
            weight_payload_bytes: weight.len() * 2,
        });
    }
    Ok(RouterVerificationReport {
        schema_version: 1,
        semantic: "qwen3_8_flash_next_real_top10_router_verification",
        model: fixture.model,
        revision: fixture.revision,
        cases_verified: reports.len(),
        exact_selected_expert_lists: reports.len(),
        maximum_selected_logit_absolute_error: reports
            .iter()
            .map(|case| case.maximum_selected_logit_absolute_error)
            .fold(0.0, f32::max),
        maximum_normalized_score_absolute_error: reports
            .iter()
            .map(|case| case.maximum_normalized_score_absolute_error)
            .fold(0.0, f32::max),
        cases: reports,
        accepted_tokens: 0,
        performance_claim: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_conversion_matches_known_value() {
        assert_eq!(from_bf16(to_bf16(0.1)), 0.100_097_656);
    }

    #[test]
    fn committed_input_specs_match_hashes() {
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../fixtures/router/qwen3_8_flash_next_real.json"
        ))
        .expect("fixture parses");
        for case in fixture.cases {
            let hidden = make_hidden(2560, &case.input_spec).expect("input builds");
            assert_eq!(
                format!("{:x}", Sha256::digest(as_bytes(&hidden))),
                case.input_bf16_sha256
            );
        }
    }
}
