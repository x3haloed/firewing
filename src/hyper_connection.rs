use crate::expert::{bf16_hash, from_bf16, linear_bf16, sigmoid_bf16, to_bf16};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

const MODEL: &str = "Qwen/Qwen3.8-Flash-Next";
const HIDDEN: usize = 2560;
const HC_COUNT: usize = 4;
const HC_LOWRANK: usize = 320;
const HC_HIDDEN: usize = HIDDEN * HC_COUNT;

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
    rms_source: String,
    config_sha256: String,
    tensor_index_sha256: String,
    model_lock_sha256: String,
}

#[derive(Deserialize)]
struct Configuration {
    hidden_size: usize,
    hc_count: usize,
    hc_lowrank: usize,
    rms_norm_eps: f32,
    boundary_dtype: String,
    use_combine: bool,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    layer: usize,
    kind: String,
    input_spec: InputSpec,
    tensors: BTreeMap<String, Tensor>,
    expected_bf16_sha256: BTreeMap<String, String>,
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
struct Tensor {
    tensor: String,
    shape: Vec<usize>,
    shard: String,
    shard_bytes: u64,
    shard_sha256: String,
    payload_sha256: String,
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
pub struct HyperConnectionVerificationReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub model: String,
    pub revision: String,
    pub layer: usize,
    pub kind: String,
    pub tensors_verified: usize,
    pub exact_capture_hashes: usize,
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

fn make_input(spec: &InputSpec) -> Result<Vec<u16>, String> {
    if spec.multiplier != 43
        || spec.add != 17
        || spec.modulus != 263
        || spec.center != 131
        || spec.divisor != 128
        || spec.sparse_stride != 1
    {
        return Err("unsupported hyper-connection input specification".to_owned());
    }
    (0..HC_HIDDEN)
        .map(|index| {
            let raw = ((index as i64 * spec.multiplier + spec.add).rem_euclid(spec.modulus)
                - spec.center) as f32;
            Ok(to_bf16(raw / spec.divisor as f32))
        })
        .collect()
}

fn read_tensor(path: &Path, tensor: &str, expected_shape: &[usize]) -> Result<Vec<u16>, String> {
    let mut file =
        File::open(path).map_err(|error| format!("cannot open {}: {error}", path.display()))?;
    let mut prefix = [0_u8; 8];
    file.read_exact(&mut prefix)
        .map_err(|error| error.to_string())?;
    let header_bytes = u64::from_le_bytes(prefix);
    if header_bytes == 0 || header_bytes > 16 * 1024 * 1024 {
        return Err("invalid safetensors header length".to_owned());
    }
    let mut header = vec![0_u8; header_bytes as usize];
    file.read_exact(&mut header)
        .map_err(|error| error.to_string())?;
    let header: Value = serde_json::from_slice(&header).map_err(|error| error.to_string())?;
    let item = header
        .get(tensor)
        .ok_or_else(|| format!("missing tensor {tensor}"))?;
    let shape = item
        .get("shape")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("tensor {tensor} has no shape"))?;
    if item.get("dtype").and_then(Value::as_str) != Some("BF16")
        || shape.len() != expected_shape.len()
        || !shape
            .iter()
            .zip(expected_shape)
            .all(|(actual, expected)| actual.as_u64() == Some(*expected as u64))
    {
        return Err(format!("tensor {tensor} has unsupported dtype or shape"));
    }
    let offsets = item
        .get("data_offsets")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("tensor {tensor} has no offsets"))?;
    let start = offsets
        .first()
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("tensor {tensor} has invalid offsets"))?;
    let end = offsets
        .get(1)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("tensor {tensor} has invalid offsets"))?;
    let count = expected_shape
        .iter()
        .try_fold(1_usize, |total, value| total.checked_mul(*value))
        .ok_or_else(|| "tensor element count overflow".to_owned())?;
    if end.checked_sub(start) != Some((count * 2) as u64) {
        return Err(format!("tensor {tensor} byte count mismatch"));
    }
    let absolute = 8_u64
        .checked_add(header_bytes)
        .and_then(|value| value.checked_add(start))
        .ok_or_else(|| "tensor offset overflow".to_owned())?;
    file.seek(SeekFrom::Start(absolute))
        .map_err(|error| error.to_string())?;
    let mut payload = vec![0_u8; count * 2];
    file.read_exact(&mut payload)
        .map_err(|error| error.to_string())?;
    Ok(payload
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect())
}

// Matches PyTorch's aarch64 contiguous F32 inner-reduction cascade. Adapted
// from Prismwing's independently fixture-tested implementation pinned in
// docs/SOURCES.md.
#[allow(clippy::needless_range_loop)]
fn pytorch_inner_square_sum(values: &[f32]) -> f32 {
    const LANES: usize = 4;
    const INTERLEAVE: usize = 4;
    const LEVELS: usize = 4;
    let vector_count = values.len() / LANES;
    let cascade_size = vector_count / INTERLEAVE;
    let ceil_log2 = if cascade_size <= 1 {
        0
    } else {
        usize::BITS as usize - (cascade_size - 1).leading_zeros() as usize
    };
    let level_power = 4_usize.max(ceil_log2 / LEVELS);
    let level_step = 1_usize << level_power;
    let level_mask = level_step - 1;
    let mut accumulators = [[[0.0_f32; LANES]; INTERLEAVE]; LEVELS];
    let mut index = 0;
    while index + level_step <= cascade_size {
        for _ in 0..level_step {
            for register in 0..INTERLEAVE {
                let base = (index * INTERLEAVE + register) * LANES;
                for lane in 0..LANES {
                    accumulators[0][register][lane] += values[base + lane] * values[base + lane];
                }
            }
            index += 1;
        }
        for level in 1..LEVELS {
            for register in 0..INTERLEAVE {
                for lane in 0..LANES {
                    accumulators[level][register][lane] += accumulators[level - 1][register][lane];
                    accumulators[level - 1][register][lane] = 0.0;
                }
            }
            if index & (level_mask << (level * level_power)) != 0 {
                break;
            }
        }
    }
    while index < cascade_size {
        for register in 0..INTERLEAVE {
            let base = (index * INTERLEAVE + register) * LANES;
            for lane in 0..LANES {
                accumulators[0][register][lane] += values[base + lane] * values[base + lane];
            }
        }
        index += 1;
    }
    for level in 1..LEVELS {
        for register in 0..INTERLEAVE {
            for lane in 0..LANES {
                accumulators[0][register][lane] += accumulators[level][register][lane];
            }
        }
    }
    for vector in cascade_size * INTERLEAVE..vector_count {
        for lane in 0..LANES {
            let value = values[vector * LANES + lane];
            accumulators[0][0][lane] += value * value;
        }
    }
    for register in 1..INTERLEAVE {
        for lane in 0..LANES {
            accumulators[0][0][lane] += accumulators[0][register][lane];
        }
    }
    let mut result = values[vector_count * LANES..]
        .iter()
        .fold(0.0_f32, |sum, value| sum + value * value);
    for lane in 0..LANES {
        result += accumulators[0][0][lane];
    }
    result
}

fn grouped_rms(input: &[u16], weight: &[u16], epsilon: f32) -> Vec<u16> {
    let mut output = Vec::with_capacity(HC_HIDDEN);
    for group in 0..HC_COUNT {
        let input = &input[group * HIDDEN..(group + 1) * HIDDEN];
        let float: Vec<_> = input.iter().map(|value| from_bf16(*value)).collect();
        let inverse = (pytorch_inner_square_sum(&float) / HIDDEN as f32 + epsilon)
            .sqrt()
            .recip();
        output.extend(
            input
                .iter()
                .zip(&weight[group * HIDDEN..(group + 1) * HIDDEN])
                .map(|(value, weight)| {
                    to_bf16(from_bf16(*value) * inverse * (1.0 + from_bf16(*weight)))
                }),
        );
    }
    output
}

fn silu_bf16(values: &[u16]) -> Vec<u16> {
    values
        .iter()
        .map(|value| {
            let value = from_bf16(*value);
            to_bf16(value / (1.0 + (-value).exp()))
        })
        .collect()
}

fn require_capture(
    captures: &BTreeMap<String, String>,
    name: &str,
    actual: &[u16],
) -> Result<(), String> {
    let expected = captures
        .get(name)
        .ok_or_else(|| format!("missing expected capture {name}"))?;
    if bf16_hash(actual) != *expected {
        return Err(format!("hyper-connection capture mismatch at {name}"));
    }
    Ok(())
}

pub fn verify_hyper_connection_fixture(
    checkpoint_dir: &Path,
    model_lock_path: &Path,
    fixture_path: &Path,
) -> Result<HyperConnectionVerificationReport, String> {
    let fixture: Fixture =
        serde_json::from_slice(&fs::read(fixture_path).map_err(|error| error.to_string())?)
            .map_err(|error| format!("malformed hyper-connection fixture: {error}"))?;
    let config = &fixture.configuration;
    let case = &fixture.case;
    if fixture.schema_version != 1
        || fixture.semantic != "qwen3_8_flash_next_real_gated_hyper_connection"
        || fixture.model != MODEL
        || fixture.reference.implementation != "huggingface_transformers_qwen4_exp"
        || fixture.reference.transformers_version != "5.16.1"
        || fixture.reference.source
            != "transformers.models.qwen4_exp.modeling_qwen4_exp.Qwen4ExpTextGatedResidual.forward"
        || fixture.reference.rms_source
            != "transformers.models.qwen4_exp.modeling_qwen4_exp.Qwen4ExpTextRMSNorm.forward"
        || config.hidden_size != HIDDEN
        || config.hc_count != HC_COUNT
        || config.hc_lowrank != HC_LOWRANK
        || config.rms_norm_eps.to_bits() != 1.0e-6_f32.to_bits()
        || config.boundary_dtype != "BF16"
        || !config.use_combine
        || case.name != "layer_0_attention_affine_mod_hyper_connection"
        || case.layer != 0
        || case.kind != "attn_hyper_connection"
        || case.tensors.len() != 4
        || case.expected_bf16_sha256.len() != 13
    {
        return Err("hyper-connection fixture identity or configuration is unsupported".to_owned());
    }
    if !case.expected_bf16_sha256.values().all(|hash| is_hash(hash))
        || sha256_file(model_lock_path)? != fixture.reference.model_lock_sha256
        || sha256_file(&checkpoint_dir.join("config.json"))? != fixture.reference.config_sha256
        || sha256_file(&checkpoint_dir.join("model.safetensors.index.json"))?
            != fixture.reference.tensor_index_sha256
    {
        return Err("hyper-connection reference identity mismatch".to_owned());
    }
    let lock: ModelLock =
        serde_json::from_slice(&fs::read(model_lock_path).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    if lock.model != fixture.model || lock.revision != fixture.revision {
        return Err("hyper-connection model lock mismatch".to_owned());
    }
    let expected = [
        ("hc_norm", vec![HC_HIDDEN], "hc_norm.weight"),
        (
            "input_mix_weight_down",
            vec![HC_LOWRANK, HC_HIDDEN],
            "input_mix_weight_down.weight",
        ),
        (
            "input_mix_weight_up",
            vec![HC_HIDDEN, HC_LOWRANK],
            "input_mix_weight_up.weight",
        ),
        (
            "block_inject_weight",
            vec![HC_COUNT, HC_HIDDEN],
            "block_inject_weight.weight",
        ),
    ];
    let mut values = BTreeMap::new();
    let mut tensor_payload_bytes = 0;
    for (key, shape, suffix) in expected {
        let tensor = case
            .tensors
            .get(key)
            .ok_or_else(|| format!("missing tensor record {key}"))?;
        let expected_name = format!("model.language_model.layers.0.attn_hyper_connection.{suffix}");
        let matches: Vec<_> = lock
            .files
            .iter()
            .filter(|entry| entry.path == tensor.shard)
            .collect();
        if tensor.tensor != expected_name
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
                "hyper-connection tensor identity mismatch for {key}"
            ));
        }
        let payload = read_tensor(&checkpoint_dir.join(&tensor.shard), &tensor.tensor, &shape)?;
        if bf16_hash(&payload) != tensor.payload_sha256 {
            return Err(format!("hyper-connection payload mismatch for {key}"));
        }
        tensor_payload_bytes += payload.len() * 2;
        values.insert(key, payload);
    }
    let input = make_input(&case.input_spec)?;
    require_capture(&case.expected_bf16_sha256, "hyper_input", &input)?;
    let normed = grouped_rms(&input, &values["hc_norm"], config.rms_norm_eps);
    require_capture(&case.expected_bf16_sha256, "hyper_input_normed", &normed)?;
    let mix_down = linear_bf16(
        &values["input_mix_weight_down"],
        &normed,
        HC_LOWRANK,
        HC_HIDDEN,
    );
    require_capture(&case.expected_bf16_sha256, "mix_down", &mix_down)?;
    let mix_down_scaled: Vec<_> = mix_down
        .iter()
        .map(|value| to_bf16(from_bf16(*value) / HC_COUNT as f32))
        .collect();
    require_capture(
        &case.expected_bf16_sha256,
        "mix_down_scaled",
        &mix_down_scaled,
    )?;
    let mix_down_silu = silu_bf16(&mix_down_scaled);
    require_capture(&case.expected_bf16_sha256, "mix_down_silu", &mix_down_silu)?;
    let mix_up = linear_bf16(
        &values["input_mix_weight_up"],
        &mix_down_silu,
        HC_HIDDEN,
        HC_LOWRANK,
    );
    require_capture(&case.expected_bf16_sha256, "mix_up", &mix_up)?;
    let input_mix: Vec<_> = mix_up.iter().map(|value| sigmoid_bf16(*value)).collect();
    require_capture(&case.expected_bf16_sha256, "input_mix_weight", &input_mix)?;
    let products: Vec<_> = input_mix
        .iter()
        .zip(&normed)
        .map(|(left, right)| to_bf16(from_bf16(*left) * from_bf16(*right)))
        .collect();
    require_capture(&case.expected_bf16_sha256, "mixed_products", &products)?;
    let mixed: Vec<_> = (0..HIDDEN)
        .map(|column| {
            let values = (0..HC_COUNT)
                .map(|group| from_bf16(products[group * HIDDEN + column]))
                .collect::<Vec<_>>();
            to_bf16(((values[0] + values[1]) + (values[2] + values[3])) / HC_COUNT as f32)
        })
        .collect();
    require_capture(&case.expected_bf16_sha256, "mixed_input", &mixed)?;
    let inject = linear_bf16(&values["block_inject_weight"], &normed, HC_COUNT, HC_HIDDEN);
    require_capture(&case.expected_bf16_sha256, "inject_projection", &inject)?;
    let inject_scaled: Vec<_> = inject
        .iter()
        .map(|value| to_bf16(from_bf16(*value) / HC_COUNT as f32))
        .collect();
    require_capture(&case.expected_bf16_sha256, "inject_scaled", &inject_scaled)?;
    let inject_sigmoid: Vec<_> = inject_scaled
        .iter()
        .map(|value| sigmoid_bf16(*value))
        .collect();
    require_capture(
        &case.expected_bf16_sha256,
        "inject_sigmoid",
        &inject_sigmoid,
    )?;
    let injection_weights: Vec<_> = inject_sigmoid
        .iter()
        .map(|value| to_bf16(2.0 * from_bf16(*value)))
        .collect();
    require_capture(
        &case.expected_bf16_sha256,
        "injection_weights",
        &injection_weights,
    )?;
    Ok(HyperConnectionVerificationReport {
        schema_version: 1,
        semantic: "qwen3_8_flash_next_real_gated_hyper_connection_verification",
        model: fixture.model,
        revision: fixture.revision,
        layer: case.layer,
        kind: case.kind.clone(),
        tensors_verified: 4,
        exact_capture_hashes: 13,
        tensor_payload_bytes,
        accepted_tokens: 0,
        performance_claim: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grouped_rms_is_independent_per_stream() {
        let input = vec![to_bf16(1.0); HC_HIDDEN];
        let weight = vec![to_bf16(0.0); HC_HIDDEN];
        let result = grouped_rms(&input, &weight, 1.0e-6);
        assert_eq!(result.len(), HC_HIDDEN);
        assert!(result.iter().all(|value| *value == to_bf16(1.0)));
    }

    #[test]
    fn four_stream_mean_uses_fixed_pair_tree() {
        let values = [1.0_f32, 2.0, 4.0, 8.0];
        assert_eq!(
            ((values[0] + values[1]) + (values[2] + values[3])) / 4.0,
            3.75
        );
    }
}
