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
    router_fixture_sha256: String,
}

#[derive(Deserialize)]
struct Configuration {
    hidden_size: usize,
    intermediate_size: usize,
    num_experts: usize,
    top_k: usize,
    activation: String,
    input_dtype: String,
    weight_dtype: String,
    boundary_dtype: String,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    layer: usize,
    expert: usize,
    route_weight_bf16: f32,
    input_spec: InputSpec,
    input_bf16_sha256: String,
    gate_up: TensorSlice,
    down: TensorSlice,
    expected_bf16_sha256: ExpectedHashes,
}

#[derive(Deserialize)]
struct TensorSlice {
    tensor: String,
    shard: String,
    shard_bytes: u64,
    shard_sha256: String,
    expert_payload_sha256: String,
}

#[derive(Deserialize)]
struct ExpectedHashes {
    gate_up: String,
    gate: String,
    up: String,
    swiglu: String,
    down: String,
    weighted_down: String,
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
pub struct ExpertVerificationReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub model: String,
    pub revision: String,
    pub layer: usize,
    pub expert: usize,
    pub route_weight_bf16: f32,
    pub exact_capture_hashes: usize,
    pub gate_up_payload_bytes: usize,
    pub down_payload_bytes: usize,
    pub total_expert_payload_bytes: usize,
    pub accepted_tokens: usize,
    pub performance_claim: Option<String>,
}

fn is_hash(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
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

fn to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn from_bf16(value: u16) -> f32 {
    f32::from_bits((value as u32) << 16)
}

fn bf16_bytes(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn bf16_hash(values: &[u16]) -> String {
    format!("{:x}", Sha256::digest(bf16_bytes(values)))
}

fn make_hidden(size: usize, spec: &InputSpec) -> Result<Vec<u16>, String> {
    if spec.modulus <= 0 || spec.divisor <= 0 || spec.sparse_stride == 0 {
        return Err("invalid expert input specification".to_owned());
    }
    (0..size)
        .map(|index| {
            let affine = (index as i64)
                .checked_mul(spec.multiplier)
                .and_then(|value| value.checked_add(spec.add))
                .ok_or("expert input overflow")?;
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

fn read_expert_slice(
    path: &Path,
    tensor: &str,
    expert: usize,
    experts: usize,
    rows: usize,
    columns: usize,
) -> Result<Vec<u16>, String> {
    if expert >= experts {
        return Err("expert index is out of range".to_owned());
    }
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
    let item = header.get(tensor).ok_or("expert tensor is missing")?;
    let shape = item
        .get("shape")
        .and_then(Value::as_array)
        .ok_or("expert shape is missing")?;
    if item.get("dtype").and_then(Value::as_str) != Some("BF16")
        || shape.len() != 3
        || shape[0].as_u64() != Some(experts as u64)
        || shape[1].as_u64() != Some(rows as u64)
        || shape[2].as_u64() != Some(columns as u64)
    {
        return Err("unsupported expert tensor dtype or shape".to_owned());
    }
    let offsets = item
        .get("data_offsets")
        .and_then(Value::as_array)
        .ok_or("expert offsets are missing")?;
    let start = offsets
        .first()
        .and_then(Value::as_u64)
        .ok_or("invalid expert offset")?;
    let end = offsets
        .get(1)
        .and_then(Value::as_u64)
        .ok_or("invalid expert offset")?;
    let values_per_expert = rows.checked_mul(columns).ok_or("expert size overflow")?;
    let bytes_per_expert = values_per_expert
        .checked_mul(2)
        .ok_or("expert byte size overflow")?;
    let total_bytes = bytes_per_expert
        .checked_mul(experts)
        .ok_or("expert tensor byte size overflow")?;
    if end.checked_sub(start) != Some(total_bytes as u64) {
        return Err("expert tensor byte count mismatch".to_owned());
    }
    let expert_offset = bytes_per_expert
        .checked_mul(expert)
        .ok_or("expert slice offset overflow")? as u64;
    let absolute = 8_u64
        .checked_add(header_len)
        .and_then(|value| value.checked_add(start))
        .and_then(|value| value.checked_add(expert_offset))
        .ok_or("expert absolute offset overflow")?;
    file.seek(SeekFrom::Start(absolute))
        .map_err(|error| error.to_string())?;
    let mut payload = vec![0; bytes_per_expert];
    file.read_exact(&mut payload)
        .map_err(|error| error.to_string())?;
    Ok(payload
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect())
}

// PyTorch's aarch64 BF16 GEMV fast path uses eight four-lane F32 vector
// accumulators over 32-value blocks, followed by a fixed reduction tree. This
// is deliberately not a conventional forward scalar sum.
fn pytorch_bf16_vector_dot(left: &[u16], right: &[u16]) -> f32 {
    debug_assert_eq!(left.len(), right.len());
    let mut accumulators = [[0.0_f32; 4]; 8];
    let complete_blocks = left.len() / 32 * 32;
    for block in (0..complete_blocks).step_by(32) {
        for (register, accumulator) in accumulators.iter_mut().enumerate() {
            for (lane, value) in accumulator.iter_mut().enumerate() {
                let index = block + register * 4 + lane;
                *value += from_bf16(left[index]) * from_bf16(right[index]);
            }
        }
    }
    for offset in [4, 2, 1] {
        for register in 0..offset {
            let source = accumulators[offset + register];
            for (target, source) in accumulators[register].iter_mut().zip(source) {
                *target += source;
            }
        }
    }
    let mut reduced =
        (accumulators[0][0] + accumulators[0][1]) + (accumulators[0][2] + accumulators[0][3]);
    let complete_vectors = left.len() / 8 * 8;
    let mut tail = [0.0_f32; 4];
    for block in (complete_blocks..complete_vectors).step_by(8) {
        for lane in 0..4 {
            tail[lane] += from_bf16(left[block + lane]) * from_bf16(right[block + lane]);
            tail[lane] += from_bf16(left[block + 4 + lane]) * from_bf16(right[block + 4 + lane]);
        }
    }
    reduced += (tail[0] + tail[1]) + (tail[2] + tail[3]);
    for index in complete_vectors..left.len() {
        reduced += from_bf16(left[index]) * from_bf16(right[index]);
    }
    reduced
}

fn linear_bf16(weight: &[u16], input: &[u16], rows: usize, columns: usize) -> Vec<u16> {
    weight
        .chunks_exact(columns)
        .take(rows)
        .map(|row| to_bf16(pytorch_bf16_vector_dot(row, input)))
        .collect()
}

fn swiglu_bf16(gate: &[u16], up: &[u16]) -> Vec<u16> {
    gate.iter()
        .zip(up)
        .map(|(gate, up)| {
            let gate = from_bf16(*gate);
            let silu = from_bf16(to_bf16(gate / (1.0 + (-gate).exp())));
            to_bf16(silu * from_bf16(*up))
        })
        .collect()
}

fn require_locked_tensor(
    checkpoint_dir: &Path,
    lock: &ModelLock,
    tensor: &TensorSlice,
) -> Result<(), String> {
    let records: Vec<_> = lock
        .files
        .iter()
        .filter(|file| file.path == tensor.shard)
        .collect();
    if records.len() != 1
        || records[0].size != tensor.shard_bytes
        || records[0].lfs_sha256.as_deref() != Some(tensor.shard_sha256.as_str())
        || fs::metadata(checkpoint_dir.join(&tensor.shard))
            .map_err(|error| error.to_string())?
            .len()
            != tensor.shard_bytes
    {
        return Err(format!(
            "expert shard {} does not match model lock",
            tensor.shard
        ));
    }
    Ok(())
}

pub fn verify_expert_fixture(
    checkpoint_dir: &Path,
    model_lock_path: &Path,
    router_fixture_path: &Path,
    fixture_path: &Path,
) -> Result<ExpertVerificationReport, String> {
    let fixture: Fixture =
        serde_json::from_slice(&fs::read(fixture_path).map_err(|error| error.to_string())?)
            .map_err(|error| format!("malformed expert fixture: {error}"))?;
    let config = &fixture.configuration;
    let case = &fixture.case;
    if fixture.schema_version != 1
        || fixture.semantic != "qwen3_8_flash_next_real_routed_expert"
        || fixture.model != MODEL
        || fixture.reference.implementation != "huggingface_transformers_qwen4_exp"
        || fixture.reference.transformers_version != "5.16.1"
        || fixture.reference.source
            != "transformers.models.qwen4_exp.modeling_qwen4_exp.Qwen4ExpTextExperts.forward"
        || config.hidden_size != 2560
        || config.intermediate_size != 640
        || config.num_experts != 512
        || config.top_k != 10
        || config.activation != "silu"
        || [
            &config.input_dtype,
            &config.weight_dtype,
            &config.boundary_dtype,
        ] != ["BF16", "BF16", "BF16"]
        || case.name != "layer_0_top_1_expert"
        || case.layer != 0
        || case.expert != 376
        || case.gate_up.tensor != "model.language_model.layers.0.mlp.experts.gate_up_proj"
        || case.down.tensor != "model.language_model.layers.0.mlp.experts.down_proj"
    {
        return Err("expert fixture identity or configuration is unsupported".to_owned());
    }
    let hashes = [
        &fixture.reference.config_sha256,
        &fixture.reference.tensor_index_sha256,
        &fixture.reference.model_lock_sha256,
        &fixture.reference.router_fixture_sha256,
        &case.input_bf16_sha256,
        &case.gate_up.shard_sha256,
        &case.gate_up.expert_payload_sha256,
        &case.down.shard_sha256,
        &case.down.expert_payload_sha256,
        &case.expected_bf16_sha256.gate_up,
        &case.expected_bf16_sha256.gate,
        &case.expected_bf16_sha256.up,
        &case.expected_bf16_sha256.swiglu,
        &case.expected_bf16_sha256.down,
        &case.expected_bf16_sha256.weighted_down,
    ];
    if !hashes.iter().all(|value| is_hash(value))
        || sha256_file(model_lock_path)? != fixture.reference.model_lock_sha256
        || sha256_file(router_fixture_path)? != fixture.reference.router_fixture_sha256
        || sha256_file(&checkpoint_dir.join("config.json"))? != fixture.reference.config_sha256
        || sha256_file(&checkpoint_dir.join("model.safetensors.index.json"))?
            != fixture.reference.tensor_index_sha256
    {
        return Err("expert reference identity mismatch".to_owned());
    }
    let lock: ModelLock =
        serde_json::from_slice(&fs::read(model_lock_path).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    if lock.model != fixture.model || lock.revision != fixture.revision {
        return Err("expert model lock mismatch".to_owned());
    }
    require_locked_tensor(checkpoint_dir, &lock, &case.gate_up)?;
    require_locked_tensor(checkpoint_dir, &lock, &case.down)?;

    let hidden = make_hidden(config.hidden_size, &case.input_spec)?;
    if bf16_hash(&hidden) != case.input_bf16_sha256 {
        return Err("expert input hash mismatch".to_owned());
    }
    if from_bf16(to_bf16(case.route_weight_bf16)) != case.route_weight_bf16 {
        return Err("expert route weight is not exactly BF16".to_owned());
    }
    let gate_up_weight = read_expert_slice(
        &checkpoint_dir.join(&case.gate_up.shard),
        &case.gate_up.tensor,
        case.expert,
        config.num_experts,
        config.intermediate_size * 2,
        config.hidden_size,
    )?;
    let down_weight = read_expert_slice(
        &checkpoint_dir.join(&case.down.shard),
        &case.down.tensor,
        case.expert,
        config.num_experts,
        config.hidden_size,
        config.intermediate_size,
    )?;
    if bf16_hash(&gate_up_weight) != case.gate_up.expert_payload_sha256
        || bf16_hash(&down_weight) != case.down.expert_payload_sha256
    {
        return Err("expert weight payload hash mismatch".to_owned());
    }

    let gate_up = linear_bf16(
        &gate_up_weight,
        &hidden,
        config.intermediate_size * 2,
        config.hidden_size,
    );
    let (gate, up) = gate_up.split_at(config.intermediate_size);
    let swiglu = swiglu_bf16(gate, up);
    let down = linear_bf16(
        &down_weight,
        &swiglu,
        config.hidden_size,
        config.intermediate_size,
    );
    let weighted_down: Vec<_> = down
        .iter()
        .map(|value| to_bf16(from_bf16(*value) * case.route_weight_bf16))
        .collect();
    for (name, actual, expected) in [
        (
            "gate_up",
            gate_up.as_slice(),
            &case.expected_bf16_sha256.gate_up,
        ),
        ("gate", gate, &case.expected_bf16_sha256.gate),
        ("up", up, &case.expected_bf16_sha256.up),
        (
            "swiglu",
            swiglu.as_slice(),
            &case.expected_bf16_sha256.swiglu,
        ),
        ("down", down.as_slice(), &case.expected_bf16_sha256.down),
        (
            "weighted_down",
            weighted_down.as_slice(),
            &case.expected_bf16_sha256.weighted_down,
        ),
    ] {
        let actual = bf16_hash(actual);
        if actual != *expected {
            return Err(format!(
                "expert {name} hash mismatch: expected {expected}, got {actual}"
            ));
        }
    }

    let gate_up_payload_bytes = gate_up_weight.len() * 2;
    let down_payload_bytes = down_weight.len() * 2;
    Ok(ExpertVerificationReport {
        schema_version: 1,
        semantic: "qwen3_8_flash_next_real_routed_expert_verification",
        model: fixture.model,
        revision: fixture.revision,
        layer: case.layer,
        expert: case.expert,
        route_weight_bf16: case.route_weight_bf16,
        exact_capture_hashes: 6,
        gate_up_payload_bytes,
        down_payload_bytes,
        total_expert_payload_bytes: gate_up_payload_bytes + down_payload_bytes,
        accepted_tokens: 0,
        performance_claim: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_expert_equation_rounds_each_bf16_boundary() {
        let input = [to_bf16(1.0), to_bf16(-2.0)];
        let gate_up_weight: Vec<_> = [1.0, 2.0, 0.5, -0.5, 2.0, 0.0, -1.0, 1.0]
            .into_iter()
            .map(to_bf16)
            .collect();
        let gate_up = linear_bf16(&gate_up_weight, &input, 4, 2);
        assert_eq!(
            gate_up
                .iter()
                .map(|value| from_bf16(*value))
                .collect::<Vec<_>>(),
            vec![-3.0, 1.5, 2.0, -3.0]
        );
        let activated = swiglu_bf16(&gate_up[..2], &gate_up[2..]);
        let down_weight: Vec<_> = [1.0, 0.0, 0.0, 1.0].into_iter().map(to_bf16).collect();
        assert_eq!(linear_bf16(&down_weight, &activated, 2, 2), activated);
    }

    #[test]
    fn vector_dot_reduction_is_not_a_forward_sum() {
        let mut state = 25_u32;
        let values: Vec<_> = (0..128)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let raw = ((state >> 8) % 65_535) as i64 - 32_768;
                to_bf16(raw as f32 / 64.0)
            })
            .collect();
        let (left, right) = values.split_at(64);
        let forward = left
            .iter()
            .zip(right)
            .fold(0.0_f32, |sum, (a, b)| sum + from_bf16(*a) * from_bf16(*b));
        assert_eq!(pytorch_bf16_vector_dot(left, right).to_bits(), 0x48e4_807c);
        assert_eq!(forward.to_bits(), 0x48e4_807e);
    }

    #[test]
    fn committed_input_hash_is_reproducible() {
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../fixtures/expert/qwen3_8_flash_next_real.json"
        ))
        .expect("fixture parses");
        let hidden = make_hidden(2560, &fixture.case.input_spec).expect("input builds");
        assert_eq!(bf16_hash(&hidden), fixture.case.input_bf16_sha256);
    }
}
