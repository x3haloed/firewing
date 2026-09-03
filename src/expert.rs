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

#[derive(Deserialize)]
struct MixtureFixture {
    schema_version: u32,
    semantic: String,
    model: String,
    revision: String,
    reference: MixtureReference,
    configuration: MixtureConfiguration,
    case: MixtureCase,
}

#[derive(Deserialize)]
struct MixtureReference {
    implementation: String,
    transformers_version: String,
    source: String,
    config_sha256: String,
    tensor_index_sha256: String,
    model_lock_sha256: String,
    router_fixture_sha256: String,
    expert_fixture_sha256: String,
}

#[derive(Deserialize)]
struct MixtureConfiguration {
    hidden_size: usize,
    intermediate_size: usize,
    num_experts: usize,
    top_k: usize,
    activation: String,
    input_dtype: String,
    weight_dtype: String,
    boundary_dtype: String,
    mixture_accumulator_dtype: String,
}

#[derive(Deserialize)]
struct MixtureCase {
    name: String,
    layer: usize,
    input_spec: InputSpec,
    input_bf16_sha256: String,
    top_k_selection_order: Vec<usize>,
    expert_execution_order: Vec<usize>,
    gate_up: TensorBank,
    down: TensorBank,
    experts: Vec<MixtureExpert>,
    mixture_bf16_sha256: String,
}

#[derive(Deserialize)]
struct TensorBank {
    tensor: String,
    shard: String,
    shard_bytes: u64,
    shard_sha256: String,
}

#[derive(Deserialize)]
struct MixtureExpert {
    expert: usize,
    route_weight_bf16: f32,
    gate_up_payload_sha256: String,
    down_payload_sha256: String,
    weighted_down_bf16_sha256: String,
}

#[derive(Debug, Serialize)]
pub struct MixtureVerificationReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub model: String,
    pub revision: String,
    pub layer: usize,
    pub top_k_selection_order: Vec<usize>,
    pub expert_execution_order: Vec<usize>,
    pub unique_experts: usize,
    pub exact_weighted_expert_hashes: usize,
    pub exact_mixture_hashes: usize,
    pub gate_up_payload_bytes: usize,
    pub down_payload_bytes: usize,
    pub total_expert_payload_bytes: usize,
    pub accepted_tokens: usize,
    pub performance_claim: Option<String>,
}

#[derive(Deserialize)]
struct SparseMoeFixture {
    schema_version: u32,
    semantic: String,
    model: String,
    revision: String,
    reference: SparseMoeReference,
    configuration: SparseMoeConfiguration,
    case: SparseMoeCase,
}

#[derive(Deserialize)]
struct SparseMoeReference {
    implementation: String,
    transformers_version: String,
    source: String,
    config_sha256: String,
    tensor_index_sha256: String,
    model_lock_sha256: String,
    router_fixture_sha256: String,
    expert_fixture_sha256: String,
    mixture_fixture_sha256: String,
}

#[derive(Deserialize)]
struct SparseMoeConfiguration {
    hidden_size: usize,
    intermediate_size: usize,
    shared_intermediate_size: usize,
    num_experts: usize,
    top_k: usize,
    activation: String,
    boundary_dtype: String,
}

#[derive(Deserialize)]
struct SparseMoeCase {
    name: String,
    layer: usize,
    input_spec: InputSpec,
    input_bf16_sha256: String,
    common_shard: String,
    common_shard_bytes: u64,
    common_shard_sha256: String,
    tensors: SharedTensors,
    expected_bf16_sha256: SparseMoeHashes,
}

#[derive(Deserialize)]
struct SharedTensors {
    shared_gate_weight: NamedTensor,
    shared_up_weight: NamedTensor,
    shared_down_weight: NamedTensor,
    shared_expert_gate_weight: NamedTensor,
}

#[derive(Deserialize)]
struct NamedTensor {
    tensor: String,
    shape: Vec<usize>,
    payload_sha256: String,
}

#[derive(Deserialize)]
struct SparseMoeHashes {
    shared_gate: String,
    shared_up: String,
    shared_swiglu: String,
    shared_down: String,
    shared_gate_logit: String,
    shared_gate_sigmoid: String,
    gated_shared: String,
    routed_mixture: String,
    combined: String,
}

#[derive(Debug, Serialize)]
pub struct SparseMoeVerificationReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub model: String,
    pub revision: String,
    pub layer: usize,
    pub exact_shared_capture_hashes: usize,
    pub exact_routed_mixture_hashes: usize,
    pub exact_combined_hashes: usize,
    pub routed_expert_payload_bytes: usize,
    pub shared_expert_payload_bytes: usize,
    pub shared_gate_payload_bytes: usize,
    pub total_moe_payload_bytes: usize,
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

pub(crate) fn to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

pub(crate) fn from_bf16(value: u16) -> f32 {
    f32::from_bits((value as u32) << 16)
}

fn bf16_bytes(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

pub(crate) fn bf16_hash(values: &[u16]) -> String {
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

pub(crate) fn read_expert_slice(
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

fn read_tensor_2d(
    path: &Path,
    tensor: &str,
    rows: usize,
    columns: usize,
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
    let item = header
        .get(tensor)
        .ok_or("shared expert tensor is missing")?;
    let shape = item
        .get("shape")
        .and_then(Value::as_array)
        .ok_or("shared expert shape is missing")?;
    if item.get("dtype").and_then(Value::as_str) != Some("BF16")
        || shape.len() != 2
        || shape[0].as_u64() != Some(rows as u64)
        || shape[1].as_u64() != Some(columns as u64)
    {
        return Err("unsupported shared expert tensor dtype or shape".to_owned());
    }
    let offsets = item
        .get("data_offsets")
        .and_then(Value::as_array)
        .ok_or("shared expert offsets are missing")?;
    let start = offsets
        .first()
        .and_then(Value::as_u64)
        .ok_or("invalid shared expert offset")?;
    let end = offsets
        .get(1)
        .and_then(Value::as_u64)
        .ok_or("invalid shared expert offset")?;
    let count = rows
        .checked_mul(columns)
        .ok_or("shared expert size overflow")?;
    if end.checked_sub(start) != Some((count * 2) as u64) {
        return Err("shared expert tensor byte count mismatch".to_owned());
    }
    let absolute = 8_u64
        .checked_add(header_len)
        .and_then(|value| value.checked_add(start))
        .ok_or("shared expert absolute offset overflow")?;
    file.seek(SeekFrom::Start(absolute))
        .map_err(|error| error.to_string())?;
    let mut payload = vec![0; count * 2];
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
pub(crate) fn pytorch_bf16_vector_dot(left: &[u16], right: &[u16]) -> f32 {
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

pub(crate) fn linear_bf16(weight: &[u16], input: &[u16], rows: usize, columns: usize) -> Vec<u16> {
    weight
        .chunks_exact(columns)
        .take(rows)
        .map(|row| to_bf16(pytorch_bf16_vector_dot(row, input)))
        .collect()
}

pub(crate) fn swiglu_bf16(gate: &[u16], up: &[u16]) -> Vec<u16> {
    gate.iter()
        .zip(up)
        .map(|(gate, up)| {
            let gate = from_bf16(*gate);
            let silu = from_bf16(to_bf16(gate / (1.0 + (-gate).exp())));
            to_bf16(silu * from_bf16(*up))
        })
        .collect()
}

pub(crate) fn sigmoid_bf16(value: u16) -> u16 {
    let value = from_bf16(value);
    to_bf16(1.0 / (1.0 + (-value).exp()))
}

pub(crate) fn add_bf16(left: u16, right: u16) -> u16 {
    to_bf16(from_bf16(left) + from_bf16(right))
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

fn require_locked_bank(
    checkpoint_dir: &Path,
    lock: &ModelLock,
    tensor: &TensorBank,
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
            "mixture shard {} does not match model lock",
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

fn verify_mixture_fixture_with_output(
    checkpoint_dir: &Path,
    model_lock_path: &Path,
    router_fixture_path: &Path,
    expert_fixture_path: &Path,
    mixture_fixture_path: &Path,
) -> Result<(MixtureVerificationReport, Vec<u16>), String> {
    let fixture: MixtureFixture =
        serde_json::from_slice(&fs::read(mixture_fixture_path).map_err(|error| error.to_string())?)
            .map_err(|error| format!("malformed mixture fixture: {error}"))?;
    let config = &fixture.configuration;
    let case = &fixture.case;
    const SELECTION: [usize; 10] = [376, 349, 384, 191, 211, 363, 337, 206, 247, 295];
    const EXECUTION: [usize; 10] = [191, 206, 211, 247, 295, 337, 349, 363, 376, 384];
    const SCORE_BITS_BY_SELECTION: [u16; 10] = [
        0x3df5, 0x3de4, 0x3dd5, 0x3dd0, 0x3dca, 0x3dc5, 0x3dc3, 0x3dc0, 0x3db8, 0x3db8,
    ];
    if fixture.schema_version != 1
        || fixture.semantic != "qwen3_8_flash_next_real_top10_expert_mixture"
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
            &config.mixture_accumulator_dtype,
        ] != ["BF16", "BF16", "BF16", "BF16"]
        || case.name != "layer_0_affine_mod_top10"
        || case.layer != 0
        || case.top_k_selection_order != SELECTION
        || case.expert_execution_order != EXECUTION
        || case.experts.len() != config.top_k
        || case.gate_up.tensor != "model.language_model.layers.0.mlp.experts.gate_up_proj"
        || case.down.tensor != "model.language_model.layers.0.mlp.experts.down_proj"
    {
        return Err("mixture fixture identity or configuration is unsupported".to_owned());
    }
    let identity_hashes = [
        &fixture.reference.config_sha256,
        &fixture.reference.tensor_index_sha256,
        &fixture.reference.model_lock_sha256,
        &fixture.reference.router_fixture_sha256,
        &fixture.reference.expert_fixture_sha256,
        &case.input_bf16_sha256,
        &case.gate_up.shard_sha256,
        &case.down.shard_sha256,
        &case.mixture_bf16_sha256,
    ];
    if !identity_hashes.iter().all(|value| is_hash(value))
        || case.experts.iter().any(|entry| {
            !is_hash(&entry.gate_up_payload_sha256)
                || !is_hash(&entry.down_payload_sha256)
                || !is_hash(&entry.weighted_down_bf16_sha256)
        })
        || sha256_file(model_lock_path)? != fixture.reference.model_lock_sha256
        || sha256_file(router_fixture_path)? != fixture.reference.router_fixture_sha256
        || sha256_file(expert_fixture_path)? != fixture.reference.expert_fixture_sha256
        || sha256_file(&checkpoint_dir.join("config.json"))? != fixture.reference.config_sha256
        || sha256_file(&checkpoint_dir.join("model.safetensors.index.json"))?
            != fixture.reference.tensor_index_sha256
    {
        return Err("mixture reference identity mismatch".to_owned());
    }
    for (&expert, &score_bits) in SELECTION.iter().zip(&SCORE_BITS_BY_SELECTION) {
        let entries: Vec<_> = case
            .experts
            .iter()
            .filter(|entry| entry.expert == expert)
            .collect();
        if entries.len() != 1
            || to_bf16(entries[0].route_weight_bf16) != score_bits
            || entries[0].route_weight_bf16 != from_bf16(score_bits)
        {
            return Err("mixture expert weights do not match the router authority".to_owned());
        }
    }
    if case
        .experts
        .iter()
        .map(|entry| entry.expert)
        .collect::<Vec<_>>()
        != EXECUTION
    {
        return Err("mixture entries are not in source expert order".to_owned());
    }

    let lock: ModelLock =
        serde_json::from_slice(&fs::read(model_lock_path).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    if lock.model != fixture.model || lock.revision != fixture.revision {
        return Err("mixture model lock mismatch".to_owned());
    }
    require_locked_bank(checkpoint_dir, &lock, &case.gate_up)?;
    require_locked_bank(checkpoint_dir, &lock, &case.down)?;
    let hidden = make_hidden(config.hidden_size, &case.input_spec)?;
    if bf16_hash(&hidden) != case.input_bf16_sha256 {
        return Err("mixture input hash mismatch".to_owned());
    }

    let mut mixture = vec![to_bf16(0.0); config.hidden_size];
    for entry in &case.experts {
        let gate_up_weight = read_expert_slice(
            &checkpoint_dir.join(&case.gate_up.shard),
            &case.gate_up.tensor,
            entry.expert,
            config.num_experts,
            config.intermediate_size * 2,
            config.hidden_size,
        )?;
        let down_weight = read_expert_slice(
            &checkpoint_dir.join(&case.down.shard),
            &case.down.tensor,
            entry.expert,
            config.num_experts,
            config.hidden_size,
            config.intermediate_size,
        )?;
        if bf16_hash(&gate_up_weight) != entry.gate_up_payload_sha256
            || bf16_hash(&down_weight) != entry.down_payload_sha256
        {
            return Err(format!("mixture expert {} payload mismatch", entry.expert));
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
        let weighted: Vec<_> = down
            .iter()
            .map(|value| to_bf16(from_bf16(*value) * entry.route_weight_bf16))
            .collect();
        if bf16_hash(&weighted) != entry.weighted_down_bf16_sha256 {
            return Err(format!(
                "mixture expert {} weighted output mismatch",
                entry.expert
            ));
        }
        for (output, contribution) in mixture.iter_mut().zip(weighted) {
            *output = to_bf16(from_bf16(*output) + from_bf16(contribution));
        }
    }
    if bf16_hash(&mixture) != case.mixture_bf16_sha256 {
        return Err("mixture output hash mismatch".to_owned());
    }

    let gate_up_payload_bytes =
        config.top_k * config.intermediate_size * 2 * config.hidden_size * 2;
    let down_payload_bytes = config.top_k * config.hidden_size * config.intermediate_size * 2;
    let report = MixtureVerificationReport {
        schema_version: 1,
        semantic: "qwen3_8_flash_next_real_top10_expert_mixture_verification",
        model: fixture.model,
        revision: fixture.revision,
        layer: case.layer,
        top_k_selection_order: case.top_k_selection_order.clone(),
        expert_execution_order: case.expert_execution_order.clone(),
        unique_experts: case.experts.len(),
        exact_weighted_expert_hashes: case.experts.len(),
        exact_mixture_hashes: 1,
        gate_up_payload_bytes,
        down_payload_bytes,
        total_expert_payload_bytes: gate_up_payload_bytes + down_payload_bytes,
        accepted_tokens: 0,
        performance_claim: None,
    };
    Ok((report, mixture))
}

pub fn verify_mixture_fixture(
    checkpoint_dir: &Path,
    model_lock_path: &Path,
    router_fixture_path: &Path,
    expert_fixture_path: &Path,
    mixture_fixture_path: &Path,
) -> Result<MixtureVerificationReport, String> {
    verify_mixture_fixture_with_output(
        checkpoint_dir,
        model_lock_path,
        router_fixture_path,
        expert_fixture_path,
        mixture_fixture_path,
    )
    .map(|(report, _)| report)
}

pub fn verify_sparse_moe_fixture(
    checkpoint_dir: &Path,
    model_lock_path: &Path,
    router_fixture_path: &Path,
    expert_fixture_path: &Path,
    mixture_fixture_path: &Path,
    sparse_moe_fixture_path: &Path,
) -> Result<SparseMoeVerificationReport, String> {
    let fixture: SparseMoeFixture = serde_json::from_slice(
        &fs::read(sparse_moe_fixture_path).map_err(|error| error.to_string())?,
    )
    .map_err(|error| format!("malformed sparse-MoE fixture: {error}"))?;
    let config = &fixture.configuration;
    let case = &fixture.case;
    let expected_names = [
        "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
        "model.language_model.layers.0.mlp.shared_expert.up_proj.weight",
        "model.language_model.layers.0.mlp.shared_expert.down_proj.weight",
        "model.language_model.layers.0.mlp.shared_expert_gate.weight",
    ];
    let tensors = [
        &case.tensors.shared_gate_weight,
        &case.tensors.shared_up_weight,
        &case.tensors.shared_down_weight,
        &case.tensors.shared_expert_gate_weight,
    ];
    if fixture.schema_version != 1
        || fixture.semantic != "qwen3_8_flash_next_real_sparse_moe_block"
        || fixture.model != MODEL
        || fixture.reference.implementation != "huggingface_transformers_qwen4_exp"
        || fixture.reference.transformers_version != "5.16.1"
        || fixture.reference.source
            != "transformers.models.qwen4_exp.modeling_qwen4_exp.Qwen4ExpTextSparseMoeBlock.forward"
        || config.hidden_size != 2560
        || config.intermediate_size != 640
        || config.shared_intermediate_size != 640
        || config.num_experts != 512
        || config.top_k != 10
        || config.activation != "silu"
        || config.boundary_dtype != "BF16"
        || case.name != "layer_0_affine_mod_sparse_moe_block"
        || case.layer != 0
        || case.common_shard != "model-00003-of-00131.safetensors"
        || tensors
            .iter()
            .map(|tensor| tensor.tensor.as_str())
            .collect::<Vec<_>>()
            != expected_names
        || case.tensors.shared_gate_weight.shape != [640, 2560]
        || case.tensors.shared_up_weight.shape != [640, 2560]
        || case.tensors.shared_down_weight.shape != [2560, 640]
        || case.tensors.shared_expert_gate_weight.shape != [1, 2560]
    {
        return Err("sparse-MoE fixture identity or configuration is unsupported".to_owned());
    }
    let hashes = [
        &fixture.reference.config_sha256,
        &fixture.reference.tensor_index_sha256,
        &fixture.reference.model_lock_sha256,
        &fixture.reference.router_fixture_sha256,
        &fixture.reference.expert_fixture_sha256,
        &fixture.reference.mixture_fixture_sha256,
        &case.input_bf16_sha256,
        &case.common_shard_sha256,
        &case.expected_bf16_sha256.shared_gate,
        &case.expected_bf16_sha256.shared_up,
        &case.expected_bf16_sha256.shared_swiglu,
        &case.expected_bf16_sha256.shared_down,
        &case.expected_bf16_sha256.shared_gate_logit,
        &case.expected_bf16_sha256.shared_gate_sigmoid,
        &case.expected_bf16_sha256.gated_shared,
        &case.expected_bf16_sha256.routed_mixture,
        &case.expected_bf16_sha256.combined,
    ];
    if !hashes.iter().all(|value| is_hash(value))
        || tensors
            .iter()
            .any(|tensor| !is_hash(&tensor.payload_sha256))
        || sha256_file(model_lock_path)? != fixture.reference.model_lock_sha256
        || sha256_file(router_fixture_path)? != fixture.reference.router_fixture_sha256
        || sha256_file(expert_fixture_path)? != fixture.reference.expert_fixture_sha256
        || sha256_file(mixture_fixture_path)? != fixture.reference.mixture_fixture_sha256
        || sha256_file(&checkpoint_dir.join("config.json"))? != fixture.reference.config_sha256
        || sha256_file(&checkpoint_dir.join("model.safetensors.index.json"))?
            != fixture.reference.tensor_index_sha256
    {
        return Err("sparse-MoE reference identity mismatch".to_owned());
    }
    let lock: ModelLock =
        serde_json::from_slice(&fs::read(model_lock_path).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    let records: Vec<_> = lock
        .files
        .iter()
        .filter(|file| file.path == case.common_shard)
        .collect();
    if lock.model != fixture.model
        || lock.revision != fixture.revision
        || records.len() != 1
        || records[0].size != case.common_shard_bytes
        || records[0].lfs_sha256.as_deref() != Some(case.common_shard_sha256.as_str())
        || fs::metadata(checkpoint_dir.join(&case.common_shard))
            .map_err(|error| error.to_string())?
            .len()
            != case.common_shard_bytes
    {
        return Err("sparse-MoE model lock or shard mismatch".to_owned());
    }
    let hidden = make_hidden(config.hidden_size, &case.input_spec)?;
    if bf16_hash(&hidden) != case.input_bf16_sha256 {
        return Err("sparse-MoE input hash mismatch".to_owned());
    }
    let (mixture_report, routed) = verify_mixture_fixture_with_output(
        checkpoint_dir,
        model_lock_path,
        router_fixture_path,
        expert_fixture_path,
        mixture_fixture_path,
    )?;
    if bf16_hash(&routed) != case.expected_bf16_sha256.routed_mixture {
        return Err("sparse-MoE routed mixture hash mismatch".to_owned());
    }
    let shard = checkpoint_dir.join(&case.common_shard);
    let shared_gate_weight = read_tensor_2d(&shard, &tensors[0].tensor, 640, 2560)?;
    let shared_up_weight = read_tensor_2d(&shard, &tensors[1].tensor, 640, 2560)?;
    let shared_down_weight = read_tensor_2d(&shard, &tensors[2].tensor, 2560, 640)?;
    let shared_expert_gate_weight = read_tensor_2d(&shard, &tensors[3].tensor, 1, 2560)?;
    for (value, tensor) in [
        (shared_gate_weight.as_slice(), tensors[0]),
        (shared_up_weight.as_slice(), tensors[1]),
        (shared_down_weight.as_slice(), tensors[2]),
        (shared_expert_gate_weight.as_slice(), tensors[3]),
    ] {
        if bf16_hash(value) != tensor.payload_sha256 {
            return Err(format!("shared tensor {} payload mismatch", tensor.tensor));
        }
    }
    let shared_gate = linear_bf16(&shared_gate_weight, &hidden, 640, 2560);
    let shared_up = linear_bf16(&shared_up_weight, &hidden, 640, 2560);
    let shared_swiglu = swiglu_bf16(&shared_gate, &shared_up);
    let shared_down = linear_bf16(&shared_down_weight, &shared_swiglu, 2560, 640);
    let shared_gate_logit = linear_bf16(&shared_expert_gate_weight, &hidden, 1, 2560);
    let shared_gate_sigmoid = [sigmoid_bf16(shared_gate_logit[0])];
    let sigmoid = from_bf16(shared_gate_sigmoid[0]);
    let gated_shared: Vec<_> = shared_down
        .iter()
        .map(|value| to_bf16(sigmoid * from_bf16(*value)))
        .collect();
    let combined: Vec<_> = routed
        .iter()
        .zip(&gated_shared)
        .map(|(routed, shared)| add_bf16(*routed, *shared))
        .collect();
    for (name, actual, expected) in [
        (
            "shared_gate",
            shared_gate.as_slice(),
            &case.expected_bf16_sha256.shared_gate,
        ),
        (
            "shared_up",
            shared_up.as_slice(),
            &case.expected_bf16_sha256.shared_up,
        ),
        (
            "shared_swiglu",
            shared_swiglu.as_slice(),
            &case.expected_bf16_sha256.shared_swiglu,
        ),
        (
            "shared_down",
            shared_down.as_slice(),
            &case.expected_bf16_sha256.shared_down,
        ),
        (
            "shared_gate_logit",
            shared_gate_logit.as_slice(),
            &case.expected_bf16_sha256.shared_gate_logit,
        ),
        (
            "shared_gate_sigmoid",
            shared_gate_sigmoid.as_slice(),
            &case.expected_bf16_sha256.shared_gate_sigmoid,
        ),
        (
            "gated_shared",
            gated_shared.as_slice(),
            &case.expected_bf16_sha256.gated_shared,
        ),
        (
            "combined",
            combined.as_slice(),
            &case.expected_bf16_sha256.combined,
        ),
    ] {
        if bf16_hash(actual) != *expected {
            return Err(format!("sparse-MoE {name} hash mismatch"));
        }
    }
    let shared_expert_payload_bytes = (640 * 2560 + 640 * 2560 + 2560 * 640) * 2;
    let shared_gate_payload_bytes = 2560 * 2;
    Ok(SparseMoeVerificationReport {
        schema_version: 1,
        semantic: "qwen3_8_flash_next_real_sparse_moe_block_verification",
        model: fixture.model,
        revision: fixture.revision,
        layer: case.layer,
        exact_shared_capture_hashes: 7,
        exact_routed_mixture_hashes: 1,
        exact_combined_hashes: 1,
        routed_expert_payload_bytes: mixture_report.total_expert_payload_bytes,
        shared_expert_payload_bytes,
        shared_gate_payload_bytes,
        total_moe_payload_bytes: mixture_report.total_expert_payload_bytes
            + shared_expert_payload_bytes
            + shared_gate_payload_bytes,
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

    #[test]
    fn bf16_mixture_order_is_observable() {
        let mut first = to_bf16(0.0);
        for value in [256.0, -256.0, 1.0] {
            first = to_bf16(from_bf16(first) + value);
        }
        let mut second = to_bf16(0.0);
        for value in [256.0, 1.0, -256.0] {
            second = to_bf16(from_bf16(second) + value);
        }
        assert_eq!(from_bf16(first), 1.0);
        assert_eq!(from_bf16(second), 0.0);
    }

    #[test]
    fn shared_gate_and_combination_round_at_bf16_boundaries() {
        assert_eq!(from_bf16(sigmoid_bf16(to_bf16(0.0))), 0.5);
        assert_eq!(from_bf16(add_bf16(to_bf16(256.0), to_bf16(1.0))), 256.0);
    }
}
