use crate::expert::{
    bf16_hash, from_bf16, linear_bf16, pytorch_bf16_vector_dot, sigmoid_bf16, to_bf16,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

const MODEL: &str = "Qwen/Qwen3.8-Flash-Next";
const HIDDEN: usize = 2560;
const K_HEADS: usize = 16;
const V_HEADS: usize = 48;
const HEAD_DIM: usize = 128;
const KEY_DIM: usize = K_HEADS * HEAD_DIM;
const VALUE_DIM: usize = V_HEADS * HEAD_DIM;
const CONV_DIM: usize = 2 * KEY_DIM + VALUE_DIM;
const CONV_KERNEL: usize = 4;
const RECURRENT_VALUES: usize = V_HEADS * HEAD_DIM * HEAD_DIM;

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
}

#[derive(Deserialize)]
struct Configuration {
    hidden_size: usize,
    linear_num_key_heads: usize,
    linear_num_value_heads: usize,
    linear_key_head_dim: usize,
    linear_value_head_dim: usize,
    linear_conv_kernel_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_dim: usize,
    activation: String,
    output_gate: String,
    weight_dtype: String,
    recurrent_state_dtype: String,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    layer: usize,
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
pub struct DeltaNetVerificationReport {
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
    pub convolution_state_bytes: usize,
    pub recurrent_state_bytes: usize,
    pub accepted_tokens: usize,
    pub performance_claim: Option<String>,
}

struct State {
    convolution: Vec<u16>,
    recurrent: Vec<f32>,
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
        .ok_or_else(|| "tensor size overflow".to_owned())?;
    if end.checked_sub(start) != Some((count * 2) as u64) {
        return Err(format!("tensor {tensor} byte count mismatch"));
    }
    let absolute = 8_u64
        .checked_add(header_bytes)
        .and_then(|value| value.checked_add(start))
        .ok_or_else(|| "tensor offset overflow".to_owned())?;
    file.seek(SeekFrom::Start(absolute))
        .map_err(|error| error.to_string())?;
    let mut bytes = vec![0_u8; count * 2];
    file.read_exact(&mut bytes)
        .map_err(|error| error.to_string())?;
    Ok(bytes
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect())
}

fn make_input(spec: &InputSpec, ordinal: usize) -> Result<Vec<u16>, String> {
    let expected = [(47, 23, 269, 134, 128), (59, 31, 271, 135, 128)];
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
            "unsupported DeltaNet input specification at step {ordinal}"
        ));
    }
    Ok((0..HIDDEN)
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
        .ok_or_else(|| format!("missing capture {name}"))?;
    if capture.dtype != "BF16"
        || capture.shape != shape
        || !is_hash(&capture.sha256)
        || bf16_hash(values) != capture.sha256
    {
        return Err(format!(
            "DeltaNet capture mismatch at step {} {name}",
            step.ordinal
        ));
    }
    Ok(())
}

fn require_f32(step: &Step, name: &str, shape: &[usize], values: &[f32]) -> Result<(), String> {
    let capture = step
        .captures
        .get(name)
        .ok_or_else(|| format!("missing capture {name}"))?;
    if capture.dtype != "F32"
        || capture.shape != shape
        || !is_hash(&capture.sha256)
        || f32_hash(values) != capture.sha256
    {
        return Err(format!(
            "DeltaNet capture mismatch at step {} {name}",
            step.ordinal
        ));
    }
    Ok(())
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

unsafe extern "C" {
    fn firewing_sleef_expf_u10(output: *mut f32, input: *const f32, count: usize);
    fn firewing_sleef_log1pf_u10(output: *mut f32, input: *const f32, count: usize);
    fn firewing_neon_sqrtf(output: *mut f32, input: *const f32, count: usize);
    fn firewing_neon_reciprocalf(output: *mut f32, input: *const f32, count: usize);
    fn firewing_neon_rsqrtf(output: *mut f32, input: *const f32, count: usize);
    fn firewing_sleef_sigmoidf(output: *mut f32, input: *const f32, count: usize);
    fn firewing_accelerate_padded_dot(left: *const f32, right: *const f32) -> f32;
}

fn sleef_exp(values: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0_f32; values.len()];
    // SAFETY: input and output cover `values.len()` F32 values and do not overlap.
    unsafe { firewing_sleef_expf_u10(output.as_mut_ptr(), values.as_ptr(), values.len()) };
    output
}

fn sleef_log1p(values: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0_f32; values.len()];
    // SAFETY: input and output cover `values.len()` F32 values and do not overlap.
    unsafe { firewing_sleef_log1pf_u10(output.as_mut_ptr(), values.as_ptr(), values.len()) };
    output
}

fn neon_sqrt(values: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0_f32; values.len()];
    // SAFETY: input and output cover `values.len()` F32 values and do not overlap.
    unsafe { firewing_neon_sqrtf(output.as_mut_ptr(), values.as_ptr(), values.len()) };
    output
}

fn neon_reciprocal(values: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0_f32; values.len()];
    // SAFETY: input and output cover `values.len()` F32 values and do not overlap.
    unsafe { firewing_neon_reciprocalf(output.as_mut_ptr(), values.as_ptr(), values.len()) };
    output
}

fn neon_rsqrt(values: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0_f32; values.len()];
    // SAFETY: input and output cover `values.len()` F32 values and do not overlap.
    unsafe { firewing_neon_rsqrtf(output.as_mut_ptr(), values.as_ptr(), values.len()) };
    output
}

fn sleef_sigmoid(values: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0_f32; values.len()];
    // SAFETY: input and output cover `values.len()` F32 values and do not overlap.
    unsafe { firewing_sleef_sigmoidf(output.as_mut_ptr(), values.as_ptr(), values.len()) };
    output
}

fn decay_values(a: &[u16], dt_bias: &[u16], a_log: &[u16]) -> Result<Vec<f32>, String> {
    let a_log_f32: Vec<_> = a_log.iter().map(|value| from_bf16(*value)).collect();
    let a_exp = sleef_exp(&a_log_f32);
    let shifted: Vec<_> = a
        .iter()
        .zip(dt_bias)
        .map(|(value, bias)| from_bf16(*value) + from_bf16(*bias))
        .collect();
    let shifted_exp = sleef_exp(&shifted);
    let mut softplus = sleef_log1p(&shifted_exp);
    for (output, input) in softplus.iter_mut().zip(&shifted) {
        if *input > 20.0 {
            *output = *input;
        }
    }
    Ok(a_exp
        .iter()
        .zip(softplus)
        .map(|(coefficient, step)| -*coefficient * step)
        .collect())
}

fn convolution(projection: &[u16], weight: &[u16], state: &mut [u16]) -> Vec<u16> {
    for channel in 0..CONV_DIM {
        let row = &mut state[channel * CONV_KERNEL..(channel + 1) * CONV_KERNEL];
        row.copy_within(1..CONV_KERNEL, 0);
        row[CONV_KERNEL - 1] = projection[channel];
    }
    (0..CONV_DIM)
        .map(|channel| {
            let row = &state[channel * CONV_KERNEL..(channel + 1) * CONV_KERNEL];
            let kernel = &weight[channel * CONV_KERNEL..(channel + 1) * CONV_KERNEL];
            let dot = pytorch_bf16_vector_dot(row, kernel);
            to_bf16(silu(from_bf16(to_bf16(dot))))
        })
        .collect()
}

fn repeat_heads(values: &[u16]) -> Vec<u16> {
    let mut output = Vec::with_capacity(V_HEADS * HEAD_DIM);
    for head in 0..K_HEADS {
        for _ in 0..V_HEADS / K_HEADS {
            output.extend_from_slice(&values[head * HEAD_DIM..(head + 1) * HEAD_DIM]);
        }
    }
    output
}

// ATen's cascade sum uses four interleaved vector accumulators for this
// 128-value contiguous BF16 reduction. Each eight-value BF16 load is first
// folded into four F32 lanes, the four streams are joined in order, and the
// final four lanes are accumulated into a scalar in order.
fn pytorch_bf16_sum(values: &[u16]) -> f32 {
    debug_assert_eq!(values.len(), HEAD_DIM);
    let mut partials = [[0.0_f32; 4]; 4];
    for (chunk, values) in values.chunks_exact(8).enumerate() {
        for lane in 0..4 {
            partials[chunk % 4][lane] += from_bf16(values[lane]) + from_bf16(values[lane + 4]);
        }
    }
    for stream in 1..4 {
        let (target, sources) = partials.split_at_mut(1);
        for (target, source) in target[0].iter_mut().zip(sources[stream - 1]) {
            *target += source;
        }
    }
    partials[0].iter().fold(0.0_f32, |sum, value| sum + value)
}

fn pytorch_f32_sum_128(values: &[f32]) -> f32 {
    debug_assert_eq!(values.len(), HEAD_DIM);
    let mut partials = [[0.0_f32; 4]; 4];
    for (chunk, values) in values.chunks_exact(4).enumerate() {
        for lane in 0..4 {
            partials[chunk % 4][lane] += values[lane];
        }
    }
    for stream in 1..4 {
        let (target, sources) = partials.split_at_mut(1);
        for (target, source) in target[0].iter_mut().zip(sources[stream - 1]) {
            *target += source;
        }
    }
    partials[0].iter().fold(0.0_f32, |sum, value| sum + value)
}

fn pytorch_f32_outer_sum_128(values: &[f32]) -> f32 {
    debug_assert_eq!(values.len(), HEAD_DIM);
    let mut half_streams = [[0.0_f32; 4]; 2];
    for (index, value) in values.iter().enumerate() {
        half_streams[index / 64][index % 4] += value;
    }
    let mut streams = [0.0_f32; 4];
    for stream in 0..4 {
        streams[stream] = half_streams[1][stream] + half_streams[0][stream];
    }
    ((streams[0] + streams[1]) + streams[2]) + streams[3]
}

fn normalize_heads(values: &[u16]) -> Vec<u16> {
    let mut output = Vec::with_capacity(values.len());
    for head in 0..V_HEADS {
        let row = &values[head * HEAD_DIM..(head + 1) * HEAD_DIM];
        let squares: Vec<_> = row
            .iter()
            .map(|value| to_bf16(from_bf16(*value) * from_bf16(*value)))
            .collect();
        let sum = to_bf16(pytorch_bf16_sum(&squares));
        let biased = from_bf16(to_bf16(from_bf16(sum) + 1.0e-6));
        let root = from_bf16(to_bf16(neon_sqrt(&[biased])[0]));
        let inverse = from_bf16(to_bf16(neon_reciprocal(&[root])[0]));
        output.extend(row.iter().map(|value| to_bf16(from_bf16(*value) * inverse)));
    }
    output
}

fn recurrent_step(
    query: &[u16],
    key: &[u16],
    value: &[u16],
    decay: &[f32],
    beta: &[u16],
    state: &mut [f32],
    initial_chunk: bool,
) -> Vec<u16> {
    let scale = 1.0_f32 / (HEAD_DIM as f32).sqrt();
    let retentions = sleef_exp(decay);
    let mut core = vec![0_u16; VALUE_DIM];
    for head in 0..V_HEADS {
        let q = &query[head * HEAD_DIM..(head + 1) * HEAD_DIM];
        let k = &key[head * HEAD_DIM..(head + 1) * HEAD_DIM];
        let v = &value[head * HEAD_DIM..(head + 1) * HEAD_DIM];
        let state = &mut state[head * HEAD_DIM * HEAD_DIM..(head + 1) * HEAD_DIM * HEAD_DIM];
        let retention = retentions[head];
        for cell in state.iter_mut() {
            *cell *= retention;
        }
        let mut memory = vec![0.0_f32; HEAD_DIM];
        for value_index in 0..HEAD_DIM {
            let products: Vec<_> = (0..HEAD_DIM)
                .map(|key_index| {
                    state[key_index * HEAD_DIM + value_index] * from_bf16(k[key_index])
                })
                .collect();
            memory[value_index] = pytorch_f32_outer_sum_128(&products);
        }
        let beta = from_bf16(beta[head]);
        let delta: Vec<_> = (0..HEAD_DIM)
            .map(|index| (from_bf16(v[index]) - memory[index]) * beta)
            .collect();
        let initial_attention = if initial_chunk {
            let scaled_query: Vec<_> = q.iter().map(|value| from_bf16(*value) * scale).collect();
            let float_key: Vec<_> = k.iter().map(|value| from_bf16(*value)).collect();
            // SAFETY: both inputs contain exactly HEAD_DIM contiguous F32 values.
            let attention = unsafe {
                firewing_accelerate_padded_dot(scaled_query.as_ptr(), float_key.as_ptr())
            };
            Some(attention)
        } else {
            None
        };
        let mut update = vec![0.0_f32; HEAD_DIM * HEAD_DIM];
        for key_index in 0..HEAD_DIM {
            for value_index in 0..HEAD_DIM {
                update[key_index * HEAD_DIM + value_index] =
                    from_bf16(k[key_index]) * delta[value_index];
            }
        }
        for (cell, update) in state.iter_mut().zip(update) {
            *cell += update;
        }
        let scaled_query: Vec<_> = q.iter().map(|value| from_bf16(*value) * scale).collect();
        for value_index in 0..HEAD_DIM {
            let sum = if let Some(attention) = initial_attention {
                attention * delta[value_index]
            } else {
                let products: Vec<_> = (0..HEAD_DIM)
                    .map(|key_index| {
                        state[key_index * HEAD_DIM + value_index] * scaled_query[key_index]
                    })
                    .collect();
                pytorch_f32_outer_sum_128(&products)
            };
            core[head * HEAD_DIM + value_index] = to_bf16(sum);
        }
    }
    core
}

fn gated_norm(core: &[u16], gate: &[u16], weight: &[u16]) -> Vec<u16> {
    let mut output = Vec::with_capacity(VALUE_DIM);
    let gate_f32: Vec<_> = gate.iter().map(|value| from_bf16(*value)).collect();
    let activated_gate = sleef_sigmoid(&gate_f32);
    for head in 0..V_HEADS {
        let row = &core[head * HEAD_DIM..(head + 1) * HEAD_DIM];
        let float: Vec<_> = row.iter().map(|value| from_bf16(*value)).collect();
        let squares: Vec<_> = float.iter().map(|value| value * value).collect();
        let variance = pytorch_f32_sum_128(&squares) / HEAD_DIM as f32;
        let inverse = neon_rsqrt(&[variance + 1.0e-6])[0];
        for index in 0..HEAD_DIM {
            let normalized = to_bf16(float[index] * inverse);
            let weighted = to_bf16(from_bf16(weight[index]) * from_bf16(normalized));
            output.push(to_bf16(
                from_bf16(weighted) * activated_gate[head * HEAD_DIM + index],
            ));
        }
    }
    output
}

pub fn verify_deltanet_fixture(
    checkpoint_dir: &Path,
    model_lock_path: &Path,
    fixture_path: &Path,
) -> Result<DeltaNetVerificationReport, String> {
    let fixture: Fixture =
        serde_json::from_slice(&fs::read(fixture_path).map_err(|error| error.to_string())?)
            .map_err(|error| format!("malformed DeltaNet fixture: {error}"))?;
    let config = &fixture.configuration;
    let case = &fixture.case;
    if fixture.schema_version != 1
        || fixture.semantic != "qwen3_8_flash_next_real_gated_deltanet_cached_decode"
        || fixture.model != MODEL
        || fixture.reference.implementation != "huggingface_transformers_qwen4_exp"
        || fixture.reference.transformers_version != "5.16.1"
        || fixture.reference.source
            != "transformers.models.qwen4_exp.modeling_qwen4_exp.Qwen4ExpTextGatedDeltaNet.forward"
        || config.hidden_size != HIDDEN
        || config.linear_num_key_heads != K_HEADS
        || config.linear_num_value_heads != V_HEADS
        || config.linear_key_head_dim != HEAD_DIM
        || config.linear_value_head_dim != HEAD_DIM
        || config.linear_conv_kernel_dim != CONV_KERNEL
        || config.key_dim != KEY_DIM
        || config.value_dim != VALUE_DIM
        || config.conv_dim != CONV_DIM
        || config.activation != "silu"
        || config.output_gate != "sigmoid"
        || config.weight_dtype != "BF16"
        || config.recurrent_state_dtype != "F32"
        || case.name != "layer_0_two_token_cached_decode"
        || case.layer != 0
        || case.tensors.len() != 9
        || case.steps.len() != 2
    {
        return Err("DeltaNet fixture identity or configuration is unsupported".to_owned());
    }
    if sha256_file(model_lock_path)? != fixture.reference.model_lock_sha256
        || sha256_file(&checkpoint_dir.join("config.json"))? != fixture.reference.config_sha256
        || sha256_file(&checkpoint_dir.join("model.safetensors.index.json"))?
            != fixture.reference.tensor_index_sha256
    {
        return Err("DeltaNet reference identity mismatch".to_owned());
    }
    let lock: ModelLock =
        serde_json::from_slice(&fs::read(model_lock_path).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    if lock.model != fixture.model || lock.revision != fixture.revision {
        return Err("DeltaNet model lock mismatch".to_owned());
    }
    let expected: [(&str, &[usize]); 9] = [
        ("A_log", &[V_HEADS]),
        ("conv1d.weight", &[CONV_DIM, 1, CONV_KERNEL]),
        ("dt_bias", &[V_HEADS]),
        ("in_proj_a.weight", &[V_HEADS, HIDDEN]),
        ("in_proj_b.weight", &[V_HEADS, HIDDEN]),
        ("in_proj_qkv.weight", &[CONV_DIM, HIDDEN]),
        ("in_proj_z.weight", &[VALUE_DIM, HIDDEN]),
        ("norm.weight", &[HEAD_DIM]),
        ("out_proj.weight", &[HIDDEN, VALUE_DIM]),
    ];
    let mut weights = BTreeMap::new();
    let mut tensor_payload_bytes = 0;
    for (local, shape) in expected {
        let tensor = case
            .tensors
            .get(local)
            .ok_or_else(|| format!("missing DeltaNet tensor {local}"))?;
        let expected_name = format!("model.language_model.layers.0.linear_attn.{local}");
        let matches: Vec<_> = lock
            .files
            .iter()
            .filter(|entry| entry.path == tensor.shard)
            .collect();
        if tensor.tensor != expected_name
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
            return Err(format!("DeltaNet tensor identity mismatch for {local}"));
        }
        let payload = read_tensor(&checkpoint_dir.join(&tensor.shard), &tensor.tensor, shape)?;
        if bf16_hash(&payload) != tensor.payload_sha256 {
            return Err(format!("DeltaNet tensor payload mismatch for {local}"));
        }
        tensor_payload_bytes += payload.len() * 2;
        weights.insert(local, payload);
    }
    let mut state = State {
        convolution: vec![to_bf16(0.0); CONV_DIM * CONV_KERNEL],
        recurrent: vec![0.0; RECURRENT_VALUES],
    };
    for (ordinal, step) in case.steps.iter().enumerate() {
        if step.ordinal != ordinal
            || step.mode
                != if ordinal == 0 {
                    "initial_chunk"
                } else {
                    "cached_recurrent"
                }
            || step.captures.len() != 20
        {
            return Err(format!("DeltaNet step {ordinal} metadata mismatch"));
        }
        let input = make_input(&step.input_spec, ordinal)?;
        require_bf16(step, "hidden_states", &[1, 1, HIDDEN], &input)?;
        let qkv = linear_bf16(&weights["in_proj_qkv.weight"], &input, CONV_DIM, HIDDEN);
        require_bf16(step, "mixed_qkv_projection", &[1, 1, CONV_DIM], &qkv)?;
        let z = linear_bf16(&weights["in_proj_z.weight"], &input, VALUE_DIM, HIDDEN);
        require_bf16(step, "z_projection", &[1, 1, V_HEADS, HEAD_DIM], &z)?;
        let b = linear_bf16(&weights["in_proj_b.weight"], &input, V_HEADS, HIDDEN);
        require_bf16(step, "b_projection", &[1, 1, V_HEADS], &b)?;
        let a = linear_bf16(&weights["in_proj_a.weight"], &input, V_HEADS, HIDDEN);
        require_bf16(step, "a_projection", &[1, 1, V_HEADS], &a)?;
        let convolved = convolution(&qkv, &weights["conv1d.weight"], &mut state.convolution);
        require_bf16(
            step,
            "convolution_state",
            &[1, CONV_DIM, CONV_KERNEL],
            &state.convolution,
        )?;
        require_bf16(step, "convolved_qkv", &[1, 1, CONV_DIM], &convolved)?;
        let query = &convolved[..KEY_DIM];
        let key = &convolved[KEY_DIM..2 * KEY_DIM];
        let value = &convolved[2 * KEY_DIM..];
        require_bf16(step, "query", &[1, 1, K_HEADS, HEAD_DIM], query)?;
        require_bf16(step, "key", &[1, 1, K_HEADS, HEAD_DIM], key)?;
        require_bf16(step, "value", &[1, 1, V_HEADS, HEAD_DIM], value)?;
        let beta: Vec<_> = b.iter().map(|value| sigmoid_bf16(*value)).collect();
        require_bf16(step, "beta", &[1, 1, V_HEADS], &beta)?;
        let decay = decay_values(&a, &weights["dt_bias"], &weights["A_log"])?;
        require_f32(step, "decay", &[1, 1, V_HEADS], &decay)?;
        let query_repeated = repeat_heads(query);
        let key_repeated = repeat_heads(key);
        require_bf16(
            step,
            "query_repeated",
            &[1, 1, V_HEADS, HEAD_DIM],
            &query_repeated,
        )?;
        require_bf16(
            step,
            "key_repeated",
            &[1, 1, V_HEADS, HEAD_DIM],
            &key_repeated,
        )?;
        let query_normalized = normalize_heads(&query_repeated);
        let key_normalized = normalize_heads(&key_repeated);
        require_bf16(
            step,
            "query_normalized",
            &[1, 1, V_HEADS, HEAD_DIM],
            &query_normalized,
        )?;
        require_bf16(
            step,
            "key_normalized",
            &[1, 1, V_HEADS, HEAD_DIM],
            &key_normalized,
        )?;
        let core = recurrent_step(
            &query_normalized,
            &key_normalized,
            value,
            &decay,
            &beta,
            &mut state.recurrent,
            ordinal == 0,
        );
        require_f32(
            step,
            "recurrent_state",
            &[1, V_HEADS, HEAD_DIM, HEAD_DIM],
            &state.recurrent,
        )?;
        require_bf16(step, "core_attention", &[1, 1, V_HEADS, HEAD_DIM], &core)?;
        let normed = gated_norm(&core, &z, &weights["norm.weight"]);
        require_bf16(step, "gated_norm", &[V_HEADS, HEAD_DIM], &normed)?;
        let output = linear_bf16(&weights["out_proj.weight"], &normed, HIDDEN, VALUE_DIM);
        require_bf16(step, "output", &[1, 1, HIDDEN], &output)?;
    }
    let bf16_count = case
        .steps
        .iter()
        .flat_map(|step| step.captures.values())
        .filter(|capture| capture.dtype == "BF16")
        .count();
    let f32_count = case
        .steps
        .iter()
        .flat_map(|step| step.captures.values())
        .filter(|capture| capture.dtype == "F32")
        .count();
    Ok(DeltaNetVerificationReport {
        schema_version: 1,
        semantic: "qwen3_8_flash_next_real_gated_deltanet_cached_decode_verification",
        model: fixture.model,
        revision: fixture.revision,
        layer: case.layer,
        steps_verified: 2,
        tensors_verified: 9,
        exact_bf16_capture_hashes: bf16_count,
        exact_f32_capture_hashes: f32_count,
        tensor_payload_bytes,
        convolution_state_bytes: CONV_DIM * CONV_KERNEL * 2,
        recurrent_state_bytes: RECURRENT_VALUES * 4,
        accepted_tokens: 0,
        performance_claim: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn state_sizes_match_configuration() {
        assert_eq!(CONV_DIM, 10_240);
        assert_eq!(RECURRENT_VALUES, 786_432);
        assert_eq!(VALUE_DIM, 6_144);
    }
    #[test]
    fn head_repeat_is_grouped() {
        let values: Vec<_> = (0..KEY_DIM).map(|value| value as u16).collect();
        let repeated = repeat_heads(&values);
        assert_eq!(&repeated[..HEAD_DIM], &values[..HEAD_DIM]);
        assert_eq!(&repeated[HEAD_DIM..2 * HEAD_DIM], &values[..HEAD_DIM]);
        assert_eq!(
            &repeated[3 * HEAD_DIM..4 * HEAD_DIM],
            &values[HEAD_DIM..2 * HEAD_DIM]
        );
    }

    #[test]
    fn sleef_softplus_matches_pytorch_vector_path() {
        let input = [-3.744_140_6_f32];
        let exp = sleef_exp(&input);
        let result = sleef_log1p(&exp);
        assert_eq!(result[0].to_bits(), 0x3cbf_886e);
    }
}
