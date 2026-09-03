use crate::deltanet::read_tensor;
use crate::expert::{bf16_hash, from_bf16, linear_bf16, pytorch_bf16_vector_dot, to_bf16};
use crate::hyper_connection::grouped_rms;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

const MODEL: &str = "Qwen/Qwen3.8-Flash-Next";
const HIDDEN: usize = 2560;
const HC_COUNT: usize = 4;
const HC_HIDDEN: usize = HIDDEN * HC_COUNT;
const HEADS: usize = 16;
const HEAD_WIDTH: usize = 160;
const CONTEXT: usize = 2;
const CONV_STATE: usize = 9;

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
    ngram_fixture_sha256: String,
    ngram_row_fixture_sha256: String,
}

#[derive(Deserialize)]
struct Configuration {
    layer: usize,
    ple_layer_index: usize,
    hidden_size: usize,
    hc_count: usize,
    embedding_width: usize,
    ngram_heads: usize,
    head_width: usize,
    context_length: usize,
    conv_kernel_size: usize,
    conv_dilation: usize,
    conv_state_length: usize,
    boundary_dtype: String,
    token_state_dtype: String,
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
    previous_context: Vec<i64>,
    input_spec: InputSpec,
    rows: Vec<Row>,
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
struct Row {
    global_row: i64,
    part: i64,
    local_row: i64,
    tensor: String,
    shard: String,
    shard_bytes: u64,
    shard_sha256: String,
    data_offsets: Vec<u64>,
    payload_sha256: String,
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

#[derive(Debug, Serialize)]
pub struct PleVerificationReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub model: String,
    pub revision: String,
    pub layer: usize,
    pub steps_verified: usize,
    pub rows_verified: usize,
    pub exact_bf16_capture_hashes: usize,
    pub exact_i64_capture_hashes: usize,
    pub dense_tensors_verified: usize,
    pub dense_tensor_payload_bytes: usize,
    pub requested_embedding_bytes: usize,
    pub total_verified_payload_bytes: usize,
    pub convolution_state_bytes: usize,
    pub token_context_state_bytes: usize,
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

fn i64_hash(values: &[i64]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn require_bf16(step: &Step, name: &str, shape: &[usize], values: &[u16]) -> Result<(), String> {
    let capture = step
        .captures
        .get(name)
        .ok_or_else(|| format!("missing PLE capture {name}"))?;
    let actual = bf16_hash(values);
    if capture.dtype != "BF16"
        || capture.shape != shape
        || !is_hash(&capture.sha256)
        || capture.sha256 != actual
    {
        return Err(format!(
            "PLE capture mismatch at step {} {name}: expected {}, got {actual}",
            step.ordinal, capture.sha256
        ));
    }
    Ok(())
}

fn require_i64(step: &Step, name: &str, shape: &[usize], values: &[i64]) -> Result<(), String> {
    let capture = step
        .captures
        .get(name)
        .ok_or_else(|| format!("missing PLE capture {name}"))?;
    let actual = i64_hash(values);
    if capture.dtype != "I64"
        || capture.shape != shape
        || !is_hash(&capture.sha256)
        || capture.sha256 != actual
    {
        return Err(format!(
            "PLE capture mismatch at step {} {name}: expected {}, got {actual}",
            step.ordinal, capture.sha256
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
            "unsupported PLE input specification at step {ordinal}"
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

#[allow(clippy::needless_range_loop)]
fn pytorch_inner_sum(values: &[u16]) -> f32 {
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
                    accumulators[0][register][lane] += from_bf16(values[base + lane]);
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
                accumulators[0][register][lane] += from_bf16(values[base + lane]);
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
            accumulators[0][0][lane] += from_bf16(values[vector * LANES + lane]);
        }
    }
    for register in 1..INTERLEAVE {
        for lane in 0..LANES {
            accumulators[0][0][lane] += accumulators[0][register][lane];
        }
    }
    let mut result = values[vector_count * LANES..]
        .iter()
        .fold(0.0_f32, |sum, value| sum + from_bf16(*value));
    for lane in 0..LANES {
        result += accumulators[0][0][lane];
    }
    result
}

fn read_row(checkpoint_dir: &Path, lock: &ModelLock, row: &Row) -> Result<Vec<u16>, String> {
    if !(0..128).contains(&row.part)
        || row.local_row < 0
        || row.local_row >= 2_500_012
        || row.global_row != row.part * 2_500_012 + row.local_row
        || row.tensor
            != format!(
                "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{}.weight",
                row.part
            )
        || row.data_offsets.len() != 2
        || row.data_offsets[1].checked_sub(row.data_offsets[0])
            != Some(2_500_012 * HEAD_WIDTH as u64 * 2)
        || !is_hash(&row.shard_sha256)
        || !is_hash(&row.payload_sha256)
    {
        return Err("PLE sparse row identity mismatch".to_owned());
    }
    let records: Vec<_> = lock
        .files
        .iter()
        .filter(|entry| entry.path == row.shard)
        .collect();
    if records.len() != 1
        || records[0].size != row.shard_bytes
        || records[0].lfs_sha256.as_deref() != Some(row.shard_sha256.as_str())
    {
        return Err("PLE sparse row shard lock mismatch".to_owned());
    }
    let path = checkpoint_dir.join(&row.shard);
    if fs::metadata(&path)
        .map_err(|error| error.to_string())?
        .len()
        != row.shard_bytes
    {
        return Err("PLE sparse row shard size mismatch".to_owned());
    }
    let mut file = File::open(&path).map_err(|error| error.to_string())?;
    let mut raw = [0_u8; 8];
    file.read_exact(&mut raw)
        .map_err(|error| error.to_string())?;
    let header_bytes = u64::from_le_bytes(raw);
    if header_bytes == 0 || header_bytes > 16 * 1024 * 1024 {
        return Err("PLE sparse row header size mismatch".to_owned());
    }
    let mut header = vec![0_u8; header_bytes as usize];
    file.read_exact(&mut header)
        .map_err(|error| error.to_string())?;
    let header: Value = serde_json::from_slice(&header).map_err(|error| error.to_string())?;
    let descriptor = header
        .get(&row.tensor)
        .ok_or("PLE sparse row tensor missing from shard")?;
    let descriptor_shape = descriptor
        .get("shape")
        .and_then(Value::as_array)
        .ok_or("PLE sparse row shape missing")?;
    let descriptor_offsets = descriptor
        .get("data_offsets")
        .and_then(Value::as_array)
        .ok_or("PLE sparse row offsets missing")?;
    if descriptor.get("dtype").and_then(Value::as_str) != Some("BF16")
        || descriptor_shape.len() != 2
        || descriptor_shape[0].as_u64() != Some(2_500_012)
        || descriptor_shape[1].as_u64() != Some(HEAD_WIDTH as u64)
        || descriptor_offsets.len() != 2
        || descriptor_offsets[0].as_u64() != Some(row.data_offsets[0])
        || descriptor_offsets[1].as_u64() != Some(row.data_offsets[1])
    {
        return Err("PLE sparse row descriptor mismatch".to_owned());
    }
    let absolute = 8_u64
        .checked_add(header_bytes)
        .and_then(|value| value.checked_add(row.data_offsets[0]))
        .and_then(|value| value.checked_add(row.local_row as u64 * HEAD_WIDTH as u64 * 2))
        .ok_or("PLE sparse row offset overflow")?;
    file.seek(SeekFrom::Start(absolute))
        .map_err(|error| error.to_string())?;
    let mut bytes = vec![0_u8; HEAD_WIDTH * 2];
    file.read_exact(&mut bytes)
        .map_err(|error| error.to_string())?;
    if format!("{:x}", Sha256::digest(&bytes)) != row.payload_sha256 {
        return Err("PLE sparse row payload mismatch".to_owned());
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect())
}

fn expected_tensors() -> Vec<(&'static str, &'static str, Vec<usize>)> {
    vec![
        (
            "key_proj",
            "model.language_model.layers.1.ple.key_proj.weight",
            vec![HC_HIDDEN, HIDDEN],
        ),
        (
            "value_proj",
            "model.language_model.layers.1.ple.value_proj.weight",
            vec![HIDDEN, HIDDEN],
        ),
        (
            "norm_key",
            "model.language_model.layers.1.ple.norm_key.weight",
            vec![HC_HIDDEN],
        ),
        (
            "norm_query",
            "model.language_model.layers.1.ple.norm_query.weight",
            vec![HC_HIDDEN],
        ),
        (
            "norm_conv",
            "model.language_model.layers.1.ple.norm_conv.weight",
            vec![HC_HIDDEN],
        ),
        (
            "conv1d",
            "model.language_model.layers.1.ple.conv1d.weight",
            vec![HC_HIDDEN, 1, 4],
        ),
    ]
}

pub(crate) fn verify_ple_fixture_bytes_with_outputs(
    checkpoint_dir: &Path,
    model_lock_path: &Path,
    ngram_fixture_path: &Path,
    ngram_row_fixture_path: &Path,
    fixture_bytes: &[u8],
    expected_semantic: &str,
    hidden_overrides: Option<&[Vec<u16>]>,
) -> Result<(PleVerificationReport, Vec<Vec<u16>>), String> {
    let fixture: Fixture = serde_json::from_slice(fixture_bytes)
        .map_err(|error| format!("malformed PLE fixture: {error}"))?;
    let config = &fixture.configuration;
    let case = &fixture.case;
    if fixture.schema_version != 1
        || fixture.semantic != expected_semantic
        || fixture.model != MODEL
        || fixture.reference.implementation != "source_derived_huggingface_transformers_qwen4_exp"
        || fixture.reference.transformers_version != "5.16.1"
        || fixture.reference.source
            != "transformers.models.qwen4_exp.modeling_qwen4_exp.Qwen4ExpTextPLELayer.forward"
        || config.layer != 1
        || config.ple_layer_index != 0
        || config.hidden_size != HIDDEN
        || config.hc_count != HC_COUNT
        || config.embedding_width != HIDDEN
        || config.ngram_heads != HEADS
        || config.head_width != HEAD_WIDTH
        || config.context_length != CONTEXT
        || config.conv_kernel_size != 4
        || config.conv_dilation != 3
        || config.conv_state_length != CONV_STATE
        || config.boundary_dtype != "BF16"
        || config.token_state_dtype != "I64"
        || case.name != "layer_1_two_token_ple"
        || case.tensors.len() != 6
        || case.steps.len() != 2
        || hidden_overrides.is_some_and(|values| values.len() != case.steps.len())
    {
        return Err("PLE fixture identity or configuration is unsupported".to_owned());
    }
    if sha256_file(model_lock_path)? != fixture.reference.model_lock_sha256
        || sha256_file(&checkpoint_dir.join("config.json"))? != fixture.reference.config_sha256
        || sha256_file(&checkpoint_dir.join("model.safetensors.index.json"))?
            != fixture.reference.tensor_index_sha256
        || sha256_file(ngram_fixture_path)? != fixture.reference.ngram_fixture_sha256
        || sha256_file(ngram_row_fixture_path)? != fixture.reference.ngram_row_fixture_sha256
    {
        return Err("PLE reference identity mismatch".to_owned());
    }
    let lock: ModelLock =
        serde_json::from_slice(&fs::read(model_lock_path).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    if lock.model != fixture.model || lock.revision != fixture.revision {
        return Err("PLE model lock mismatch".to_owned());
    }

    let mut tensors = BTreeMap::new();
    let mut dense_tensor_payload_bytes = 0;
    for (key, name, shape) in expected_tensors() {
        let tensor = case
            .tensors
            .get(key)
            .ok_or_else(|| format!("missing PLE tensor {key}"))?;
        let records: Vec<_> = lock
            .files
            .iter()
            .filter(|entry| entry.path == tensor.shard)
            .collect();
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
            return Err(format!("PLE tensor identity mismatch for {key}"));
        }
        let payload = read_tensor(&checkpoint_dir.join(&tensor.shard), &tensor.tensor, &shape)?;
        if bf16_hash(&payload) != tensor.payload_sha256 {
            return Err(format!("PLE tensor payload mismatch for {key}"));
        }
        dense_tensor_payload_bytes += payload.len() * 2;
        tensors.insert(key.to_owned(), payload);
    }

    let mut context = vec![248_044_i64; CONTEXT];
    let mut convolution_state = vec![to_bf16(0.0); HC_HIDDEN * CONV_STATE];
    let mut unique_rows = BTreeSet::new();
    let mut outputs = Vec::with_capacity(case.steps.len());
    for (ordinal, step) in case.steps.iter().enumerate() {
        if step.ordinal != ordinal
            || step.mode
                != if ordinal == 0 {
                    "initial_chunk"
                } else {
                    "cached_recurrent"
                }
            || step.token_id != [42, 43][ordinal]
            || step.previous_context != context
            || step.rows.len() != HEADS
            || step.captures.len() != 16
        {
            return Err(format!("PLE step {ordinal} metadata mismatch"));
        }
        let generated_hidden = make_input(&step.input_spec, ordinal)?;
        let hidden = hidden_overrides
            .map(|values| values[ordinal].clone())
            .unwrap_or(generated_hidden);
        if hidden.len() != HC_HIDDEN {
            return Err(format!(
                "PLE hidden override shape mismatch at step {ordinal}"
            ));
        }
        require_bf16(step, "hidden_states", &[1, 1, HC_HIDDEN], &hidden)?;
        let mut embedding = Vec::with_capacity(HIDDEN);
        for row in &step.rows {
            unique_rows.insert(row.global_row);
            embedding.extend(read_row(checkpoint_dir, &lock, row)?);
        }
        require_bf16(step, "embedding", &[1, 1, HIDDEN], &embedding)?;
        let key_projection = linear_bf16(&tensors["key_proj"], &embedding, HC_HIDDEN, HIDDEN);
        require_bf16(step, "key_projection", &[1, 1, HC_HIDDEN], &key_projection)?;
        let key_normed = grouped_rms(&key_projection, &tensors["norm_key"], 1.0e-6);
        require_bf16(step, "key_normed", &[1, 1, HC_HIDDEN], &key_normed)?;
        let value = linear_bf16(&tensors["value_proj"], &embedding, HIDDEN, HIDDEN);
        require_bf16(step, "value", &[1, 1, HIDDEN], &value)?;
        let query_normed = grouped_rms(&hidden, &tensors["norm_query"], 1.0e-6);
        require_bf16(step, "query_normed", &[1, 1, HC_HIDDEN], &query_normed)?;
        let products: Vec<_> = key_normed
            .iter()
            .zip(&query_normed)
            .map(|(key, query)| to_bf16(from_bf16(*key) * from_bf16(*query)))
            .collect();
        require_bf16(
            step,
            "key_query_products",
            &[1, 1, HC_COUNT, HIDDEN],
            &products,
        )?;
        let gate: Vec<_> = products
            .chunks_exact(HIDDEN)
            .map(|group| {
                let sum = from_bf16(to_bf16(pytorch_inner_sum(group)));
                to_bf16(sum / (HIDDEN as f32).sqrt())
            })
            .collect();
        require_bf16(step, "gate", &[1, 1, HC_COUNT, 1], &gate)?;
        let transformed_gate: Vec<_> = gate
            .iter()
            .map(|value| {
                let value = from_bf16(*value);
                let root = from_bf16(to_bf16(value.abs().max(1.0e-6).sqrt()));
                to_bf16(root * value.signum())
            })
            .collect();
        require_bf16(
            step,
            "transformed_gate",
            &[1, 1, HC_COUNT, 1],
            &transformed_gate,
        )?;
        let gate_sigmoid: Vec<_> = transformed_gate
            .iter()
            .map(|value| {
                let value = from_bf16(*value);
                to_bf16(1.0 / (1.0 + (-value).exp()))
            })
            .collect();
        require_bf16(step, "gate_sigmoid", &[1, 1, HC_COUNT, 1], &gate_sigmoid)?;
        let gated_value: Vec<_> = gate_sigmoid
            .iter()
            .flat_map(|gate| {
                value
                    .iter()
                    .map(|value| to_bf16(from_bf16(*gate) * from_bf16(*value)))
            })
            .collect();
        require_bf16(step, "gated_value", &[1, 1, HC_HIDDEN], &gated_value)?;
        let gated_value_normed = grouped_rms(&gated_value, &tensors["norm_conv"], 1.0e-6);
        require_bf16(
            step,
            "gated_value_normed",
            &[1, 1, HC_HIDDEN],
            &gated_value_normed,
        )?;

        let convolution: Vec<_> = (0..HC_HIDDEN)
            .map(|channel| {
                let state = &convolution_state[channel * CONV_STATE..(channel + 1) * CONV_STATE];
                let dilated = [state[0], state[3], state[6], gated_value_normed[channel]];
                let kernel = &tensors["conv1d"][channel * 4..(channel + 1) * 4];
                let dot = from_bf16(to_bf16(pytorch_bf16_vector_dot(&dilated, kernel)));
                to_bf16(dot / (1.0 + (-dot).exp()))
            })
            .collect();
        for channel in 0..HC_HIDDEN {
            let state = &mut convolution_state[channel * CONV_STATE..(channel + 1) * CONV_STATE];
            state.copy_within(1..CONV_STATE, 0);
            state[CONV_STATE - 1] = gated_value_normed[channel];
        }
        require_bf16(
            step,
            "convolution_state",
            &[1, HC_HIDDEN, CONV_STATE],
            &convolution_state,
        )?;
        require_bf16(step, "convolution", &[1, 1, HC_HIDDEN], &convolution)?;
        let output: Vec<_> = gated_value
            .iter()
            .zip(&convolution)
            .map(|(left, right)| to_bf16(from_bf16(*left) + from_bf16(*right)))
            .collect();
        require_bf16(step, "output", &[1, 1, HC_HIDDEN], &output)?;
        outputs.push(output);
        context.remove(0);
        context.push(step.token_id);
        require_i64(step, "token_context_state", &[1, CONTEXT], &context)?;
    }

    let requested_embedding_bytes =
        case.steps.iter().map(|step| step.rows.len()).sum::<usize>() * HEAD_WIDTH * 2;
    Ok((
        PleVerificationReport {
            schema_version: 1,
            semantic: "qwen3_8_flash_next_layer1_ple_cached_decode_verification",
            model: fixture.model,
            revision: fixture.revision,
            layer: 1,
            steps_verified: 2,
            rows_verified: unique_rows.len(),
            exact_bf16_capture_hashes: 30,
            exact_i64_capture_hashes: 2,
            dense_tensors_verified: 6,
            dense_tensor_payload_bytes,
            requested_embedding_bytes,
            total_verified_payload_bytes: dense_tensor_payload_bytes + requested_embedding_bytes,
            convolution_state_bytes: convolution_state.len() * 2,
            token_context_state_bytes: context.len() * 8,
            accepted_tokens: 0,
            performance_claim: None,
        },
        outputs,
    ))
}

pub(crate) fn verify_ple_fixture_with_outputs(
    checkpoint_dir: &Path,
    model_lock_path: &Path,
    ngram_fixture_path: &Path,
    ngram_row_fixture_path: &Path,
    fixture_path: &Path,
) -> Result<(PleVerificationReport, Vec<Vec<u16>>), String> {
    let bytes = fs::read(fixture_path).map_err(|error| error.to_string())?;
    verify_ple_fixture_bytes_with_outputs(
        checkpoint_dir,
        model_lock_path,
        ngram_fixture_path,
        ngram_row_fixture_path,
        &bytes,
        "qwen3_8_flash_next_layer1_ple_cached_decode",
        None,
    )
}

pub fn verify_ple_fixture(
    checkpoint_dir: &Path,
    model_lock_path: &Path,
    ngram_fixture_path: &Path,
    ngram_row_fixture_path: &Path,
    fixture_path: &Path,
) -> Result<PleVerificationReport, String> {
    verify_ple_fixture_with_outputs(
        checkpoint_dir,
        model_lock_path,
        ngram_fixture_path,
        ngram_row_fixture_path,
        fixture_path,
    )
    .map(|(report, _)| report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dilation_three_uses_old_positions_zero_three_six_and_current() {
        let state: Vec<_> = (0..CONV_STATE).map(|value| to_bf16(value as f32)).collect();
        let selected = [state[0], state[3], state[6], to_bf16(9.0)];
        assert_eq!(
            selected.map(from_bf16),
            [0.0_f32, 3.0_f32, 6.0_f32, 9.0_f32]
        );
    }
}
