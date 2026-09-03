use crate::ngram::{
    AlignedBuffer, aligned_read_plan, invalidate_plan, process_disk_bytes_read, read_exact_at,
    resident_pages, set_uncached,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::Read;
use std::path::Path;
use std::time::Instant;

const MODEL: &str = "Qwen/Qwen3.8-Flash-Next";
const SEMANTIC: &str = "qwen3_8_flash_next_all_layer_selected_expert_acquisition";
const LAYERS: usize = 48;
const EXPERTS: usize = 512;
const TOP_K: usize = 10;
const HIDDEN: usize = 2560;
const INTERMEDIATE: usize = 640;
const GATE_UP_BYTES: usize = 6_553_600;
const DOWN_BYTES: usize = 3_276_800;
const TRACE_BYTES: usize = 4_718_592_000;
const EXTENTS: usize = LAYERS * TOP_K * 2;
const PAGE_BYTES: usize = 16 * 1024;
const MAX_DESTINATION_BYTES: usize = 256 * 1024 * 1024;
const WORKER_COUNTS: [usize; 4] = [1, 2, 4, 8];
const ORDERS: [[usize; 4]; 3] = [[1, 2, 4, 8], [8, 4, 2, 1], [2, 8, 1, 4]];

#[derive(Debug, Deserialize)]
struct Fixture {
    schema_version: u32,
    semantic: String,
    model: String,
    revision: String,
    reference: Reference,
    configuration: Configuration,
    gate_up_bytes_per_expert: usize,
    down_bytes_per_expert: usize,
    bytes_per_expert: usize,
    logical_bytes_per_trace: usize,
    shards: BTreeMap<String, Shard>,
    layers: Vec<Layer>,
}

#[derive(Debug, Deserialize)]
struct Reference {
    implementation: String,
    transformers_version: String,
    source: String,
    config_sha256: String,
    tensor_index_sha256: String,
    model_lock_sha256: String,
}

#[derive(Debug, Deserialize)]
struct Configuration {
    layers: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_experts: usize,
    top_k: usize,
    input_dtype: String,
    weight_dtype: String,
    expert_execution_order: String,
}

#[derive(Debug, Deserialize)]
struct Shard {
    bytes: u64,
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct Layer {
    layer: usize,
    input_spec: InputSpec,
    input_bf16_sha256: String,
    router_tensor: String,
    router_shard: String,
    top_k_selection_order: Vec<usize>,
    expert_execution_order: Vec<usize>,
    experts: Vec<Expert>,
}

#[derive(Debug, Deserialize)]
struct InputSpec {
    multiplier: i64,
    add: i64,
    modulus: i64,
    center: i64,
    divisor: i64,
    sparse_stride: usize,
}

#[derive(Debug, Deserialize)]
struct Expert {
    expert: usize,
    gate_up: Extent,
    down: Extent,
}

#[derive(Clone, Debug, Deserialize)]
struct Extent {
    tensor: String,
    shard: String,
    absolute_offset: u64,
    logical_bytes: usize,
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct ModelLock {
    model: String,
    revision: String,
    files: Vec<LockedFile>,
}

#[derive(Debug, Deserialize)]
struct LockedFile {
    path: String,
    size: u64,
    lfs_sha256: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ExpertAcquisitionTrial {
    pub cache_state: &'static str,
    pub order_ordinal: usize,
    pub workers: usize,
    pub complete_wall_ms: f64,
    pub summed_pread_ms: f64,
    pub summed_integrity_ms: f64,
    pub logical_bytes: usize,
    pub widened_bytes: usize,
    pub pread_calls: usize,
    pub extents_verified: usize,
    pub process_disk_bytes_read: u64,
    pub cold_prepare_ms: f64,
    pub resident_page_instances_before: Option<u64>,
    pub resident_page_instances_after: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct ExpertAcquisitionSummary {
    pub cache_state: &'static str,
    pub workers: usize,
    pub samples: usize,
    pub complete_wall_ms_p10: f64,
    pub complete_wall_ms_median: f64,
    pub complete_wall_ms_p90: f64,
    pub logical_gb_per_second_median: f64,
    pub process_disk_bytes_read_median: u64,
}

#[derive(Debug, Serialize)]
pub struct ExpertAcquisitionBenchmarkReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub model: String,
    pub revision: String,
    pub commit: String,
    pub fixture_sha256: String,
    pub model_lock_sha256: String,
    pub hardware: &'static str,
    pub checkpoint_storage: &'static str,
    pub page_bytes: usize,
    pub layers: usize,
    pub selected_experts_per_layer: usize,
    pub layer_expert_identities: usize,
    pub extents_per_trace: usize,
    pub logical_bytes_per_trace: usize,
    pub maximum_destination_capacity_bytes: usize,
    pub worker_counts: [usize; 4],
    pub interleaved_orders: [[usize; 4]; 3],
    pub warm_prefault_wall_ms: f64,
    pub trials: Vec<ExpertAcquisitionTrial>,
    pub summaries: Vec<ExpertAcquisitionSummary>,
    pub batch_size: usize,
    pub concurrency: usize,
    pub accepted_tokens: usize,
    #[serde(rename = "A")]
    pub accepted_per_verification: usize,
    #[serde(rename = "U")]
    pub expert_union: usize,
    pub performance_claim: Option<String>,
}

#[derive(Clone, Copy)]
struct TensorLayout {
    absolute_offset: u64,
    payload_bytes: u64,
}

struct WorkerResult {
    pread_ms: f64,
    integrity_ms: f64,
    logical_bytes: usize,
    widened_bytes: usize,
    extents_verified: usize,
}

fn require_hex(value: &str, length: usize) -> bool {
    value.len() == length && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let bytes =
        fs::read(path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn read_tensor_layout(
    path: &Path,
    tensor: &str,
    expected_shape: &[usize],
) -> Result<TensorLayout, String> {
    let mut file =
        File::open(path).map_err(|error| format!("cannot open {}: {error}", path.display()))?;
    let mut prefix = [0_u8; 8];
    file.read_exact(&mut prefix)
        .map_err(|error| format!("cannot read {} header: {error}", path.display()))?;
    let header_bytes = u64::from_le_bytes(prefix);
    if header_bytes == 0 || header_bytes > 16 * 1024 * 1024 {
        return Err(format!(
            "{} has an invalid safetensors header length",
            path.display()
        ));
    }
    let mut bytes = vec![0_u8; header_bytes as usize];
    file.read_exact(&mut bytes)
        .map_err(|error| format!("cannot read {} header: {error}", path.display()))?;
    let header: Value = serde_json::from_slice(&bytes).map_err(|error| {
        format!(
            "malformed safetensors header in {}: {error}",
            path.display()
        )
    })?;
    let descriptor = header
        .get(tensor)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("missing tensor {tensor}"))?;
    if descriptor.get("dtype").and_then(Value::as_str) != Some("BF16") {
        return Err(format!("tensor {tensor} is not BF16"));
    }
    let shape = descriptor
        .get("shape")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("tensor {tensor} has no shape"))?;
    if shape.len() != expected_shape.len()
        || !shape
            .iter()
            .zip(expected_shape)
            .all(|(actual, expected)| actual.as_u64() == Some(*expected as u64))
    {
        return Err(format!("tensor {tensor} has an unsupported shape"));
    }
    let offsets = descriptor
        .get("data_offsets")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("tensor {tensor} has no data offsets"))?;
    if offsets.len() != 2 {
        return Err(format!("tensor {tensor} has malformed data offsets"));
    }
    let start = offsets[0]
        .as_u64()
        .ok_or_else(|| format!("tensor {tensor} has malformed start offset"))?;
    let end = offsets[1]
        .as_u64()
        .ok_or_else(|| format!("tensor {tensor} has malformed end offset"))?;
    Ok(TensorLayout {
        absolute_offset: 8_u64
            .checked_add(header_bytes)
            .and_then(|value| value.checked_add(start))
            .ok_or_else(|| format!("tensor {tensor} offset overflow"))?,
        payload_bytes: end
            .checked_sub(start)
            .ok_or_else(|| format!("tensor {tensor} has reversed offsets"))?,
    })
}

fn validate_fixture(
    checkpoint_dir: &Path,
    model_lock_path: &Path,
    fixture_path: &Path,
) -> Result<(Fixture, Vec<Extent>, String, String), String> {
    let fixture_bytes = fs::read(fixture_path)
        .map_err(|error| format!("cannot read {}: {error}", fixture_path.display()))?;
    let fixture_hash = format!("{:x}", Sha256::digest(&fixture_bytes));
    let fixture: Fixture = serde_json::from_slice(&fixture_bytes)
        .map_err(|error| format!("malformed acquisition fixture: {error}"))?;
    if fixture.schema_version != 1
        || fixture.semantic != SEMANTIC
        || fixture.model != MODEL
        || !require_hex(&fixture.revision, 40)
        || fixture.reference.implementation != "huggingface_transformers_qwen4_exp"
        || fixture.reference.transformers_version != "5.16.1"
        || fixture.reference.source
            != "transformers.models.qwen4_exp.modeling_qwen4_exp.Qwen4ExpTextTopKRouter.forward"
        || fixture.configuration.layers != LAYERS
        || fixture.configuration.hidden_size != HIDDEN
        || fixture.configuration.intermediate_size != INTERMEDIATE
        || fixture.configuration.num_experts != EXPERTS
        || fixture.configuration.top_k != TOP_K
        || fixture.configuration.input_dtype != "BF16"
        || fixture.configuration.weight_dtype != "BF16"
        || fixture.configuration.expert_execution_order != "ascending_expert_id"
        || fixture.gate_up_bytes_per_expert != GATE_UP_BYTES
        || fixture.down_bytes_per_expert != DOWN_BYTES
        || fixture.bytes_per_expert != GATE_UP_BYTES + DOWN_BYTES
        || fixture.logical_bytes_per_trace != TRACE_BYTES
        || fixture.layers.len() != LAYERS
    {
        return Err(
            "acquisition fixture identity, reference, or configuration is unsupported".to_owned(),
        );
    }
    let lock_hash = sha256_file(model_lock_path)?;
    if lock_hash != fixture.reference.model_lock_sha256
        || sha256_file(&checkpoint_dir.join("config.json"))? != fixture.reference.config_sha256
        || sha256_file(&checkpoint_dir.join("model.safetensors.index.json"))?
            != fixture.reference.tensor_index_sha256
    {
        return Err("checkpoint metadata content identity mismatch".to_owned());
    }
    let lock: ModelLock = serde_json::from_slice(
        &fs::read(model_lock_path)
            .map_err(|error| format!("cannot read {}: {error}", model_lock_path.display()))?,
    )
    .map_err(|error| format!("malformed model lock: {error}"))?;
    if lock.model != fixture.model || lock.revision != fixture.revision {
        return Err("model lock identity does not match acquisition fixture".to_owned());
    }
    let locked: BTreeMap<_, _> = lock
        .files
        .iter()
        .map(|file| (file.path.as_str(), file))
        .collect();
    for (name, shard) in &fixture.shards {
        let entry = locked
            .get(name.as_str())
            .ok_or_else(|| format!("model lock has no entry for {name}"))?;
        if entry.size != shard.bytes
            || entry.lfs_sha256.as_deref() != Some(shard.sha256.as_str())
            || !require_hex(&shard.sha256, 64)
            || fs::metadata(checkpoint_dir.join(name))
                .map_err(|error| format!("cannot stat {name}: {error}"))?
                .len()
                != shard.bytes
        {
            return Err(format!("checkpoint shard identity mismatch for {name}"));
        }
    }
    let index: Value = serde_json::from_slice(
        &fs::read(checkpoint_dir.join("model.safetensors.index.json"))
            .map_err(|error| error.to_string())?,
    )
    .map_err(|error| format!("malformed tensor index: {error}"))?;
    let weight_map = index
        .get("weight_map")
        .and_then(Value::as_object)
        .ok_or_else(|| "tensor index has no weight_map".to_owned())?;
    let mut extents = Vec::with_capacity(EXTENTS);
    for (layer_index, layer) in fixture.layers.iter().enumerate() {
        let expected_router = format!("model.language_model.layers.{layer_index}.mlp.gate.weight");
        let expected_gate_up =
            format!("model.language_model.layers.{layer_index}.mlp.experts.gate_up_proj");
        let expected_down =
            format!("model.language_model.layers.{layer_index}.mlp.experts.down_proj");
        if layer.layer != layer_index
            || layer.router_tensor != expected_router
            || !fixture.shards.contains_key(&layer.router_shard)
            || weight_map.get(&expected_router).and_then(Value::as_str)
                != Some(layer.router_shard.as_str())
            || layer.input_spec.multiplier != 37 + 2 * layer_index as i64
            || layer.input_spec.add != 11 + 13 * layer_index as i64
            || layer.input_spec.modulus != 257
            || layer.input_spec.center != 128
            || layer.input_spec.divisor != 128
            || layer.input_spec.sparse_stride != 1
            || !require_hex(&layer.input_bf16_sha256, 64)
            || layer.top_k_selection_order.len() != TOP_K
            || layer.expert_execution_order.len() != TOP_K
            || layer.experts.len() != TOP_K
        {
            return Err(format!("layer {layer_index} metadata is unsupported"));
        }
        let selected: BTreeSet<_> = layer.top_k_selection_order.iter().copied().collect();
        let mut sorted = layer.top_k_selection_order.clone();
        sorted.sort_unstable();
        if selected.len() != TOP_K
            || selected.iter().any(|expert| *expert >= EXPERTS)
            || sorted != layer.expert_execution_order
        {
            return Err(format!("layer {layer_index} expert selection is invalid"));
        }
        let gate_up_shard = weight_map
            .get(&expected_gate_up)
            .and_then(Value::as_str)
            .ok_or_else(|| format!("tensor index is missing {expected_gate_up}"))?;
        let down_shard = weight_map
            .get(&expected_down)
            .and_then(Value::as_str)
            .ok_or_else(|| format!("tensor index is missing {expected_down}"))?;
        let gate_up_layout = read_tensor_layout(
            &checkpoint_dir.join(gate_up_shard),
            &expected_gate_up,
            &[EXPERTS, 2 * INTERMEDIATE, HIDDEN],
        )?;
        let down_layout = read_tensor_layout(
            &checkpoint_dir.join(down_shard),
            &expected_down,
            &[EXPERTS, HIDDEN, INTERMEDIATE],
        )?;
        if gate_up_layout.payload_bytes != (EXPERTS * GATE_UP_BYTES) as u64
            || down_layout.payload_bytes != (EXPERTS * DOWN_BYTES) as u64
        {
            return Err(format!("layer {layer_index} tensor payload size mismatch"));
        }
        for (position, expert) in layer.experts.iter().enumerate() {
            if expert.expert != layer.expert_execution_order[position] {
                return Err(format!(
                    "layer {layer_index} expert execution order mismatch"
                ));
            }
            for (extent, tensor, shard, layout, logical_bytes) in [
                (
                    &expert.gate_up,
                    expected_gate_up.as_str(),
                    gate_up_shard,
                    gate_up_layout,
                    GATE_UP_BYTES,
                ),
                (
                    &expert.down,
                    expected_down.as_str(),
                    down_shard,
                    down_layout,
                    DOWN_BYTES,
                ),
            ] {
                let expected_offset = layout
                    .absolute_offset
                    .checked_add((expert.expert * logical_bytes) as u64)
                    .ok_or_else(|| "expert offset overflow".to_owned())?;
                if extent.tensor != tensor
                    || extent.shard != shard
                    || extent.logical_bytes != logical_bytes
                    || extent.absolute_offset != expected_offset
                    || !require_hex(&extent.sha256, 64)
                {
                    return Err(format!(
                        "layer {layer_index} expert {} extent mismatch",
                        expert.expert
                    ));
                }
                extents.push(extent.clone());
            }
        }
    }
    if extents.len() != EXTENTS
        || extents
            .iter()
            .map(|extent| extent.logical_bytes)
            .sum::<usize>()
            != TRACE_BYTES
    {
        return Err("acquisition extent ledger mismatch".to_owned());
    }
    Ok((fixture, extents, fixture_hash, lock_hash))
}

fn open_handles(
    checkpoint_dir: &Path,
    extents: &[Extent],
    uncached: bool,
) -> Result<BTreeMap<String, File>, String> {
    let mut handles = BTreeMap::new();
    for extent in extents {
        if !handles.contains_key(&extent.shard) {
            let path = checkpoint_dir.join(&extent.shard);
            let file = File::open(&path)
                .map_err(|error| format!("cannot open {}: {error}", path.display()))?;
            if uncached {
                set_uncached(&file)?;
            }
            handles.insert(extent.shard.clone(), file);
        }
    }
    Ok(handles)
}

fn run_trial(
    checkpoint_dir: &Path,
    extents: &[Extent],
    workers: usize,
    cold: bool,
    order_ordinal: usize,
) -> Result<ExpertAcquisitionTrial, String> {
    if !WORKER_COUNTS.contains(&workers) {
        return Err("unsupported acquisition worker count".to_owned());
    }
    let handles = open_handles(checkpoint_dir, extents, false)?;
    let plans = extents
        .iter()
        .map(|extent| {
            let size = handles[&extent.shard]
                .metadata()
                .map_err(|error| error.to_string())?
                .len();
            aligned_read_plan(
                extent.absolute_offset,
                extent.logical_bytes,
                size,
                PAGE_BYTES,
            )
        })
        .collect::<Result<Vec<_>, String>>()?;
    let destination_capacity = workers
        .checked_mul(
            plans
                .iter()
                .map(|plan| plan.physical_bytes)
                .max()
                .unwrap_or(0),
        )
        .ok_or_else(|| "destination capacity overflow".to_owned())?;
    if destination_capacity >= MAX_DESTINATION_BYTES {
        return Err("acquisition destination exceeds the frozen safety bound".to_owned());
    }
    let (cold_prepare_ms, resident_before, resident_after) = if cold {
        let count = || {
            extents
                .iter()
                .zip(&plans)
                .try_fold(0_u64, |total, (extent, plan)| {
                    resident_pages(&handles[&extent.shard], *plan, PAGE_BYTES).and_then(|value| {
                        total
                            .checked_add(value)
                            .ok_or_else(|| "resident page count overflow".to_owned())
                    })
                })
        };
        let before = count()?;
        let started = Instant::now();
        for (extent, plan) in extents.iter().zip(&plans) {
            invalidate_plan(&handles[&extent.shard], *plan)?;
        }
        let elapsed = started.elapsed().as_secs_f64() * 1000.0;
        let after = count()?;
        (elapsed, Some(before), Some(after))
    } else {
        (0.0, None, None)
    };
    drop(handles);
    let handles = open_handles(checkpoint_dir, extents, cold)?;
    let disk_before = process_disk_bytes_read()?;
    let started = Instant::now();
    let results = std::thread::scope(|scope| {
        let mut joins = Vec::new();
        for worker in 0..workers {
            let handles = &handles;
            let plans = &plans;
            joins.push(scope.spawn(move || -> Result<WorkerResult, String> {
                let capacity = (worker..extents.len())
                    .step_by(workers)
                    .map(|index| {
                        if cold {
                            plans[index].physical_bytes
                        } else {
                            extents[index].logical_bytes
                        }
                    })
                    .max()
                    .unwrap_or(1);
                let mut buffer = AlignedBuffer::new(capacity, PAGE_BYTES)?;
                let mut result = WorkerResult {
                    pread_ms: 0.0,
                    integrity_ms: 0.0,
                    logical_bytes: 0,
                    widened_bytes: 0,
                    extents_verified: 0,
                };
                for index in (worker..extents.len()).step_by(workers) {
                    let extent = &extents[index];
                    let plan = plans[index];
                    let read_bytes = if cold {
                        plan.physical_bytes
                    } else {
                        extent.logical_bytes
                    };
                    let read_offset = if cold {
                        plan.physical_offset
                    } else {
                        extent.absolute_offset
                    };
                    let read_started = Instant::now();
                    read_exact_at(
                        &handles[&extent.shard],
                        &mut buffer.bytes_mut()[..read_bytes],
                        read_offset,
                    )?;
                    result.pread_ms += read_started.elapsed().as_secs_f64() * 1000.0;
                    let logical = if cold {
                        &buffer.bytes_mut()
                            [plan.logical_offset..plan.logical_offset + extent.logical_bytes]
                    } else {
                        &buffer.bytes_mut()[..extent.logical_bytes]
                    };
                    let hash_started = Instant::now();
                    let actual = format!("{:x}", Sha256::digest(logical));
                    result.integrity_ms += hash_started.elapsed().as_secs_f64() * 1000.0;
                    if actual != extent.sha256 {
                        return Err(format!("payload hash mismatch at extent {index}"));
                    }
                    result.logical_bytes += extent.logical_bytes;
                    result.widened_bytes += read_bytes;
                    result.extents_verified += 1;
                }
                Ok(result)
            }));
        }
        joins
            .into_iter()
            .map(|join| {
                join.join()
                    .map_err(|_| "acquisition worker panicked".to_owned())?
            })
            .collect::<Result<Vec<_>, String>>()
    })?;
    let complete_wall_ms = started.elapsed().as_secs_f64() * 1000.0;
    let disk_after = process_disk_bytes_read()?;
    let logical_bytes = results.iter().map(|result| result.logical_bytes).sum();
    let extents_verified = results.iter().map(|result| result.extents_verified).sum();
    if logical_bytes != TRACE_BYTES || extents_verified != EXTENTS {
        return Err("worker acquisition ledger mismatch".to_owned());
    }
    Ok(ExpertAcquisitionTrial {
        cache_state: if cold {
            "range_invalidated_page_aligned_f_nocache_f_rdahead_zero"
        } else {
            "prefaulted_cacheable_exact_pread"
        },
        order_ordinal,
        workers,
        complete_wall_ms,
        summed_pread_ms: results.iter().map(|result| result.pread_ms).sum(),
        summed_integrity_ms: results.iter().map(|result| result.integrity_ms).sum(),
        logical_bytes,
        widened_bytes: results.iter().map(|result| result.widened_bytes).sum(),
        pread_calls: EXTENTS,
        extents_verified,
        process_disk_bytes_read: disk_after
            .checked_sub(disk_before)
            .ok_or_else(|| "process disk byte counter moved backwards".to_owned())?,
        cold_prepare_ms,
        resident_page_instances_before: resident_before,
        resident_page_instances_after: resident_after,
    })
}

fn quantile(values: &[f64], fraction: f64) -> f64 {
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    ordered[((ordered.len() - 1) as f64 * fraction).round() as usize]
}

fn median_u64(values: &[u64]) -> u64 {
    let mut ordered = values.to_vec();
    ordered.sort_unstable();
    ordered[ordered.len() / 2]
}

pub fn benchmark_expert_acquisition(
    checkpoint_dir: &Path,
    model_lock_path: &Path,
    fixture_path: &Path,
    commit: &str,
) -> Result<ExpertAcquisitionBenchmarkReport, String> {
    if !require_hex(commit, 40) {
        return Err("implementation commit must be a full hexadecimal Git commit".to_owned());
    }
    let (fixture, extents, fixture_hash, model_lock_hash) =
        validate_fixture(checkpoint_dir, model_lock_path, fixture_path)?;
    let mut trials = Vec::with_capacity(24);
    for (order_ordinal, order) in ORDERS.iter().enumerate() {
        for workers in order {
            trials.push(run_trial(
                checkpoint_dir,
                &extents,
                *workers,
                true,
                order_ordinal,
            )?);
        }
    }
    let prefault_started = Instant::now();
    let _prefault = run_trial(checkpoint_dir, &extents, 1, false, usize::MAX)?;
    let warm_prefault_wall_ms = prefault_started.elapsed().as_secs_f64() * 1000.0;
    for (order_ordinal, order) in ORDERS.iter().enumerate() {
        for workers in order {
            trials.push(run_trial(
                checkpoint_dir,
                &extents,
                *workers,
                false,
                order_ordinal,
            )?);
        }
    }
    let mut summaries = Vec::new();
    for cache_state in [
        "range_invalidated_page_aligned_f_nocache_f_rdahead_zero",
        "prefaulted_cacheable_exact_pread",
    ] {
        for workers in WORKER_COUNTS {
            let matching: Vec<_> = trials
                .iter()
                .filter(|trial| trial.cache_state == cache_state && trial.workers == workers)
                .collect();
            let walls: Vec<_> = matching
                .iter()
                .map(|trial| trial.complete_wall_ms)
                .collect();
            let disk: Vec<_> = matching
                .iter()
                .map(|trial| trial.process_disk_bytes_read)
                .collect();
            let median = quantile(&walls, 0.5);
            summaries.push(ExpertAcquisitionSummary {
                cache_state,
                workers,
                samples: matching.len(),
                complete_wall_ms_p10: quantile(&walls, 0.1),
                complete_wall_ms_median: median,
                complete_wall_ms_p90: quantile(&walls, 0.9),
                logical_gb_per_second_median: TRACE_BYTES as f64
                    / 1_000_000_000.0
                    / (median / 1000.0),
                process_disk_bytes_read_median: median_u64(&disk),
            });
        }
    }
    let maximum_plan = extents
        .iter()
        .map(|extent| {
            aligned_read_plan(
                extent.absolute_offset,
                extent.logical_bytes,
                fixture.shards[&extent.shard].bytes,
                PAGE_BYTES,
            )
        })
        .collect::<Result<Vec<_>, _>>()?
        .iter()
        .map(|plan| plan.physical_bytes)
        .max()
        .unwrap_or(0);
    Ok(ExpertAcquisitionBenchmarkReport {
        schema_version: 1,
        semantic: "qwen3_8_flash_next_all_layer_source_expert_acquisition_diagnostic",
        model: fixture.model,
        revision: fixture.revision,
        commit: commit.to_owned(),
        fixture_sha256: fixture_hash,
        model_lock_sha256: model_lock_hash,
        hardware: "Apple M1 Mac mini Macmini9,1 16 GiB",
        checkpoint_storage: "internal_ssd",
        page_bytes: PAGE_BYTES,
        layers: LAYERS,
        selected_experts_per_layer: TOP_K,
        layer_expert_identities: LAYERS * TOP_K,
        extents_per_trace: EXTENTS,
        logical_bytes_per_trace: TRACE_BYTES,
        maximum_destination_capacity_bytes: maximum_plan * WORKER_COUNTS[3],
        worker_counts: WORKER_COUNTS,
        interleaved_orders: ORDERS,
        warm_prefault_wall_ms,
        trials,
        summaries,
        batch_size: 1,
        concurrency: 1,
        accepted_tokens: 0,
        accepted_per_verification: 0,
        expert_union: LAYERS * TOP_K,
        performance_claim: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_byte_ledger_is_exact() {
        assert_eq!(LAYERS * TOP_K * (GATE_UP_BYTES + DOWN_BYTES), TRACE_BYTES);
        assert_eq!(EXTENTS, 960);
    }

    #[test]
    fn aligned_plan_contains_logical_range() {
        let plan = aligned_read_plan(20_001, GATE_UP_BYTES, 10_000_000, PAGE_BYTES).unwrap();
        assert_eq!(plan.physical_offset % PAGE_BYTES as u64, 0);
        assert!(plan.logical_offset + GATE_UP_BYTES <= plan.physical_bytes);
    }
}
