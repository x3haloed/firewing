use crate::expert_acquisition::read_tensor_layout;
use crate::host_safety::{
    HostSafetyMonitor, HostSafetyPolicy, HostSafetySnapshot, PersistentResidencyDeclaration,
};
use crate::metal_moe::ExactResidentTop10Runner;
use crate::ngram::{
    AlignedBuffer, AlignedReadPlan, aligned_read_plan, invalidate_plan, process_disk_bytes_read,
    read_exact_at, resident_pages, set_uncached,
};
use crate::verify_mixture_fixture;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs::{self, File};
use std::io::Read;
use std::path::Path;
use std::sync::{Arc, Barrier};
use std::time::Instant;

const MODEL: &str = "Qwen/Qwen3.8-Flash-Next";
const REVISION: &str = "de4b8e4d43b917e7706784d8bb445c9af86a3540";
const ENDPOINT_SHA256: &str = "2eb3712c7959837a24fe6db5b5d7b1f87c9926b4dd80eae524590e43287331ca";
const Q2_ENDPOINT_SHA256: &str = "e2ccf01a37cc5cb2cf44a30185850b8910b06233bc32d7ddaaeb537204daa899";
const Q2_TRANSACTION_SHA256: &str =
    "9954668a28b64944c0830760a799383082e834be22106ec1613df12d748b9757";
const MODEL_LOCK_SHA256: &str = "f87399e8659ab3274601fcd455b78b73c600f57e5fc1e91499eec3ac1f4b9444";
const LAYERS: usize = 48;
const TOP_K: usize = 10;
const EXPERTS: usize = 512;
const HIDDEN: usize = 2560;
const INTERMEDIATE: usize = 640;
const EXPERT_BYTES: usize = 9_830_400;
const GATE_UP_BYTES: usize = 6_553_600;
const DOWN_BYTES: usize = 3_276_800;
const CACHE_EXPERTS: usize = 433;
const WORKERS: usize = 8;
const PAGE_BYTES: usize = 16 * 1024;
const SAMPLES: usize = 3;

#[derive(Deserialize)]
struct Endpoint {
    schema_version: u32,
    semantic: String,
    model: String,
    revision: String,
    reference: EndpointReference,
    layers: Vec<EndpointLayer>,
}

#[derive(Deserialize)]
struct EndpointReference {
    config_sha256: String,
    tensor_index_sha256: String,
    model_lock_sha256: String,
}

#[derive(Deserialize)]
struct EndpointLayer {
    layer: usize,
    decoder: Decoder,
}

#[derive(Deserialize)]
struct Decoder {
    expert_banks: ExpertBanks,
    steps: Vec<Step>,
}

#[derive(Deserialize)]
struct ExpertBanks {
    gate_up: TensorBank,
    down: TensorBank,
}

#[derive(Deserialize)]
struct TensorBank {
    tensor: String,
    shard: String,
    shard_bytes: u64,
    shard_sha256: String,
    shape: Vec<usize>,
    dtype: String,
}

#[derive(Deserialize)]
struct Step {
    ordinal: usize,
    selected_experts: Vec<usize>,
    expert_execution_order: Vec<usize>,
    experts: Vec<StepExpert>,
}

#[derive(Deserialize)]
struct StepExpert {
    expert: usize,
    gate_up_payload_sha256: String,
    down_payload_sha256: String,
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

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Identity {
    layer: usize,
    expert: usize,
}

#[derive(Clone)]
struct ReadExtent {
    identity: Identity,
    role: &'static str,
    shard: String,
    absolute_offset: u64,
    logical_bytes: usize,
    sha256: String,
}

#[derive(Default)]
struct WorkerResult {
    pread_ns: u128,
    install_copy_ns: u128,
    logical_bytes: usize,
    widened_bytes: usize,
    extents: usize,
}

#[derive(Debug, Serialize)]
pub struct OverlapTrial {
    pub token_ordinal: usize,
    pub mode: &'static str,
    pub sample_ordinal: usize,
    pub misses: usize,
    pub logical_bytes: usize,
    pub widened_bytes: usize,
    pub extents: usize,
    pub workers: usize,
    pub complete_wall_time_ns: u128,
    pub compute_wall_time_ns: Option<u128>,
    pub summed_pread_time_ns: u128,
    pub maximum_worker_pread_time_ns: u128,
    pub summed_install_copy_time_ns: u128,
    pub maximum_worker_install_copy_time_ns: u128,
    pub process_disk_bytes_read: u64,
    pub cold_prepare_wall_time_ns: u128,
    pub resident_page_instances_before: u64,
    pub resident_page_instances_after: u64,
}

#[derive(Debug, Serialize)]
pub struct OverlapSummary {
    pub token_ordinal: usize,
    pub mode: &'static str,
    pub samples: usize,
    pub complete_p10_wall_time_ns: u128,
    pub complete_median_wall_time_ns: u128,
    pub complete_p90_wall_time_ns: u128,
    pub process_disk_bytes_read_median: u64,
}

#[derive(Debug, Serialize)]
pub struct ExactOverlapBoundReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub implementation_commit: String,
    pub model: &'static str,
    pub revision: &'static str,
    pub endpoint_fixture_sha256: &'static str,
    pub model_lock_sha256: &'static str,
    pub device_name: String,
    pub cache_capacity_experts: usize,
    pub free_future_aware_initial_experts: usize,
    pub misses_by_token: [usize; 2],
    pub miss_bytes_by_token: [usize; 2],
    pub unique_miss_extents_verified: usize,
    pub unique_miss_bytes_verified: usize,
    pub authority_verification_wall_time_ns: u128,
    pub metal_warmups: usize,
    pub metal_executions_per_overlap_trial: usize,
    pub storage_workers: usize,
    pub maximum_phase_scoped_staging_bytes: usize,
    pub cache_state: &'static str,
    pub trials: Vec<OverlapTrial>,
    pub summaries: Vec<OverlapSummary>,
    pub paired_overlap_tps: Vec<String>,
    pub paired_overlap_tps_p10: String,
    pub paired_overlap_tps_median: String,
    pub paired_overlap_tps_p90: String,
    pub batch_size: usize,
    pub concurrency: usize,
    pub accepted_tokens: usize,
    #[serde(rename = "A")]
    pub accepted_per_verification: usize,
    #[serde(rename = "U")]
    pub expert_union: usize,
    pub favorable_grants: Vec<&'static str>,
    pub host_safety_policy: HostSafetyPolicy,
    pub host_safety_snapshots: Vec<HostSafetySnapshot>,
    pub performance_claim: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Q2ExactOverlapBoundReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub implementation_commit: String,
    pub model: &'static str,
    pub revision: &'static str,
    pub endpoint_fixture_sha256: &'static str,
    pub transaction_fixture_sha256: &'static str,
    pub model_lock_sha256: &'static str,
    pub device_name: String,
    pub target_step_ordinals: [usize; 2],
    pub cache_capacity_experts: usize,
    pub free_future_aware_initial_experts: usize,
    pub misses_by_target_row: [usize; 2],
    pub miss_bytes_by_target_row: [usize; 2],
    pub unique_miss_extents_verified: usize,
    pub unique_miss_bytes_verified: usize,
    pub authority_verification_wall_time_ns: u128,
    pub metal_warmups: usize,
    pub metal_executions_per_overlap_trial: usize,
    pub storage_workers: usize,
    pub maximum_phase_scoped_staging_bytes: usize,
    pub cache_state: &'static str,
    pub trials: Vec<OverlapTrial>,
    pub summaries: Vec<OverlapSummary>,
    pub paired_accepted_bound_tps: Vec<String>,
    pub paired_accepted_bound_tps_p10: String,
    pub paired_accepted_bound_tps_median: String,
    pub paired_accepted_bound_tps_p90: String,
    pub batch_size: usize,
    pub concurrency: usize,
    pub sampling: &'static str,
    pub q: usize,
    #[serde(rename = "A")]
    pub accepted_per_verification: usize,
    #[serde(rename = "U")]
    pub expert_union: f64,
    #[serde(rename = "A_over_U")]
    pub accepted_over_union: f64,
    pub target_union_expert_rows: usize,
    pub draft_unique_expert_rows: usize,
    pub favorable_grants: Vec<&'static str>,
    pub host_safety_policy: HostSafetyPolicy,
    pub host_safety_snapshots: Vec<HostSafetySnapshot>,
    pub performance_claim: Option<String>,
}

struct AuthoritySpec<'a> {
    endpoint_sha256: &'a str,
    endpoint_semantic: &'a str,
    step_start: usize,
    total_steps: usize,
    expected_misses: [usize; 2],
}

struct OverlapMeasurement {
    misses: [usize; 2],
    miss_bytes: [usize; 2],
    unique_miss_extents_verified: usize,
    unique_miss_bytes_verified: usize,
    authority_verification_wall_time_ns: u128,
    device_name: String,
    trials: Vec<OverlapTrial>,
    summaries: Vec<OverlapSummary>,
    paired_tps: Vec<f64>,
    ordered_tps: Vec<f64>,
    host_safety_policy: HostSafetyPolicy,
    host_safety_snapshots: Vec<HostSafetySnapshot>,
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file =
        File::open(path).map_err(|error| format!("cannot open {}: {error}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
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

fn load_authority(
    checkpoint_dir: &Path,
    model_lock_path: &Path,
    endpoint_path: &Path,
    spec: &AuthoritySpec<'_>,
) -> Result<(Endpoint, BTreeMap<String, LockedFile>), String> {
    if sha256_file(endpoint_path)? != spec.endpoint_sha256
        || sha256_file(model_lock_path)? != MODEL_LOCK_SHA256
    {
        return Err("overlap-bound authority hash mismatch".to_owned());
    }
    let endpoint: Endpoint =
        serde_json::from_slice(&fs::read(endpoint_path).map_err(|error| error.to_string())?)
            .map_err(|error| format!("malformed endpoint fixture: {error}"))?;
    if endpoint.schema_version != 1
        || endpoint.semantic != spec.endpoint_semantic
        || endpoint.model != MODEL
        || endpoint.revision != REVISION
        || endpoint.layers.len() != LAYERS
        || endpoint.reference.model_lock_sha256 != MODEL_LOCK_SHA256
        || !is_hash(&endpoint.reference.config_sha256)
        || !is_hash(&endpoint.reference.tensor_index_sha256)
        || sha256_file(&checkpoint_dir.join("config.json"))? != endpoint.reference.config_sha256
        || sha256_file(&checkpoint_dir.join("model.safetensors.index.json"))?
            != endpoint.reference.tensor_index_sha256
    {
        return Err("overlap-bound endpoint identity mismatch".to_owned());
    }
    let lock: ModelLock =
        serde_json::from_slice(&fs::read(model_lock_path).map_err(|error| error.to_string())?)
            .map_err(|error| format!("malformed model lock: {error}"))?;
    if lock.model != MODEL || lock.revision != REVISION {
        return Err("overlap-bound model lock identity mismatch".to_owned());
    }
    Ok((
        endpoint,
        lock.files
            .into_iter()
            .map(|entry| (entry.path.clone(), entry))
            .collect(),
    ))
}

fn validate_routes(
    endpoint: &Endpoint,
    step_start: usize,
    total_steps: usize,
) -> Result<Vec<BTreeSet<Identity>>, String> {
    let mut events = Vec::with_capacity(LAYERS * 2);
    for token in step_start..step_start + 2 {
        for (layer_index, layer) in endpoint.layers.iter().enumerate() {
            if layer.layer != layer_index || layer.decoder.steps.len() != total_steps {
                return Err("overlap-bound layer schedule mismatch".to_owned());
            }
            let step = &layer.decoder.steps[token];
            let selected = step
                .selected_experts
                .iter()
                .copied()
                .collect::<BTreeSet<_>>();
            let mut sorted = step.selected_experts.clone();
            sorted.sort_unstable();
            if step.ordinal != token
                || selected.len() != TOP_K
                || selected.iter().any(|expert| *expert >= EXPERTS)
                || sorted != step.expert_execution_order
                || step
                    .experts
                    .iter()
                    .map(|entry| entry.expert)
                    .collect::<Vec<_>>()
                    != step.expert_execution_order
                || step.experts.iter().any(|entry| {
                    !is_hash(&entry.gate_up_payload_sha256) || !is_hash(&entry.down_payload_sha256)
                })
            {
                return Err("overlap-bound route authority mismatch".to_owned());
            }
            events.push(
                selected
                    .into_iter()
                    .map(|expert| Identity {
                        layer: layer_index,
                        expert,
                    })
                    .collect(),
            );
        }
    }
    Ok(events)
}

fn validate_bank(
    checkpoint_dir: &Path,
    locked: &BTreeMap<String, LockedFile>,
    bank: &TensorBank,
    expected_tensor: &str,
    expected_shape: &[usize],
) -> Result<u64, String> {
    let entry = locked
        .get(&bank.shard)
        .ok_or_else(|| format!("model lock is missing {}", bank.shard))?;
    if bank.tensor != expected_tensor
        || bank.dtype != "BF16"
        || bank.shape != expected_shape
        || entry.size != bank.shard_bytes
        || entry.lfs_sha256.as_deref() != Some(bank.shard_sha256.as_str())
        || fs::metadata(checkpoint_dir.join(&bank.shard))
            .map_err(|error| error.to_string())?
            .len()
            != bank.shard_bytes
    {
        return Err(format!(
            "overlap-bound bank identity mismatch: {expected_tensor}"
        ));
    }
    let layout = read_tensor_layout(
        &checkpoint_dir.join(&bank.shard),
        &bank.tensor,
        expected_shape,
    )?;
    Ok(layout.absolute_offset)
}

fn miss_schedule(
    checkpoint_dir: &Path,
    locked: &BTreeMap<String, LockedFile>,
    endpoint: &Endpoint,
    spec: &AuthoritySpec<'_>,
) -> Result<[Vec<ReadExtent>; 2], String> {
    let events = validate_routes(endpoint, spec.step_start, spec.total_steps)?;
    let misses_by_event = belady_misses(&events, CACHE_EXPERTS)?;
    let mut result = [Vec::new(), Vec::new()];
    let mut layouts = Vec::with_capacity(LAYERS);
    for (layer_index, layer) in endpoint.layers.iter().enumerate() {
        let gate_tensor =
            format!("model.language_model.layers.{layer_index}.mlp.experts.gate_up_proj");
        let down_tensor =
            format!("model.language_model.layers.{layer_index}.mlp.experts.down_proj");
        let gate_base = validate_bank(
            checkpoint_dir,
            locked,
            &layer.decoder.expert_banks.gate_up,
            &gate_tensor,
            &[EXPERTS, INTERMEDIATE * 2, HIDDEN],
        )?;
        let down_base = validate_bank(
            checkpoint_dir,
            locked,
            &layer.decoder.expert_banks.down,
            &down_tensor,
            &[EXPERTS, HIDDEN, INTERMEDIATE],
        )?;
        layouts.push((gate_base, down_base));
    }

    for (position, misses) in misses_by_event.iter().enumerate() {
        let layer_index = position % LAYERS;
        let token = position / LAYERS;
        let step = &endpoint.layers[layer_index].decoder.steps[spec.step_start + token];
        let records = step
            .experts
            .iter()
            .map(|entry| (entry.expert, entry))
            .collect::<BTreeMap<_, _>>();
        for identity in misses {
            let record = records
                .get(&identity.expert)
                .ok_or_else(|| "overlap-bound expert record missing".to_owned())?;
            let banks = &endpoint.layers[layer_index].decoder.expert_banks;
            result[token].push(ReadExtent {
                identity: *identity,
                role: "gate_up",
                shard: banks.gate_up.shard.clone(),
                absolute_offset: layouts[layer_index].0 + (identity.expert * GATE_UP_BYTES) as u64,
                logical_bytes: GATE_UP_BYTES,
                sha256: record.gate_up_payload_sha256.clone(),
            });
            result[token].push(ReadExtent {
                identity: *identity,
                role: "down",
                shard: banks.down.shard.clone(),
                absolute_offset: layouts[layer_index].1 + (identity.expert * DOWN_BYTES) as u64,
                logical_bytes: DOWN_BYTES,
                sha256: record.down_payload_sha256.clone(),
            });
        }
    }
    let misses = [result[0].len() / 2, result[1].len() / 2];
    if misses != spec.expected_misses
        || result
            .iter()
            .flatten()
            .map(|extent| extent.logical_bytes)
            .sum::<usize>()
            != spec.expected_misses.iter().sum::<usize>() * EXPERT_BYTES
    {
        return Err("overlap-bound miss ledger differs from its frozen authority".to_owned());
    }
    Ok(result)
}

fn belady_misses(
    events: &[BTreeSet<Identity>],
    capacity: usize,
) -> Result<Vec<BTreeSet<Identity>>, String> {
    if events.is_empty()
        || events
            .iter()
            .any(|event| event.is_empty() || event.len() > capacity)
    {
        return Err("overlap-bound cache capacity cannot serve an event".to_owned());
    }
    let mut future: BTreeMap<Identity, VecDeque<usize>> = BTreeMap::new();
    for (position, demand) in events.iter().enumerate() {
        for identity in demand {
            future.entry(*identity).or_default().push_back(position);
        }
    }
    let mut initial = future.keys().copied().collect::<Vec<_>>();
    initial.sort_by_key(|identity| (future[identity][0], *identity));
    let mut resident = initial.into_iter().take(capacity).collect::<BTreeSet<_>>();
    let mut misses_by_event = Vec::with_capacity(events.len());
    for (position, demand) in events.iter().enumerate() {
        let misses = demand
            .difference(&resident)
            .copied()
            .collect::<BTreeSet<_>>();
        misses_by_event.push(misses.clone());
        resident.extend(misses);
        for identity in demand {
            if future.get_mut(identity).and_then(VecDeque::pop_front) != Some(position) {
                return Err("overlap-bound future-use replay mismatch".to_owned());
            }
        }
        while resident.len() > capacity {
            let victim = resident
                .difference(demand)
                .copied()
                .max_by_key(|identity| {
                    (
                        future[identity].front().copied().unwrap_or(usize::MAX),
                        *identity,
                    )
                })
                .ok_or_else(|| "overlap-bound cache cannot retain current demand".to_owned())?;
            resident.remove(&victim);
        }
    }
    Ok(misses_by_event)
}

fn open_handles(
    checkpoint_dir: &Path,
    extents: &[ReadExtent],
    uncached: bool,
) -> Result<BTreeMap<String, File>, String> {
    let mut handles = BTreeMap::new();
    for extent in extents {
        if !handles.contains_key(&extent.shard) {
            let file = File::open(checkpoint_dir.join(&extent.shard))
                .map_err(|error| format!("cannot open {}: {error}", extent.shard))?;
            if uncached {
                set_uncached(&file)?;
            }
            handles.insert(extent.shard.clone(), file);
        }
    }
    Ok(handles)
}

fn plans(
    handles: &BTreeMap<String, File>,
    extents: &[ReadExtent],
) -> Result<Vec<AlignedReadPlan>, String> {
    extents
        .iter()
        .map(|extent| {
            aligned_read_plan(
                extent.absolute_offset,
                extent.logical_bytes,
                handles[&extent.shard]
                    .metadata()
                    .map_err(|error| error.to_string())?
                    .len(),
                PAGE_BYTES,
            )
        })
        .collect()
}

fn verify_extents(checkpoint_dir: &Path, extents: &[ReadExtent]) -> Result<(usize, usize), String> {
    let handles = open_handles(checkpoint_dir, extents, false)?;
    let mut unique = BTreeMap::new();
    for extent in extents {
        unique
            .entry((
                extent.shard.clone(),
                extent.absolute_offset,
                extent.logical_bytes,
            ))
            .or_insert(extent);
    }
    let maximum = unique
        .values()
        .map(|extent| extent.logical_bytes)
        .max()
        .unwrap_or(1);
    let mut buffer = vec![0_u8; maximum];
    for extent in unique.values() {
        read_exact_at(
            &handles[&extent.shard],
            &mut buffer[..extent.logical_bytes],
            extent.absolute_offset,
        )?;
        let actual = format!("{:x}", Sha256::digest(&buffer[..extent.logical_bytes]));
        if actual != extent.sha256 {
            return Err(format!(
                "overlap-bound payload mismatch at layer {} expert {} {}",
                extent.identity.layer, extent.identity.expert, extent.role
            ));
        }
    }
    Ok((
        unique.len(),
        unique.values().map(|extent| extent.logical_bytes).sum(),
    ))
}

fn cold_prepare(
    handles: &BTreeMap<String, File>,
    extents: &[ReadExtent],
    plans: &[AlignedReadPlan],
) -> Result<(u128, u64, u64), String> {
    let count = || {
        extents
            .iter()
            .zip(plans)
            .try_fold(0_u64, |total, (extent, plan)| {
                total
                    .checked_add(resident_pages(&handles[&extent.shard], *plan, PAGE_BYTES)?)
                    .ok_or_else(|| "overlap-bound resident-page count overflow".to_owned())
            })
    };
    let before = count()?;
    let started = Instant::now();
    for (extent, plan) in extents.iter().zip(plans) {
        invalidate_plan(&handles[&extent.shard], *plan)?;
    }
    let elapsed = started.elapsed().as_nanos();
    let after = count()?;
    if after != 0 {
        return Err(format!(
            "overlap-bound cold preparation retained {after} page instances"
        ));
    }
    Ok((elapsed, before, after))
}

fn run_trial(
    checkpoint_dir: &Path,
    extents: &[ReadExtent],
    token_ordinal: usize,
    sample_ordinal: usize,
    overlap: bool,
    runner: &ExactResidentTop10Runner,
) -> Result<OverlapTrial, String> {
    let cached_handles = open_handles(checkpoint_dir, extents, false)?;
    let read_plans = plans(&cached_handles, extents)?;
    let (cold_prepare_wall_time_ns, resident_before, resident_after) =
        cold_prepare(&cached_handles, extents, &read_plans)?;
    drop(cached_handles);
    let handles = open_handles(checkpoint_dir, extents, true)?;
    let ready = Arc::new(Barrier::new(WORKERS + 1));
    let start = Arc::new(Barrier::new(WORKERS + 1));
    let joins = std::thread::scope(
        |scope| -> Result<(Vec<WorkerResult>, u128, Option<u128>, u64), String> {
            let mut joins = Vec::with_capacity(WORKERS);
            for worker in 0..WORKERS {
                let ready = Arc::clone(&ready);
                let start = Arc::clone(&start);
                let handles = &handles;
                let read_plans = &read_plans;
                joins.push(scope.spawn(move || -> Result<WorkerResult, String> {
                    let capacity = (worker..extents.len())
                        .step_by(WORKERS)
                        .map(|index| read_plans[index].physical_bytes)
                        .max()
                        .unwrap_or(1);
                    let install_capacity = (worker..extents.len())
                        .step_by(WORKERS)
                        .map(|index| extents[index].logical_bytes)
                        .max()
                        .unwrap_or(1);
                    let mut source = AlignedBuffer::new(capacity, PAGE_BYTES)?;
                    let mut installed = AlignedBuffer::new(install_capacity, PAGE_BYTES)?;
                    ready.wait();
                    start.wait();
                    let mut result = WorkerResult::default();
                    for index in (worker..extents.len()).step_by(WORKERS) {
                        let extent = &extents[index];
                        let plan = read_plans[index];
                        let pread_started = Instant::now();
                        read_exact_at(
                            &handles[&extent.shard],
                            &mut source.bytes_mut()[..plan.physical_bytes],
                            plan.physical_offset,
                        )?;
                        result.pread_ns += pread_started.elapsed().as_nanos();
                        let install_started = Instant::now();
                        installed.bytes_mut()[..extent.logical_bytes].copy_from_slice(
                            &source.bytes_mut()
                                [plan.logical_offset..plan.logical_offset + extent.logical_bytes],
                        );
                        std::hint::black_box(&installed.bytes_mut()[..extent.logical_bytes]);
                        result.install_copy_ns += install_started.elapsed().as_nanos();
                        result.logical_bytes += extent.logical_bytes;
                        result.widened_bytes += plan.physical_bytes;
                        result.extents += 1;
                    }
                    Ok(result)
                }));
            }
            ready.wait();
            let disk_before = process_disk_bytes_read()?;
            let wall_started = Instant::now();
            start.wait();
            let compute_wall_time_ns = if overlap {
                let compute_started = Instant::now();
                for _ in 0..LAYERS {
                    runner.execute_exact()?;
                }
                Some(compute_started.elapsed().as_nanos())
            } else {
                None
            };
            let results = joins
                .into_iter()
                .map(|join| {
                    join.join()
                        .map_err(|_| "overlap-bound storage worker panicked".to_owned())?
                })
                .collect::<Result<Vec<_>, String>>()?;
            let wall = wall_started.elapsed().as_nanos();
            let disk = process_disk_bytes_read()?
                .checked_sub(disk_before)
                .ok_or_else(|| "overlap-bound disk counter moved backwards".to_owned())?;
            Ok((results, wall, compute_wall_time_ns, disk))
        },
    )?;
    let (results, complete_wall_time_ns, compute_wall_time_ns, disk_bytes) = joins;
    let logical_bytes = results
        .iter()
        .map(|result| result.logical_bytes)
        .sum::<usize>();
    let extent_count = results.iter().map(|result| result.extents).sum::<usize>();
    if logical_bytes
        != extents
            .iter()
            .map(|extent| extent.logical_bytes)
            .sum::<usize>()
        || extent_count != extents.len()
        || disk_bytes == 0
    {
        return Err("overlap-bound timed transport ledger mismatch".to_owned());
    }
    Ok(OverlapTrial {
        token_ordinal,
        mode: if overlap {
            "storage_compute_overlap"
        } else {
            "storage_only_control"
        },
        sample_ordinal,
        misses: extents.len() / 2,
        logical_bytes,
        widened_bytes: results.iter().map(|result| result.widened_bytes).sum(),
        extents: extent_count,
        workers: WORKERS,
        complete_wall_time_ns,
        compute_wall_time_ns,
        summed_pread_time_ns: results.iter().map(|result| result.pread_ns).sum(),
        maximum_worker_pread_time_ns: results
            .iter()
            .map(|result| result.pread_ns)
            .max()
            .unwrap_or(0),
        summed_install_copy_time_ns: results.iter().map(|result| result.install_copy_ns).sum(),
        maximum_worker_install_copy_time_ns: results
            .iter()
            .map(|result| result.install_copy_ns)
            .max()
            .unwrap_or(0),
        process_disk_bytes_read: disk_bytes,
        cold_prepare_wall_time_ns,
        resident_page_instances_before: resident_before,
        resident_page_instances_after: resident_after,
    })
}

fn quantile_u128(values: &[u128], numerator: usize, denominator: usize) -> u128 {
    let mut ordered = values.to_vec();
    ordered.sort_unstable();
    ordered[((ordered.len() - 1) * numerator + denominator / 2) / denominator]
}

fn median_u64(values: &[u64]) -> u64 {
    let mut ordered = values.to_vec();
    ordered.sort_unstable();
    ordered[ordered.len() / 2]
}

fn summarize(trials: &[OverlapTrial], token: usize, mode: &'static str) -> OverlapSummary {
    let matching = trials
        .iter()
        .filter(|trial| trial.token_ordinal == token && trial.mode == mode)
        .collect::<Vec<_>>();
    let walls = matching
        .iter()
        .map(|trial| trial.complete_wall_time_ns)
        .collect::<Vec<_>>();
    let disks = matching
        .iter()
        .map(|trial| trial.process_disk_bytes_read)
        .collect::<Vec<_>>();
    OverlapSummary {
        token_ordinal: token,
        mode,
        samples: matching.len(),
        complete_p10_wall_time_ns: quantile_u128(&walls, 1, 10),
        complete_median_wall_time_ns: quantile_u128(&walls, 1, 2),
        complete_p90_wall_time_ns: quantile_u128(&walls, 9, 10),
        process_disk_bytes_read_median: median_u64(&disks),
    }
}

#[allow(clippy::too_many_arguments)]
fn measure_overlap(
    checkpoint_dir: &Path,
    model_lock_path: &Path,
    router_fixture_path: &Path,
    expert_fixture_path: &Path,
    mixture_fixture_path: &Path,
    endpoint_fixture_path: &Path,
    kernel_path: &Path,
    spec: &AuthoritySpec<'_>,
) -> Result<OverlapMeasurement, String> {
    let mut safety = HostSafetyMonitor::start_normative(vec![PersistentResidencyDeclaration {
        object: "exact_layer0_top10_bf16_metal_contention_probe".to_owned(),
        maximum_bytes: 98_398_736,
        lifetime: "overlap_measurement_series".to_owned(),
        eviction_order: 1,
    }])?;
    let authority_started = Instant::now();
    verify_mixture_fixture(
        checkpoint_dir,
        model_lock_path,
        router_fixture_path,
        expert_fixture_path,
        mixture_fixture_path,
    )?;
    let (endpoint, locked) =
        load_authority(checkpoint_dir, model_lock_path, endpoint_fixture_path, spec)?;
    let miss_extents = miss_schedule(checkpoint_dir, &locked, &endpoint, spec)?;
    let all_extents = miss_extents.iter().flatten().cloned().collect::<Vec<_>>();
    let (unique_miss_extents_verified, unique_miss_bytes_verified) =
        verify_extents(checkpoint_dir, &all_extents)?;
    let authority_verification_wall_time_ns = authority_started.elapsed().as_nanos();
    safety.checkpoint("authority_complete", true)?;

    let runner =
        ExactResidentTop10Runner::install(checkpoint_dir, mixture_fixture_path, kernel_path)?;
    for _ in 0..3 {
        runner.execute_exact()?;
    }
    safety.checkpoint("metal_warmups_complete", false)?;

    let order = [false, true, true, false, false, true];
    let mut trials = Vec::with_capacity(2 * order.len());
    for (token, extents) in miss_extents.iter().enumerate() {
        let mut controls = 0;
        let mut candidates = 0;
        for overlap in order {
            let ordinal = if overlap {
                let value = candidates;
                candidates += 1;
                value
            } else {
                let value = controls;
                controls += 1;
                value
            };
            trials.push(run_trial(
                checkpoint_dir,
                extents,
                token,
                ordinal,
                overlap,
                &runner,
            )?);
        }
        safety.checkpoint(&format!("token_{token}_trials_complete"), false)?;
    }
    let mut summaries = Vec::new();
    for token in 0..2 {
        summaries.push(summarize(&trials, token, "storage_only_control"));
        summaries.push(summarize(&trials, token, "storage_compute_overlap"));
    }
    let paired_ns = (0..SAMPLES)
        .map(|sample| {
            (0..2)
                .map(|token| {
                    trials
                        .iter()
                        .find(|trial| {
                            trial.token_ordinal == token
                                && trial.mode == "storage_compute_overlap"
                                && trial.sample_ordinal == sample
                        })
                        .map(|trial| trial.complete_wall_time_ns)
                        .ok_or_else(|| "overlap-bound paired trial missing".to_owned())
                })
                .sum::<Result<u128, String>>()
        })
        .collect::<Result<Vec<_>, String>>()?;
    let paired_tps = paired_ns
        .iter()
        .map(|elapsed| 2_000_000_000_f64 / *elapsed as f64)
        .collect::<Vec<_>>();
    let mut ordered_tps = paired_tps.clone();
    ordered_tps.sort_by(f64::total_cmp);
    let device_name = runner.device_name().to_owned();
    drop(runner);
    let (host_safety_policy, host_safety_snapshots) = safety.finish()?;
    Ok(OverlapMeasurement {
        misses: [miss_extents[0].len() / 2, miss_extents[1].len() / 2],
        miss_bytes: [
            miss_extents[0]
                .iter()
                .map(|extent| extent.logical_bytes)
                .sum(),
            miss_extents[1]
                .iter()
                .map(|extent| extent.logical_bytes)
                .sum(),
        ],
        unique_miss_extents_verified,
        unique_miss_bytes_verified,
        authority_verification_wall_time_ns,
        device_name,
        trials,
        summaries,
        paired_tps,
        ordered_tps,
        host_safety_policy,
        host_safety_snapshots,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn benchmark_exact_overlap_bound(
    checkpoint_dir: &Path,
    model_lock_path: &Path,
    router_fixture_path: &Path,
    expert_fixture_path: &Path,
    mixture_fixture_path: &Path,
    endpoint_fixture_path: &Path,
    kernel_path: &Path,
    implementation_commit: &str,
) -> Result<ExactOverlapBoundReport, String> {
    if implementation_commit.len() != 40
        || !implementation_commit
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        return Err("implementation commit must be a full hexadecimal Git hash".to_owned());
    }
    let measurement = measure_overlap(
        checkpoint_dir,
        model_lock_path,
        router_fixture_path,
        expert_fixture_path,
        mixture_fixture_path,
        endpoint_fixture_path,
        kernel_path,
        &AuthoritySpec {
            endpoint_sha256: ENDPOINT_SHA256,
            endpoint_semantic: "qwen3_8_flash_next_firewing_two_token_cached_text_logits",
            step_start: 0,
            total_steps: 2,
            expected_misses: [47, 379],
        },
    )?;
    let format_tps = |value: f64| format!("{value:.6}");
    Ok(ExactOverlapBoundReport {
        schema_version: 1,
        semantic: "qwen3_8_flash_next_exact_bf16_12gib_belady_miss_metal_overlap_favorable_bound",
        implementation_commit: implementation_commit.to_owned(),
        model: MODEL,
        revision: REVISION,
        endpoint_fixture_sha256: ENDPOINT_SHA256,
        model_lock_sha256: MODEL_LOCK_SHA256,
        device_name: measurement.device_name,
        cache_capacity_experts: CACHE_EXPERTS,
        free_future_aware_initial_experts: CACHE_EXPERTS,
        misses_by_token: measurement.misses,
        miss_bytes_by_token: measurement.miss_bytes,
        unique_miss_extents_verified: measurement.unique_miss_extents_verified,
        unique_miss_bytes_verified: measurement.unique_miss_bytes_verified,
        authority_verification_wall_time_ns: measurement.authority_verification_wall_time_ns,
        metal_warmups: 3,
        metal_executions_per_overlap_trial: LAYERS,
        storage_workers: WORKERS,
        maximum_phase_scoped_staging_bytes: WORKERS * 2 * 6_569_984,
        cache_state: "range_invalidated_page_aligned_f_nocache_f_rdahead_zero_into_preallocated_page_aligned_install_staging",
        trials: measurement.trials,
        summaries: measurement.summaries,
        paired_overlap_tps: measurement.paired_tps.into_iter().map(format_tps).collect(),
        paired_overlap_tps_p10: format_tps(measurement.ordered_tps[0]),
        paired_overlap_tps_median: format_tps(measurement.ordered_tps[1]),
        paired_overlap_tps_p90: format_tps(measurement.ordered_tps[2]),
        batch_size: 1,
        concurrency: 1,
        accepted_tokens: 0,
        accepted_per_verification: 0,
        expert_union: 0,
        favorable_grants: vec![
            "all fixed matrices and cache hits are free and consume no measured memory",
            "the initial 433-expert cache is free and future-aware",
            "Belady eviction metadata and victim handling are free",
            "miss reads may start for the whole token before layer dependencies reveal routes",
            "the exact layer-0 top10 Metal workload stands in for all 48 routed layers",
            "staging install copies are charged but cache-slot binding and eviction are free",
            "attention shared experts routing ngram final projection and sampling are free",
        ],
        host_safety_policy: measurement.host_safety_policy,
        host_safety_snapshots: measurement.host_safety_snapshots,
        performance_claim: None,
    })
}

fn validate_q2_transaction(path: &Path) -> Result<(), String> {
    if sha256_file(path)? != Q2_TRANSACTION_SHA256 {
        return Err("q2 overlap-bound transaction hash mismatch".to_owned());
    }
    let fixture: serde_json::Value =
        serde_json::from_slice(&fs::read(path).map_err(|error| error.to_string())?)
            .map_err(|error| format!("malformed q2 transaction fixture: {error}"))?;
    let number = |pointer: &str| fixture.pointer(pointer).and_then(serde_json::Value::as_u64);
    let text = |pointer: &str| fixture.pointer(pointer).and_then(serde_json::Value::as_str);
    let float = |pointer: &str| fixture.pointer(pointer).and_then(serde_json::Value::as_f64);
    if number("/schema_version") != Some(1)
        || text("/semantic") != Some("qwen3_8_flash_next_first_greedy_mtp_transaction")
        || text("/model") != Some(MODEL)
        || text("/revision") != Some(REVISION)
        || text("/reference/target_fixture_sha256") != Some(Q2_ENDPOINT_SHA256)
        || text("/configuration/sampling") != Some("greedy")
        || number("/configuration/batch_size") != Some(1)
        || number("/configuration/concurrency") != Some(1)
        || number("/configuration/q") != Some(2)
        || number("/configuration/target_layers") != Some(LAYERS as u64)
        || number("/configuration/top_k_experts") != Some(TOP_K as u64)
        || number("/configuration/expert_payload_bytes") != Some(EXPERT_BYTES as u64)
        || number("/decision/accepted_tokens") != Some(2)
        || number("/decision/rolled_back_proposal_rows") != Some(0)
        || number("/expert_union/target_union_expert_rows") != Some(687)
        || number("/expert_union/draft_unique_expert_rows") != Some(10)
        || number("/expert_union/combined_union_expert_rows") != Some(697)
        || number("/expert_union/one_token_expert_rows") != Some(480)
        || float("/expert_union/U") != Some(697.0 / 480.0)
        || float("/expert_union/A_over_U") != Some(2.0 / (697.0 / 480.0))
        || number("/claims/accepted_tokens") != Some(2)
        || !fixture
            .pointer("/claims/performance_claim")
            .is_some_and(serde_json::Value::is_null)
    {
        return Err("q2 overlap-bound transaction identity mismatch".to_owned());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn benchmark_q2_exact_overlap_bound(
    checkpoint_dir: &Path,
    model_lock_path: &Path,
    router_fixture_path: &Path,
    expert_fixture_path: &Path,
    mixture_fixture_path: &Path,
    endpoint_fixture_path: &Path,
    transaction_fixture_path: &Path,
    kernel_path: &Path,
    implementation_commit: &str,
) -> Result<Q2ExactOverlapBoundReport, String> {
    if implementation_commit.len() != 40
        || !implementation_commit
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        return Err("implementation commit must be a full hexadecimal Git hash".to_owned());
    }
    validate_q2_transaction(transaction_fixture_path)?;
    let measurement = measure_overlap(
        checkpoint_dir,
        model_lock_path,
        router_fixture_path,
        expert_fixture_path,
        mixture_fixture_path,
        endpoint_fixture_path,
        kernel_path,
        &AuthoritySpec {
            endpoint_sha256: Q2_ENDPOINT_SHA256,
            endpoint_semantic: "qwen3_8_flash_next_firewing_four_token_cached_text_logits",
            step_start: 2,
            total_steps: 4,
            expected_misses: [47, 207],
        },
    )?;
    let format_tps = |value: f64| format!("{value:.6}");
    Ok(Q2ExactOverlapBoundReport {
        schema_version: 1,
        semantic: "qwen3_8_flash_next_q2_exact_bf16_12gib_belady_miss_metal_overlap_favorable_bound",
        implementation_commit: implementation_commit.to_owned(),
        model: MODEL,
        revision: REVISION,
        endpoint_fixture_sha256: Q2_ENDPOINT_SHA256,
        transaction_fixture_sha256: Q2_TRANSACTION_SHA256,
        model_lock_sha256: MODEL_LOCK_SHA256,
        device_name: measurement.device_name,
        target_step_ordinals: [2, 3],
        cache_capacity_experts: CACHE_EXPERTS,
        free_future_aware_initial_experts: CACHE_EXPERTS,
        misses_by_target_row: measurement.misses,
        miss_bytes_by_target_row: measurement.miss_bytes,
        unique_miss_extents_verified: measurement.unique_miss_extents_verified,
        unique_miss_bytes_verified: measurement.unique_miss_bytes_verified,
        authority_verification_wall_time_ns: measurement.authority_verification_wall_time_ns,
        metal_warmups: 3,
        metal_executions_per_overlap_trial: LAYERS,
        storage_workers: WORKERS,
        maximum_phase_scoped_staging_bytes: WORKERS * 2 * 6_569_984,
        cache_state: "range_invalidated_page_aligned_f_nocache_f_rdahead_zero_into_preallocated_page_aligned_install_staging",
        trials: measurement.trials,
        summaries: measurement.summaries,
        paired_accepted_bound_tps: measurement.paired_tps.into_iter().map(format_tps).collect(),
        paired_accepted_bound_tps_p10: format_tps(measurement.ordered_tps[0]),
        paired_accepted_bound_tps_median: format_tps(measurement.ordered_tps[1]),
        paired_accepted_bound_tps_p90: format_tps(measurement.ordered_tps[2]),
        batch_size: 1,
        concurrency: 1,
        sampling: "greedy",
        q: 2,
        accepted_per_verification: 2,
        expert_union: 697.0 / 480.0,
        accepted_over_union: 2.0 / (697.0 / 480.0),
        target_union_expert_rows: 687,
        draft_unique_expert_rows: 10,
        favorable_grants: vec![
            "all MTP drafter work and its ten expert rows are free",
            "all fixed matrices and cache hits are free and consume no measured memory",
            "the initial 433-expert cache is free and future-aware",
            "Belady eviction metadata and victim handling are free",
            "miss reads may start for each exact target row before layer dependencies reveal routes",
            "the exact layer-0 top10 Metal workload stands in for all 48 routed layers per target row",
            "staging install copies are charged but cache-slot binding and eviction are free",
            "attention shared experts routing ngram final projection sampling rollback and synchronization are free",
        ],
        host_safety_policy: measurement.host_safety_policy,
        host_safety_snapshots: measurement.host_safety_snapshots,
        performance_claim: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantile_indices_are_stable_for_three_samples() {
        assert_eq!(quantile_u128(&[30, 10, 20], 1, 10), 10);
        assert_eq!(quantile_u128(&[30, 10, 20], 1, 2), 20);
        assert_eq!(quantile_u128(&[30, 10, 20], 9, 10), 30);
    }

    #[test]
    fn belady_uses_free_earliest_initial_contents_and_farthest_eviction() {
        let identity = |expert| Identity { layer: 0, expert };
        let events = [
            BTreeSet::from([identity(0), identity(1)]),
            BTreeSet::from([identity(1), identity(2)]),
            BTreeSet::from([identity(0), identity(2)]),
        ];
        let misses = belady_misses(&events, 2).unwrap();
        assert_eq!(misses[0], BTreeSet::new());
        assert_eq!(misses[1], BTreeSet::from([identity(2)]));
        assert_eq!(misses[2], BTreeSet::from([identity(0)]));
    }

    #[test]
    fn first_q2_target_rows_have_frozen_belady_miss_counts() {
        let endpoint: Endpoint = serde_json::from_str(include_str!(
            "../fixtures/endpoint/qwen3_8_flash_next_firewing_four_token.json"
        ))
        .unwrap();
        let events = validate_routes(&endpoint, 2, 4).unwrap();
        let misses = belady_misses(&events, CACHE_EXPERTS).unwrap();
        assert_eq!(
            [
                misses[..LAYERS].iter().map(BTreeSet::len).sum::<usize>(),
                misses[LAYERS..].iter().map(BTreeSet::len).sum::<usize>(),
            ],
            [47, 207]
        );
    }

    #[test]
    fn first_q2_transaction_is_bound_to_the_accepted_authority() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/mtp/qwen3_8_flash_next_first_transaction.json");
        validate_q2_transaction(&path).unwrap();
    }
}
