use crate::host_safety::{
    HostSafetyMonitor, HostSafetyPolicy, HostSafetySnapshot, PersistentResidencyDeclaration,
};
use crate::metal_moe::ExactResidentTop10Runner;
use crate::ngram::{
    AlignedBuffer, AlignedReadPlan, aligned_read_plan, invalidate_plan, process_disk_bytes_read,
    read_exact_at, resident_pages, set_uncached,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::Read;
use std::path::Path;
use std::sync::{Arc, Barrier};
use std::time::Instant;

const MODEL: &str = "Qwen/Qwen3.8-Flash-Next";
const REVISION: &str = "de4b8e4d43b917e7706784d8bb445c9af86a3540";
const MODEL_LOCK_SHA256: &str = "f87399e8659ab3274601fcd455b78b73c600f57e5fc1e91499eec3ac1f4b9444";
const ENDPOINT_SHA256: &str = "e2ccf01a37cc5cb2cf44a30185850b8910b06233bc32d7ddaaeb537204daa899";
const TRANSACTION_SHA256: &str = "9954668a28b64944c0830760a799383082e834be22106ec1613df12d748b9757";
const FW0044_SHA256: &str = "6aa3f7cc04d35a8686ceb6c3c0b55f22b548129b67db242975b6693c79d5d6f9";
const CONTAINER_MANIFEST_SHA256: &str =
    "893fa5739e4d4e22f23f5306d2e32ef33bb17af54a7e631fdf5b1286e63cc863";
const CONTAINER_SHA256: &str = "bcc410a162445937641f4b5c894eccab9547c23e2cf4e9a3bf233a41edb93b87";
const BUILDER_COMMIT: &str = "a782e771f3ec4067ad4430865938defcc591108b";
const EXPERT_BYTES: usize = 9_830_400;
const PAGE_BYTES: usize = 16 * 1024;
const LAYERS: usize = 48;
const TOP_K: usize = 10;
const RECORDS: usize = 687;
const COMPRESSED_BYTES: usize = 5_251_840_172;
const PHYSICAL_BYTES: usize = 5_257_854_976;
const CACHE_BYTES: usize = 4_260_902_888;
const SAMPLES: usize = 3;
const WORKERS: usize = 8;

#[derive(Deserialize)]
struct Manifest {
    schema_version: u32,
    semantic: String,
    implementation_commit: String,
    model: String,
    revision: String,
    authorities: Authorities,
    codec: Codec,
    page_bytes: usize,
    source_bytes_per_expert: usize,
    fixed_resident_bytes: usize,
    resident_limit_bytes: usize,
    compressed_cache_bytes: usize,
    records: Vec<Record>,
    target_rows: Vec<Vec<Vec<String>>>,
    source_bytes: usize,
    compressed_bytes: usize,
    physical_bytes: usize,
    container_file: String,
    container_sha256: String,
    exact_round_trips: usize,
    batch_size: usize,
    concurrency: usize,
    sampling: String,
    q: usize,
    #[serde(rename = "A")]
    accepted: usize,
    #[serde(rename = "U")]
    union: f64,
    performance_claim: Option<String>,
}

#[derive(Deserialize)]
struct Authorities {
    model_lock_sha256: String,
    endpoint_fixture_sha256: String,
    transaction_fixture_sha256: String,
    fw_0044_receipt_sha256: String,
}

#[derive(Deserialize)]
struct Codec {
    name: String,
    python_package_version: String,
    level: i32,
    frame_content_size: bool,
    independent_frames: bool,
}

#[derive(Clone, Deserialize)]
struct Record {
    identity: String,
    layer: usize,
    expert: usize,
    offset: u64,
    compressed_bytes: usize,
    physical_bytes: usize,
    frame_sha256: String,
    source_sha256: String,
}

#[derive(Default)]
struct WorkerResult {
    pread_ns: u128,
    decompression_ns: u128,
    compressed_bytes: usize,
    physical_bytes: usize,
    source_bytes: usize,
    frames: usize,
}

#[derive(Debug, Serialize)]
pub struct CompressedOverlapTrial {
    pub mode: &'static str,
    pub sample_ordinal: usize,
    pub workers: usize,
    pub frames: usize,
    pub compressed_bytes: usize,
    pub physical_bytes: usize,
    pub source_bytes: usize,
    pub complete_wall_time_ns: u128,
    pub compute_wall_time_ns: Option<u128>,
    pub summed_pread_time_ns: u128,
    pub maximum_worker_pread_time_ns: u128,
    pub summed_decompression_time_ns: u128,
    pub maximum_worker_decompression_time_ns: u128,
    pub process_disk_bytes_read: u64,
    pub cold_prepare_wall_time_ns: u128,
    pub resident_page_instances_before: u64,
    pub resident_page_instances_after: u64,
}

#[derive(Debug, Serialize)]
pub struct ParallelZstdOverlapReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub implementation_commit: String,
    pub model: &'static str,
    pub revision: &'static str,
    pub builder_commit: &'static str,
    pub manifest_sha256: &'static str,
    pub container_sha256: &'static str,
    pub device_name: String,
    pub codec: &'static str,
    pub page_bytes: usize,
    pub expert_frames: usize,
    pub compressed_cache_bytes: usize,
    pub first_target_row_frames: usize,
    pub repeated_frames: usize,
    pub second_only_frames: usize,
    pub free_initial_frames: usize,
    pub free_initial_compressed_bytes: usize,
    pub miss_frames: usize,
    pub miss_compressed_bytes: usize,
    pub miss_physical_bytes: usize,
    pub miss_source_bytes: usize,
    pub authority_verification_wall_time_ns: u128,
    pub exact_round_trips: usize,
    pub diagnostic_worker_trials: Vec<CompressedOverlapTrial>,
    pub interleaved_trials: Vec<CompressedOverlapTrial>,
    pub control_p10_wall_time_ns: u128,
    pub control_median_wall_time_ns: u128,
    pub control_p90_wall_time_ns: u128,
    pub candidate_p10_wall_time_ns: u128,
    pub candidate_median_wall_time_ns: u128,
    pub candidate_p90_wall_time_ns: u128,
    pub accepted_bound_tps: Vec<String>,
    pub accepted_bound_tps_p10: String,
    pub accepted_bound_tps_median: String,
    pub accepted_bound_tps_p90: String,
    pub metal_warmups: usize,
    pub metal_executions_per_candidate: usize,
    pub maximum_phase_scoped_staging_bytes: usize,
    pub cache_state: &'static str,
    pub batch_size: usize,
    pub concurrency: usize,
    pub sampling: &'static str,
    pub q: usize,
    #[serde(rename = "A")]
    pub accepted: usize,
    #[serde(rename = "U")]
    pub union: f64,
    pub rolled_back_proposal_rows: usize,
    pub favorable_grants: Vec<&'static str>,
    pub host_safety_policy: HostSafetyPolicy,
    pub host_safety_snapshots: Vec<HostSafetySnapshot>,
    pub performance_claim: Option<String>,
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

fn load_manifest(path: &Path, container_path: &Path) -> Result<Manifest, String> {
    if sha256_file(path)? != CONTAINER_MANIFEST_SHA256 {
        return Err("compressed-overlap manifest hash mismatch".to_owned());
    }
    let manifest: Manifest =
        serde_json::from_slice(&fs::read(path).map_err(|error| error.to_string())?)
            .map_err(|error| format!("malformed compressed-overlap manifest: {error}"))?;
    if manifest.schema_version != 1
        || manifest.semantic != "qwen3_8_flash_next_q2_exact_zstd1_page_aligned_expert_container"
        || manifest.implementation_commit != BUILDER_COMMIT
        || manifest.model != MODEL
        || manifest.revision != REVISION
        || manifest.authorities.model_lock_sha256 != MODEL_LOCK_SHA256
        || manifest.authorities.endpoint_fixture_sha256 != ENDPOINT_SHA256
        || manifest.authorities.transaction_fixture_sha256 != TRANSACTION_SHA256
        || manifest.authorities.fw_0044_receipt_sha256 != FW0044_SHA256
        || manifest.codec.name != "zstandard"
        || manifest.codec.python_package_version != "0.25.0"
        || manifest.codec.level != 1
        || !manifest.codec.frame_content_size
        || !manifest.codec.independent_frames
        || manifest.page_bytes != PAGE_BYTES
        || manifest.source_bytes_per_expert != EXPERT_BYTES
        || manifest.fixed_resident_bytes != 8_623_999_000
        || manifest.resident_limit_bytes != 12 * 1024usize.pow(3)
        || manifest.compressed_cache_bytes != CACHE_BYTES
        || manifest.records.len() != RECORDS
        || manifest.source_bytes != RECORDS * EXPERT_BYTES
        || manifest.compressed_bytes != COMPRESSED_BYTES
        || manifest.physical_bytes != PHYSICAL_BYTES
        || manifest.container_file != "q2-zstd1.fwz"
        || manifest.container_sha256 != CONTAINER_SHA256
        || manifest.exact_round_trips != RECORDS
        || manifest.batch_size != 1
        || manifest.concurrency != 1
        || manifest.sampling != "greedy"
        || manifest.q != 2
        || manifest.accepted != 2
        || manifest.union != 697.0 / 480.0
        || manifest.performance_claim.is_some()
        || fs::metadata(container_path)
            .map_err(|error| error.to_string())?
            .len()
            != PHYSICAL_BYTES as u64
        || sha256_file(container_path)? != CONTAINER_SHA256
    {
        return Err("compressed-overlap manifest identity mismatch".to_owned());
    }
    let mut expected_offset = 0_u64;
    let mut identities = BTreeSet::new();
    for record in &manifest.records {
        if record.identity != format!("{}:{}", record.layer, record.expert)
            || record.layer >= LAYERS
            || record.expert >= 512
            || record.offset != expected_offset
            || record.offset % PAGE_BYTES as u64 != 0
            || record.compressed_bytes == 0
            || record.physical_bytes < record.compressed_bytes
            || record.physical_bytes % PAGE_BYTES != 0
            || !is_hash(&record.frame_sha256)
            || !is_hash(&record.source_sha256)
            || !identities.insert(record.identity.clone())
        {
            return Err("compressed-overlap record layout mismatch".to_owned());
        }
        expected_offset = expected_offset
            .checked_add(record.physical_bytes as u64)
            .ok_or_else(|| "compressed-overlap offset overflow".to_owned())?;
    }
    if expected_offset != PHYSICAL_BYTES as u64
        || manifest.target_rows.len() != 2
        || manifest
            .target_rows
            .iter()
            .any(|row| row.len() != LAYERS || row.iter().any(|event| event.len() != TOP_K))
        || manifest
            .target_rows
            .iter()
            .flatten()
            .flatten()
            .any(|identity| !identities.contains(identity))
        || manifest
            .target_rows
            .iter()
            .flatten()
            .flatten()
            .cloned()
            .collect::<BTreeSet<_>>()
            != identities
    {
        return Err("compressed-overlap route union mismatch".to_owned());
    }
    Ok(manifest)
}

fn verify_frames(container: &File, records: &[Record]) -> Result<(), String> {
    let maximum = records
        .iter()
        .map(|record| record.compressed_bytes)
        .max()
        .unwrap_or(1);
    let mut encoded = vec![0_u8; maximum];
    let mut decoded = vec![0_u8; EXPERT_BYTES];
    let mut decompressor = zstd::bulk::Decompressor::new().map_err(|error| error.to_string())?;
    for record in records {
        let source = &mut encoded[..record.compressed_bytes];
        read_exact_at(container, source, record.offset)?;
        if format!("{:x}", Sha256::digest(&*source)) != record.frame_sha256 {
            return Err(format!(
                "compressed frame hash mismatch: {}",
                record.identity
            ));
        }
        let count = decompressor
            .decompress_to_buffer(source, &mut decoded)
            .map_err(|error| error.to_string())?;
        if count != EXPERT_BYTES
            || format!("{:x}", Sha256::digest(&decoded[..count])) != record.source_sha256
        {
            return Err(format!("decompressed source mismatch: {}", record.identity));
        }
    }
    Ok(())
}

struct Schedule {
    misses: Vec<Record>,
    first_count: usize,
    repeated_count: usize,
    second_only_count: usize,
    initial_count: usize,
    initial_bytes: usize,
}

fn choose_initial(
    records: &BTreeMap<String, &Record>,
    first: &BTreeSet<String>,
    second: &BTreeSet<String>,
    capacity: usize,
) -> Result<(BTreeSet<String>, usize), String> {
    let first_bytes = first
        .iter()
        .map(|identity| records[identity].compressed_bytes)
        .sum::<usize>();
    if first_bytes > capacity {
        return Err("compressed-overlap first target row does not fit cache".to_owned());
    }
    let mut candidates = second.difference(first).cloned().collect::<Vec<_>>();
    candidates.sort_by_key(|identity| {
        (
            std::cmp::Reverse(records[identity].compressed_bytes),
            identity.clone(),
        )
    });
    let mut initial = first.clone();
    let mut initial_bytes = first_bytes;
    for identity in candidates {
        let bytes = records[&identity].compressed_bytes;
        if initial_bytes + bytes <= capacity {
            initial.insert(identity);
            initial_bytes += bytes;
        }
    }
    Ok((initial, initial_bytes))
}

fn build_schedule(manifest: &Manifest) -> Result<Schedule, String> {
    let records = manifest
        .records
        .iter()
        .map(|record| (record.identity.clone(), record))
        .collect::<BTreeMap<_, _>>();
    let first = manifest.target_rows[0]
        .iter()
        .flatten()
        .cloned()
        .collect::<BTreeSet<_>>();
    let second = manifest.target_rows[1]
        .iter()
        .flatten()
        .cloned()
        .collect::<BTreeSet<_>>();
    let second_only = second.difference(&first).cloned().collect::<BTreeSet<_>>();
    let repeated = first.intersection(&second).count();
    if first.len() != 480 || second.len() != 480 || repeated != 273 || second_only.len() != 207 {
        return Err("compressed-overlap target-row set mismatch".to_owned());
    }
    let (initial, initial_bytes) = choose_initial(&records, &first, &second, CACHE_BYTES)?;
    let misses = second_only
        .difference(&initial)
        .map(|identity| (*records[identity]).clone())
        .collect::<Vec<_>>();
    let second_bytes = second
        .iter()
        .map(|identity| records[identity].compressed_bytes)
        .sum::<usize>();
    if second_bytes > CACHE_BYTES || misses.is_empty() {
        return Err("compressed-overlap second target row cache mismatch".to_owned());
    }
    Ok(Schedule {
        misses,
        first_count: 480,
        repeated_count: repeated,
        second_only_count: second_only.len(),
        initial_count: initial.len(),
        initial_bytes,
    })
}

fn cold_prepare(file: &File, plans: &[AlignedReadPlan]) -> Result<(u128, u64, u64), String> {
    let count = || {
        plans.iter().try_fold(0_u64, |total, plan| {
            total
                .checked_add(resident_pages(file, *plan, PAGE_BYTES)?)
                .ok_or_else(|| "compressed-overlap resident-page overflow".to_owned())
        })
    };
    let before = count()?;
    let started = Instant::now();
    for plan in plans {
        invalidate_plan(file, *plan)?;
    }
    let elapsed = started.elapsed().as_nanos();
    let after = count()?;
    if after != 0 {
        return Err(format!(
            "compressed-overlap cold preparation retained {after} page instances"
        ));
    }
    Ok((elapsed, before, after))
}

fn run_trial(
    container_path: &Path,
    misses: &[Record],
    workers: usize,
    sample_ordinal: usize,
    overlap: bool,
    runner: &ExactResidentTop10Runner,
) -> Result<CompressedOverlapTrial, String> {
    if workers == 0 || workers > WORKERS {
        return Err("compressed-overlap worker count out of bounds".to_owned());
    }
    let file = File::open(container_path).map_err(|error| error.to_string())?;
    let file_bytes = file.metadata().map_err(|error| error.to_string())?.len();
    let plans = misses
        .iter()
        .map(|record| {
            aligned_read_plan(
                record.offset,
                record.compressed_bytes,
                file_bytes,
                PAGE_BYTES,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    if misses
        .iter()
        .zip(&plans)
        .any(|(record, plan)| plan.physical_bytes != record.physical_bytes)
    {
        return Err("compressed-overlap aligned plan mismatch".to_owned());
    }
    let (cold_prepare_wall_time_ns, resident_before, resident_after) = cold_prepare(&file, &plans)?;
    set_uncached(&file)?;
    let ready = Arc::new(Barrier::new(workers + 1));
    let start = Arc::new(Barrier::new(workers + 1));
    let (results, complete_wall_time_ns, compute_wall_time_ns, disk_bytes) =
        std::thread::scope(|scope| -> Result<_, String> {
            let mut joins = Vec::with_capacity(workers);
            for worker in 0..workers {
                let ready = Arc::clone(&ready);
                let start = Arc::clone(&start);
                let file = &file;
                let plans = &plans;
                joins.push(scope.spawn(move || -> Result<WorkerResult, String> {
                    let capacity = (worker..misses.len())
                        .step_by(workers)
                        .map(|index| plans[index].physical_bytes)
                        .max()
                        .unwrap_or(1);
                    let mut encoded = AlignedBuffer::new(capacity, PAGE_BYTES)?;
                    let mut decoded = vec![0_u8; EXPERT_BYTES];
                    let mut decompressor =
                        zstd::bulk::Decompressor::new().map_err(|error| error.to_string())?;
                    ready.wait();
                    start.wait();
                    let mut result = WorkerResult::default();
                    for index in (worker..misses.len()).step_by(workers) {
                        let record = &misses[index];
                        let plan = plans[index];
                        let pread_started = Instant::now();
                        read_exact_at(
                            file,
                            &mut encoded.bytes_mut()[..plan.physical_bytes],
                            plan.physical_offset,
                        )?;
                        result.pread_ns += pread_started.elapsed().as_nanos();
                        let source = &encoded.bytes_mut()
                            [plan.logical_offset..plan.logical_offset + record.compressed_bytes];
                        let decode_started = Instant::now();
                        let count = decompressor
                            .decompress_to_buffer(source, &mut decoded)
                            .map_err(|error| error.to_string())?;
                        result.decompression_ns += decode_started.elapsed().as_nanos();
                        if count != EXPERT_BYTES {
                            return Err("compressed-overlap decoded byte mismatch".to_owned());
                        }
                        std::hint::black_box(&decoded[..count]);
                        result.compressed_bytes += record.compressed_bytes;
                        result.physical_bytes += plan.physical_bytes;
                        result.source_bytes += count;
                        result.frames += 1;
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
                for _ in 0..LAYERS * 2 {
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
                        .map_err(|_| "compressed-overlap worker panicked".to_owned())?
                })
                .collect::<Result<Vec<_>, String>>()?;
            let wall = wall_started.elapsed().as_nanos();
            let disk = process_disk_bytes_read()?
                .checked_sub(disk_before)
                .ok_or_else(|| "compressed-overlap disk counter moved backwards".to_owned())?;
            Ok((results, wall, compute_wall_time_ns, disk))
        })?;
    let compressed_bytes = results.iter().map(|result| result.compressed_bytes).sum();
    let physical_bytes = results.iter().map(|result| result.physical_bytes).sum();
    let source_bytes = results.iter().map(|result| result.source_bytes).sum();
    let frames = results.iter().map(|result| result.frames).sum();
    if compressed_bytes
        != misses
            .iter()
            .map(|record| record.compressed_bytes)
            .sum::<usize>()
        || physical_bytes
            != misses
                .iter()
                .map(|record| record.physical_bytes)
                .sum::<usize>()
        || source_bytes != misses.len() * EXPERT_BYTES
        || frames != misses.len()
        || disk_bytes != physical_bytes as u64
    {
        return Err("compressed-overlap timed ledger mismatch".to_owned());
    }
    Ok(CompressedOverlapTrial {
        mode: if overlap {
            "parallel_storage_decode_exact_metal_overlap"
        } else {
            "parallel_storage_decode_control"
        },
        sample_ordinal,
        workers,
        frames,
        compressed_bytes,
        physical_bytes,
        source_bytes,
        complete_wall_time_ns,
        compute_wall_time_ns,
        summed_pread_time_ns: results.iter().map(|result| result.pread_ns).sum(),
        maximum_worker_pread_time_ns: results
            .iter()
            .map(|result| result.pread_ns)
            .max()
            .unwrap_or(0),
        summed_decompression_time_ns: results.iter().map(|result| result.decompression_ns).sum(),
        maximum_worker_decompression_time_ns: results
            .iter()
            .map(|result| result.decompression_ns)
            .max()
            .unwrap_or(0),
        process_disk_bytes_read: disk_bytes,
        cold_prepare_wall_time_ns,
        resident_page_instances_before: resident_before,
        resident_page_instances_after: resident_after,
    })
}

fn quantile(values: &[u128], numerator: usize, denominator: usize) -> u128 {
    let mut ordered = values.to_vec();
    ordered.sort_unstable();
    ordered[((ordered.len() - 1) * numerator + denominator / 2) / denominator]
}

#[allow(clippy::too_many_arguments)]
pub fn benchmark_parallel_zstd_overlap(
    checkpoint_dir: &Path,
    mixture_fixture_path: &Path,
    kernel_path: &Path,
    manifest_path: &Path,
    container_path: &Path,
    implementation_commit: &str,
) -> Result<ParallelZstdOverlapReport, String> {
    if implementation_commit.len() != 40
        || !implementation_commit
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        return Err("implementation commit must be a full hexadecimal Git hash".to_owned());
    }
    let mut safety = HostSafetyMonitor::start_normative(vec![PersistentResidencyDeclaration {
        object: "exact_layer0_top10_bf16_metal_contention_probe".to_owned(),
        maximum_bytes: 98_398_736,
        lifetime: "compressed_overlap_measurement_series".to_owned(),
        eviction_order: 1,
    }])?;
    let authority_started = Instant::now();
    let manifest = load_manifest(manifest_path, container_path)?;
    let container = File::open(container_path).map_err(|error| error.to_string())?;
    verify_frames(&container, &manifest.records)?;
    drop(container);
    let schedule = build_schedule(&manifest)?;
    let authority_verification_wall_time_ns = authority_started.elapsed().as_nanos();
    safety.checkpoint("authority_complete", true)?;

    let runner =
        ExactResidentTop10Runner::install(checkpoint_dir, mixture_fixture_path, kernel_path)?;
    for _ in 0..3 {
        runner.execute_exact()?;
    }
    safety.checkpoint("metal_warmups_complete", false)?;

    let mut diagnostic_worker_trials = Vec::new();
    for workers in [1, 2, 4] {
        diagnostic_worker_trials.push(run_trial(
            container_path,
            &schedule.misses,
            workers,
            0,
            false,
            &runner,
        )?);
        safety.checkpoint(&format!("diagnostic_workers_{workers}_complete"), false)?;
    }
    let order = [false, true, true, false, false, true];
    let mut controls = 0;
    let mut candidates = 0;
    let mut interleaved_trials = Vec::with_capacity(order.len());
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
        interleaved_trials.push(run_trial(
            container_path,
            &schedule.misses,
            WORKERS,
            ordinal,
            overlap,
            &runner,
        )?);
    }
    safety.checkpoint("interleaved_trials_complete", false)?;
    let control = interleaved_trials
        .iter()
        .filter(|trial| trial.mode == "parallel_storage_decode_control")
        .map(|trial| trial.complete_wall_time_ns)
        .collect::<Vec<_>>();
    let candidate = interleaved_trials
        .iter()
        .filter(|trial| trial.mode == "parallel_storage_decode_exact_metal_overlap")
        .map(|trial| trial.complete_wall_time_ns)
        .collect::<Vec<_>>();
    if control.len() != SAMPLES || candidate.len() != SAMPLES {
        return Err("compressed-overlap sample ledger mismatch".to_owned());
    }
    let accepted_tps = candidate
        .iter()
        .map(|wall| 2_000_000_000_f64 / *wall as f64)
        .collect::<Vec<_>>();
    let mut ordered_tps = accepted_tps.clone();
    ordered_tps.sort_by(f64::total_cmp);
    let format_tps = |value: f64| format!("{value:.6}");
    let device_name = runner.device_name().to_owned();
    drop(runner);
    let (host_safety_policy, host_safety_snapshots) = safety.finish()?;
    Ok(ParallelZstdOverlapReport {
        schema_version: 1,
        semantic: "qwen3_8_flash_next_q2_exact_zstd1_parallel_physical_metal_overlap_favorable_bound",
        implementation_commit: implementation_commit.to_owned(),
        model: MODEL,
        revision: REVISION,
        builder_commit: BUILDER_COMMIT,
        manifest_sha256: CONTAINER_MANIFEST_SHA256,
        container_sha256: CONTAINER_SHA256,
        device_name,
        codec: "zstd_0.13.3_bulk_decompressor_independent_frames",
        page_bytes: PAGE_BYTES,
        expert_frames: RECORDS,
        compressed_cache_bytes: CACHE_BYTES,
        first_target_row_frames: schedule.first_count,
        repeated_frames: schedule.repeated_count,
        second_only_frames: schedule.second_only_count,
        free_initial_frames: schedule.initial_count,
        free_initial_compressed_bytes: schedule.initial_bytes,
        miss_frames: schedule.misses.len(),
        miss_compressed_bytes: schedule
            .misses
            .iter()
            .map(|record| record.compressed_bytes)
            .sum(),
        miss_physical_bytes: schedule
            .misses
            .iter()
            .map(|record| record.physical_bytes)
            .sum(),
        miss_source_bytes: schedule.misses.len() * EXPERT_BYTES,
        authority_verification_wall_time_ns,
        exact_round_trips: RECORDS,
        diagnostic_worker_trials,
        interleaved_trials,
        control_p10_wall_time_ns: quantile(&control, 1, 10),
        control_median_wall_time_ns: quantile(&control, 1, 2),
        control_p90_wall_time_ns: quantile(&control, 9, 10),
        candidate_p10_wall_time_ns: quantile(&candidate, 1, 10),
        candidate_median_wall_time_ns: quantile(&candidate, 1, 2),
        candidate_p90_wall_time_ns: quantile(&candidate, 9, 10),
        accepted_bound_tps: accepted_tps.into_iter().map(format_tps).collect(),
        accepted_bound_tps_p10: format_tps(ordered_tps[0]),
        accepted_bound_tps_median: format_tps(ordered_tps[1]),
        accepted_bound_tps_p90: format_tps(ordered_tps[2]),
        metal_warmups: 3,
        metal_executions_per_candidate: LAYERS * 2,
        maximum_phase_scoped_staging_bytes: WORKERS * (7_749_632 + EXPERT_BYTES),
        cache_state: "range_invalidated_page_aligned_f_nocache_f_rdahead_zero_into_preallocated_parallel_zstd_install_buffers",
        batch_size: 1,
        concurrency: 1,
        sampling: "greedy",
        q: 2,
        accepted: 2,
        union: 697.0 / 480.0,
        rolled_back_proposal_rows: 0,
        favorable_grants: vec![
            "the future-known initial compressed cache is free and contains every first-row frame plus the largest fitting second-only frames",
            "cache metadata eviction and all cache-hit traffic are free",
            "all compressed misses may begin before target layer dependencies reveal routes",
            "the exact layer-0 top10 Metal workload stands in for all 96 target layer-row executions",
            "MTP fixed endpoint work attention shared experts routing ngram sampling rollback and synchronization are free",
        ],
        host_safety_policy,
        host_safety_snapshots,
        performance_claim: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantiles_are_stable_for_three_samples() {
        assert_eq!(quantile(&[30, 10, 20], 1, 10), 10);
        assert_eq!(quantile(&[30, 10, 20], 1, 2), 20);
        assert_eq!(quantile(&[30, 10, 20], 9, 10), 30);
    }

    #[test]
    fn initial_cache_covers_first_row_then_largest_fitting_second_only_frames() {
        let record = |identity: &str, bytes| Record {
            identity: identity.to_owned(),
            layer: 0,
            expert: 0,
            offset: 0,
            compressed_bytes: bytes,
            physical_bytes: PAGE_BYTES,
            frame_sha256: "0".repeat(64),
            source_sha256: "0".repeat(64),
        };
        let owned = [record("first", 4), record("large", 3), record("small", 2)];
        let records = owned
            .iter()
            .map(|record| (record.identity.clone(), record))
            .collect::<BTreeMap<_, _>>();
        let first = BTreeSet::from(["first".to_owned()]);
        let second = BTreeSet::from(["large".to_owned(), "small".to_owned()]);
        let (initial, bytes) = choose_initial(&records, &first, &second, 7).unwrap();
        assert_eq!(
            initial,
            BTreeSet::from(["first".to_owned(), "large".to_owned()])
        );
        assert_eq!(bytes, 7);
        assert!(choose_initial(&records, &first, &second, 3).is_err());
    }
}
