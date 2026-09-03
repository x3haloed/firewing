use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::time::Instant;

const MODEL: &str = "Qwen/Qwen3.8-Flash-Next";
const SEMANTIC: &str = "qwen3_8_flash_next_ngram_addresses";
const BUFFER_NAMES: [&str; 3] = [
    "layer_multipliers",
    "ngram_heads_offsets",
    "ngram_heads_vocab_sizes",
];

#[derive(Debug, Deserialize)]
struct NGramFixture {
    schema_version: u32,
    semantic: String,
    model: String,
    revision: String,
    reference: NGramReference,
    configuration: NGramConfiguration,
    checkpoint_buffers: BTreeMap<String, CheckpointBuffer>,
    table_parts: Vec<TablePart>,
    cases: Vec<NGramCase>,
}

#[derive(Debug, Deserialize)]
struct NGramReference {
    implementation: String,
    transformers_version: String,
    config_sha256: String,
    conversion_mapping_sha256: String,
    layout_source: String,
    model_lock_sha256: String,
    tensor_index_sha256: String,
    source: String,
}

#[derive(Debug, Deserialize)]
struct NGramConfiguration {
    seed: i64,
    eos_token_id: i64,
    unigram_vocab_size: i64,
    ngram_size: usize,
    heads_per_ngram: usize,
    ngram_heads: usize,
    ngram_vocab_size_base: i64,
    embedding_width: usize,
    head_width: usize,
    padded_rows: i64,
    split_parts: i64,
    rows_per_shard: i64,
    useful_bf16_bytes_per_token: usize,
}

#[derive(Debug, Deserialize)]
struct CheckpointBuffer {
    tensor: String,
    shard: String,
    shard_bytes: u64,
    shard_sha256: String,
    values: Vec<i64>,
}

#[derive(Debug, Deserialize)]
struct TablePart {
    part: i64,
    tensor: String,
    shard: String,
    shard_bytes: u64,
    shard_sha256: String,
    data_offsets: Vec<u64>,
}

#[derive(Debug, Deserialize)]
struct NGramCase {
    name: String,
    input_ids: Vec<i64>,
    previous_context: Vec<i64>,
    global_rows: Vec<Vec<i64>>,
    physical_rows: Vec<Vec<PhysicalRow>>,
}

#[derive(Debug, Deserialize, Eq, PartialEq)]
struct PhysicalRow {
    shard: i64,
    row: i64,
}

#[derive(Debug, Deserialize)]
struct RowHashFixture {
    schema_version: u32,
    semantic: String,
    model: String,
    revision: String,
    address_fixture_sha256: String,
    row_bytes: usize,
    cases: Vec<RowHashCase>,
}

#[derive(Debug, Deserialize)]
struct RowHashCase {
    name: String,
    row_sha256: Vec<Vec<String>>,
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

#[derive(Debug, Eq, PartialEq, Serialize)]
pub struct NGramVerificationReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub model: String,
    pub revision: String,
    pub cases_verified: usize,
    pub token_positions_verified: usize,
    pub addresses_verified: usize,
    pub checkpoint_buffers_verified: usize,
    pub table_parts_verified: usize,
    pub rows_per_shard: i64,
    pub useful_bf16_bytes_per_token: usize,
    pub accepted_tokens: usize,
    pub performance_claim: Option<String>,
}

#[derive(Debug, Eq, PartialEq, Serialize)]
pub struct NGramRowVerificationReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub model: String,
    pub revision: String,
    pub cases_verified: usize,
    pub rows_verified: usize,
    pub requested_payload_bytes: usize,
    pub row_bytes: usize,
    pub accepted_tokens: usize,
    pub performance_claim: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct NGramTransportTrial {
    pub transport: &'static str,
    pub ordinal: usize,
    pub wall_ms: f64,
    pub logical_bytes: usize,
    pub widened_bytes: usize,
    pub pread_calls: usize,
    pub process_disk_bytes_read: u64,
    pub stream_sha256: String,
}

#[derive(Debug, Serialize)]
pub struct NGramTransportSummary {
    pub transport: &'static str,
    pub samples: usize,
    pub wall_ms_p10: f64,
    pub wall_ms_median: f64,
    pub wall_ms_p90: f64,
    pub disk_bytes_median: u64,
    pub logical_bytes_per_trial: usize,
    pub widened_bytes_per_trial: usize,
}

#[derive(Debug, Serialize)]
pub struct NGramTransportBenchmarkReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub model: String,
    pub revision: String,
    pub commit: String,
    pub hardware: &'static str,
    pub checkpoint_storage: &'static str,
    pub page_bytes: usize,
    pub row_bytes: usize,
    pub rows_per_trial: usize,
    pub warmups_per_transport: usize,
    pub measurements_per_transport: usize,
    pub initial_cache_state: &'static str,
    pub trials: Vec<NGramTransportTrial>,
    pub summaries: Vec<NGramTransportSummary>,
    pub batch_size: usize,
    pub concurrency: usize,
    pub accepted_tokens: usize,
    #[serde(rename = "A")]
    pub accepted_per_verification: usize,
    #[serde(rename = "U")]
    pub expert_union: usize,
    pub performance_claim: Option<String>,
}

#[derive(Clone)]
struct SparseRequest {
    shard: String,
    absolute_offset: u64,
    expected_sha256: String,
}

#[derive(Clone, Copy)]
struct AlignedReadPlan {
    physical_offset: u64,
    physical_bytes: usize,
    logical_offset: usize,
}

struct AlignedBuffer {
    pointer: *mut u8,
    capacity: usize,
}

impl AlignedBuffer {
    fn new(capacity: usize, alignment: usize) -> Result<Self, String> {
        let mut pointer = std::ptr::null_mut();
        // SAFETY: posix_memalign initializes `pointer` on success; both values
        // are nonzero powers/multiples fixed by the validated read plan.
        let result = unsafe { libc::posix_memalign(&mut pointer, alignment, capacity) };
        if result != 0 || pointer.is_null() {
            return Err(format!("aligned allocation failed with {result}"));
        }
        Ok(Self {
            pointer: pointer.cast(),
            capacity,
        })
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: the allocation is live, exclusive, and exactly `capacity` bytes.
        unsafe { std::slice::from_raw_parts_mut(self.pointer, self.capacity) }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        // SAFETY: this pointer came from posix_memalign and is freed exactly once.
        unsafe { libc::free(self.pointer.cast()) };
    }
}

#[repr(C)]
#[derive(Default)]
struct RusageInfoV2 {
    uuid: [u8; 16],
    user_time: u64,
    system_time: u64,
    pkg_idle_wkups: u64,
    interrupt_wkups: u64,
    pageins: u64,
    wired_size: u64,
    resident_size: u64,
    phys_footprint: u64,
    proc_start_abstime: u64,
    proc_exit_abstime: u64,
    child_user_time: u64,
    child_system_time: u64,
    child_pkg_idle_wkups: u64,
    child_interrupt_wkups: u64,
    child_pageins: u64,
    child_elapsed_abstime: u64,
    diskio_bytesread: u64,
    diskio_byteswritten: u64,
}

#[cfg(target_os = "macos")]
#[link(name = "proc")]
unsafe extern "C" {
    fn proc_pid_rusage(
        pid: libc::c_int,
        flavor: libc::c_int,
        buffer: *mut libc::c_void,
    ) -> libc::c_int;
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file =
        File::open(path).map_err(|error| format!("cannot open {}: {error}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn require_hex(value: &str, len: usize) -> bool {
    value.len() == len && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn validate_identity(fixture: &NGramFixture) -> Result<(), String> {
    let config = &fixture.configuration;
    if fixture.schema_version != 1
        || fixture.semantic != SEMANTIC
        || fixture.model != MODEL
        || !require_hex(&fixture.revision, 40)
        || fixture.reference.implementation != "huggingface_transformers_qwen4_exp"
        || fixture.reference.transformers_version != "5.16.1"
        || fixture.reference.source != "transformers.models.qwen4_exp.modeling_qwen4_exp"
        || !require_hex(&fixture.reference.config_sha256, 64)
        || !require_hex(&fixture.reference.conversion_mapping_sha256, 64)
        || fixture.reference.layout_source
            != "transformers.conversion_mapping.qwen4_exp_text.Concatenate(dim=0)"
        || !require_hex(&fixture.reference.model_lock_sha256, 64)
        || !require_hex(&fixture.reference.tensor_index_sha256, 64)
        || config.seed != 1234
        || config.eos_token_id != 248_044
        || config.unigram_vocab_size != 248_320
        || config.ngram_size != 3
        || config.heads_per_ngram != 8
        || config.ngram_heads != 16
        || config.ngram_vocab_size_base != 20_000_000
        || config.embedding_width != 2560
        || config.head_width != 160
        || config.padded_rows != 320_001_536
        || config.split_parts != 128
        || config.rows_per_shard != 2_500_012
        || config.useful_bf16_bytes_per_token != 5120
        || fixture.checkpoint_buffers.len() != BUFFER_NAMES.len()
        || fixture.table_parts.len() != config.split_parts as usize
    {
        return Err(
            "n-gram fixture identity, reference, or configuration is unsupported".to_owned(),
        );
    }
    Ok(())
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn build_layer_multipliers(vocab_size: i64, ngram_size: usize, seed: i64) -> Vec<i64> {
    let multiplier_max = i64::MAX / vocab_size.max(1);
    let half_bound = (multiplier_max / 2).max(1) as u64;
    (0..ngram_size)
        .map(|index| {
            let value = (seed as u64)
                .wrapping_add(0x9e37_79b9_7f4a_7c15_u64.wrapping_mul((index + 1) as u64));
            (2 * (splitmix64(value) % half_bound) + 1) as i64
        })
        .collect()
}

fn is_prime(value: i64) -> bool {
    if value < 2 {
        return false;
    }
    if value % 2 == 0 {
        return value == 2;
    }
    let mut divisor = 3;
    while divisor <= value / divisor {
        if value % divisor == 0 {
            return false;
        }
        divisor += 2;
    }
    true
}

fn nth_prime_after(start: i64, count: usize) -> i64 {
    let mut prime = start;
    for _ in 0..count {
        prime += 1;
        while !is_prime(prime) {
            prime += 1;
        }
    }
    prime
}

fn derived_buffers(config: &NGramConfiguration) -> BTreeMap<String, Vec<i64>> {
    let sizes: Vec<i64> = (0..config.ngram_heads)
        .map(|index| nth_prime_after(config.ngram_vocab_size_base - 1, index + 1))
        .collect();
    let mut running = 0_i64;
    let offsets = sizes
        .iter()
        .map(|size| {
            let current = running;
            running += size;
            current
        })
        .collect();
    BTreeMap::from([
        (
            "layer_multipliers".to_owned(),
            build_layer_multipliers(config.unigram_vocab_size, config.ngram_size, config.seed),
        ),
        ("ngram_heads_offsets".to_owned(), offsets),
        ("ngram_heads_vocab_sizes".to_owned(), sizes),
    ])
}

fn locked_file<'a>(lock: &'a ModelLock, path: &str) -> Result<&'a LockedFile, String> {
    let matches: Vec<_> = lock.files.iter().filter(|file| file.path == path).collect();
    if matches.len() != 1 {
        return Err(format!(
            "model lock must contain exactly one entry for {path}"
        ));
    }
    Ok(matches[0])
}

fn read_i64_safetensor(
    path: &Path,
    tensor_name: &str,
    expected_len: usize,
) -> Result<Vec<i64>, String> {
    let mut file =
        File::open(path).map_err(|error| format!("cannot open {}: {error}", path.display()))?;
    let mut length_bytes = [0_u8; 8];
    file.read_exact(&mut length_bytes)
        .map_err(|error| format!("cannot read safetensors header length: {error}"))?;
    let header_len = u64::from_le_bytes(length_bytes);
    if header_len == 0 || header_len > 16 * 1024 * 1024 {
        return Err("safetensors header length is unsupported".to_owned());
    }
    let mut header_bytes = vec![0_u8; header_len as usize];
    file.read_exact(&mut header_bytes)
        .map_err(|error| format!("cannot read safetensors header: {error}"))?;
    let header: Value = serde_json::from_slice(&header_bytes)
        .map_err(|error| format!("malformed safetensors header: {error}"))?;
    let entry = header
        .get(tensor_name)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("missing checkpoint tensor {tensor_name}"))?;
    if entry.get("dtype").and_then(Value::as_str) != Some("I64") {
        return Err(format!("checkpoint tensor {tensor_name} is not I64"));
    }
    let shape = entry
        .get("shape")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("checkpoint tensor {tensor_name} has no shape"))?;
    if shape.len() != 1 || shape[0].as_u64() != Some(expected_len as u64) {
        return Err(format!(
            "checkpoint tensor {tensor_name} has an unsupported shape"
        ));
    }
    let offsets = entry
        .get("data_offsets")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("checkpoint tensor {tensor_name} has no data offsets"))?;
    if offsets.len() != 2 {
        return Err(format!(
            "checkpoint tensor {tensor_name} has invalid data offsets"
        ));
    }
    let start = offsets[0]
        .as_u64()
        .ok_or_else(|| "invalid tensor start offset".to_owned())?;
    let end = offsets[1]
        .as_u64()
        .ok_or_else(|| "invalid tensor end offset".to_owned())?;
    let expected_bytes = (expected_len * 8) as u64;
    if end.checked_sub(start) != Some(expected_bytes) {
        return Err(format!(
            "checkpoint tensor {tensor_name} byte length disagrees with shape"
        ));
    }
    file.seek(SeekFrom::Start(8 + header_len + start))
        .map_err(|error| format!("cannot seek to checkpoint tensor: {error}"))?;
    let mut payload = vec![0_u8; expected_bytes as usize];
    file.read_exact(&mut payload)
        .map_err(|error| format!("cannot read checkpoint tensor: {error}"))?;
    Ok(payload
        .chunks_exact(8)
        .map(|bytes| i64::from_le_bytes(bytes.try_into().expect("chunk has eight bytes")))
        .collect())
}

fn read_safetensor_descriptor(path: &Path, tensor_name: &str) -> Result<Value, String> {
    let mut file =
        File::open(path).map_err(|error| format!("cannot open {}: {error}", path.display()))?;
    let mut length_bytes = [0_u8; 8];
    file.read_exact(&mut length_bytes)
        .map_err(|error| format!("cannot read safetensors header length: {error}"))?;
    let header_len = u64::from_le_bytes(length_bytes);
    if header_len == 0 || header_len > 16 * 1024 * 1024 {
        return Err("safetensors header length is unsupported".to_owned());
    }
    let mut header_bytes = vec![0_u8; header_len as usize];
    file.read_exact(&mut header_bytes)
        .map_err(|error| format!("cannot read safetensors header: {error}"))?;
    let header: Value = serde_json::from_slice(&header_bytes)
        .map_err(|error| format!("malformed safetensors header: {error}"))?;
    header
        .get(tensor_name)
        .cloned()
        .ok_or_else(|| format!("missing checkpoint tensor {tensor_name}"))
}

fn read_sparse_row<R: Read + Seek>(
    reader: &mut R,
    tensor_start: u64,
    local_row: i64,
    rows_per_part: i64,
    row_bytes: usize,
) -> Result<Vec<u8>, String> {
    if !(0..rows_per_part).contains(&local_row) || row_bytes == 0 {
        return Err("sparse row request is out of bounds".to_owned());
    }
    reader
        .seek(SeekFrom::Start(0))
        .map_err(|error| format!("cannot seek to safetensors header: {error}"))?;
    let mut length_bytes = [0_u8; 8];
    reader
        .read_exact(&mut length_bytes)
        .map_err(|error| format!("cannot read safetensors header length: {error}"))?;
    let header_len = u64::from_le_bytes(length_bytes);
    if header_len == 0 || header_len > 16 * 1024 * 1024 {
        return Err("safetensors header length is unsupported".to_owned());
    }
    let row_offset = (local_row as u64)
        .checked_mul(row_bytes as u64)
        .and_then(|offset| offset.checked_add(tensor_start))
        .and_then(|offset| offset.checked_add(8 + header_len))
        .ok_or_else(|| "sparse row byte offset overflow".to_owned())?;
    reader
        .seek(SeekFrom::Start(row_offset))
        .map_err(|error| format!("cannot seek to sparse row: {error}"))?;
    let mut row = vec![0_u8; row_bytes];
    reader
        .read_exact(&mut row)
        .map_err(|error| format!("cannot read sparse row: {error}"))?;
    Ok(row)
}

fn shifted_token(history: &[i64], index: usize, shift: usize, eos: i64) -> i64 {
    if shift == 0 {
        return history[index];
    }
    let segment_start = history[..index]
        .iter()
        .rposition(|token| *token == eos)
        .map_or(0, |position| position + 1);
    if index >= shift && index - segment_start >= shift {
        history[index - shift]
    } else {
        eos
    }
}

fn compute_addresses(
    input_ids: &[i64],
    previous_context: &[i64],
    config: &NGramConfiguration,
    buffers: &BTreeMap<String, Vec<i64>>,
) -> Vec<Vec<i64>> {
    let multipliers = &buffers["layer_multipliers"];
    let offsets = &buffers["ngram_heads_offsets"];
    let sizes = &buffers["ngram_heads_vocab_sizes"];
    let mut history = previous_context.to_vec();
    history.extend_from_slice(input_ids);
    (previous_context.len()..history.len())
        .map(|index| {
            (0..config.ngram_heads)
                .map(|head| {
                    let ngram = 2 + head / config.heads_per_ngram;
                    let mut mixed = shifted_token(&history, index, 0, config.eos_token_id)
                        .wrapping_mul(multipliers[0]);
                    for (position, multiplier) in multipliers.iter().enumerate().take(ngram).skip(1)
                    {
                        mixed ^= shifted_token(&history, index, position, config.eos_token_id)
                            .wrapping_mul(*multiplier);
                    }
                    mixed.rem_euclid(sizes[head]) + offsets[head]
                })
                .collect()
        })
        .collect()
}

pub fn verify_ngram_fixture(
    checkpoint_dir: &Path,
    model_lock_path: &Path,
    fixture_path: &Path,
) -> Result<NGramVerificationReport, String> {
    let fixture: NGramFixture = serde_json::from_slice(
        &fs::read(fixture_path)
            .map_err(|error| format!("cannot read fixture {}: {error}", fixture_path.display()))?,
    )
    .map_err(|error| format!("malformed n-gram fixture: {error}"))?;
    validate_identity(&fixture)?;

    if sha256_file(model_lock_path)? != fixture.reference.model_lock_sha256 {
        return Err("model lock content identity mismatch".to_owned());
    }
    let lock: ModelLock = serde_json::from_slice(&fs::read(model_lock_path).map_err(|error| {
        format!(
            "cannot read model lock {}: {error}",
            model_lock_path.display()
        )
    })?)
    .map_err(|error| format!("malformed model lock: {error}"))?;
    if lock.model != fixture.model || lock.revision != fixture.revision {
        return Err("model lock identity does not match fixture".to_owned());
    }
    if sha256_file(&checkpoint_dir.join("config.json"))? != fixture.reference.config_sha256 {
        return Err("checkpoint config content identity mismatch".to_owned());
    }
    if sha256_file(&checkpoint_dir.join("model.safetensors.index.json"))?
        != fixture.reference.tensor_index_sha256
    {
        return Err("checkpoint tensor-index content identity mismatch".to_owned());
    }

    let derived = derived_buffers(&fixture.configuration);
    for name in BUFFER_NAMES {
        let buffer = fixture
            .checkpoint_buffers
            .get(name)
            .ok_or_else(|| format!("fixture is missing checkpoint buffer {name}"))?;
        if derived[name] != buffer.values {
            return Err(format!(
                "fixture buffer {name} disagrees with native derivation"
            ));
        }
        if !require_hex(&buffer.shard_sha256, 64) {
            return Err(format!("fixture buffer {name} has an invalid shard hash"));
        }
        let locked = locked_file(&lock, &buffer.shard)?;
        if locked.size != buffer.shard_bytes
            || locked.lfs_sha256.as_deref() != Some(buffer.shard_sha256.as_str())
        {
            return Err(format!("model lock identity mismatch for {}", buffer.shard));
        }
        let shard_path = checkpoint_dir.join(&buffer.shard);
        if fs::metadata(&shard_path)
            .map_err(|error| format!("cannot stat {}: {error}", shard_path.display()))?
            .len()
            != buffer.shard_bytes
        {
            return Err(format!(
                "checkpoint shard size mismatch for {}",
                buffer.shard
            ));
        }
        let actual = read_i64_safetensor(&shard_path, &buffer.tensor, buffer.values.len())?;
        if actual != buffer.values {
            return Err(format!("checkpoint payload mismatch for {}", buffer.tensor));
        }
    }

    for (expected_part, part) in fixture.table_parts.iter().enumerate() {
        let expected_tensor = format!(
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{expected_part}.weight"
        );
        if part.part != expected_part as i64
            || part.tensor != expected_tensor
            || part.data_offsets.len() != 2
            || part.data_offsets[1].checked_sub(part.data_offsets[0])
                != Some((fixture.configuration.rows_per_shard * 160 * 2) as u64)
            || !require_hex(&part.shard_sha256, 64)
        {
            return Err(format!(
                "n-gram table part {expected_part} has invalid metadata"
            ));
        }
        let locked = locked_file(&lock, &part.shard)?;
        if locked.size != part.shard_bytes
            || locked.lfs_sha256.as_deref() != Some(part.shard_sha256.as_str())
        {
            return Err(format!(
                "model lock identity mismatch for table part {expected_part}"
            ));
        }
        let shard_path = checkpoint_dir.join(&part.shard);
        if fs::metadata(&shard_path)
            .map_err(|error| format!("cannot stat {}: {error}", shard_path.display()))?
            .len()
            != part.shard_bytes
        {
            return Err(format!("checkpoint shard size mismatch for {}", part.shard));
        }
        let descriptor = read_safetensor_descriptor(&shard_path, &part.tensor)?;
        let shape = descriptor
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("table part {expected_part} has no shape"))?;
        let offsets = descriptor
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("table part {expected_part} has no data offsets"))?;
        let shape_matches = shape.len() == 2
            && shape[0].as_i64() == Some(fixture.configuration.rows_per_shard)
            && shape[1].as_i64() == Some(160);
        let offsets_match = offsets.len() == 2
            && offsets[0].as_u64() == Some(part.data_offsets[0])
            && offsets[1].as_u64() == Some(part.data_offsets[1]);
        if descriptor.get("dtype").and_then(Value::as_str) != Some("BF16")
            || !shape_matches
            || !offsets_match
        {
            return Err(format!(
                "checkpoint header mismatch for table part {expected_part}"
            ));
        }
    }

    let mut token_positions = 0;
    for case in &fixture.cases {
        if case.name.is_empty()
            || case.input_ids.is_empty()
            || case.previous_context.len() != fixture.configuration.ngram_size - 1
            || case
                .input_ids
                .iter()
                .chain(&case.previous_context)
                .any(|token| !(0..fixture.configuration.unigram_vocab_size).contains(token))
        {
            return Err(format!("case {} has invalid token inputs", case.name));
        }
        let actual = compute_addresses(
            &case.input_ids,
            &case.previous_context,
            &fixture.configuration,
            &derived,
        );
        if actual != case.global_rows {
            return Err(format!("case {} global n-gram address mismatch", case.name));
        }
        let physical: Vec<Vec<PhysicalRow>> = actual
            .iter()
            .map(|rows| {
                rows.iter()
                    .map(|row| PhysicalRow {
                        shard: row / fixture.configuration.rows_per_shard,
                        row: row % fixture.configuration.rows_per_shard,
                    })
                    .collect()
            })
            .collect();
        if physical != case.physical_rows
            || physical.iter().flatten().any(|location| {
                location.shard < 0
                    || location.shard >= fixture.configuration.split_parts
                    || location.row < 0
                    || location.row >= fixture.configuration.rows_per_shard
            })
        {
            return Err(format!(
                "case {} physical n-gram address mismatch",
                case.name
            ));
        }
        token_positions += case.input_ids.len();
    }

    Ok(NGramVerificationReport {
        schema_version: 1,
        semantic: "qwen3_8_flash_next_ngram_fixture_verification",
        model: fixture.model,
        revision: fixture.revision,
        cases_verified: fixture.cases.len(),
        token_positions_verified: token_positions,
        addresses_verified: token_positions * fixture.configuration.ngram_heads,
        checkpoint_buffers_verified: BUFFER_NAMES.len(),
        table_parts_verified: fixture.table_parts.len(),
        rows_per_shard: fixture.configuration.rows_per_shard,
        useful_bf16_bytes_per_token: fixture.configuration.useful_bf16_bytes_per_token,
        accepted_tokens: 0,
        performance_claim: None,
    })
}

pub fn verify_ngram_rows(
    checkpoint_dir: &Path,
    model_lock_path: &Path,
    address_fixture_path: &Path,
    row_fixture_path: &Path,
) -> Result<NGramRowVerificationReport, String> {
    verify_ngram_fixture(checkpoint_dir, model_lock_path, address_fixture_path)?;
    let address: NGramFixture =
        serde_json::from_slice(&fs::read(address_fixture_path).map_err(|error| {
            format!(
                "cannot read address fixture {}: {error}",
                address_fixture_path.display()
            )
        })?)
        .map_err(|error| format!("malformed n-gram address fixture: {error}"))?;
    let rows: RowHashFixture =
        serde_json::from_slice(&fs::read(row_fixture_path).map_err(|error| {
            format!(
                "cannot read row fixture {}: {error}",
                row_fixture_path.display()
            )
        })?)
        .map_err(|error| format!("malformed n-gram row fixture: {error}"))?;
    if rows.schema_version != 1
        || rows.semantic != "qwen3_8_flash_next_ngram_row_hashes"
        || rows.model != MODEL
        || rows.model != address.model
        || rows.revision != address.revision
        || !require_hex(&rows.address_fixture_sha256, 64)
        || sha256_file(address_fixture_path)? != rows.address_fixture_sha256
        || rows.row_bytes != address.configuration.head_width * 2
        || rows.row_bytes != 320
        || rows.cases.len() != address.cases.len()
    {
        return Err("n-gram row fixture identity or configuration is unsupported".to_owned());
    }

    let mut handles: BTreeMap<String, File> = BTreeMap::new();
    let mut rows_verified = 0;
    for (address_case, row_case) in address.cases.iter().zip(&rows.cases) {
        if address_case.name != row_case.name
            || row_case.row_sha256.len() != address_case.global_rows.len()
        {
            return Err(format!(
                "row fixture case {} has invalid dimensions",
                row_case.name
            ));
        }
        for ((global_rows, physical_rows), expected_hashes) in address_case
            .global_rows
            .iter()
            .zip(&address_case.physical_rows)
            .zip(&row_case.row_sha256)
        {
            if global_rows.len() != address.configuration.ngram_heads
                || physical_rows.len() != global_rows.len()
                || expected_hashes.len() != global_rows.len()
            {
                return Err(format!(
                    "row fixture case {} has invalid head dimensions",
                    row_case.name
                ));
            }
            for ((global_row, physical), expected_hash) in
                global_rows.iter().zip(physical_rows).zip(expected_hashes)
            {
                if !require_hex(expected_hash, 64) {
                    return Err(format!(
                        "row fixture case {} has an invalid hash",
                        row_case.name
                    ));
                }
                let expected_part = global_row / address.configuration.rows_per_shard;
                let expected_local_row = global_row % address.configuration.rows_per_shard;
                if physical.shard != expected_part || physical.row != expected_local_row {
                    return Err(format!(
                        "row fixture case {} has an invalid location",
                        row_case.name
                    ));
                }
                let part = address
                    .table_parts
                    .get(expected_part as usize)
                    .ok_or_else(|| "global row selects a missing table part".to_owned())?;
                if !handles.contains_key(&part.shard) {
                    let path = checkpoint_dir.join(&part.shard);
                    let file = File::open(&path)
                        .map_err(|error| format!("cannot open {}: {error}", path.display()))?;
                    handles.insert(part.shard.clone(), file);
                }
                let row = read_sparse_row(
                    handles
                        .get_mut(&part.shard)
                        .expect("opened shard handle must remain present"),
                    part.data_offsets[0],
                    physical.row,
                    address.configuration.rows_per_shard,
                    rows.row_bytes,
                )?;
                let actual_hash = format!("{:x}", Sha256::digest(&row));
                if actual_hash != *expected_hash {
                    return Err(format!(
                        "checkpoint row hash mismatch in case {} at global row {global_row}",
                        row_case.name
                    ));
                }
                rows_verified += 1;
            }
        }
    }

    Ok(NGramRowVerificationReport {
        schema_version: 1,
        semantic: "qwen3_8_flash_next_ngram_sparse_row_verification",
        model: rows.model,
        revision: rows.revision,
        cases_verified: rows.cases.len(),
        rows_verified,
        requested_payload_bytes: rows_verified * rows.row_bytes,
        row_bytes: rows.row_bytes,
        accepted_tokens: 0,
        performance_claim: None,
    })
}

#[cfg(target_os = "macos")]
fn process_disk_bytes_read() -> Result<u64, String> {
    let mut usage = RusageInfoV2::default();
    // SAFETY: `usage` has Darwin's rusage_info_v2 layout and is exclusively borrowed.
    let result = unsafe {
        proc_pid_rusage(
            std::process::id() as libc::c_int,
            2,
            (&mut usage as *mut RusageInfoV2).cast(),
        )
    };
    if result != 0 {
        return Err(format!("proc_pid_rusage failed with {result}"));
    }
    Ok(usage.diskio_bytesread)
}

#[cfg(not(target_os = "macos"))]
fn process_disk_bytes_read() -> Result<u64, String> {
    Err("Darwin process disk counters are required".to_owned())
}

fn aligned_read_plan(
    offset: u64,
    logical_bytes: usize,
    file_bytes: u64,
    page_bytes: usize,
) -> Result<AlignedReadPlan, String> {
    if logical_bytes == 0 || !page_bytes.is_power_of_two() {
        return Err("invalid aligned sparse read parameters".to_owned());
    }
    let logical_end = offset
        .checked_add(logical_bytes as u64)
        .ok_or_else(|| "aligned sparse read range overflow".to_owned())?;
    if logical_end > file_bytes {
        return Err("aligned sparse read exceeds checkpoint shard".to_owned());
    }
    let mask = page_bytes as u64 - 1;
    let physical_offset = offset & !mask;
    let physical_end = logical_end
        .checked_add(mask)
        .map(|end| end & !mask)
        .ok_or_else(|| "aligned sparse read rounding overflow".to_owned())?
        .min(file_bytes);
    Ok(AlignedReadPlan {
        physical_offset,
        physical_bytes: usize::try_from(physical_end - physical_offset)
            .map_err(|_| "aligned sparse read length does not fit usize".to_owned())?,
        logical_offset: usize::try_from(offset - physical_offset)
            .map_err(|_| "aligned sparse logical offset does not fit usize".to_owned())?,
    })
}

fn read_exact_at(file: &File, mut destination: &mut [u8], mut offset: u64) -> Result<(), String> {
    while !destination.is_empty() {
        let count = file
            .read_at(destination, offset)
            .map_err(|error| format!("pread failed: {error}"))?;
        if count == 0 {
            return Err("pread reached EOF before completing request".to_owned());
        }
        offset = offset
            .checked_add(count as u64)
            .ok_or_else(|| "pread offset overflow".to_owned())?;
        destination = &mut destination[count..];
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn set_uncached(file: &File) -> Result<(), String> {
    use std::os::fd::AsRawFd;
    let descriptor = file.as_raw_fd();
    // SAFETY: fcntl receives a live descriptor and Darwin's documented flags.
    if unsafe { libc::fcntl(descriptor, libc::F_NOCACHE, 1) } == -1
        || unsafe { libc::fcntl(descriptor, libc::F_RDAHEAD, 0) } == -1
    {
        return Err(format!(
            "uncached transport fcntl failed: {}",
            std::io::Error::last_os_error()
        ));
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn set_uncached(_file: &File) -> Result<(), String> {
    Err("Darwin F_NOCACHE transport is required".to_owned())
}

fn quantile_f64(values: &[f64], fraction: f64) -> f64 {
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    ordered[((ordered.len() - 1) as f64 * fraction).round() as usize]
}

fn median_u64(values: &[u64]) -> u64 {
    let mut ordered = values.to_vec();
    ordered.sort_unstable();
    ordered[ordered.len() / 2]
}

fn run_transport_trial(
    checkpoint_dir: &Path,
    requests: &[SparseRequest],
    row_bytes: usize,
    page_bytes: usize,
    uncached: bool,
    ordinal: usize,
) -> Result<NGramTransportTrial, String> {
    let mut handles = BTreeMap::new();
    for request in requests {
        if !handles.contains_key(&request.shard) {
            let path = checkpoint_dir.join(&request.shard);
            let file = File::open(&path)
                .map_err(|error| format!("cannot open {}: {error}", path.display()))?;
            if uncached {
                set_uncached(&file)?;
            }
            handles.insert(request.shard.clone(), file);
        }
    }
    let plans = if uncached {
        requests
            .iter()
            .map(|request| {
                let file = &handles[&request.shard];
                aligned_read_plan(
                    request.absolute_offset,
                    row_bytes,
                    file.metadata().map_err(|error| error.to_string())?.len(),
                    page_bytes,
                )
            })
            .collect::<Result<Vec<_>, String>>()?
    } else {
        Vec::new()
    };
    let maximum_widened = plans
        .iter()
        .map(|plan| plan.physical_bytes)
        .max()
        .unwrap_or(row_bytes);
    let mut aligned_buffer = AlignedBuffer::new(maximum_widened, page_bytes)?;
    let mut exact_buffer = vec![0_u8; row_bytes];
    let disk_before = process_disk_bytes_read()?;
    let started = Instant::now();
    let mut stream_digest = Sha256::new();
    let mut widened_bytes = 0_usize;
    for (index, request) in requests.iter().enumerate() {
        let row = if uncached {
            let plan = plans[index];
            let buffer = &mut aligned_buffer.bytes_mut()[..plan.physical_bytes];
            read_exact_at(&handles[&request.shard], buffer, plan.physical_offset)?;
            widened_bytes = widened_bytes
                .checked_add(plan.physical_bytes)
                .ok_or_else(|| "widened byte count overflow".to_owned())?;
            &buffer[plan.logical_offset..plan.logical_offset + row_bytes]
        } else {
            read_exact_at(
                &handles[&request.shard],
                &mut exact_buffer,
                request.absolute_offset,
            )?;
            exact_buffer.as_slice()
        };
        let row_hash = format!("{:x}", Sha256::digest(row));
        if row_hash != request.expected_sha256 {
            return Err(format!("transport row hash mismatch at request {index}"));
        }
        stream_digest.update(row);
    }
    let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
    let disk_after = process_disk_bytes_read()?;
    let process_disk_bytes_read = disk_after
        .checked_sub(disk_before)
        .ok_or_else(|| "process disk byte counter moved backwards".to_owned())?;
    Ok(NGramTransportTrial {
        transport: if uncached {
            "page_aligned_f_nocache_f_rdahead_zero"
        } else {
            "cacheable_exact_pread"
        },
        ordinal,
        wall_ms,
        logical_bytes: requests.len() * row_bytes,
        widened_bytes: if uncached {
            widened_bytes
        } else {
            requests.len() * row_bytes
        },
        pread_calls: requests.len(),
        process_disk_bytes_read,
        stream_sha256: format!("{:x}", stream_digest.finalize()),
    })
}

pub fn benchmark_ngram_transport(
    checkpoint_dir: &Path,
    model_lock_path: &Path,
    address_fixture_path: &Path,
    row_fixture_path: &Path,
    commit: &str,
) -> Result<NGramTransportBenchmarkReport, String> {
    const PAGE_BYTES: usize = 16 * 1024;
    const WARMUPS: usize = 5;
    const MEASUREMENTS: usize = 30;
    if !require_hex(commit, 40) {
        return Err("benchmark commit must be exactly 40 hexadecimal characters".to_owned());
    }
    verify_ngram_rows(
        checkpoint_dir,
        model_lock_path,
        address_fixture_path,
        row_fixture_path,
    )?;
    let address: NGramFixture =
        serde_json::from_slice(&fs::read(address_fixture_path).map_err(|error| error.to_string())?)
            .map_err(|error| format!("malformed n-gram address fixture: {error}"))?;
    let rows: RowHashFixture =
        serde_json::from_slice(&fs::read(row_fixture_path).map_err(|error| error.to_string())?)
            .map_err(|error| format!("malformed n-gram row fixture: {error}"))?;
    let mut payload_starts = BTreeMap::new();
    for part in &address.table_parts {
        if !payload_starts.contains_key(&part.shard) {
            let mut file =
                File::open(checkpoint_dir.join(&part.shard)).map_err(|error| error.to_string())?;
            let mut raw = [0_u8; 8];
            file.read_exact(&mut raw)
                .map_err(|error| error.to_string())?;
            payload_starts.insert(part.shard.clone(), 8 + u64::from_le_bytes(raw));
        }
    }
    let mut requests = Vec::new();
    for (address_case, row_case) in address.cases.iter().zip(&rows.cases) {
        for ((global_rows, physical_rows), hashes) in address_case
            .global_rows
            .iter()
            .zip(&address_case.physical_rows)
            .zip(&row_case.row_sha256)
        {
            for ((global_row, physical), expected_sha256) in
                global_rows.iter().zip(physical_rows).zip(hashes)
            {
                let part = &address.table_parts[physical.shard as usize];
                if global_row
                    != &(physical.shard * address.configuration.rows_per_shard + physical.row)
                {
                    return Err("benchmark request physical address mismatch".to_owned());
                }
                let absolute_offset = payload_starts[&part.shard]
                    .checked_add(part.data_offsets[0])
                    .and_then(|offset| {
                        offset.checked_add(physical.row as u64 * rows.row_bytes as u64)
                    })
                    .ok_or_else(|| "benchmark request byte offset overflow".to_owned())?;
                requests.push(SparseRequest {
                    shard: part.shard.clone(),
                    absolute_offset,
                    expected_sha256: expected_sha256.clone(),
                });
            }
        }
    }

    let mut trials = Vec::new();
    for &uncached in &[false, true] {
        for ordinal in 0..WARMUPS {
            run_transport_trial(
                checkpoint_dir,
                &requests,
                rows.row_bytes,
                PAGE_BYTES,
                uncached,
                ordinal,
            )?;
        }
        for ordinal in 0..MEASUREMENTS {
            trials.push(run_transport_trial(
                checkpoint_dir,
                &requests,
                rows.row_bytes,
                PAGE_BYTES,
                uncached,
                ordinal,
            )?);
        }
    }
    let mut summaries = Vec::new();
    for transport in [
        "cacheable_exact_pread",
        "page_aligned_f_nocache_f_rdahead_zero",
    ] {
        let matching: Vec<_> = trials
            .iter()
            .filter(|trial| trial.transport == transport)
            .collect();
        let walls: Vec<_> = matching.iter().map(|trial| trial.wall_ms).collect();
        let disks: Vec<_> = matching
            .iter()
            .map(|trial| trial.process_disk_bytes_read)
            .collect();
        summaries.push(NGramTransportSummary {
            transport,
            samples: matching.len(),
            wall_ms_p10: quantile_f64(&walls, 0.1),
            wall_ms_median: quantile_f64(&walls, 0.5),
            wall_ms_p90: quantile_f64(&walls, 0.9),
            disk_bytes_median: median_u64(&disks),
            logical_bytes_per_trial: matching[0].logical_bytes,
            widened_bytes_per_trial: matching[0].widened_bytes,
        });
    }
    Ok(NGramTransportBenchmarkReport {
        schema_version: 1,
        semantic: "qwen3_8_flash_next_ngram_sparse_transport_diagnostic",
        model: rows.model,
        revision: rows.revision,
        commit: commit.to_owned(),
        hardware: "Apple M1 Mac mini Macmini9,1 16 GiB",
        checkpoint_storage: "internal_ssd",
        page_bytes: PAGE_BYTES,
        row_bytes: rows.row_bytes,
        rows_per_trial: requests.len(),
        warmups_per_transport: WARMUPS,
        measurements_per_transport: MEASUREMENTS,
        initial_cache_state: "cache-influenced after identity verification; F_NOCACHE trials bypass file cache",
        trials,
        summaries,
        batch_size: 1,
        concurrency: 1,
        accepted_tokens: 0,
        accepted_per_verification: 0,
        expert_union: 0,
        performance_claim: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[derive(Deserialize)]
    struct SyntheticRowFixture {
        schema_version: u32,
        semantic: String,
        header_json: Value,
        payload_prefix_hex: String,
        tensor_hex: String,
        reads: Vec<SyntheticRead>,
    }

    #[derive(Deserialize)]
    struct SyntheticRead {
        expected_hex: String,
        row: i64,
    }

    fn decode_hex(value: &str) -> Vec<u8> {
        assert!(value.len().is_multiple_of(2));
        value
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| {
                u8::from_str_radix(std::str::from_utf8(pair).expect("fixture hex is ASCII"), 16)
                    .expect("fixture hex is valid")
            })
            .collect()
    }

    #[test]
    fn multipliers_match_pinned_checkpoint_values() {
        assert_eq!(
            build_layer_multipliers(248_320, 3, 1234),
            vec![23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071]
        );
    }

    #[test]
    fn eos_shift_stays_inside_segment() {
        let history = [9, 10, 99, 20, 21];
        assert_eq!(shifted_token(&history, 4, 1, 99), 20);
        assert_eq!(shifted_token(&history, 4, 2, 99), 99);
    }

    #[test]
    fn committed_fixture_matches_independent_scalar_addresses() {
        let fixture: NGramFixture =
            serde_json::from_str(include_str!("../fixtures/ngram/qwen3_8_flash_next.json"))
                .expect("committed fixture must parse");
        validate_identity(&fixture).expect("committed fixture identity must be supported");
        let buffers = derived_buffers(&fixture.configuration);
        for case in fixture.cases {
            assert_eq!(
                compute_addresses(
                    &case.input_ids,
                    &case.previous_context,
                    &fixture.configuration,
                    &buffers,
                ),
                case.global_rows,
                "case {}",
                case.name
            );
        }
    }

    #[test]
    fn unsupported_seed_fails_closed() {
        let mut fixture: NGramFixture =
            serde_json::from_str(include_str!("../fixtures/ngram/qwen3_8_flash_next.json"))
                .expect("committed fixture must parse");
        fixture.configuration.seed += 1;
        assert_eq!(
            validate_identity(&fixture),
            Err("n-gram fixture identity, reference, or configuration is unsupported".to_owned())
        );
    }

    #[test]
    fn sparse_reader_matches_synthetic_fixture() {
        let fixture: SyntheticRowFixture = serde_json::from_str(include_str!(
            "../fixtures/ngram/sparse_row_reader_synthetic.json"
        ))
        .expect("synthetic row fixture must parse");
        assert_eq!(fixture.schema_version, 1);
        assert_eq!(fixture.semantic, "firewing_sparse_row_reader_synthetic");
        let header = serde_json::to_vec(&fixture.header_json).expect("header must serialize");
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&decode_hex(&fixture.payload_prefix_hex));
        bytes.extend_from_slice(&decode_hex(&fixture.tensor_hex));
        let mut cursor = Cursor::new(bytes);
        for read in fixture.reads {
            assert_eq!(
                read_sparse_row(&mut cursor, 3, read.row, 3, 4).expect("row read must succeed"),
                decode_hex(&read.expected_hex)
            );
        }
        assert_eq!(
            read_sparse_row(&mut cursor, 3, 3, 3, 4),
            Err("sparse row request is out of bounds".to_owned())
        );
    }

    #[test]
    fn uncached_plan_contains_cross_page_row() {
        let plan = aligned_read_plan(16_384 - 100, 320, 1_000_000, 16_384)
            .expect("cross-page row must have a valid plan");
        assert_eq!(plan.physical_offset, 0);
        assert_eq!(plan.physical_bytes, 32_768);
        assert_eq!(plan.logical_offset, 16_284);
    }
}
