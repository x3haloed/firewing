use crate::expert::{
    InputSpec, bf16_hash, from_bf16, linear_bf16, make_hidden, swiglu_bf16, to_bf16,
};
use crate::host_safety::{
    HostSafetyMonitor, HostSafetyPolicy, HostSafetySnapshot, PersistentResidencyDeclaration,
};
use crate::verify_expert_fixture;
use memmap2::{Mmap, MmapOptions, UncheckedAdvice};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::os::unix::fs::MetadataExt;
use std::path::{Component, Path};
use std::sync::OnceLock;
use std::time::Instant;

const MODEL: &str = "Qwen/Qwen3.8-Flash-Next";
const REVISION: &str = "de4b8e4d43b917e7706784d8bb445c9af86a3540";
const HIDDEN: usize = 2560;
const INTERMEDIATE: usize = 640;
const EXPERTS: usize = 512;
const SHARDS: usize = 131;
const TENSORS: usize = 1658;
const MEASUREMENTS: usize = 30;

static ACTIVE_CATALOG: OnceLock<CheckpointCatalog> = OnceLock::new();

#[derive(Deserialize)]
struct IdentityBinding {
    schema_version: u32,
    semantic: String,
    model: String,
    revision: String,
    checkpoint_dir: String,
    model_lock_sha256: String,
    verification_receipt: VerificationReceipt,
    files: Vec<IdentityFile>,
    file_count: usize,
    total_bytes: u64,
}

#[derive(Deserialize)]
struct VerificationReceipt {
    path: String,
    sha256: String,
    bytes_hashed: u64,
}

#[derive(Deserialize)]
struct IdentityFile {
    path: String,
    bytes: u64,
    sha256: String,
    device: u64,
    inode: u64,
    modified_ns: i128,
    changed_ns: i128,
}

#[derive(Deserialize)]
struct ModelLock {
    schema_version: u32,
    model: String,
    revision: String,
    expected_file_count: usize,
    expected_total_bytes: u64,
    files: Vec<LockedFile>,
    local_small_file_sha256: BTreeMap<String, String>,
}

#[derive(Deserialize)]
struct LockedFile {
    path: String,
    size: u64,
    kind: String,
    lfs_sha256: Option<String>,
}

#[derive(Deserialize)]
struct Index {
    weight_map: BTreeMap<String, String>,
}

#[derive(Debug)]
struct TensorMetadata {
    dtype: String,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

struct MappedShard {
    mapping: Mmap,
    tensors: BTreeMap<String, TensorMetadata>,
    payload_start: usize,
}

pub(crate) struct TensorView<'a> {
    pub(crate) bytes: &'a [u8],
    pub(crate) shape: &'a [usize],
}

pub(crate) struct CheckpointCatalog {
    weight_map: BTreeMap<String, String>,
    shards: BTreeMap<String, MappedShard>,
    identity_sha256: String,
    total_checkpoint_bytes: u64,
    mapped_shard_bytes: u64,
    header_bytes: u64,
}

fn hash_file(path: &Path) -> Result<String, String> {
    fs::read(path)
        .map(|bytes| format!("{:x}", Sha256::digest(bytes)))
        .map_err(|error| format!("cannot read {}: {error}", path.display()))
}

fn valid_hash(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn safe_relative(name: &str) -> bool {
    let mut components = Path::new(name).components();
    matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none()
}

fn metadata_times(metadata: &fs::Metadata) -> (i128, i128) {
    let modified = i128::from(metadata.mtime()) * 1_000_000_000 + i128::from(metadata.mtime_nsec());
    let changed = i128::from(metadata.ctime()) * 1_000_000_000 + i128::from(metadata.ctime_nsec());
    (modified, changed)
}

fn dtype_bytes(dtype: &str) -> Option<usize> {
    match dtype {
        "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E4M3FN" | "F8_E5M2" => Some(1),
        "U16" | "I16" | "F16" | "BF16" => Some(2),
        "U32" | "I32" | "F32" => Some(4),
        "U64" | "I64" | "F64" => Some(8),
        _ => None,
    }
}

impl MappedShard {
    fn open(path: &Path) -> Result<Self, String> {
        let file = File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
        let file_bytes = file.metadata().map_err(|error| error.to_string())?.len() as usize;
        if file_bytes < 16 {
            return Err(format!("{} is too short", path.display()));
        }
        // SAFETY: the file is read-only and the mapping exposes immutable slices.
        let mapping = unsafe { MmapOptions::new().map(&file) }
            .map_err(|error| format!("cannot map {}: {error}", path.display()))?;
        let header_bytes = u64::from_le_bytes(
            mapping[..8]
                .try_into()
                .map_err(|_| "missing safetensors header prefix")?,
        ) as usize;
        let payload_start = 8_usize
            .checked_add(header_bytes)
            .ok_or("safetensors header overflow")?;
        if header_bytes == 0 || header_bytes > 16 * 1024 * 1024 || payload_start > mapping.len() {
            return Err(format!("invalid safetensors header in {}", path.display()));
        }
        let value: Value = serde_json::from_slice(&mapping[8..payload_start])
            .map_err(|error| format!("{} header: {error}", path.display()))?;
        let object = value
            .as_object()
            .ok_or("safetensors header is not an object")?;
        let payload_bytes = mapping.len() - payload_start;
        let mut tensors = BTreeMap::new();
        let mut ranges = Vec::new();
        for (name, item) in object {
            if name == "__metadata__" {
                if !item.is_object() {
                    return Err("safetensors __metadata__ is not an object".to_owned());
                }
                continue;
            }
            let dtype = item
                .get("dtype")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("{name}: missing dtype"))?;
            let element_bytes =
                dtype_bytes(dtype).ok_or_else(|| format!("{name}: unknown dtype"))?;
            let shape = item
                .get("shape")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("{name}: missing shape"))?
                .iter()
                .map(|value| {
                    value
                        .as_u64()
                        .and_then(|value| usize::try_from(value).ok())
                        .ok_or_else(|| format!("{name}: invalid shape"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            if shape.is_empty() || shape.contains(&0) {
                return Err(format!("{name}: empty shape"));
            }
            let expected = shape.iter().try_fold(element_bytes, |bytes, dimension| {
                bytes.checked_mul(*dimension).ok_or("tensor size overflow")
            })?;
            let offsets = item
                .get("data_offsets")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("{name}: missing offsets"))?;
            if offsets.len() != 2 {
                return Err(format!("{name}: invalid offsets"));
            }
            let start = offsets[0]
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| format!("{name}: invalid start"))?;
            let end = offsets[1]
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| format!("{name}: invalid end"))?;
            if end.checked_sub(start) != Some(expected) || end > payload_bytes {
                return Err(format!("{name}: payload layout mismatch"));
            }
            ranges.push((start, end, name.clone()));
            if tensors
                .insert(
                    name.clone(),
                    TensorMetadata {
                        dtype: dtype.to_owned(),
                        shape,
                        start,
                        end,
                    },
                )
                .is_some()
            {
                return Err(format!("duplicate tensor {name}"));
            }
        }
        ranges.sort_by_key(|range| (range.0, range.1));
        let mut previous_end = 0;
        for (start, end, name) in ranges {
            if start < previous_end {
                return Err(format!("overlapping tensor {name}"));
            }
            previous_end = end;
        }
        Ok(Self {
            mapping,
            tensors,
            payload_start,
        })
    }

    fn tensor(&self, name: &str) -> Result<TensorView<'_>, String> {
        let metadata = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("tensor absent from shard: {name}"))?;
        if metadata.dtype != "BF16" {
            return Err(format!("{name}: expected BF16, got {}", metadata.dtype));
        }
        let start = self.payload_start + metadata.start;
        let end = self.payload_start + metadata.end;
        Ok(TensorView {
            bytes: &self.mapping[start..end],
            shape: &metadata.shape,
        })
    }
}

impl CheckpointCatalog {
    pub(crate) fn open(
        root: &Path,
        lock_path: &Path,
        identity_path: &Path,
        expected_identity_sha256: &str,
    ) -> Result<Self, String> {
        let root = root.canonicalize().map_err(|error| error.to_string())?;
        if !valid_hash(expected_identity_sha256)
            || hash_file(identity_path)? != expected_identity_sha256
        {
            return Err("checkpoint identity binding hash mismatch".to_owned());
        }
        let binding: IdentityBinding =
            serde_json::from_slice(&fs::read(identity_path).map_err(|error| error.to_string())?)
                .map_err(|error| format!("checkpoint identity binding: {error}"))?;
        let lock_bytes = fs::read(lock_path).map_err(|error| error.to_string())?;
        let lock_hash = format!("{:x}", Sha256::digest(&lock_bytes));
        let lock: ModelLock =
            serde_json::from_slice(&lock_bytes).map_err(|error| error.to_string())?;
        if binding.schema_version != 1
            || binding.semantic != "firewing_verified_checkpoint_live_identity_binding"
            || binding.model != MODEL
            || binding.revision != REVISION
            || binding.checkpoint_dir != root.to_string_lossy()
            || binding.model_lock_sha256 != lock_hash
            || binding.file_count != 144
            || binding.files.len() != binding.file_count
            || binding.total_bytes != 360_023_351_514
            || binding.verification_receipt.bytes_hashed != binding.total_bytes
            || !valid_hash(&binding.verification_receipt.sha256)
            || hash_file(Path::new(&binding.verification_receipt.path))?
                != binding.verification_receipt.sha256
            || lock.schema_version != 1
            || lock.model != MODEL
            || lock.revision != REVISION
            || lock.expected_file_count != binding.file_count
            || lock.files.len() != binding.file_count
            || lock.expected_total_bytes != binding.total_bytes
        {
            return Err("checkpoint catalog authority mismatch".to_owned());
        }
        let locked = lock
            .files
            .iter()
            .map(|file| (file.path.as_str(), file))
            .collect::<BTreeMap<_, _>>();
        let mut identities = BTreeMap::new();
        for identity in &binding.files {
            if !safe_relative(&identity.path)
                || identities
                    .insert(identity.path.as_str(), identity)
                    .is_some()
            {
                return Err("unsafe or duplicate identity path".to_owned());
            }
            let expected = locked
                .get(identity.path.as_str())
                .ok_or("identity file absent from lock")?;
            let expected_hash = if expected.kind == "metadata" {
                lock.local_small_file_sha256.get(&expected.path)
            } else {
                expected.lfs_sha256.as_ref()
            };
            let path = root.join(&identity.path);
            let metadata = fs::symlink_metadata(&path)
                .map_err(|error| format!("{}: {error}", path.display()))?;
            let (modified_ns, changed_ns) = metadata_times(&metadata);
            if !metadata.file_type().is_file()
                || metadata.file_type().is_symlink()
                || expected.size != identity.bytes
                || expected_hash.map(String::as_str) != Some(identity.sha256.as_str())
                || metadata.len() != identity.bytes
                || metadata.dev() != identity.device
                || metadata.ino() != identity.inode
                || modified_ns != identity.modified_ns
                || changed_ns != identity.changed_ns
            {
                return Err(format!("live checkpoint identity drift: {}", identity.path));
            }
        }
        if identities.len() != locked.len() {
            return Err("live identity inventory mismatch".to_owned());
        }
        let index: Index = serde_json::from_slice(
            &fs::read(root.join("model.safetensors.index.json"))
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| format!("checkpoint index: {error}"))?;
        if index.weight_map.len() != TENSORS {
            return Err("checkpoint index tensor count mismatch".to_owned());
        }
        let shard_names = index.weight_map.values().cloned().collect::<BTreeSet<_>>();
        if shard_names.len() != SHARDS {
            return Err("checkpoint index shard count mismatch".to_owned());
        }
        let mut shards = BTreeMap::new();
        let mut mapped_shard_bytes = 0_u64;
        let mut header_bytes = 0_u64;
        for shard in shard_names {
            if !safe_relative(&shard) || !identities.contains_key(shard.as_str()) {
                return Err(format!("unbound checkpoint shard: {shard}"));
            }
            let path = root.join(&shard);
            mapped_shard_bytes += path.metadata().map_err(|error| error.to_string())?.len();
            let mapped = MappedShard::open(&path)?;
            header_bytes += mapped.payload_start as u64;
            shards.insert(shard, mapped);
        }
        for (tensor, shard) in &index.weight_map {
            if !shards
                .get(shard)
                .is_some_and(|mapped| mapped.tensors.contains_key(tensor))
            {
                return Err(format!("indexed tensor absent: {tensor}"));
            }
        }
        for (shard, mapped) in &shards {
            for tensor in mapped.tensors.keys() {
                if index.weight_map.get(tensor) != Some(shard) {
                    return Err(format!("unindexed tensor in {shard}: {tensor}"));
                }
            }
        }
        Ok(Self {
            weight_map: index.weight_map,
            shards,
            identity_sha256: expected_identity_sha256.to_owned(),
            total_checkpoint_bytes: binding.total_bytes,
            mapped_shard_bytes,
            header_bytes,
        })
    }

    pub(crate) fn tensor(&self, name: &str) -> Result<TensorView<'_>, String> {
        let shard = self
            .weight_map
            .get(name)
            .ok_or_else(|| format!("tensor absent from index: {name}"))?;
        self.shards
            .get(shard)
            .ok_or("mapped shard absent")?
            .tensor(name)
    }

    pub(crate) fn expert_bf16(
        &self,
        name: &str,
        expert: usize,
        rows: usize,
        columns: usize,
    ) -> Result<&[u16], String> {
        let view = self.tensor(name)?;
        if view.shape != [EXPERTS, rows, columns] || expert >= EXPERTS {
            return Err(format!("{name}: expert shape mismatch"));
        }
        let values = rows.checked_mul(columns).ok_or("expert size overflow")?;
        let start = expert
            .checked_mul(values)
            .and_then(|value| value.checked_mul(2))
            .ok_or("expert offset overflow")?;
        let end = start.checked_add(values * 2).ok_or("expert end overflow")?;
        let bytes = view
            .bytes
            .get(start..end)
            .ok_or("expert slice outside tensor")?;
        if !(bytes.as_ptr() as usize).is_multiple_of(std::mem::align_of::<u16>()) {
            return Err("BF16 expert view is unaligned".to_owned());
        }
        // SAFETY: alignment and exact byte length are checked; u16 accepts all bit patterns.
        Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<u16>(), values) })
    }
}

pub(crate) fn install_active_catalog(
    root: &Path,
    lock_path: &Path,
    identity_path: &Path,
    identity_sha256: &str,
) -> Result<u128, String> {
    if ACTIVE_CATALOG.get().is_some() {
        return Err("checkpoint catalog is already installed in this process".to_owned());
    }
    let started = Instant::now();
    let catalog = CheckpointCatalog::open(root, lock_path, identity_path, identity_sha256)?;
    let elapsed = started.elapsed().as_nanos();
    ACTIVE_CATALOG
        .set(catalog)
        .map_err(|_| "checkpoint catalog installation raced".to_owned())?;
    Ok(elapsed)
}

pub(crate) fn catalog_payloads_authenticated() -> bool {
    ACTIVE_CATALOG.get().is_some()
}

pub(crate) fn active_bf16_tensor(
    path: &Path,
    name: &str,
    expected_shape: &[usize],
) -> Option<Result<Vec<u16>, String>> {
    let catalog = ACTIVE_CATALOG.get()?;
    Some((|| {
        let requested_shard = path
            .file_name()
            .and_then(|value| value.to_str())
            .ok_or("catalog tensor path has no UTF-8 file name")?;
        if catalog.weight_map.get(name).map(String::as_str) != Some(requested_shard) {
            return Err(format!("catalog shard mismatch for {name}"));
        }
        let shard = catalog
            .shards
            .get(requested_shard)
            .ok_or("mapped shard absent")?;
        let metadata = shard
            .tensors
            .get(name)
            .ok_or_else(|| format!("tensor absent from shard: {name}"))?;
        if metadata.dtype != "BF16" || metadata.shape != expected_shape {
            return Err(format!("catalog shape mismatch for {name}"));
        }
        let start = shard.payload_start + metadata.start;
        let len = metadata.end - metadata.start;
        let values = shard.mapping[start..start + len]
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect();
        // SAFETY: the source slice is no longer borrowed after collection and
        // the read-only file mapping remains valid if Darwin evicts its pages.
        unsafe {
            shard
                .mapping
                .unchecked_advise_range(UncheckedAdvice::DontNeed, start, len)
        }
        .map_err(|error| format!("cannot release mapped tensor {name}: {error}"))?;
        Ok(values)
    })())
}

pub(crate) fn active_bf16_expert(
    path: &Path,
    name: &str,
    expert: usize,
    experts: usize,
    rows: usize,
    columns: usize,
) -> Option<Result<Vec<u16>, String>> {
    let catalog = ACTIVE_CATALOG.get()?;
    Some((|| {
        if experts != EXPERTS || expert >= experts {
            return Err(format!("catalog expert count mismatch for {name}"));
        }
        let requested_shard = path
            .file_name()
            .and_then(|value| value.to_str())
            .ok_or("catalog expert path has no UTF-8 file name")?;
        if catalog.weight_map.get(name).map(String::as_str) != Some(requested_shard) {
            return Err(format!("catalog shard mismatch for {name}"));
        }
        let shard = catalog
            .shards
            .get(requested_shard)
            .ok_or("mapped shard absent")?;
        let metadata = shard
            .tensors
            .get(name)
            .ok_or_else(|| format!("tensor absent from shard: {name}"))?;
        if metadata.dtype != "BF16" || metadata.shape != [experts, rows, columns] {
            return Err(format!("catalog expert shape mismatch for {name}"));
        }
        let expert_bytes = rows
            .checked_mul(columns)
            .and_then(|value| value.checked_mul(2))
            .ok_or("catalog expert size overflow")?;
        let start = shard
            .payload_start
            .checked_add(metadata.start)
            .and_then(|value| value.checked_add(expert * expert_bytes))
            .ok_or("catalog expert offset overflow")?;
        let end = start
            .checked_add(expert_bytes)
            .ok_or("catalog expert end overflow")?;
        let bytes = shard
            .mapping
            .get(start..end)
            .ok_or("catalog expert outside tensor")?;
        let values = bytes
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect();
        // SAFETY: the source slice is no longer borrowed after collection and
        // the read-only file mapping remains valid if Darwin evicts its pages.
        unsafe {
            shard
                .mapping
                .unchecked_advise_range(UncheckedAdvice::DontNeed, start, expert_bytes)
        }
        .map_err(|error| format!("cannot release mapped expert {name}: {error}"))?;
        Ok(values)
    })())
}

pub(crate) fn active_bf16_row(
    path: &Path,
    name: &str,
    expected_shape: &[usize],
    row: usize,
    columns: usize,
) -> Option<Result<Vec<u16>, String>> {
    let catalog = ACTIVE_CATALOG.get()?;
    Some((|| {
        let requested_shard = path
            .file_name()
            .and_then(|value| value.to_str())
            .ok_or("catalog row path has no UTF-8 file name")?;
        if catalog.weight_map.get(name).map(String::as_str) != Some(requested_shard) {
            return Err(format!("catalog shard mismatch for {name}"));
        }
        let shard = catalog
            .shards
            .get(requested_shard)
            .ok_or("mapped shard absent")?;
        let metadata = shard
            .tensors
            .get(name)
            .ok_or_else(|| format!("tensor absent from shard: {name}"))?;
        if metadata.dtype != "BF16"
            || metadata.shape != expected_shape
            || expected_shape.last() != Some(&columns)
        {
            return Err(format!("catalog row shape mismatch for {name}"));
        }
        let relative_start = row
            .checked_mul(columns)
            .and_then(|value| value.checked_mul(2))
            .ok_or("catalog row offset overflow")?;
        let relative_end = relative_start
            .checked_add(columns * 2)
            .ok_or("catalog row end overflow")?;
        let bytes = shard
            .mapping
            .get(
                shard.payload_start + metadata.start + relative_start
                    ..shard.payload_start + metadata.start + relative_end,
            )
            .ok_or("catalog row outside tensor")?;
        let values = bytes
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect();
        // SAFETY: the source slice is no longer borrowed after collection and
        // the read-only file mapping remains valid if Darwin evicts its pages.
        unsafe {
            shard.mapping.unchecked_advise_range(
                UncheckedAdvice::DontNeed,
                shard.payload_start + metadata.start + relative_start,
                columns * 2,
            )
        }
        .map_err(|error| format!("cannot release mapped row {name}: {error}"))?;
        Ok(values)
    })())
}

#[derive(Deserialize)]
struct ExpertFixture {
    case: ExpertCase,
}

#[derive(Deserialize)]
struct ExpertCase {
    expert: usize,
    route_weight_bf16: f32,
    input_spec: InputSpec,
    input_bf16_sha256: String,
    expected_bf16_sha256: ExpertHashes,
}

#[derive(Deserialize)]
struct ExpertHashes {
    gate_up: String,
    swiglu: String,
    down: String,
    weighted_down: String,
}

#[derive(Debug, Eq, PartialEq, Serialize)]
pub struct CheckpointCatalogReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub implementation_commit: String,
    pub identity_binding_sha256: String,
    pub checkpoint_bytes_bound: u64,
    pub mapped_shards: usize,
    pub mapped_shard_bytes: u64,
    pub reconciled_tensors: usize,
    pub parsed_header_bytes: u64,
    pub catalog_open_wall_time_ns: u128,
    pub exact_measurements: usize,
    pub hot_expert_wall_times_ns: Vec<u128>,
    pub hot_expert_p10_wall_time_ns: u128,
    pub hot_expert_median_wall_time_ns: u128,
    pub hot_expert_p90_wall_time_ns: u128,
    pub cache_state: &'static str,
    pub host_safety_policy: HostSafetyPolicy,
    pub host_safety_snapshots: Vec<HostSafetySnapshot>,
    pub accepted_tokens: usize,
    pub performance_claim: Option<String>,
}

fn quantile(values: &[u128], numerator: usize, denominator: usize) -> u128 {
    let mut ordered = values.to_vec();
    ordered.sort_unstable();
    ordered[((ordered.len() - 1) * numerator + denominator / 2) / denominator]
}

#[allow(clippy::too_many_arguments)]
pub fn benchmark_checkpoint_catalog(
    checkpoint_dir: &Path,
    model_lock_path: &Path,
    identity_path: &Path,
    identity_sha256: &str,
    router_fixture_path: &Path,
    expert_fixture_path: &Path,
    implementation_commit: &str,
) -> Result<CheckpointCatalogReport, String> {
    if implementation_commit.len() != 40
        || !implementation_commit
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        return Err("implementation commit must be a full hexadecimal Git hash".to_owned());
    }
    let mut safety = HostSafetyMonitor::start_normative(vec![PersistentResidencyDeclaration {
        object: "checkpoint_catalog_headers_maps_and_one_expert_working_set".to_owned(),
        maximum_bytes: 128 * 1024 * 1024,
        lifetime: "catalog_benchmark".to_owned(),
        eviction_order: 1,
    }])?;
    verify_expert_fixture(
        checkpoint_dir,
        model_lock_path,
        router_fixture_path,
        expert_fixture_path,
    )?;
    safety.checkpoint("scalar_authority_complete", true)?;
    let started = Instant::now();
    let catalog = CheckpointCatalog::open(
        checkpoint_dir,
        model_lock_path,
        identity_path,
        identity_sha256,
    )?;
    let catalog_open_wall_time_ns = started.elapsed().as_nanos();
    safety.checkpoint("catalog_open_complete", true)?;
    let fixture: ExpertFixture =
        serde_json::from_slice(&fs::read(expert_fixture_path).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    let case = fixture.case;
    let hidden = make_hidden(HIDDEN, &case.input_spec)?;
    if bf16_hash(&hidden) != case.input_bf16_sha256 {
        return Err("catalog expert input mismatch".to_owned());
    }
    let mut walls = Vec::with_capacity(MEASUREMENTS);
    for _ in 0..MEASUREMENTS {
        let started = Instant::now();
        let gate_up_weight = catalog.expert_bf16(
            "model.language_model.layers.0.mlp.experts.gate_up_proj",
            case.expert,
            INTERMEDIATE * 2,
            HIDDEN,
        )?;
        let down_weight = catalog.expert_bf16(
            "model.language_model.layers.0.mlp.experts.down_proj",
            case.expert,
            HIDDEN,
            INTERMEDIATE,
        )?;
        let gate_up = linear_bf16(gate_up_weight, &hidden, INTERMEDIATE * 2, HIDDEN);
        let swiglu = swiglu_bf16(&gate_up[..INTERMEDIATE], &gate_up[INTERMEDIATE..]);
        let down = linear_bf16(down_weight, &swiglu, HIDDEN, INTERMEDIATE);
        let weighted = down
            .iter()
            .map(|value| to_bf16(from_bf16(*value) * case.route_weight_bf16))
            .collect::<Vec<_>>();
        walls.push(started.elapsed().as_nanos());
        for (name, actual, expected) in [
            ("gate_up", &gate_up, &case.expected_bf16_sha256.gate_up),
            ("swiglu", &swiglu, &case.expected_bf16_sha256.swiglu),
            ("down", &down, &case.expected_bf16_sha256.down),
            (
                "weighted_down",
                &weighted,
                &case.expected_bf16_sha256.weighted_down,
            ),
        ] {
            if bf16_hash(actual) != *expected {
                return Err(format!("catalog expert {name} mismatch"));
            }
        }
    }
    safety.checkpoint("measurements_complete", true)?;
    let report_values = (
        catalog.identity_sha256.clone(),
        catalog.total_checkpoint_bytes,
        catalog.shards.len(),
        catalog.mapped_shard_bytes,
        catalog.weight_map.len(),
        catalog.header_bytes,
    );
    drop(catalog);
    let (host_safety_policy, host_safety_snapshots) = safety.finish()?;
    Ok(CheckpointCatalogReport {
        schema_version: 1,
        semantic: "qwen3_8_flash_next_once_authenticated_mapped_tensor_catalog",
        implementation_commit: implementation_commit.to_owned(),
        identity_binding_sha256: report_values.0,
        checkpoint_bytes_bound: report_values.1,
        mapped_shards: report_values.2,
        mapped_shard_bytes: report_values.3,
        reconciled_tensors: report_values.4,
        parsed_header_bytes: report_values.5,
        catalog_open_wall_time_ns,
        exact_measurements: MEASUREMENTS,
        hot_expert_p10_wall_time_ns: quantile(&walls, 1, 10),
        hot_expert_median_wall_time_ns: quantile(&walls, 1, 2),
        hot_expert_p90_wall_time_ns: quantile(&walls, 9, 10),
        hot_expert_wall_times_ns: walls,
        cache_state: "verified_live_identity_read_only_mmaps_warm_expert_pages_no_per_use_hash_or_copy",
        host_safety_policy,
        host_safety_snapshots,
        accepted_tokens: 0,
        performance_claim: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsafe_catalog_paths_fail() {
        assert!(safe_relative("model.safetensors"));
        assert!(!safe_relative("../model.safetensors"));
        assert!(!safe_relative("nested/model.safetensors"));
    }

    #[test]
    fn catalog_quantiles_are_nearest_index() {
        let values = (0..30).map(|value| value * 10).collect::<Vec<_>>();
        assert_eq!(quantile(&values, 1, 10), 30);
        assert_eq!(quantile(&values, 1, 2), 150);
        assert_eq!(quantile(&values, 9, 10), 260);
    }
}
