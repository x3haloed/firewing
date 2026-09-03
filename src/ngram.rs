use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
