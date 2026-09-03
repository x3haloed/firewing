use crate::attention_residual::verify_attention_residual_fixture_bytes_with_outputs;
use crate::decoder_layer3::verify_decoder_mlp_fixture_bytes_with_outputs;
use crate::expert::{bf16_hash, from_bf16};
use crate::full_attention_residual::verify_full_attention_residual_fixture_bytes_with_outputs;
use crate::host_safety::{
    HostSafetyMonitor, HostSafetyPolicy, HostSafetySnapshot, PersistentResidencyDeclaration,
};
use crate::ple::verify_ple_fixture_bytes_with_outputs;
use crate::ple_attention_residual::verify_ple_attention_residual_fixture_bytes_with_outputs;
use crate::text_output::verify_embedded_text_output_fixture;
use crate::verify_tokenizer_fixture;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::time::Instant;

const MODEL: &str = "Qwen/Qwen3.8-Flash-Next";
const HIDDEN: usize = 2560;
const HC_COUNT: usize = 4;
const HC_HIDDEN: usize = HIDDEN * HC_COUNT;
const VOCAB: usize = 248_320;
const TOKEN_IDS: [usize; 2] = [16_207, 22_856];

#[derive(Deserialize)]
struct Fixture {
    schema_version: u32,
    semantic: String,
    model: String,
    revision: String,
    reference: Reference,
    configuration: Configuration,
    embedding: Embedding,
    embedding_root_hashes: Vec<String>,
    layers: Vec<Layer>,
    output: Value,
}

#[derive(Deserialize)]
struct Reference {
    implementation: String,
    transformers_version: String,
    config_sha256: String,
    tensor_index_sha256: String,
    model_lock_sha256: String,
    tokenizer_fixture_sha256: String,
}

#[derive(Deserialize)]
struct Configuration {
    text: String,
    token_ids: Vec<usize>,
    hidden_size: usize,
    hc_count: usize,
    vocab_size: usize,
    layers: usize,
    boundary_dtype: String,
    cache_mode: String,
}

#[derive(Deserialize)]
struct Embedding {
    tensor: String,
    shape: Vec<usize>,
    shard: String,
    shard_bytes: u64,
    shard_sha256: String,
    selected_rows: Vec<EmbeddingRow>,
}

#[derive(Deserialize)]
struct EmbeddingRow {
    token_id: usize,
    payload_sha256: String,
}

#[derive(Deserialize)]
struct Layer {
    layer: usize,
    layer_type: String,
    #[serde(default)]
    ple: Option<Value>,
    attention: Value,
    decoder: Value,
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
pub struct EndpointLayerTiming {
    pub layer: usize,
    pub layer_type: String,
    pub attention_wall_time_ns: u128,
    pub decoder_wall_time_ns: u128,
    pub safety_checkpoint_wall_time_ns: u128,
    pub complete_layer_wall_time_ns: u128,
}

#[derive(Debug, Eq, PartialEq, Serialize)]
pub struct TokenTextEndpointVerificationReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub model: String,
    pub revision: String,
    pub source_text: String,
    pub token_ids: Vec<usize>,
    pub tokenizer_raw_cases_verified: usize,
    pub tokenizer_chat_cases_verified: usize,
    pub embedding_rows_verified: usize,
    pub embedding_root_hashes_verified: usize,
    pub decoder_layers_verified: usize,
    pub linear_layers_verified: usize,
    pub full_attention_layers_verified: usize,
    pub dynamic_expert_selections_verified: usize,
    pub embedding_verified_payload_bytes: usize,
    pub decoder_verified_payload_bytes: usize,
    pub output_verified_payload_bytes: usize,
    pub total_verified_payload_bytes: usize,
    pub top20_token_ids_by_step: Vec<Vec<usize>>,
    pub top20_cutoff_tie_counts_by_step: Vec<usize>,
    pub cache_state: &'static str,
    pub setup_wall_time_ns: u128,
    pub embedding_wall_time_ns: u128,
    pub layer_timings: Vec<EndpointLayerTiming>,
    pub output_wall_time_ns: u128,
    pub final_safety_wall_time_ns: u128,
    pub complete_wall_time_ns: u128,
    pub host_safety_policy: HostSafetyPolicy,
    pub host_safety_snapshots: Vec<HostSafetySnapshot>,
    pub accepted_tokens: usize,
    pub performance_claim: Option<String>,
}

fn sha256_file(path: &Path) -> Result<String, String> {
    fs::read(path)
        .map(|bytes| format!("{:x}", Sha256::digest(bytes)))
        .map_err(|error| format!("cannot read {}: {error}", path.display()))
}

fn is_hash(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn read_embedding_row(
    checkpoint_dir: &Path,
    lock: &ModelLock,
    embedding: &Embedding,
    row: &EmbeddingRow,
) -> Result<Vec<u16>, String> {
    let locked = lock
        .files
        .iter()
        .filter(|file| file.path == embedding.shard)
        .collect::<Vec<_>>();
    if embedding.tensor != "model.language_model.embed_tokens.weight"
        || embedding.shape != [VOCAB, HIDDEN]
        || !is_hash(&embedding.shard_sha256)
        || row.token_id >= VOCAB
        || !is_hash(&row.payload_sha256)
        || locked.len() != 1
        || locked[0].size != embedding.shard_bytes
        || locked[0].lfs_sha256.as_deref() != Some(embedding.shard_sha256.as_str())
        || fs::metadata(checkpoint_dir.join(&embedding.shard))
            .map_err(|error| error.to_string())?
            .len()
            != embedding.shard_bytes
    {
        return Err("token embedding identity mismatch".to_owned());
    }
    let path = checkpoint_dir.join(&embedding.shard);
    let mut file =
        File::open(&path).map_err(|error| format!("cannot open {}: {error}", path.display()))?;
    let mut prefix = [0_u8; 8];
    file.read_exact(&mut prefix)
        .map_err(|error| error.to_string())?;
    let header_bytes = u64::from_le_bytes(prefix);
    if header_bytes == 0 || header_bytes > 16 * 1024 * 1024 {
        return Err("invalid embedding safetensors header length".to_owned());
    }
    let mut header = vec![0_u8; header_bytes as usize];
    file.read_exact(&mut header)
        .map_err(|error| error.to_string())?;
    let header: Value = serde_json::from_slice(&header).map_err(|error| error.to_string())?;
    let item = header
        .get(&embedding.tensor)
        .ok_or("embedding tensor missing from shard")?;
    let shape = item
        .get("shape")
        .and_then(Value::as_array)
        .ok_or("embedding shape missing")?;
    let offsets = item
        .get("data_offsets")
        .and_then(Value::as_array)
        .ok_or("embedding offsets missing")?;
    if item.get("dtype").and_then(Value::as_str) != Some("BF16")
        || shape.iter().filter_map(Value::as_u64).collect::<Vec<_>>()
            != [VOCAB as u64, HIDDEN as u64]
        || offsets.len() != 2
    {
        return Err("embedding shard metadata mismatch".to_owned());
    }
    let tensor_start = offsets[0].as_u64().ok_or("invalid embedding start")?;
    let tensor_end = offsets[1].as_u64().ok_or("invalid embedding end")?;
    if tensor_end.checked_sub(tensor_start) != Some((VOCAB * HIDDEN * 2) as u64) {
        return Err("embedding tensor byte count mismatch".to_owned());
    }
    let absolute = 8_u64
        .checked_add(header_bytes)
        .and_then(|value| value.checked_add(tensor_start))
        .and_then(|value| value.checked_add((row.token_id * HIDDEN * 2) as u64))
        .ok_or("embedding row offset overflow")?;
    file.seek(SeekFrom::Start(absolute))
        .map_err(|error| error.to_string())?;
    let mut bytes = vec![0_u8; HIDDEN * 2];
    file.read_exact(&mut bytes)
        .map_err(|error| error.to_string())?;
    if format!("{:x}", Sha256::digest(&bytes)) != row.payload_sha256 {
        return Err(format!("embedding row {} payload mismatch", row.token_id));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect())
}

fn four_stream_root(row: &[u16]) -> Result<Vec<u16>, String> {
    if row.len() != HIDDEN || row.iter().any(|value| !from_bf16(*value).is_finite()) {
        return Err("invalid token embedding row".to_owned());
    }
    let mut output = Vec::with_capacity(HC_HIDDEN);
    for _ in 0..HC_COUNT {
        output.extend_from_slice(row);
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn verify_token_text_endpoint_fixture(
    checkpoint_dir: &Path,
    model_lock_path: &Path,
    tokenizer_fixture_path: &Path,
    ngram_fixture_path: &Path,
    ngram_row_fixture_path: &Path,
    ple_fixture_path: &Path,
    fixture_path: &Path,
) -> Result<TokenTextEndpointVerificationReport, String> {
    let total_started = Instant::now();
    let setup_started = Instant::now();
    let mut safety = HostSafetyMonitor::start_normative(vec![
        PersistentResidencyDeclaration {
            object: "endpoint_fixture_metadata_and_hash_authority".to_owned(),
            maximum_bytes: 3 * 1024 * 1024,
            lifetime: "complete_verification".to_owned(),
            eviction_order: 2,
        },
        PersistentResidencyDeclaration {
            object: "two_current_four_stream_hidden_roots".to_owned(),
            maximum_bytes: (2 * 4 * 2560 * 2) as u64,
            lifetime: "replaced_at_each_layer_and_released_after_output".to_owned(),
            eviction_order: 1,
        },
    ])?;
    let fixture: Fixture = serde_json::from_slice(
        &fs::read(fixture_path)
            .map_err(|error| format!("cannot read endpoint fixture: {error}"))?,
    )
    .map_err(|error| format!("malformed endpoint fixture: {error}"))?;
    let config = &fixture.configuration;
    let expected_types = (0..48)
        .map(|layer| {
            if layer % 4 == 3 {
                "full_attention"
            } else {
                "linear_attention"
            }
        })
        .collect::<Vec<_>>();
    if fixture.schema_version != 1
        || fixture.semantic != "qwen3_8_flash_next_firewing_two_token_cached_text_logits"
        || fixture.model != MODEL
        || fixture.reference.implementation
            != "source_derived_and_official_huggingface_transformers_qwen4_exp"
        || fixture.reference.transformers_version != "5.16.1"
        || config.text != "Firewing"
        || config.token_ids != TOKEN_IDS
        || config.hidden_size != HIDDEN
        || config.hc_count != HC_COUNT
        || config.vocab_size != VOCAB
        || config.layers != 48
        || config.boundary_dtype != "BF16"
        || config.cache_mode != "sequential_incremental"
        || fixture.embedding.selected_rows.len() != 2
        || fixture.embedding_root_hashes.len() != 2
        || fixture.layers.len() != 48
        || fixture.layers.iter().enumerate().any(|(layer, record)| {
            record.layer != layer
                || record.layer_type != expected_types[layer]
                || (layer == 1) != record.ple.is_some()
        })
        || sha256_file(model_lock_path)? != fixture.reference.model_lock_sha256
        || sha256_file(&checkpoint_dir.join("config.json"))? != fixture.reference.config_sha256
        || sha256_file(&checkpoint_dir.join("model.safetensors.index.json"))?
            != fixture.reference.tensor_index_sha256
        || sha256_file(tokenizer_fixture_path)? != fixture.reference.tokenizer_fixture_sha256
    {
        return Err("token text endpoint identity or configuration is unsupported".to_owned());
    }
    let tokenizer = verify_tokenizer_fixture(checkpoint_dir, tokenizer_fixture_path)?;
    let lock: ModelLock =
        serde_json::from_slice(&fs::read(model_lock_path).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    if lock.model != fixture.model || lock.revision != fixture.revision {
        return Err("token text endpoint model lock mismatch".to_owned());
    }
    let setup_wall_time_ns = setup_started.elapsed().as_nanos();
    let embedding_started = Instant::now();
    let mut current_outputs = Vec::with_capacity(2);
    for (ordinal, row) in fixture.embedding.selected_rows.iter().enumerate() {
        if row.token_id != TOKEN_IDS[ordinal] {
            return Err("token embedding row order mismatch".to_owned());
        }
        let root = four_stream_root(&read_embedding_row(
            checkpoint_dir,
            &lock,
            &fixture.embedding,
            row,
        )?)?;
        if bf16_hash(&root) != fixture.embedding_root_hashes[ordinal] {
            return Err(format!("embedding root mismatch at step {ordinal}"));
        }
        current_outputs.push(root);
    }
    safety.checkpoint("embedding_complete", true)?;
    let embedding_wall_time_ns = embedding_started.elapsed().as_nanos();

    let mut decoder_bytes = 0_usize;
    let mut selections = 0_usize;
    let mut linear_layers = 0_usize;
    let mut full_layers = 0_usize;
    let mut layer_timings = Vec::with_capacity(48);
    for layer in &fixture.layers {
        let layer_started = Instant::now();
        let attention_started = Instant::now();
        let attention_bytes =
            serde_json::to_vec(&layer.attention).map_err(|error| error.to_string())?;
        let modes = if layer.layer_type == "linear_attention" {
            ["initial_chunk", "cached_recurrent"]
        } else {
            ["initial", "cached_incremental"]
        };
        let (post_attention, attention_payload_bytes) = if layer.layer == 1 {
            let ple_value = layer.ple.as_ref().ok_or("missing layer-1 PLE")?;
            let ple_steps = ple_value
                .pointer("/case/steps")
                .and_then(Value::as_array)
                .ok_or("layer-1 PLE token steps missing")?;
            let attention_steps = layer
                .attention
                .pointer("/case/steps")
                .and_then(Value::as_array)
                .ok_or("layer-1 attention token steps missing")?;
            if ple_steps.len() != 2
                || attention_steps.len() != 2
                || (0..2).any(|ordinal| {
                    ple_steps[ordinal].get("token_id").and_then(Value::as_u64)
                        != Some(TOKEN_IDS[ordinal] as u64)
                        || attention_steps[ordinal]
                            .get("token_id")
                            .and_then(Value::as_u64)
                            != Some(TOKEN_IDS[ordinal] as u64)
                })
            {
                return Err("layer-1 token identity mismatch".to_owned());
            }
            let ple_bytes = serde_json::to_vec(ple_value).map_err(|error| error.to_string())?;
            let ple_execution = verify_ple_fixture_bytes_with_outputs(
                checkpoint_dir,
                model_lock_path,
                ngram_fixture_path,
                ngram_row_fixture_path,
                &ple_bytes,
                "qwen3_8_flash_next_token_layer1_ple",
                [TOKEN_IDS[0] as i64, TOKEN_IDS[1] as i64],
                Some(&current_outputs),
            )?;
            let execution = verify_ple_attention_residual_fixture_bytes_with_outputs(
                checkpoint_dir,
                model_lock_path,
                ngram_fixture_path,
                ngram_row_fixture_path,
                ple_fixture_path,
                &attention_bytes,
                "qwen3_8_flash_next_token_layer1_attention",
                [TOKEN_IDS[0] as i64, TOKEN_IDS[1] as i64],
                Some(&current_outputs),
                Some(ple_execution),
            )?;
            linear_layers += 1;
            let bytes = execution.0.total_verified_payload_bytes;
            (execution.1, bytes)
        } else if layer.layer_type == "linear_attention" {
            let execution = verify_attention_residual_fixture_bytes_with_outputs(
                checkpoint_dir,
                model_lock_path,
                &attention_bytes,
                layer.layer,
                &format!("qwen3_8_flash_next_token_layer{}_attention", layer.layer),
                &format!("layer_{}_two_token_attention_residual", layer.layer),
                "qwen3_8_flash_next_token_endpoint_linear_attention_verification",
                Some(&current_outputs),
            )?;
            linear_layers += 1;
            (execution.1, execution.0.tensor_payload_bytes)
        } else {
            let execution = verify_full_attention_residual_fixture_bytes_with_outputs(
                checkpoint_dir,
                model_lock_path,
                &attention_bytes,
                &format!("qwen3_8_flash_next_token_layer{}_attention", layer.layer),
                "qwen3_8_flash_next_token_endpoint_full_attention_verification",
                layer.layer,
                [0, 1],
                modes,
                true,
                None,
                Some(&current_outputs),
            )?;
            full_layers += 1;
            (execution.1, execution.0.total_verified_payload_bytes)
        };
        let attention_wall_time_ns = attention_started.elapsed().as_nanos();
        let decoder_started = Instant::now();
        let decoder_fixture =
            serde_json::to_vec(&layer.decoder).map_err(|error| error.to_string())?;
        let execution = verify_decoder_mlp_fixture_bytes_with_outputs(
            checkpoint_dir,
            model_lock_path,
            &decoder_fixture,
            layer.layer,
            &layer.layer_type,
            &format!("qwen3_8_flash_next_token_layer{}_decoder", layer.layer),
            "qwen3_8_flash_next_token_endpoint_decoder_verification",
            modes,
            attention_payload_bytes,
            post_attention,
        )?;
        selections += execution
            .0
            .selected_experts_by_step
            .iter()
            .map(Vec::len)
            .sum::<usize>();
        decoder_bytes += execution.0.total_verified_payload_bytes;
        current_outputs = execution.1;
        let decoder_wall_time_ns = decoder_started.elapsed().as_nanos();
        let safety_started = Instant::now();
        safety.checkpoint(&format!("layer_{}_complete", layer.layer), true)?;
        let safety_checkpoint_wall_time_ns = safety_started.elapsed().as_nanos();
        layer_timings.push(EndpointLayerTiming {
            layer: layer.layer,
            layer_type: layer.layer_type.clone(),
            attention_wall_time_ns,
            decoder_wall_time_ns,
            safety_checkpoint_wall_time_ns,
            complete_layer_wall_time_ns: layer_started.elapsed().as_nanos(),
        });
    }
    let output_started = Instant::now();
    let output = verify_embedded_text_output_fixture(
        checkpoint_dir,
        model_lock_path,
        &fixture.output,
        &fixture.model,
        &fixture.revision,
        &current_outputs,
    )?;
    let output_wall_time_ns = output_started.elapsed().as_nanos();
    let final_safety_started = Instant::now();
    let (host_safety_policy, host_safety_snapshots) = safety.finish()?;
    let final_safety_wall_time_ns = final_safety_started.elapsed().as_nanos();
    let complete_wall_time_ns = total_started.elapsed().as_nanos();
    let embedding_bytes = TOKEN_IDS.len() * HIDDEN * 2;
    Ok(TokenTextEndpointVerificationReport {
        schema_version: 1,
        semantic: "qwen3_8_flash_next_firewing_two_token_cached_text_logits_verification",
        model: fixture.model,
        revision: fixture.revision,
        source_text: config.text.clone(),
        token_ids: config.token_ids.clone(),
        tokenizer_raw_cases_verified: tokenizer.raw_cases_verified,
        tokenizer_chat_cases_verified: tokenizer.chat_cases_verified,
        embedding_rows_verified: 2,
        embedding_root_hashes_verified: 2,
        decoder_layers_verified: 48,
        linear_layers_verified: linear_layers,
        full_attention_layers_verified: full_layers,
        dynamic_expert_selections_verified: selections,
        embedding_verified_payload_bytes: embedding_bytes,
        decoder_verified_payload_bytes: decoder_bytes,
        output_verified_payload_bytes: output.output_verified_payload_bytes,
        total_verified_payload_bytes: embedding_bytes
            + decoder_bytes
            + output.output_verified_payload_bytes,
        top20_token_ids_by_step: output.top20_token_ids_by_step,
        top20_cutoff_tie_counts_by_step: output.top20_cutoff_tie_counts_by_step,
        cache_state: "uncontrolled_mixed_os_cache_no_application_tensor_cache",
        setup_wall_time_ns,
        embedding_wall_time_ns,
        layer_timings,
        output_wall_time_ns,
        final_safety_wall_time_ns,
        complete_wall_time_ns,
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
    fn committed_endpoint_has_exact_schedule_and_tokenizer_ids() {
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../fixtures/endpoint/qwen3_8_flash_next_firewing_two_token.json"
        ))
        .unwrap();
        assert_eq!(fixture.configuration.token_ids, TOKEN_IDS);
        assert_eq!(fixture.layers.len(), 48);
        assert_eq!(
            fixture
                .layers
                .iter()
                .filter(|layer| layer.ple.is_some())
                .count(),
            1
        );
        assert_eq!(fixture.layers[1].layer, 1);
    }
}
