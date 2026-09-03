use crate::decoder_layer3::verify_decoder_mlp_fixture_bytes_with_prefix;
use crate::expert::{bf16_hash, bf16_payload_matches, from_bf16, linear_bf16, to_bf16};
use crate::full_attention_residual::verify_full_attention_residual_fixture_bytes_with_prefix;
use crate::hyper_connection::pytorch_inner_square_sum;
use crate::text_output::verify_embedded_text_output_fixture_with_names;
use crate::token_text_endpoint::{
    TokenTextEndpointVerificationReport, verify_token_text_endpoint_fixture_with_outputs,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

const MODEL: &str = "Qwen/Qwen3.8-Flash-Next";
const REVISION: &str = "de4b8e4d43b917e7706784d8bb445c9af86a3540";
const SGLANG_COMMIT: &str = "78c5024e9d9f589dcb4deb7f4ba4fb23f7e85385";
const SGLANG_MTP_SHA256: &str = "2b2ec09230875279a75ae651a1d9e1d88999bc89748e9d0cb6b4a768ffc0e54e";
const HIDDEN: usize = 2560;
const HC_COUNT: usize = 4;
const HC_HIDDEN: usize = HIDDEN * HC_COUNT;
const VOCAB: usize = 248_320;
const EAGLE_WORKER_SHA256: &str =
    "9a66d31868385646b9fb9f78053730f55d2e885e72382a8c8dc6db9f07709271";
const EAGLE_UTILS_SHA256: &str = "87e9dc749e94f5899140457393389397840a2258978c021fd3ac490e9da4c053";

#[derive(Deserialize)]
struct Fixture {
    schema_version: u32,
    semantic: String,
    model: String,
    revision: String,
    reference: Reference,
    configuration: Configuration,
    case: Case,
    sequence_case: Option<SequenceCase>,
}

#[derive(Deserialize)]
struct Reference {
    implementation: String,
    commit: String,
    source: String,
    source_sha256: String,
    source_lock_sha256: String,
    config_sha256: String,
    tensor_index_sha256: String,
    model_lock_sha256: String,
}

#[derive(Deserialize)]
struct Configuration {
    hidden_size: usize,
    hc_count: usize,
    target_hidden_size: usize,
    rms_norm_eps: f32,
    boundary_dtype: String,
    mtp_num_hidden_layers: usize,
    mtp_use_dedicated_embeddings: bool,
    mtp_layer_types: Vec<String>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    input_specs: BTreeMap<String, InputSpec>,
    tensors: BTreeMap<String, Tensor>,
    expected_bf16_sha256: BTreeMap<String, String>,
}

#[derive(Deserialize)]
struct SequenceCase {
    name: String,
    input_specs: BTreeMap<String, InputSpec>,
    expected_bf16_sha256: BTreeMap<String, String>,
}

#[derive(Deserialize)]
struct InputSpec {
    multiplier: i64,
    add: i64,
    modulus: i64,
    center: i64,
    divisor: i64,
}

#[derive(Deserialize)]
struct Tensor {
    tensor: String,
    shape: Vec<usize>,
    shard: String,
    shard_bytes: u64,
    shard_sha256: String,
    payload_sha256: String,
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

#[derive(Deserialize)]
struct SourceLock {
    schema_version: u32,
    repository: String,
    pull_request: String,
    commit: String,
    files: Vec<SourceFile>,
}

#[derive(Deserialize)]
struct SourceFile {
    path: String,
    git_blob: String,
    sha256: String,
}

#[derive(Debug, Eq, PartialEq, Serialize)]
pub struct MtpInputFusionVerificationReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub model: String,
    pub revision: String,
    pub source_commit: String,
    pub case: String,
    pub target_hidden_streams: usize,
    pub tensors_verified: usize,
    pub tensor_payload_bytes: usize,
    pub exact_capture_hashes: usize,
    pub accepted_tokens: usize,
    pub performance_claim: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct MtpProposalVerificationReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub model: String,
    pub revision: String,
    pub source_commit: String,
    pub steps_verified: usize,
    pub exact_input_fusion_capture_hashes: usize,
    pub exact_bf16_capture_hashes: usize,
    pub exact_f32_capture_hashes: usize,
    pub exact_i64_capture_hashes: usize,
    pub dense_tensors_verified: usize,
    pub unique_experts_verified: usize,
    pub total_verified_payload_bytes: usize,
    pub selected_experts_by_step: Vec<Vec<usize>>,
    pub vocab_size: usize,
    pub exact_full_logit_hashes: usize,
    pub exact_mixer_capture_hashes: usize,
    pub exact_ranked_logit_entries: usize,
    pub top20_token_ids_by_step: Vec<Vec<usize>>,
    pub top20_cutoff_tie_counts_by_step: Vec<usize>,
    pub accepted_tokens: usize,
    #[serde(rename = "A")]
    pub accepted_per_transaction: usize,
    #[serde(rename = "U")]
    pub expert_union: usize,
    pub performance_claim: Option<String>,
}

#[derive(Deserialize)]
struct CausalSeedFixture {
    schema_version: u32,
    semantic: String,
    model: String,
    revision: String,
    reference: CausalSeedReference,
    configuration: CausalSeedConfiguration,
    embedding: CausalEmbedding,
    tensors: BTreeMap<String, Tensor>,
    steps: Vec<CausalFusionStep>,
}

#[derive(Deserialize)]
struct CausalSeedReference {
    implementation: String,
    commit: String,
    mtp_source_lock_sha256: String,
    scheduler_source_lock_sha256: String,
    endpoint_fixture_sha256: String,
    fusion_fixture_sha256: String,
    model_lock_sha256: String,
    tensor_index_sha256: String,
}

#[derive(Deserialize)]
struct CausalSeedConfiguration {
    target_input_token_ids: Vec<usize>,
    target_next_token_id: usize,
    mtp_prefill_token_ids: Vec<usize>,
    mtp_positions: Vec<usize>,
    cache_mode: String,
    boundary_dtype: String,
}

#[derive(Deserialize)]
struct CausalEmbedding {
    tensor: String,
    shape: Vec<usize>,
    shard: String,
    shard_bytes: u64,
    shard_sha256: String,
}

#[derive(Deserialize)]
struct CausalFusionStep {
    ordinal: usize,
    mtp_input_token_id: usize,
    target_hidden_endpoint_ordinal: usize,
    captures: BTreeMap<String, String>,
}

#[derive(Debug, Serialize)]
pub struct MtpCausalPrefillVerificationReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub model: String,
    pub revision: String,
    pub source_commit: String,
    pub target_input_token_ids: Vec<usize>,
    pub target_next_token_id: usize,
    pub mtp_prefill_token_ids: Vec<usize>,
    pub proposal_token_id: usize,
    pub target_endpoint: TokenTextEndpointVerificationReport,
    pub fusion_steps_verified: usize,
    pub exact_fusion_capture_hashes: usize,
    pub exact_bf16_capture_hashes: usize,
    pub exact_f32_capture_hashes: usize,
    pub exact_i64_capture_hashes: usize,
    pub dense_tensors_verified: usize,
    pub unique_experts_verified: usize,
    pub total_verified_payload_bytes: usize,
    pub selected_experts_by_step: Vec<Vec<usize>>,
    pub top20_token_ids_by_step: Vec<Vec<usize>>,
    pub accepted_tokens: usize,
    #[serde(rename = "A")]
    pub accepted_per_transaction: usize,
    #[serde(rename = "U")]
    pub expert_union: usize,
    pub performance_claim: Option<String>,
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let bytes =
        fs::read(path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn make_input(size: usize, spec: &InputSpec) -> Result<Vec<u16>, String> {
    if spec.modulus <= 0 || spec.divisor <= 0 {
        return Err("invalid MTP input specification".to_owned());
    }
    Ok((0..size)
        .map(|index| {
            let value = ((index as i64 * spec.multiplier + spec.add).rem_euclid(spec.modulus)
                - spec.center) as f32
                / spec.divisor as f32;
            to_bf16(value)
        })
        .collect())
}

fn rms_norm(input: &[u16], weight: &[u16], epsilon: f32) -> Result<Vec<u16>, String> {
    if input.len() != weight.len() || input.is_empty() {
        return Err("MTP RMSNorm shape mismatch".to_owned());
    }
    let values: Vec<_> = input.iter().map(|value| from_bf16(*value)).collect();
    let inverse = (pytorch_inner_square_sum(&values) / input.len() as f32 + epsilon)
        .sqrt()
        .recip();
    Ok(input
        .iter()
        .zip(weight)
        .map(|(value, weight)| to_bf16(from_bf16(*value) * inverse * (1.0 + from_bf16(*weight))))
        .collect())
}

fn read_tensor(path: &Path, tensor: &str, expected_shape: &[usize]) -> Result<Vec<u16>, String> {
    if let Some(result) =
        crate::checkpoint_catalog::active_bf16_tensor(path, tensor, expected_shape)
    {
        return result;
    }
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
        .ok_or("tensor element count overflow")?;
    if end.checked_sub(start) != Some((count * 2) as u64) {
        return Err(format!("tensor {tensor} byte count mismatch"));
    }
    let absolute = 8_u64
        .checked_add(header_bytes)
        .and_then(|value| value.checked_add(start))
        .ok_or("tensor offset overflow")?;
    file.seek(SeekFrom::Start(absolute))
        .map_err(|error| error.to_string())?;
    let mut payload = vec![0_u8; count * 2];
    file.read_exact(&mut payload)
        .map_err(|error| error.to_string())?;
    Ok(payload
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect())
}

fn require_capture(
    captures: &BTreeMap<String, String>,
    name: &str,
    actual: &[u16],
) -> Result<(), String> {
    let expected = captures
        .get(name)
        .ok_or_else(|| format!("missing MTP capture {name}"))?;
    if bf16_hash(actual) != *expected {
        return Err(format!("MTP input-fusion capture mismatch at {name}"));
    }
    Ok(())
}

fn verify_mtp_input_fusion_fixture_with_output(
    checkpoint_dir: &Path,
    model_lock_path: &Path,
    source_lock_path: &Path,
    fixture_path: &Path,
) -> Result<(MtpInputFusionVerificationReport, Vec<Vec<u16>>), String> {
    let fixture: Fixture =
        serde_json::from_slice(&fs::read(fixture_path).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    let model_lock: ModelLock =
        serde_json::from_slice(&fs::read(model_lock_path).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    let source_lock: SourceLock =
        serde_json::from_slice(&fs::read(source_lock_path).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;

    if fixture.schema_version != 1
        || fixture.semantic != "qwen3_8_flash_next_real_mtp_input_fusion"
        || fixture.model != MODEL
        || fixture.revision != REVISION
        || model_lock.model != MODEL
        || model_lock.revision != REVISION
    {
        return Err("MTP input-fusion identity mismatch".to_owned());
    }
    if source_lock.schema_version != 1
        || source_lock.repository != "https://github.com/sgl-project/sglang"
        || source_lock.pull_request != "https://github.com/sgl-project/sglang/pull/36497"
        || source_lock.commit != SGLANG_COMMIT
        || fixture.reference.implementation != "sglang_qwen4_exp_mtp_source_derived"
        || fixture.reference.commit != SGLANG_COMMIT
        || fixture.reference.source
            != "python/sglang/srt/models/qwen4_exp_mtp.py:Qwen4ExpForCausalLMMTP._fuse_residual_linear_shared"
        || fixture.reference.source_sha256 != SGLANG_MTP_SHA256
        || fixture.reference.source_lock_sha256 != sha256_file(source_lock_path)?
    {
        return Err("MTP source authority mismatch".to_owned());
    }
    let source = source_lock
        .files
        .iter()
        .find(|file| file.path == "python/sglang/srt/models/qwen4_exp_mtp.py")
        .ok_or("MTP source file absent from source lock")?;
    if source.sha256 != SGLANG_MTP_SHA256
        || source.git_blob != "5f0becb2bdf032fdc07b37441aa90d5aea61c250"
    {
        return Err("MTP source file identity mismatch".to_owned());
    }
    if fixture.reference.model_lock_sha256 != sha256_file(model_lock_path)?
        || fixture.reference.config_sha256 != sha256_file(&checkpoint_dir.join("config.json"))?
        || fixture.reference.tensor_index_sha256
            != sha256_file(&checkpoint_dir.join("model.safetensors.index.json"))?
    {
        return Err("MTP checkpoint metadata hash mismatch".to_owned());
    }
    let config = &fixture.configuration;
    if config.hidden_size != HIDDEN
        || config.hc_count != HC_COUNT
        || config.target_hidden_size != HC_HIDDEN
        || config.rms_norm_eps.to_bits() != 1.0e-6_f32.to_bits()
        || config.boundary_dtype != "BF16"
        || config.mtp_num_hidden_layers != 1
        || config.mtp_use_dedicated_embeddings
        || config.mtp_layer_types != ["full_attention"]
    {
        return Err("unsupported MTP input-fusion configuration".to_owned());
    }

    let expected = [
        (
            "pre_fc_norm_embedding",
            "mtp.pre_fc_norm_embedding.weight",
            vec![HIDDEN],
        ),
        (
            "pre_fc_norm_hidden",
            "mtp.pre_fc_norm_hidden.weight",
            vec![HC_HIDDEN],
        ),
        (
            "fc_embedding",
            "mtp.fc_embedding.weight",
            vec![HIDDEN, HIDDEN],
        ),
        ("fc_hidden", "mtp.fc_hidden.weight", vec![HIDDEN, HIDDEN]),
    ];
    let mut values = BTreeMap::new();
    let mut payload_bytes = 0_usize;
    for (key, tensor_name, shape) in expected {
        let record = fixture
            .case
            .tensors
            .get(key)
            .ok_or_else(|| format!("missing MTP tensor record {key}"))?;
        if record.tensor != tensor_name
            || record.shape != shape
            || record.shard.contains('/')
            || record.shard.contains("..")
        {
            return Err(format!("invalid MTP tensor record {key}"));
        }
        let locked = model_lock
            .files
            .iter()
            .find(|file| file.path == record.shard)
            .ok_or_else(|| format!("MTP shard {} absent from model lock", record.shard))?;
        if locked.size != record.shard_bytes
            || locked.lfs_sha256.as_deref() != Some(&record.shard_sha256)
        {
            return Err(format!("MTP shard identity mismatch for {key}"));
        }
        let value = read_tensor(&checkpoint_dir.join(&record.shard), tensor_name, &shape)?;
        if !bf16_payload_matches(&value, &record.payload_sha256) {
            return Err(format!("MTP tensor payload mismatch for {key}"));
        }
        payload_bytes = payload_bytes
            .checked_add(value.len() * 2)
            .ok_or("MTP payload byte count overflow")?;
        values.insert(key, value);
    }

    let embedding = make_input(
        HIDDEN,
        fixture
            .case
            .input_specs
            .get("embedding")
            .ok_or("missing embedding input specification")?,
    )?;
    let target_hidden = make_input(
        HC_HIDDEN,
        fixture
            .case
            .input_specs
            .get("target_hidden")
            .ok_or("missing target-hidden input specification")?,
    )?;
    require_capture(&fixture.case.expected_bf16_sha256, "embedding", &embedding)?;
    require_capture(
        &fixture.case.expected_bf16_sha256,
        "target_hidden",
        &target_hidden,
    )?;
    let embedding_normed = rms_norm(&embedding, &values["pre_fc_norm_embedding"], 1.0e-6)?;
    let target_hidden_normed = rms_norm(&target_hidden, &values["pre_fc_norm_hidden"], 1.0e-6)?;
    require_capture(
        &fixture.case.expected_bf16_sha256,
        "embedding_normed",
        &embedding_normed,
    )?;
    require_capture(
        &fixture.case.expected_bf16_sha256,
        "target_hidden_normed",
        &target_hidden_normed,
    )?;
    let embedding_projected =
        linear_bf16(&values["fc_embedding"], &embedding_normed, HIDDEN, HIDDEN);
    let mut target_hidden_projected = Vec::with_capacity(HC_HIDDEN);
    for stream in target_hidden_normed.chunks_exact(HIDDEN) {
        target_hidden_projected.extend(linear_bf16(&values["fc_hidden"], stream, HIDDEN, HIDDEN));
    }
    require_capture(
        &fixture.case.expected_bf16_sha256,
        "embedding_projected",
        &embedding_projected,
    )?;
    require_capture(
        &fixture.case.expected_bf16_sha256,
        "target_hidden_projected",
        &target_hidden_projected,
    )?;
    let fused_hidden: Vec<_> = target_hidden_projected
        .chunks_exact(HIDDEN)
        .flat_map(|stream| {
            stream
                .iter()
                .zip(&embedding_projected)
                .map(|(hidden, embedding)| to_bf16(from_bf16(*hidden) + from_bf16(*embedding)))
        })
        .collect();
    require_capture(
        &fixture.case.expected_bf16_sha256,
        "fused_hidden",
        &fused_hidden,
    )?;

    let mut fused_outputs = vec![fused_hidden];
    if let Some(sequence) = &fixture.sequence_case {
        if sequence.name != "real_mtp_second_position_input_fusion"
            || sequence.expected_bf16_sha256.len() != 7
        {
            return Err("unsupported MTP sequence-fusion case".to_owned());
        }
        let embedding = make_input(
            HIDDEN,
            sequence
                .input_specs
                .get("embedding")
                .ok_or("missing sequence embedding input specification")?,
        )?;
        let target_hidden = make_input(
            HC_HIDDEN,
            sequence
                .input_specs
                .get("target_hidden")
                .ok_or("missing sequence target-hidden input specification")?,
        )?;
        require_capture(&sequence.expected_bf16_sha256, "embedding", &embedding)?;
        require_capture(
            &sequence.expected_bf16_sha256,
            "target_hidden",
            &target_hidden,
        )?;
        let embedding_normed = rms_norm(&embedding, &values["pre_fc_norm_embedding"], 1.0e-6)?;
        let target_hidden_normed = rms_norm(&target_hidden, &values["pre_fc_norm_hidden"], 1.0e-6)?;
        require_capture(
            &sequence.expected_bf16_sha256,
            "embedding_normed",
            &embedding_normed,
        )?;
        require_capture(
            &sequence.expected_bf16_sha256,
            "target_hidden_normed",
            &target_hidden_normed,
        )?;
        let embedding_projected =
            linear_bf16(&values["fc_embedding"], &embedding_normed, HIDDEN, HIDDEN);
        let mut target_hidden_projected = Vec::with_capacity(HC_HIDDEN);
        for stream in target_hidden_normed.chunks_exact(HIDDEN) {
            target_hidden_projected.extend(linear_bf16(
                &values["fc_hidden"],
                stream,
                HIDDEN,
                HIDDEN,
            ));
        }
        require_capture(
            &sequence.expected_bf16_sha256,
            "embedding_projected",
            &embedding_projected,
        )?;
        require_capture(
            &sequence.expected_bf16_sha256,
            "target_hidden_projected",
            &target_hidden_projected,
        )?;
        let fused_hidden = target_hidden_projected
            .chunks_exact(HIDDEN)
            .flat_map(|stream| {
                stream
                    .iter()
                    .zip(&embedding_projected)
                    .map(|(hidden, embedding)| to_bf16(from_bf16(*hidden) + from_bf16(*embedding)))
            })
            .collect::<Vec<_>>();
        require_capture(
            &sequence.expected_bf16_sha256,
            "fused_hidden",
            &fused_hidden,
        )?;
        fused_outputs.push(fused_hidden);
    }

    Ok((
        MtpInputFusionVerificationReport {
            schema_version: 1,
            semantic: "qwen3_8_flash_next_real_mtp_input_fusion_verification",
            model: fixture.model,
            revision: fixture.revision,
            source_commit: fixture.reference.commit,
            case: fixture.case.name,
            target_hidden_streams: HC_COUNT,
            tensors_verified: fixture.case.tensors.len(),
            tensor_payload_bytes: payload_bytes,
            exact_capture_hashes: fixture.case.expected_bf16_sha256.len(),
            accepted_tokens: 0,
            performance_claim: None,
        },
        fused_outputs,
    ))
}

pub fn verify_mtp_input_fusion_fixture(
    checkpoint_dir: &Path,
    model_lock_path: &Path,
    source_lock_path: &Path,
    fixture_path: &Path,
) -> Result<MtpInputFusionVerificationReport, String> {
    verify_mtp_input_fusion_fixture_with_output(
        checkpoint_dir,
        model_lock_path,
        source_lock_path,
        fixture_path,
    )
    .map(|(report, _)| report)
}

fn reference_hash(value: &Value, name: &str) -> Result<String, String> {
    value
        .get("reference")
        .and_then(|reference| reference.get(name))
        .and_then(Value::as_str)
        .filter(|hash| hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .map(str::to_owned)
        .ok_or_else(|| format!("MTP decoder fixture lacks reference hash {name}"))
}

pub fn verify_mtp_proposal_fixture(
    checkpoint_dir: &Path,
    model_lock_path: &Path,
    source_lock_path: &Path,
    fusion_fixture_path: &Path,
    attention_fixture_path: &Path,
    decoder_fixture_path: &Path,
    output_fixture_path: &Path,
) -> Result<MtpProposalVerificationReport, String> {
    let source_lock_hash = sha256_file(source_lock_path)?;
    let fusion_fixture_hash = sha256_file(fusion_fixture_path)?;
    let attention_fixture_hash = sha256_file(attention_fixture_path)?;
    let decoder_fixture_hash = sha256_file(decoder_fixture_path)?;
    let attention_bytes = fs::read(attention_fixture_path)
        .map_err(|error| format!("cannot read MTP attention fixture: {error}"))?;
    let decoder_bytes = fs::read(decoder_fixture_path)
        .map_err(|error| format!("cannot read MTP decoder fixture: {error}"))?;
    let attention_value: Value =
        serde_json::from_slice(&attention_bytes).map_err(|error| error.to_string())?;
    let decoder_value: Value =
        serde_json::from_slice(&decoder_bytes).map_err(|error| error.to_string())?;
    let output_value: Value = serde_json::from_slice(
        &fs::read(output_fixture_path)
            .map_err(|error| format!("cannot read MTP output fixture: {error}"))?,
    )
    .map_err(|error| error.to_string())?;
    if reference_hash(&attention_value, "source_lock_sha256")? != source_lock_hash
        || reference_hash(&attention_value, "mtp_input_fusion_fixture_sha256")?
            != fusion_fixture_hash
        || reference_hash(&decoder_value, "source_lock_sha256")? != source_lock_hash
        || reference_hash(&decoder_value, "mtp_input_fusion_fixture_sha256")? != fusion_fixture_hash
        || reference_hash(&decoder_value, "attention_residual_fixture_sha256")?
            != attention_fixture_hash
        || reference_hash(&output_value, "source_lock_sha256")? != source_lock_hash
        || reference_hash(&output_value, "mtp_input_fusion_fixture_sha256")? != fusion_fixture_hash
        || reference_hash(&output_value, "decoder_fixture_sha256")? != decoder_fixture_hash
    {
        return Err("MTP decoder component authority mismatch".to_owned());
    }

    let (fusion_report, fused_hidden) = verify_mtp_input_fusion_fixture_with_output(
        checkpoint_dir,
        model_lock_path,
        source_lock_path,
        fusion_fixture_path,
    )?;
    if fused_hidden.len() != 2 {
        return Err("MTP decoder requires two independently fused input cases".to_owned());
    }
    let hidden_overrides = [fused_hidden[0].clone(), fused_hidden[1].clone()];
    let (attention_report, post_attention) =
        verify_full_attention_residual_fixture_bytes_with_prefix(
            checkpoint_dir,
            model_lock_path,
            &attention_bytes,
            "qwen3_8_flash_next_mtp_full_attention_residual",
            "qwen3_8_flash_next_mtp_full_attention_residual_verification",
            0,
            [0, 1],
            ["mtp_initial", "mtp_cached_decode"],
            true,
            None,
            Some(&hidden_overrides),
            "mtp.layers.0",
        )?;
    let (decoder_report, decoder_outputs) = verify_decoder_mlp_fixture_bytes_with_prefix(
        checkpoint_dir,
        model_lock_path,
        &decoder_bytes,
        0,
        "full_attention",
        "qwen3_8_flash_next_mtp_complete_decoder",
        "qwen3_8_flash_next_mtp_complete_decoder_verification",
        ["mtp_initial", "mtp_cached_decode"],
        attention_report.total_verified_payload_bytes,
        post_attention,
        "mtp.layers.0",
    )?;
    let output_report = verify_embedded_text_output_fixture_with_names(
        checkpoint_dir,
        model_lock_path,
        &output_value,
        MODEL,
        REVISION,
        &decoder_outputs,
        "mtp.hyper_connection_mixer",
        "lm_head.weight",
    )?;

    Ok(MtpProposalVerificationReport {
        schema_version: 1,
        semantic: "qwen3_8_flash_next_mtp_proposal_path_verification",
        model: fusion_report.model,
        revision: fusion_report.revision,
        source_commit: fusion_report.source_commit,
        steps_verified: decoder_report.steps_verified,
        exact_input_fusion_capture_hashes: fusion_report.exact_capture_hashes + 7,
        exact_bf16_capture_hashes: fusion_report.exact_capture_hashes
            + 7
            + attention_report.exact_bf16_capture_hashes
            + decoder_report.exact_bf16_capture_hashes,
        exact_f32_capture_hashes: attention_report.exact_f32_capture_hashes,
        exact_i64_capture_hashes: attention_report.exact_i64_capture_hashes,
        dense_tensors_verified: fusion_report.tensors_verified
            + attention_report.dense_tensors_verified
            + decoder_report.dense_tensors_verified,
        unique_experts_verified: decoder_report.unique_experts_verified,
        total_verified_payload_bytes: fusion_report.tensor_payload_bytes
            + decoder_report.total_verified_payload_bytes
            + output_report.output_verified_payload_bytes,
        selected_experts_by_step: decoder_report.selected_experts_by_step,
        vocab_size: 248_320,
        exact_full_logit_hashes: 2,
        exact_mixer_capture_hashes: 18,
        exact_ranked_logit_entries: 40,
        top20_token_ids_by_step: output_report.top20_token_ids_by_step,
        top20_cutoff_tie_counts_by_step: output_report.top20_cutoff_tie_counts_by_step,
        accepted_tokens: 0,
        accepted_per_transaction: 0,
        expert_union: 0,
        performance_claim: None,
    })
}

fn read_embedding_row_from_record(
    checkpoint_dir: &Path,
    model_lock: &ModelLock,
    embedding: &CausalEmbedding,
    token_id: usize,
) -> Result<Vec<u16>, String> {
    if embedding.tensor != "model.language_model.embed_tokens.weight"
        || embedding.shape != [VOCAB, HIDDEN]
        || token_id >= VOCAB
        || embedding.shard.contains('/')
        || embedding.shard.contains("..")
    {
        return Err("invalid causal MTP embedding record".to_owned());
    }
    let locked = model_lock
        .files
        .iter()
        .find(|file| file.path == embedding.shard)
        .ok_or("causal MTP embedding shard absent from model lock")?;
    if locked.size != embedding.shard_bytes
        || locked.lfs_sha256.as_deref() != Some(&embedding.shard_sha256)
    {
        return Err("causal MTP embedding shard identity mismatch".to_owned());
    }
    let path = checkpoint_dir.join(&embedding.shard);
    let mut file =
        File::open(&path).map_err(|error| format!("cannot open {}: {error}", path.display()))?;
    let mut prefix = [0_u8; 8];
    file.read_exact(&mut prefix)
        .map_err(|error| error.to_string())?;
    let header_bytes = u64::from_le_bytes(prefix);
    if header_bytes == 0 || header_bytes > 16 * 1024 * 1024 {
        return Err("invalid causal MTP embedding header".to_owned());
    }
    let mut header = vec![0_u8; header_bytes as usize];
    file.read_exact(&mut header)
        .map_err(|error| error.to_string())?;
    let header: Value = serde_json::from_slice(&header).map_err(|error| error.to_string())?;
    let item = header
        .get(&embedding.tensor)
        .ok_or("causal MTP embedding tensor missing")?;
    let shape = item
        .get("shape")
        .and_then(Value::as_array)
        .ok_or("causal MTP embedding shape missing")?;
    let offsets = item
        .get("data_offsets")
        .and_then(Value::as_array)
        .ok_or("causal MTP embedding offsets missing")?;
    if item.get("dtype").and_then(Value::as_str) != Some("BF16")
        || shape.len() != 2
        || shape[0].as_u64() != Some(VOCAB as u64)
        || shape[1].as_u64() != Some(HIDDEN as u64)
        || offsets.len() != 2
    {
        return Err("causal MTP embedding metadata mismatch".to_owned());
    }
    let tensor_start = offsets[0].as_u64().ok_or("invalid embedding start")?;
    let tensor_end = offsets[1].as_u64().ok_or("invalid embedding end")?;
    if tensor_end.checked_sub(tensor_start) != Some((VOCAB * HIDDEN * 2) as u64) {
        return Err("causal MTP embedding byte count mismatch".to_owned());
    }
    let absolute = 8_u64
        .checked_add(header_bytes)
        .and_then(|value| value.checked_add(tensor_start))
        .and_then(|value| value.checked_add((token_id * HIDDEN * 2) as u64))
        .ok_or("causal MTP embedding row offset overflow")?;
    file.seek(SeekFrom::Start(absolute))
        .map_err(|error| error.to_string())?;
    let mut payload = vec![0_u8; HIDDEN * 2];
    file.read_exact(&mut payload)
        .map_err(|error| error.to_string())?;
    Ok(payload
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect())
}

#[allow(clippy::too_many_arguments)]
pub fn verify_mtp_causal_prefill_fixture(
    checkpoint_dir: &Path,
    model_lock_path: &Path,
    mtp_source_lock_path: &Path,
    scheduler_lock_path: &Path,
    tokenizer_fixture_path: &Path,
    ngram_fixture_path: &Path,
    ngram_row_fixture_path: &Path,
    ple_fixture_path: &Path,
    endpoint_fixture_path: &Path,
    fusion_fixture_path: &Path,
    seed_fixture_path: &Path,
    attention_fixture_path: &Path,
    decoder_fixture_path: &Path,
    output_fixture_path: &Path,
) -> Result<MtpCausalPrefillVerificationReport, String> {
    let seed: CausalSeedFixture = serde_json::from_slice(
        &fs::read(seed_fixture_path)
            .map_err(|error| format!("cannot read causal MTP seed fixture: {error}"))?,
    )
    .map_err(|error| format!("malformed causal MTP seed fixture: {error}"))?;
    let model_lock: ModelLock =
        serde_json::from_slice(&fs::read(model_lock_path).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    let scheduler_lock: SourceLock =
        serde_json::from_slice(&fs::read(scheduler_lock_path).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    let worker = scheduler_lock
        .files
        .iter()
        .find(|file| file.path == "python/sglang/srt/speculative/eagle_worker_v2.py")
        .ok_or("EAGLE worker absent from scheduler source lock")?;
    let utils = scheduler_lock
        .files
        .iter()
        .find(|file| file.path == "python/sglang/srt/speculative/eagle_utils.py")
        .ok_or("EAGLE utilities absent from scheduler source lock")?;
    if scheduler_lock.schema_version != 1
        || scheduler_lock.repository != "https://github.com/sgl-project/sglang"
        || scheduler_lock.pull_request != "https://github.com/sgl-project/sglang/pull/36497"
        || scheduler_lock.commit != SGLANG_COMMIT
        || worker.sha256 != EAGLE_WORKER_SHA256
        || worker.git_blob != "93fdd61761c4f976305d7af4e4aecd65430e0539"
        || utils.sha256 != EAGLE_UTILS_SHA256
        || utils.git_blob != "18c9f0cbdd849667f9b743f704e79ff08d2e5827"
    {
        return Err("unsupported EAGLE scheduler source authority".to_owned());
    }
    let config = &seed.configuration;
    if seed.schema_version != 1
        || seed.semantic != "qwen3_8_flash_next_target_derived_mtp_prefill_fusion"
        || seed.model != MODEL
        || seed.revision != REVISION
        || model_lock.model != MODEL
        || model_lock.revision != REVISION
        || seed.reference.implementation != "sglang_eagle_prefill_rotation_and_qwen4_exp_mtp_fusion"
        || seed.reference.commit != SGLANG_COMMIT
        || seed.reference.mtp_source_lock_sha256 != sha256_file(mtp_source_lock_path)?
        || seed.reference.scheduler_source_lock_sha256 != sha256_file(scheduler_lock_path)?
        || seed.reference.endpoint_fixture_sha256 != sha256_file(endpoint_fixture_path)?
        || seed.reference.fusion_fixture_sha256 != sha256_file(fusion_fixture_path)?
        || seed.reference.model_lock_sha256 != sha256_file(model_lock_path)?
        || seed.reference.tensor_index_sha256
            != sha256_file(&checkpoint_dir.join("model.safetensors.index.json"))?
        || config.target_input_token_ids != [16_207, 22_856]
        || config.mtp_prefill_token_ids.len() != 2
        || config.mtp_prefill_token_ids[0] != 22_856
        || config.mtp_prefill_token_ids[1] != config.target_next_token_id
        || config.mtp_positions != [0, 1]
        || config.cache_mode != "sequential_mtp_prefill"
        || config.boundary_dtype != "BF16"
        || seed.steps.len() != 2
    {
        return Err("causal MTP seed identity or configuration mismatch".to_owned());
    }

    let (target_report, target_hiddens) = verify_token_text_endpoint_fixture_with_outputs(
        checkpoint_dir,
        model_lock_path,
        tokenizer_fixture_path,
        ngram_fixture_path,
        ngram_row_fixture_path,
        ple_fixture_path,
        endpoint_fixture_path,
    )?;
    if target_hiddens.len() != 2
        || target_report.top20_token_ids_by_step.len() != 2
        || target_report.top20_token_ids_by_step[1].first().copied()
            != Some(config.target_next_token_id)
    {
        return Err("target endpoint does not produce the causal MTP bonus token".to_owned());
    }

    let expected_tensors = [
        (
            "pre_fc_norm_embedding",
            "mtp.pre_fc_norm_embedding.weight",
            vec![HIDDEN],
        ),
        (
            "pre_fc_norm_hidden",
            "mtp.pre_fc_norm_hidden.weight",
            vec![HC_HIDDEN],
        ),
        (
            "fc_embedding",
            "mtp.fc_embedding.weight",
            vec![HIDDEN, HIDDEN],
        ),
        ("fc_hidden", "mtp.fc_hidden.weight", vec![HIDDEN, HIDDEN]),
    ];
    let mut tensor_values = BTreeMap::new();
    let mut fusion_payload_bytes = 0_usize;
    for (key, name, shape) in expected_tensors {
        let record = seed
            .tensors
            .get(key)
            .ok_or_else(|| format!("causal MTP seed lacks tensor {key}"))?;
        if record.tensor != name || record.shape != shape || record.shard.contains('/') {
            return Err(format!("invalid causal MTP tensor record {key}"));
        }
        let locked = model_lock
            .files
            .iter()
            .find(|file| file.path == record.shard)
            .ok_or_else(|| format!("causal MTP shard absent for {key}"))?;
        if locked.size != record.shard_bytes
            || locked.lfs_sha256.as_deref() != Some(&record.shard_sha256)
        {
            return Err(format!("causal MTP shard identity mismatch for {key}"));
        }
        let value = read_tensor(&checkpoint_dir.join(&record.shard), name, &shape)?;
        if !bf16_payload_matches(&value, &record.payload_sha256) {
            return Err(format!("causal MTP tensor payload mismatch for {key}"));
        }
        fusion_payload_bytes += value.len() * 2;
        tensor_values.insert(key, value);
    }

    let mut fused_hiddens = Vec::with_capacity(2);
    for (ordinal, step) in seed.steps.iter().enumerate() {
        if step.ordinal != ordinal
            || step.target_hidden_endpoint_ordinal != ordinal
            || step.mtp_input_token_id != config.mtp_prefill_token_ids[ordinal]
            || step.captures.len() != 7
        {
            return Err("causal MTP fusion step layout mismatch".to_owned());
        }
        let embedding = read_embedding_row_from_record(
            checkpoint_dir,
            &model_lock,
            &seed.embedding,
            step.mtp_input_token_id,
        )?;
        let target_hidden = &target_hiddens[ordinal];
        require_capture(&step.captures, "embedding", &embedding)?;
        require_capture(&step.captures, "target_hidden", target_hidden)?;
        let embedding_normed =
            rms_norm(&embedding, &tensor_values["pre_fc_norm_embedding"], 1.0e-6)?;
        let target_hidden_normed =
            rms_norm(target_hidden, &tensor_values["pre_fc_norm_hidden"], 1.0e-6)?;
        require_capture(&step.captures, "embedding_normed", &embedding_normed)?;
        require_capture(
            &step.captures,
            "target_hidden_normed",
            &target_hidden_normed,
        )?;
        let embedding_projected = linear_bf16(
            &tensor_values["fc_embedding"],
            &embedding_normed,
            HIDDEN,
            HIDDEN,
        );
        let mut target_hidden_projected = Vec::with_capacity(HC_HIDDEN);
        for stream in target_hidden_normed.chunks_exact(HIDDEN) {
            target_hidden_projected.extend(linear_bf16(
                &tensor_values["fc_hidden"],
                stream,
                HIDDEN,
                HIDDEN,
            ));
        }
        require_capture(&step.captures, "embedding_projected", &embedding_projected)?;
        require_capture(
            &step.captures,
            "target_hidden_projected",
            &target_hidden_projected,
        )?;
        let fused = target_hidden_projected
            .chunks_exact(HIDDEN)
            .flat_map(|stream| {
                stream
                    .iter()
                    .zip(&embedding_projected)
                    .map(|(hidden, embedding)| to_bf16(from_bf16(*hidden) + from_bf16(*embedding)))
            })
            .collect::<Vec<_>>();
        require_capture(&step.captures, "fused_hidden", &fused)?;
        fused_hiddens.push(fused);
    }

    let seed_hash = sha256_file(seed_fixture_path)?;
    let scheduler_hash = sha256_file(scheduler_lock_path)?;
    let endpoint_hash = sha256_file(endpoint_fixture_path)?;
    let attention_bytes = fs::read(attention_fixture_path).map_err(|error| error.to_string())?;
    let decoder_bytes = fs::read(decoder_fixture_path).map_err(|error| error.to_string())?;
    let attention_value: Value =
        serde_json::from_slice(&attention_bytes).map_err(|error| error.to_string())?;
    let decoder_value: Value =
        serde_json::from_slice(&decoder_bytes).map_err(|error| error.to_string())?;
    let output_value: Value =
        serde_json::from_slice(&fs::read(output_fixture_path).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    for component in [&attention_value, &decoder_value, &output_value] {
        if reference_hash(component, "scheduler_source_lock_sha256")? != scheduler_hash
            || reference_hash(component, "endpoint_fixture_sha256")? != endpoint_hash
            || reference_hash(component, "causal_seed_fixture_sha256")? != seed_hash
        {
            return Err("causal MTP component authority mismatch".to_owned());
        }
    }
    if attention_value.get("semantic").and_then(Value::as_str)
        != Some("qwen3_8_flash_next_target_derived_mtp_prefill_attention")
        || decoder_value.get("semantic").and_then(Value::as_str)
            != Some("qwen3_8_flash_next_target_derived_mtp_prefill_decoder")
        || output_value.get("semantic").and_then(Value::as_str)
            != Some("qwen3_8_flash_next_target_derived_mtp_prefill_logits")
        || reference_hash(&decoder_value, "attention_residual_fixture_sha256")?
            != sha256_file(attention_fixture_path)?
        || reference_hash(&output_value, "decoder_fixture_sha256")?
            != sha256_file(decoder_fixture_path)?
    {
        return Err("causal MTP component chain mismatch".to_owned());
    }

    let hidden_overrides = [fused_hiddens[0].clone(), fused_hiddens[1].clone()];
    let (attention_report, post_attention) =
        verify_full_attention_residual_fixture_bytes_with_prefix(
            checkpoint_dir,
            model_lock_path,
            &attention_bytes,
            "qwen3_8_flash_next_target_derived_mtp_prefill_attention",
            "qwen3_8_flash_next_target_derived_mtp_prefill_attention_verification",
            0,
            [0, 1],
            ["mtp_prefill_initial", "mtp_prefill_cached"],
            true,
            None,
            Some(&hidden_overrides),
            "mtp.layers.0",
        )?;
    let (decoder_report, decoder_outputs) = verify_decoder_mlp_fixture_bytes_with_prefix(
        checkpoint_dir,
        model_lock_path,
        &decoder_bytes,
        0,
        "full_attention",
        "qwen3_8_flash_next_target_derived_mtp_prefill_decoder",
        "qwen3_8_flash_next_target_derived_mtp_prefill_decoder_verification",
        ["mtp_prefill_initial", "mtp_prefill_cached"],
        attention_report.total_verified_payload_bytes,
        post_attention,
        "mtp.layers.0",
    )?;
    let output_report = verify_embedded_text_output_fixture_with_names(
        checkpoint_dir,
        model_lock_path,
        &output_value,
        MODEL,
        REVISION,
        &decoder_outputs,
        "mtp.hyper_connection_mixer",
        "lm_head.weight",
    )?;
    let proposal_token_id = output_report
        .top20_token_ids_by_step
        .last()
        .and_then(|tokens| tokens.first())
        .copied()
        .ok_or("causal MTP proposal logits are empty")?;
    let total_verified_payload_bytes = target_report
        .total_verified_payload_bytes
        .checked_add(fusion_payload_bytes)
        .and_then(|value| value.checked_add(2 * HIDDEN * 2))
        .and_then(|value| value.checked_add(decoder_report.total_verified_payload_bytes))
        .and_then(|value| value.checked_add(output_report.output_verified_payload_bytes))
        .ok_or("causal MTP payload byte count overflow")?;

    Ok(MtpCausalPrefillVerificationReport {
        schema_version: 1,
        semantic: "qwen3_8_flash_next_target_derived_mtp_prefill_verification",
        model: seed.model,
        revision: seed.revision,
        source_commit: seed.reference.commit,
        target_input_token_ids: config.target_input_token_ids.clone(),
        target_next_token_id: config.target_next_token_id,
        mtp_prefill_token_ids: config.mtp_prefill_token_ids.clone(),
        proposal_token_id,
        target_endpoint: target_report,
        fusion_steps_verified: 2,
        exact_fusion_capture_hashes: 14,
        exact_bf16_capture_hashes: 14
            + attention_report.exact_bf16_capture_hashes
            + decoder_report.exact_bf16_capture_hashes,
        exact_f32_capture_hashes: attention_report.exact_f32_capture_hashes,
        exact_i64_capture_hashes: attention_report.exact_i64_capture_hashes,
        dense_tensors_verified: seed.tensors.len()
            + attention_report.dense_tensors_verified
            + decoder_report.dense_tensors_verified,
        unique_experts_verified: decoder_report.unique_experts_verified,
        total_verified_payload_bytes,
        selected_experts_by_step: decoder_report.selected_experts_by_step,
        top20_token_ids_by_step: output_report.top20_token_ids_by_step,
        accepted_tokens: 0,
        accepted_per_transaction: 0,
        expert_union: 0,
        performance_claim: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_hidden_norm_is_one_wide_norm_not_four_grouped_norms() {
        let input = [to_bf16(1.0), to_bf16(1.0), to_bf16(3.0), to_bf16(3.0)];
        let weight = [0_u16; 4];
        let output = rms_norm(&input, &weight, 0.0).unwrap();
        assert_eq!(output[0], output[1]);
        assert_eq!(output[2], output[3]);
        assert_ne!(output[0], output[2]);
    }

    #[test]
    fn embedding_projection_is_shared_across_all_hyper_streams() {
        let embedding = [to_bf16(0.5), to_bf16(-0.25)];
        let hidden = [to_bf16(1.0), to_bf16(2.0), to_bf16(3.0), to_bf16(4.0)];
        let fused: Vec<_> = hidden
            .chunks_exact(2)
            .flat_map(|stream| {
                stream
                    .iter()
                    .zip(embedding)
                    .map(|(left, right)| to_bf16(from_bf16(*left) + from_bf16(right)))
            })
            .collect();
        assert_eq!(
            fused,
            vec![to_bf16(1.5), to_bf16(1.75), to_bf16(3.5), to_bf16(3.75)]
        );
    }

    #[test]
    fn committed_mtp_sequence_exercises_distinct_routes_and_logits() {
        let decoder: Value = serde_json::from_str(include_str!(
            "../fixtures/mtp/qwen3_8_flash_next_decoder.json"
        ))
        .unwrap();
        let output: Value = serde_json::from_str(include_str!(
            "../fixtures/mtp/qwen3_8_flash_next_logits.json"
        ))
        .unwrap();
        let routes = decoder["steps"]
            .as_array()
            .unwrap()
            .iter()
            .map(|step| step["selected_experts"].clone())
            .collect::<Vec<_>>();
        let logits = output["steps"]
            .as_array()
            .unwrap()
            .iter()
            .map(|step| step["captures"]["logits"].clone())
            .collect::<Vec<_>>();
        assert_eq!(routes.len(), 2);
        assert_ne!(routes[0], routes[1]);
        assert_ne!(logits[0], logits[1]);
    }

    #[test]
    fn causal_prefill_fixture_uses_eagle_shifted_tokens() {
        let seed: Value = serde_json::from_str(include_str!(
            "../fixtures/mtp/qwen3_8_flash_next_causal_prefill_seed.json"
        ))
        .unwrap();
        let logits: Value = serde_json::from_str(include_str!(
            "../fixtures/mtp/qwen3_8_flash_next_causal_prefill_logits.json"
        ))
        .unwrap();
        assert_eq!(
            seed["configuration"]["target_input_token_ids"],
            serde_json::json!([16207, 22856])
        );
        assert_eq!(seed["configuration"]["target_next_token_id"], 369);
        assert_eq!(
            seed["configuration"]["mtp_prefill_token_ids"],
            serde_json::json!([22856, 369])
        );
        assert_eq!(logits["steps"][1]["top20_token_ids"][0], 264);
    }
}
