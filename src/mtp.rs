use crate::decoder_layer3::verify_decoder_mlp_fixture_bytes_with_prefix;
use crate::expert::{bf16_hash, bf16_payload_matches, from_bf16, linear_bf16, to_bf16};
use crate::full_attention_residual::verify_full_attention_residual_fixture_bytes_with_prefix;
use crate::hyper_connection::pytorch_inner_square_sum;
use crate::text_output::verify_embedded_text_output_fixture_with_names;
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
}
