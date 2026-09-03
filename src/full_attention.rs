use crate::deltanet::read_tensor;
use crate::expert::{bf16_hash, from_bf16, linear_bf16, to_bf16};
use crate::hyper_connection::pytorch_inner_square_sum;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

const MODEL: &str = "Qwen/Qwen3.8-Flash-Next";
const HIDDEN: usize = 2560;
const HEADS: usize = 24;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 256;
const INDEX_HEADS: usize = 4;
const INDEX_DIM: usize = 128;
const LONG_PAST: usize = 2080;

#[derive(Deserialize)]
struct Fixture {
    schema_version: u32,
    semantic: String,
    model: String,
    revision: String,
    reference: Reference,
    configuration: Configuration,
    tensors: BTreeMap<String, TensorRecord>,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Reference {
    implementation: String,
    transformers_version: String,
    source: Vec<String>,
    config_sha256: String,
    tensor_index_sha256: String,
    model_lock_sha256: String,
}

#[derive(Deserialize)]
struct Configuration {
    layer: usize,
    hidden_size: usize,
    attention_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    rope_theta: u64,
    mrope_section: Vec<usize>,
    indexer_heads: usize,
    indexer_kv_heads: usize,
    indexer_head_dim: usize,
    indexer_budget: usize,
    indexer_compress_ratio: usize,
    boundary_dtype: String,
}

#[derive(Deserialize)]
struct TensorRecord {
    tensor: String,
    dtype: String,
    shape: Vec<usize>,
    shard: String,
    shard_bytes: u64,
    shard_sha256: String,
    payload_sha256: String,
}

#[derive(Deserialize)]
struct Case {
    ordinal: usize,
    mode: String,
    position: usize,
    past_length: usize,
    input_spec: ArithmeticSpec,
    state_specs: BTreeMap<String, ArithmeticSpec>,
    captures: BTreeMap<String, Capture>,
}

#[derive(Deserialize, Eq, PartialEq)]
struct ArithmeticSpec {
    multiplier: i64,
    add: i64,
    modulus: i64,
    center: i64,
    divisor: i64,
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
pub struct FullAttentionProjectionReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub model: String,
    pub revision: String,
    pub layer: usize,
    pub cases_verified: usize,
    pub exact_bf16_capture_hashes: usize,
    pub dense_tensors_verified: usize,
    pub dense_tensor_payload_bytes: usize,
    pub synthetic_cache_bytes: usize,
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

fn make_bf16(shape: &[usize], spec: &ArithmeticSpec) -> Result<Vec<u16>, String> {
    if spec.modulus <= 0 || spec.divisor <= 0 {
        return Err("invalid full-attention arithmetic specification".to_owned());
    }
    let count = shape
        .iter()
        .try_fold(1_usize, |total, value| total.checked_mul(*value))
        .ok_or("full-attention synthetic tensor size overflow")?;
    Ok((0..count)
        .map(|index| {
            let raw =
                (index as i64 * spec.multiplier + spec.add).rem_euclid(spec.modulus) - spec.center;
            to_bf16(raw as f32 / spec.divisor as f32)
        })
        .collect())
}

fn require_bf16(case: &Case, name: &str, shape: &[usize], values: &[u16]) -> Result<(), String> {
    let expected = case
        .captures
        .get(name)
        .ok_or_else(|| format!("missing full-attention capture {name}"))?;
    let actual = bf16_hash(values);
    if expected.dtype != "BF16"
        || expected.shape != shape
        || !is_hash(&expected.sha256)
        || expected.sha256 != actual
    {
        return Err(format!(
            "full-attention projection mismatch at case {} {name}: expected {}, got {actual}",
            case.ordinal, expected.sha256
        ));
    }
    Ok(())
}

fn expected_tensors() -> Vec<(&'static str, String, Vec<usize>)> {
    let prefix = "model.language_model.layers.3.self_attn";
    vec![
        (
            "q_proj.weight",
            "q_proj.weight",
            vec![HEADS * HEAD_DIM * 2, HIDDEN],
        ),
        (
            "k_proj.weight",
            "k_proj.weight",
            vec![KV_HEADS * HEAD_DIM, HIDDEN],
        ),
        (
            "v_proj.weight",
            "v_proj.weight",
            vec![KV_HEADS * HEAD_DIM, HIDDEN],
        ),
        (
            "o_proj.weight",
            "o_proj.weight",
            vec![HIDDEN, HEADS * HEAD_DIM],
        ),
        ("q_norm.weight", "q_norm.weight", vec![HEAD_DIM]),
        ("k_norm.weight", "k_norm.weight", vec![HEAD_DIM]),
        (
            "indexer.index_qk_proj.weight",
            "indexer.index_qk_proj.weight",
            vec![(INDEX_HEADS + 1) * INDEX_DIM, HIDDEN],
        ),
        (
            "indexer.q_layernorm.weight",
            "indexer.q_layernorm.weight",
            vec![INDEX_DIM],
        ),
        (
            "indexer.k_layernorm.weight",
            "indexer.k_layernorm.weight",
            vec![INDEX_DIM],
        ),
    ]
    .into_iter()
    .map(|(key, suffix, shape)| (key, format!("{prefix}.{suffix}"), shape))
    .collect()
}

pub fn verify_full_attention_projections(
    checkpoint_dir: &Path,
    model_lock_path: &Path,
    fixture_path: &Path,
) -> Result<FullAttentionProjectionReport, String> {
    let fixture: Fixture = serde_json::from_slice(
        &fs::read(fixture_path).map_err(|error| format!("cannot read fixture: {error}"))?,
    )
    .map_err(|error| format!("malformed full-attention fixture: {error}"))?;
    let config = &fixture.configuration;
    if fixture.schema_version != 1
        || fixture.semantic != "qwen3_8_flash_next_layer3_full_attention_qsa"
        || fixture.model != MODEL
        || fixture.reference.implementation
            != "source_derived_and_official_huggingface_transformers_qwen4_exp"
        || fixture.reference.transformers_version != "5.16.1"
        || fixture.reference.source.len() != 2
        || config.layer != 3
        || config.hidden_size != HIDDEN
        || config.attention_heads != HEADS
        || config.kv_heads != KV_HEADS
        || config.head_dim != HEAD_DIM
        || config.rotary_dim != 64
        || config.rope_theta != 10_000_000
        || config.mrope_section != [11, 11, 10]
        || config.indexer_heads != INDEX_HEADS
        || config.indexer_kv_heads != 1
        || config.indexer_head_dim != INDEX_DIM
        || config.indexer_budget != 2048
        || config.indexer_compress_ratio != 4
        || config.boundary_dtype != "BF16"
        || fixture.tensors.len() != 9
        || fixture.cases.len() != 2
    {
        return Err("full-attention fixture identity or configuration is unsupported".to_owned());
    }
    if sha256_file(model_lock_path)? != fixture.reference.model_lock_sha256
        || sha256_file(&checkpoint_dir.join("config.json"))? != fixture.reference.config_sha256
        || sha256_file(&checkpoint_dir.join("model.safetensors.index.json"))?
            != fixture.reference.tensor_index_sha256
    {
        return Err("full-attention reference identity mismatch".to_owned());
    }
    let lock: ModelLock = serde_json::from_slice(
        &fs::read(model_lock_path).map_err(|error| format!("cannot read model lock: {error}"))?,
    )
    .map_err(|error| format!("malformed model lock: {error}"))?;
    if lock.model != fixture.model || lock.revision != fixture.revision {
        return Err("full-attention model lock mismatch".to_owned());
    }

    let mut tensors = BTreeMap::new();
    let mut dense_bytes = 0;
    for (key, name, shape) in expected_tensors() {
        let record = fixture
            .tensors
            .get(key)
            .ok_or_else(|| format!("missing full-attention tensor {key}"))?;
        let locked = lock
            .files
            .iter()
            .filter(|entry| entry.path == record.shard)
            .collect::<Vec<_>>();
        if record.tensor != name
            || record.dtype != "BF16"
            || record.shape != shape
            || !is_hash(&record.shard_sha256)
            || !is_hash(&record.payload_sha256)
            || locked.len() != 1
            || locked[0].size != record.shard_bytes
            || locked[0].lfs_sha256.as_deref() != Some(record.shard_sha256.as_str())
            || fs::metadata(checkpoint_dir.join(&record.shard))
                .map_err(|error| error.to_string())?
                .len()
                != record.shard_bytes
        {
            return Err(format!("full-attention tensor identity mismatch for {key}"));
        }
        let payload = read_tensor(&checkpoint_dir.join(&record.shard), &name, &shape)?;
        if bf16_hash(&payload) != record.payload_sha256 {
            return Err(format!("full-attention tensor payload mismatch for {key}"));
        }
        dense_bytes += payload.len() * 2;
        tensors.insert(key, payload);
    }

    let expected_inputs = [
        ArithmeticSpec {
            multiplier: 47,
            add: 19,
            modulus: 269,
            center: 134,
            divisor: 128,
        },
        ArithmeticSpec {
            multiplier: 71,
            add: 31,
            modulus: 281,
            center: 140,
            divisor: 128,
        },
    ];
    let expected_index_state = ArithmeticSpec {
        multiplier: 29,
        add: 11,
        modulus: 65521,
        center: 32760,
        divisor: 32768,
    };
    let mut synthetic_cache_bytes = 0;
    for (ordinal, case) in fixture.cases.iter().enumerate() {
        let past = if ordinal == 0 { 0 } else { LONG_PAST };
        if case.ordinal != ordinal
            || case.mode
                != if ordinal == 0 {
                    "initial"
                } else {
                    "active_qsa_pruning"
                }
            || case.position != past
            || case.past_length != past
            || case.input_spec != expected_inputs[ordinal]
            || case.captures.len() != 31
            || (ordinal == 0 && !case.state_specs.is_empty())
        {
            return Err(format!("full-attention case {ordinal} metadata mismatch"));
        }
        let hidden = make_bf16(&[HIDDEN], &case.input_spec)?;
        require_bf16(case, "hidden_states", &[1, 1, HIDDEN], &hidden)?;
        let index_qk = linear_bf16(
            &tensors["indexer.index_qk_proj.weight"],
            &hidden,
            (INDEX_HEADS + 1) * INDEX_DIM,
            HIDDEN,
        );
        require_bf16(
            case,
            "index_qk_projection",
            &[1, 1, (INDEX_HEADS + 1) * INDEX_DIM],
            &index_qk,
        )?;
        let mut raw_cache = if past == 0 {
            Vec::new()
        } else {
            let spec = case
                .state_specs
                .get("indexer_keys")
                .ok_or("missing indexer state spec")?;
            if spec != &expected_index_state {
                return Err("unsupported indexer state spec".to_owned());
            }
            make_bf16(&[past, INDEX_DIM], spec)?
        };
        raw_cache.extend_from_slice(&index_qk[INDEX_HEADS * INDEX_DIM..]);
        synthetic_cache_bytes += raw_cache.len() * 2;
        require_bf16(
            case,
            "raw_indexer_cache",
            &[1, past + 1, INDEX_DIM],
            &raw_cache,
        )?;

        let q_projection = linear_bf16(
            &tensors["q_proj.weight"],
            &hidden,
            HEADS * HEAD_DIM * 2,
            HIDDEN,
        );
        require_bf16(
            case,
            "q_projection",
            &[1, 1, HEADS * HEAD_DIM * 2],
            &q_projection,
        )?;
        let gate = q_projection
            .chunks_exact(HEAD_DIM * 2)
            .flat_map(|head| head[HEAD_DIM..].iter().copied())
            .collect::<Vec<_>>();
        require_bf16(case, "gate", &[1, 1, HEADS * HEAD_DIM], &gate)?;
        let key = linear_bf16(
            &tensors["k_proj.weight"],
            &hidden,
            KV_HEADS * HEAD_DIM,
            HIDDEN,
        );
        require_bf16(case, "key_projection", &[1, 1, KV_HEADS * HEAD_DIM], &key)?;
        let value = linear_bf16(
            &tensors["v_proj.weight"],
            &hidden,
            KV_HEADS * HEAD_DIM,
            HIDDEN,
        );
        require_bf16(
            case,
            "value_projection",
            &[1, KV_HEADS, 1, HEAD_DIM],
            &value,
        )?;
    }
    Ok(FullAttentionProjectionReport {
        schema_version: 1,
        semantic: "qwen3_8_flash_next_layer3_full_attention_projection_verification",
        model: fixture.model,
        revision: fixture.revision,
        layer: 3,
        cases_verified: 2,
        exact_bf16_capture_hashes: 12,
        dense_tensors_verified: 9,
        dense_tensor_payload_bytes: dense_bytes,
        synthetic_cache_bytes,
        accepted_tokens: 0,
        performance_claim: None,
    })
}

pub fn rms_norm_heads(
    input: &[u16],
    weight: &[u16],
    head_dim: usize,
    epsilon: f32,
) -> Result<Vec<u16>, String> {
    if head_dim == 0 || !input.len().is_multiple_of(head_dim) || weight.len() != head_dim {
        return Err("full-attention RMSNorm shape mismatch".to_owned());
    }
    let mut output = Vec::with_capacity(input.len());
    for head in input.chunks_exact(head_dim) {
        let float = head
            .iter()
            .map(|value| from_bf16(*value))
            .collect::<Vec<_>>();
        let inverse = (pytorch_inner_square_sum(&float) / head_dim as f32 + epsilon)
            .sqrt()
            .recip();
        output.extend(head.iter().zip(weight).map(|(value, weight)| {
            to_bf16(from_bf16(*value) * inverse * (1.0 + from_bf16(*weight)))
        }));
    }
    Ok(output)
}

pub fn apply_partial_rope(
    values: &mut [u16],
    heads: usize,
    head_dim: usize,
    cos: &[u16],
    sin: &[u16],
) -> Result<(), String> {
    let rotary_dim = cos.len();
    if heads == 0
        || head_dim == 0
        || values.len() != heads * head_dim
        || rotary_dim == 0
        || !rotary_dim.is_multiple_of(2)
        || rotary_dim > head_dim
        || sin.len() != rotary_dim
    {
        return Err("full-attention RoPE shape mismatch".to_owned());
    }
    let half = rotary_dim / 2;
    for head in values.chunks_exact_mut(head_dim) {
        for pair in 0..half {
            let first = from_bf16(head[pair]);
            let second = from_bf16(head[pair + half]);
            let cosine = from_bf16(cos[pair]);
            let sine = from_bf16(sin[pair]);
            let first_cosine = to_bf16(first * cosine);
            let second_sine = to_bf16(second * sine);
            let second_cosine = to_bf16(second * cosine);
            let first_sine = to_bf16(first * sine);
            head[pair] = to_bf16(from_bf16(first_cosine) - from_bf16(second_sine));
            head[pair + half] = to_bf16(from_bf16(second_cosine) + from_bf16(first_sine));
        }
    }
    Ok(())
}

pub fn select_qsa_blocks(scores: &[f32], block_topk: usize) -> Result<Vec<usize>, String> {
    if block_topk > scores.len() || scores.iter().any(|score| !score.is_finite()) {
        return Err("invalid QSA score vector".to_owned());
    }
    if block_topk < scores.len() {
        let mut ordered = scores.to_vec();
        ordered.sort_by(|left, right| right.total_cmp(left));
        if ordered[block_topk - 1] == ordered[block_topk] {
            return Err("QSA top-k boundary is tied".to_owned());
        }
    }
    let mut indices = (0..scores.len()).collect::<Vec<_>>();
    indices.sort_by(|left, right| scores[*right].total_cmp(&scores[*left]));
    indices.truncate(block_topk);
    Ok(indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_rms_is_independent_and_uses_one_plus_weight() {
        let input = [to_bf16(2.0), to_bf16(2.0), to_bf16(4.0), to_bf16(4.0)];
        let weight = [to_bf16(0.0), to_bf16(1.0)];
        let output = rms_norm_heads(&input, &weight, 2, 0.0).unwrap();
        assert_eq!(
            output,
            [to_bf16(1.0), to_bf16(2.0), to_bf16(1.0), to_bf16(2.0)]
        );
    }

    #[test]
    fn partial_rope_preserves_tail_and_stages_bf16_operations() {
        let mut values = [
            to_bf16(1.0),
            to_bf16(2.0),
            to_bf16(3.0),
            to_bf16(4.0),
            to_bf16(5.0),
            to_bf16(6.0),
        ];
        let tail = values[4..].to_vec();
        apply_partial_rope(&mut values, 1, 6, &[to_bf16(0.5); 4], &[to_bf16(0.25); 4]).unwrap();
        assert_eq!(&values[4..], tail);
        assert_eq!(values[0], to_bf16(-0.25));
        assert_eq!(values[2], to_bf16(1.75));
    }

    #[test]
    fn qsa_selection_is_score_ordered_and_rejects_boundary_ties() {
        assert_eq!(select_qsa_blocks(&[0.25, 2.0, 1.0], 2).unwrap(), [1, 2]);
        assert!(select_qsa_blocks(&[2.0, 1.0, 1.0], 2).is_err());
    }

    #[test]
    fn committed_synthetic_authority_regenerates_exactly() {
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../fixtures/full_attention/qwen3_8_flash_next_layer3.json"
        ))
        .unwrap();
        for case in &fixture.cases {
            let hidden = make_bf16(&[HIDDEN], &case.input_spec).unwrap();
            assert_eq!(bf16_hash(&hidden), case.captures["hidden_states"].sha256);
        }
        let long = &fixture.cases[1];
        let raw = make_bf16(&[LONG_PAST, INDEX_DIM], &long.state_specs["indexer_keys"]).unwrap();
        assert_eq!(raw.len() * 2, LONG_PAST * INDEX_DIM * 2);
    }
}
