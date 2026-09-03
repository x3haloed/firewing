use crate::decoder_layer::route;
use crate::deltanet::read_tensor;
use crate::expert::{
    add_bf16, bf16_hash, from_bf16, linear_bf16, read_expert_slice, sigmoid_bf16, swiglu_bf16,
    to_bf16,
};
use crate::hyper_connection::run_hyper_connection;
use crate::ple_attention_residual::verify_ple_attention_residual_fixture_with_outputs;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

const MODEL: &str = "Qwen/Qwen3.8-Flash-Next";
const LAYER: usize = 1;
const HIDDEN: usize = 2560;
const HC_COUNT: usize = 4;
const HC_HIDDEN: usize = HIDDEN * HC_COUNT;
const EXPERTS: usize = 512;
const TOP_K: usize = 10;
const INTERMEDIATE: usize = 640;

#[derive(Deserialize)]
struct Fixture {
    schema_version: u32,
    semantic: String,
    model: String,
    revision: String,
    reference: Reference,
    configuration: Configuration,
    tensors: BTreeMap<String, Tensor>,
    expert_banks: BTreeMap<String, Tensor>,
    steps: Vec<Step>,
}

#[derive(Deserialize)]
struct Reference {
    implementation: String,
    transformers_version: String,
    source: String,
    config_sha256: String,
    tensor_index_sha256: String,
    model_lock_sha256: String,
    ngram_fixture_sha256: String,
    ngram_row_fixture_sha256: String,
    ple_fixture_sha256: String,
    attention_residual_fixture_sha256: String,
}

#[derive(Deserialize)]
struct Configuration {
    layer: usize,
    layer_type: String,
    hidden_size: usize,
    hc_count: usize,
    num_experts: usize,
    top_k: usize,
    intermediate_size: usize,
    shared_intermediate_size: usize,
    boundary_dtype: String,
    router_softmax_dtype: String,
}

#[derive(Deserialize)]
struct Tensor {
    tensor: String,
    dtype: String,
    shape: Vec<usize>,
    shard: String,
    shard_bytes: u64,
    shard_sha256: String,
    payload_sha256: Option<String>,
}

#[derive(Deserialize)]
struct Step {
    ordinal: usize,
    mode: String,
    selected_experts: Vec<usize>,
    expert_execution_order: Vec<usize>,
    experts: Vec<SelectedExpert>,
    captures: BTreeMap<String, Capture>,
}

#[derive(Deserialize)]
struct SelectedExpert {
    expert: usize,
    route_weight_bf16: f32,
    gate_up_payload_sha256: String,
    down_payload_sha256: String,
    weighted_down_bf16_sha256: String,
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
pub struct DecoderLayer1VerificationReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub model: String,
    pub revision: String,
    pub layer: usize,
    pub steps_verified: usize,
    pub exact_bf16_capture_hashes: usize,
    pub exact_weighted_expert_hashes: usize,
    pub dense_tensors_verified: usize,
    pub unique_experts_verified: usize,
    pub attention_residual_tensor_payload_bytes: usize,
    pub dense_tensor_payload_bytes: usize,
    pub selected_expert_payload_bytes: usize,
    pub total_verified_payload_bytes: usize,
    pub selected_experts_by_step: Vec<Vec<usize>>,
    pub accepted_tokens: usize,
    pub performance_claim: Option<String>,
}

fn is_hash(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn sha256_file(path: &Path) -> Result<String, String> {
    fs::read(path)
        .map(|bytes| format!("{:x}", Sha256::digest(bytes)))
        .map_err(|error| format!("cannot read {}: {error}", path.display()))
}

fn require_capture(step: &Step, name: &str, shape: &[usize], values: &[u16]) -> Result<(), String> {
    let expected = step
        .captures
        .get(name)
        .ok_or_else(|| format!("missing layer-1 decoder capture {name}"))?;
    let actual = bf16_hash(values);
    if expected.dtype != "BF16"
        || expected.shape != shape
        || !is_hash(&expected.sha256)
        || expected.sha256 != actual
    {
        return Err(format!(
            "layer-1 decoder capture mismatch at step {} {name}: expected {}, got {actual}",
            step.ordinal, expected.sha256
        ));
    }
    Ok(())
}

fn expected_dense() -> Vec<(String, String, Vec<usize>, String)> {
    let hyper = [
        ("hc_norm", vec![HC_HIDDEN]),
        ("input_mix_weight_down", vec![320, HC_HIDDEN]),
        ("input_mix_weight_up", vec![HC_HIDDEN, 320]),
        ("block_inject_weight", vec![HC_COUNT, HC_HIDDEN]),
    ];
    let mut expected = hyper
        .into_iter()
        .map(|(local, shape)| {
            (
                format!("mlp_hyper_connection.{local}"),
                format!("model.language_model.layers.{LAYER}.mlp_hyper_connection.{local}.weight"),
                shape,
                local.to_owned(),
            )
        })
        .collect::<Vec<_>>();
    expected.extend([
        (
            "router".to_owned(),
            format!("model.language_model.layers.{LAYER}.mlp.gate.weight"),
            vec![EXPERTS, HIDDEN],
            "router".to_owned(),
        ),
        (
            "shared_gate_weight".to_owned(),
            format!("model.language_model.layers.{LAYER}.mlp.shared_expert.gate_proj.weight"),
            vec![INTERMEDIATE, HIDDEN],
            "shared_gate_weight".to_owned(),
        ),
        (
            "shared_up_weight".to_owned(),
            format!("model.language_model.layers.{LAYER}.mlp.shared_expert.up_proj.weight"),
            vec![INTERMEDIATE, HIDDEN],
            "shared_up_weight".to_owned(),
        ),
        (
            "shared_down_weight".to_owned(),
            format!("model.language_model.layers.{LAYER}.mlp.shared_expert.down_proj.weight"),
            vec![HIDDEN, INTERMEDIATE],
            "shared_down_weight".to_owned(),
        ),
        (
            "shared_expert_gate_weight".to_owned(),
            format!("model.language_model.layers.{LAYER}.mlp.shared_expert_gate.weight"),
            vec![1, HIDDEN],
            "shared_expert_gate_weight".to_owned(),
        ),
    ]);
    expected
}

fn require_locked_tensor(
    checkpoint_dir: &Path,
    lock: &ModelLock,
    tensor: &Tensor,
) -> Result<(), String> {
    let records = lock
        .files
        .iter()
        .filter(|entry| entry.path == tensor.shard)
        .collect::<Vec<_>>();
    if tensor.dtype != "BF16"
        || !is_hash(&tensor.shard_sha256)
        || records.len() != 1
        || records[0].size != tensor.shard_bytes
        || records[0].lfs_sha256.as_deref() != Some(tensor.shard_sha256.as_str())
        || fs::metadata(checkpoint_dir.join(&tensor.shard))
            .map_err(|error| error.to_string())?
            .len()
            != tensor.shard_bytes
    {
        return Err(format!(
            "layer-1 decoder tensor lock mismatch for {}",
            tensor.tensor
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn verify_decoder_layer1_fixture(
    checkpoint_dir: &Path,
    model_lock_path: &Path,
    ngram_fixture_path: &Path,
    ngram_row_fixture_path: &Path,
    ple_fixture_path: &Path,
    attention_residual_fixture_path: &Path,
    fixture_path: &Path,
) -> Result<DecoderLayer1VerificationReport, String> {
    let fixture: Fixture = serde_json::from_slice(
        &fs::read(fixture_path).map_err(|error| format!("cannot read fixture: {error}"))?,
    )
    .map_err(|error| format!("malformed layer-1 decoder fixture: {error}"))?;
    let config = &fixture.configuration;
    if fixture.schema_version != 1
        || fixture.semantic != "qwen3_8_flash_next_layer1_ple_complete_decoder"
        || fixture.model != MODEL
        || fixture.reference.implementation != "source_derived_huggingface_transformers_qwen4_exp"
        || fixture.reference.transformers_version != "5.16.1"
        || fixture.reference.source
            != "transformers.models.qwen4_exp.modeling_qwen4_exp.Qwen4ExpTextDecoderLayer.forward"
        || config.layer != LAYER
        || config.layer_type != "linear_attention"
        || config.hidden_size != HIDDEN
        || config.hc_count != HC_COUNT
        || config.num_experts != EXPERTS
        || config.top_k != TOP_K
        || config.intermediate_size != INTERMEDIATE
        || config.shared_intermediate_size != INTERMEDIATE
        || config.boundary_dtype != "BF16"
        || config.router_softmax_dtype != "F32"
        || fixture.tensors.len() != 9
        || fixture.expert_banks.len() != 2
        || fixture.steps.len() != 2
    {
        return Err("layer-1 decoder fixture identity or configuration is unsupported".to_owned());
    }
    if sha256_file(model_lock_path)? != fixture.reference.model_lock_sha256
        || sha256_file(&checkpoint_dir.join("config.json"))? != fixture.reference.config_sha256
        || sha256_file(&checkpoint_dir.join("model.safetensors.index.json"))?
            != fixture.reference.tensor_index_sha256
        || sha256_file(ngram_fixture_path)? != fixture.reference.ngram_fixture_sha256
        || sha256_file(ngram_row_fixture_path)? != fixture.reference.ngram_row_fixture_sha256
        || sha256_file(ple_fixture_path)? != fixture.reference.ple_fixture_sha256
        || sha256_file(attention_residual_fixture_path)?
            != fixture.reference.attention_residual_fixture_sha256
    {
        return Err("layer-1 decoder reference identity mismatch".to_owned());
    }
    let lock: ModelLock = serde_json::from_slice(
        &fs::read(model_lock_path).map_err(|error| format!("cannot read model lock: {error}"))?,
    )
    .map_err(|error| format!("malformed model lock: {error}"))?;
    if lock.model != fixture.model || lock.revision != fixture.revision {
        return Err("layer-1 decoder model lock mismatch".to_owned());
    }
    let (attention_report, post_attention) = verify_ple_attention_residual_fixture_with_outputs(
        checkpoint_dir,
        model_lock_path,
        ngram_fixture_path,
        ngram_row_fixture_path,
        ple_fixture_path,
        attention_residual_fixture_path,
    )?;

    let mut dense = BTreeMap::new();
    let mut hyper_weights = BTreeMap::new();
    let mut dense_bytes = 0;
    for (key, name, shape, local) in expected_dense() {
        let tensor = fixture
            .tensors
            .get(&key)
            .ok_or_else(|| format!("missing layer-1 decoder tensor {key}"))?;
        if tensor.tensor != name || tensor.shape != shape {
            return Err(format!(
                "layer-1 decoder tensor identity mismatch for {key}"
            ));
        }
        require_locked_tensor(checkpoint_dir, &lock, tensor)?;
        let payload = read_tensor(&checkpoint_dir.join(&tensor.shard), &name, &shape)?;
        if tensor
            .payload_sha256
            .as_deref()
            .is_none_or(|hash| !is_hash(hash) || bf16_hash(&payload) != hash)
        {
            return Err(format!("layer-1 decoder tensor payload mismatch for {key}"));
        }
        dense_bytes += payload.len() * 2;
        if key.starts_with("mlp_hyper_connection.") {
            hyper_weights.insert(local, payload);
        } else {
            dense.insert(local, payload);
        }
    }

    let gate_up_bank = fixture
        .expert_banks
        .get("gate_up")
        .ok_or("missing gate-up bank")?;
    let down_bank = fixture
        .expert_banks
        .get("down")
        .ok_or("missing down bank")?;
    if gate_up_bank.tensor != "model.language_model.layers.1.mlp.experts.gate_up_proj"
        || gate_up_bank.shape != [EXPERTS, INTERMEDIATE * 2, HIDDEN]
        || gate_up_bank.payload_sha256.is_some()
        || down_bank.tensor != "model.language_model.layers.1.mlp.experts.down_proj"
        || down_bank.shape != [EXPERTS, HIDDEN, INTERMEDIATE]
        || down_bank.payload_sha256.is_some()
    {
        return Err("layer-1 decoder expert bank identity mismatch".to_owned());
    }
    require_locked_tensor(checkpoint_dir, &lock, gate_up_bank)?;
    require_locked_tensor(checkpoint_dir, &lock, down_bank)?;

    let mut unique_experts = BTreeSet::new();
    let mut selected_experts_by_step = Vec::new();
    for (ordinal, (step, post)) in fixture.steps.iter().zip(post_attention).enumerate() {
        if step.ordinal != ordinal
            || step.mode
                != if ordinal == 0 {
                    "initial_chunk"
                } else {
                    "cached_recurrent"
                }
            || step.selected_experts.len() != TOP_K
            || step.experts.len() != TOP_K
            || step.captures.len() != 16
        {
            return Err(format!("layer-1 decoder step {ordinal} metadata mismatch"));
        }
        require_capture(step, "post_attention", &[1, 1, HC_HIDDEN], &post)?;
        let hyper = run_hyper_connection(&post, &hyper_weights)?;
        require_capture(step, "mlp_input", &[1, 1, HIDDEN], &hyper.mixed)?;
        require_capture(
            step,
            "mlp_injection_weights",
            &[1, 1, HC_COUNT],
            &hyper.injection_weights,
        )?;
        let logits = linear_bf16(&dense["router"], &hyper.mixed, EXPERTS, HIDDEN);
        require_capture(step, "router_logits", &[1, 1, EXPERTS], &logits)?;
        let (selection, scores) = route(&logits)?;
        if selection != step.selected_experts {
            return Err(format!("layer-1 decoder route mismatch at step {ordinal}"));
        }
        require_capture(step, "selected_scores", &[1, 1, TOP_K], &scores)?;
        let score_by_expert = selection
            .iter()
            .copied()
            .zip(scores.iter().copied())
            .collect::<BTreeMap<_, _>>();
        let mut execution = selection.clone();
        execution.sort_unstable();
        if execution != step.expert_execution_order
            || step
                .experts
                .iter()
                .map(|entry| entry.expert)
                .collect::<Vec<_>>()
                != execution
        {
            return Err(format!(
                "layer-1 decoder expert order mismatch at step {ordinal}"
            ));
        }

        let mut routed = vec![to_bf16(0.0); HIDDEN];
        for entry in &step.experts {
            if entry.expert >= EXPERTS
                || !is_hash(&entry.gate_up_payload_sha256)
                || !is_hash(&entry.down_payload_sha256)
                || !is_hash(&entry.weighted_down_bf16_sha256)
                || to_bf16(entry.route_weight_bf16) != score_by_expert[&entry.expert]
                || from_bf16(score_by_expert[&entry.expert]) != entry.route_weight_bf16
            {
                return Err(format!(
                    "layer-1 decoder expert metadata mismatch for {}",
                    entry.expert
                ));
            }
            unique_experts.insert(entry.expert);
            let gate_up = read_expert_slice(
                &checkpoint_dir.join(&gate_up_bank.shard),
                &gate_up_bank.tensor,
                entry.expert,
                EXPERTS,
                INTERMEDIATE * 2,
                HIDDEN,
            )?;
            let down = read_expert_slice(
                &checkpoint_dir.join(&down_bank.shard),
                &down_bank.tensor,
                entry.expert,
                EXPERTS,
                HIDDEN,
                INTERMEDIATE,
            )?;
            if bf16_hash(&gate_up) != entry.gate_up_payload_sha256
                || bf16_hash(&down) != entry.down_payload_sha256
            {
                return Err(format!(
                    "layer-1 decoder expert payload mismatch for {}",
                    entry.expert
                ));
            }
            let gate_up_output = linear_bf16(&gate_up, &hyper.mixed, INTERMEDIATE * 2, HIDDEN);
            let activated = swiglu_bf16(
                &gate_up_output[..INTERMEDIATE],
                &gate_up_output[INTERMEDIATE..],
            );
            let output = linear_bf16(&down, &activated, HIDDEN, INTERMEDIATE);
            let weighted = output
                .iter()
                .map(|value| to_bf16(from_bf16(*value) * entry.route_weight_bf16))
                .collect::<Vec<_>>();
            if bf16_hash(&weighted) != entry.weighted_down_bf16_sha256 {
                return Err(format!(
                    "layer-1 decoder weighted expert mismatch for {}",
                    entry.expert
                ));
            }
            for (sum, contribution) in routed.iter_mut().zip(weighted) {
                *sum = add_bf16(*sum, contribution);
            }
        }
        require_capture(step, "routed_mixture", &[HIDDEN], &routed)?;

        let shared_gate = linear_bf16(
            &dense["shared_gate_weight"],
            &hyper.mixed,
            INTERMEDIATE,
            HIDDEN,
        );
        let shared_up = linear_bf16(
            &dense["shared_up_weight"],
            &hyper.mixed,
            INTERMEDIATE,
            HIDDEN,
        );
        let shared_swiglu = swiglu_bf16(&shared_gate, &shared_up);
        let shared_down = linear_bf16(
            &dense["shared_down_weight"],
            &shared_swiglu,
            HIDDEN,
            INTERMEDIATE,
        );
        let shared_gate_logit =
            linear_bf16(&dense["shared_expert_gate_weight"], &hyper.mixed, 1, HIDDEN);
        let shared_gate_sigmoid = [sigmoid_bf16(shared_gate_logit[0])];
        let gated_shared = shared_down
            .iter()
            .map(|value| to_bf16(from_bf16(*value) * from_bf16(shared_gate_sigmoid[0])))
            .collect::<Vec<_>>();
        require_capture(step, "shared_gate", &[INTERMEDIATE], &shared_gate)?;
        require_capture(step, "shared_up", &[INTERMEDIATE], &shared_up)?;
        require_capture(step, "shared_swiglu", &[INTERMEDIATE], &shared_swiglu)?;
        require_capture(step, "shared_down", &[HIDDEN], &shared_down)?;
        require_capture(step, "shared_gate_logit", &[1], &shared_gate_logit)?;
        require_capture(step, "shared_gate_sigmoid", &[1], &shared_gate_sigmoid)?;
        require_capture(step, "gated_shared", &[HIDDEN], &gated_shared)?;
        let moe_output = routed
            .iter()
            .zip(&gated_shared)
            .map(|(left, right)| add_bf16(*left, *right))
            .collect::<Vec<_>>();
        require_capture(step, "moe_output", &[1, 1, HIDDEN], &moe_output)?;
        let injection = hyper
            .injection_weights
            .iter()
            .flat_map(|weight| {
                moe_output
                    .iter()
                    .map(|value| to_bf16(from_bf16(*value) * from_bf16(*weight)))
            })
            .collect::<Vec<_>>();
        require_capture(
            step,
            "mlp_injection_products",
            &[1, 1, HC_COUNT, HIDDEN],
            &injection,
        )?;
        let output = post
            .iter()
            .zip(&injection)
            .map(|(left, right)| add_bf16(*left, *right))
            .collect::<Vec<_>>();
        require_capture(step, "layer_output", &[1, 1, HC_HIDDEN], &output)?;
        selected_experts_by_step.push(selection);
    }

    let selected_expert_bytes = unique_experts.len() * 9_830_400;
    let parent_bytes = attention_report.total_verified_payload_bytes;
    Ok(DecoderLayer1VerificationReport {
        schema_version: 1,
        semantic: "qwen3_8_flash_next_layer1_ple_complete_decoder_verification",
        model: fixture.model,
        revision: fixture.revision,
        layer: LAYER,
        steps_verified: 2,
        exact_bf16_capture_hashes: 32,
        exact_weighted_expert_hashes: 20,
        dense_tensors_verified: 9,
        unique_experts_verified: unique_experts.len(),
        attention_residual_tensor_payload_bytes: parent_bytes,
        dense_tensor_payload_bytes: dense_bytes,
        selected_expert_payload_bytes: selected_expert_bytes,
        total_verified_payload_bytes: parent_bytes + dense_bytes + selected_expert_bytes,
        selected_experts_by_step,
        accepted_tokens: 0,
        performance_claim: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer1_fixture_uses_distinct_dynamic_routes() {
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../fixtures/decoder_layer/qwen3_8_flash_next_layer1_ple.json"
        ))
        .unwrap();
        assert_ne!(
            fixture.steps[0].selected_experts,
            fixture.steps[1].selected_experts
        );
    }
}
