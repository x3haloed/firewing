use crate::accumulated_layers4_47::{
    AccumulatedLayers4Through47VerificationReport,
    verify_accumulated_layers4_through47_fixture_with_outputs,
};
use crate::decoder_layer::pytorch_topk_bf16;
use crate::deltanet::read_tensor;
use crate::expert::{bf16_hash, bf16_payload_matches, from_bf16, linear_bf16};
use crate::hyper_connection::{FinalMixerOutputs, run_final_mixer};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

const MODEL: &str = "Qwen/Qwen3.8-Flash-Next";
const HIDDEN: usize = 2560;
const HC_COUNT: usize = 4;
const HC_HIDDEN: usize = HIDDEN * HC_COUNT;
const HC_LOWRANK: usize = 320;
const VOCAB: usize = 248_320;

#[derive(Deserialize)]
struct Fixture {
    schema_version: u32,
    semantic: String,
    model: String,
    revision: String,
    reference: Reference,
    configuration: Configuration,
    tensors: BTreeMap<String, Tensor>,
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
    decoder_fixture_sha256: String,
}

#[derive(Deserialize)]
struct Configuration {
    hidden_size: usize,
    hc_count: usize,
    hc_lowrank: usize,
    vocab_size: usize,
    rms_norm_eps: f32,
    tie_word_embeddings: bool,
    boundary_dtype: String,
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
struct Step {
    ordinal: usize,
    captures: BTreeMap<String, String>,
    top20_token_ids: Vec<usize>,
    top20_logit_bf16_u16: Vec<u16>,
    top20_cutoff_bf16_u16: u16,
    strictly_above_cutoff_token_ids: Vec<usize>,
    cutoff_tie_token_ids: Vec<usize>,
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
struct EmbeddedOutput {
    tensors: BTreeMap<String, Tensor>,
    steps: Vec<Step>,
}

pub(crate) struct EmbeddedTextOutputVerification {
    pub output_verified_payload_bytes: usize,
    pub top20_token_ids_by_step: Vec<Vec<usize>>,
    pub top20_cutoff_tie_counts_by_step: Vec<usize>,
}

#[derive(Debug, Eq, PartialEq, Serialize)]
pub struct TextOutputVerificationReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub model: String,
    pub revision: String,
    pub decoder_layers_verified: usize,
    pub steps_verified: usize,
    pub vocab_size: usize,
    pub exact_mixer_capture_hashes: usize,
    pub exact_full_logit_hashes: usize,
    pub exact_ranked_logit_entries: usize,
    pub parent_verified_payload_bytes: usize,
    pub output_verified_payload_bytes: usize,
    pub total_verified_payload_bytes: usize,
    pub top20_token_ids_by_step: Vec<Vec<usize>>,
    pub top20_cutoff_tie_counts_by_step: Vec<usize>,
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

fn require_capture(step: &Step, name: &str, values: &[u16]) -> Result<(), String> {
    let expected = step
        .captures
        .get(name)
        .ok_or_else(|| format!("missing output capture {name}"))?;
    if bf16_hash(values) != *expected {
        return Err(format!(
            "text output capture mismatch at step {} {name}",
            step.ordinal
        ));
    }
    Ok(())
}

fn require_mixer_captures(step: &Step, output: &FinalMixerOutputs) -> Result<(), String> {
    for (name, values) in [
        ("hyper_input_normed", output.hyper_input_normed.as_slice()),
        ("mix_down", output.mix_down.as_slice()),
        ("mix_down_scaled", output.mix_down_scaled.as_slice()),
        ("mix_down_silu", output.mix_down_silu.as_slice()),
        ("mix_up", output.mix_up.as_slice()),
        ("input_mix_weight", output.input_mix.as_slice()),
        ("mixed_products", output.products.as_slice()),
        ("mixed_hidden", output.mixed.as_slice()),
    ] {
        require_capture(step, name, values)?;
    }
    Ok(())
}

fn load_locked_tensor(
    checkpoint_dir: &Path,
    lock: &ModelLock,
    record: &Tensor,
    expected_name: &str,
    expected_shape: &[usize],
) -> Result<Vec<u16>, String> {
    let locked = lock
        .files
        .iter()
        .filter(|file| file.path == record.shard)
        .collect::<Vec<_>>();
    if record.tensor != expected_name
        || record.shape != expected_shape
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
        return Err(format!(
            "text output tensor identity mismatch for {expected_name}"
        ));
    }
    let values = read_tensor(
        &checkpoint_dir.join(&record.shard),
        expected_name,
        expected_shape,
    )?;
    if !bf16_payload_matches(&values, &record.payload_sha256) {
        return Err(format!(
            "text output tensor payload mismatch for {expected_name}"
        ));
    }
    Ok(values)
}

fn verify_output_core(
    checkpoint_dir: &Path,
    lock: &ModelLock,
    tensors: &BTreeMap<String, Tensor>,
    steps: &[Step],
    decoder_outputs: &[Vec<u16>],
) -> Result<EmbeddedTextOutputVerification, String> {
    if tensors.len() != 4
        || steps.len() != 2
        || decoder_outputs.len() != 2
        || steps.iter().any(|step| {
            step.captures.len() != 10
                || !step.captures.values().all(|value| is_hash(value))
                || step.top20_token_ids.len() != 20
                || step.top20_logit_bf16_u16.len() != 20
                || step.strictly_above_cutoff_token_ids.len() >= 20
                || step.strictly_above_cutoff_token_ids.len() + step.cutoff_tie_token_ids.len() < 20
        })
    {
        return Err("text output capture schema mismatch".to_owned());
    }
    let tensor = |key: &str| {
        tensors
            .get(key)
            .ok_or_else(|| format!("missing text output tensor record {key}"))
    };
    let mixer_prefix = "model.language_model.hyper_connection_mixer";
    let hc_norm = load_locked_tensor(
        checkpoint_dir,
        lock,
        tensor("hc_norm")?,
        &format!("{mixer_prefix}.hc_norm.weight"),
        &[HC_HIDDEN],
    )?;
    let mix_down = load_locked_tensor(
        checkpoint_dir,
        lock,
        tensor("input_mix_weight_down")?,
        &format!("{mixer_prefix}.input_mix_weight_down.weight"),
        &[HC_LOWRANK, HC_HIDDEN],
    )?;
    let mix_up = load_locked_tensor(
        checkpoint_dir,
        lock,
        tensor("input_mix_weight_up")?,
        &format!("{mixer_prefix}.input_mix_weight_up.weight"),
        &[HC_HIDDEN, HC_LOWRANK],
    )?;
    let head = load_locked_tensor(
        checkpoint_dir,
        lock,
        tensor("lm_head")?,
        "lm_head.weight",
        &[VOCAB, HIDDEN],
    )?;
    let output_bytes = (hc_norm.len() + mix_down.len() + mix_up.len() + head.len()) * 2;
    let mut ranked = Vec::with_capacity(2);
    let mut tie_counts = Vec::with_capacity(2);
    for (ordinal, (step, decoder_output)) in steps.iter().zip(decoder_outputs).enumerate() {
        if step.ordinal != ordinal {
            return Err("text output step ordinal mismatch".to_owned());
        }
        require_capture(step, "decoder_output", decoder_output)?;
        let mixed = run_final_mixer(decoder_output, &hc_norm, &mix_down, &mix_up)?;
        require_mixer_captures(step, &mixed)?;
        let logits = linear_bf16(&head, &mixed.mixed, VOCAB, HIDDEN);
        require_capture(step, "logits", &logits)?;
        let indices = pytorch_topk_bf16(&logits, 20)?;
        let values = indices
            .iter()
            .map(|index| logits[*index])
            .collect::<Vec<_>>();
        if indices != step.top20_token_ids || values != step.top20_logit_bf16_u16 {
            return Err(format!(
                "text output ranked logits mismatch at step {ordinal}"
            ));
        }
        if values[19] != step.top20_cutoff_bf16_u16 {
            return Err(format!("text output cutoff mismatch at step {ordinal}"));
        }
        let strictly_above = logits
            .iter()
            .enumerate()
            .filter_map(|(index, value)| {
                (*value != step.top20_cutoff_bf16_u16
                    && from_bf16(*value) > from_bf16(step.top20_cutoff_bf16_u16))
                .then_some(index)
            })
            .collect::<Vec<_>>();
        let ties = logits
            .iter()
            .enumerate()
            .filter_map(|(index, value)| (*value == step.top20_cutoff_bf16_u16).then_some(index))
            .collect::<Vec<_>>();
        if strictly_above != step.strictly_above_cutoff_token_ids
            || ties != step.cutoff_tie_token_ids
        {
            return Err(format!(
                "text output cutoff partition mismatch at step {ordinal}"
            ));
        }
        tie_counts.push(ties.len());
        ranked.push(indices);
    }
    Ok(EmbeddedTextOutputVerification {
        output_verified_payload_bytes: output_bytes,
        top20_token_ids_by_step: ranked,
        top20_cutoff_tie_counts_by_step: tie_counts,
    })
}

pub(crate) fn verify_embedded_text_output_fixture(
    checkpoint_dir: &Path,
    model_lock_path: &Path,
    output: &serde_json::Value,
    expected_model: &str,
    expected_revision: &str,
    decoder_outputs: &[Vec<u16>],
) -> Result<EmbeddedTextOutputVerification, String> {
    let fixture: EmbeddedOutput = serde_json::from_value(output.clone())
        .map_err(|error| format!("malformed embedded text output: {error}"))?;
    let lock: ModelLock =
        serde_json::from_slice(&fs::read(model_lock_path).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    if lock.model != expected_model || lock.revision != expected_revision {
        return Err("embedded text output model lock mismatch".to_owned());
    }
    verify_output_core(
        checkpoint_dir,
        &lock,
        &fixture.tensors,
        &fixture.steps,
        decoder_outputs,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn verify_text_output_fixture(
    checkpoint_dir: &Path,
    model_lock_path: &Path,
    ngram_fixture_path: &Path,
    ngram_row_fixture_path: &Path,
    layer0_hyper_fixture_path: &Path,
    layer0_deltanet_fixture_path: &Path,
    layer0_attention_fixture_path: &Path,
    layer0_sparse_moe_fixture_path: &Path,
    layer0_fixture_path: &Path,
    ple_fixture_path: &Path,
    layer1_attention_fixture_path: &Path,
    layer1_fixture_path: &Path,
    layers01_fixture_path: &Path,
    layer2_fixture_path: &Path,
    layer3_fixture_path: &Path,
    decoder_fixture_path: &Path,
    fixture_path: &Path,
) -> Result<TextOutputVerificationReport, String> {
    let fixture: Fixture = serde_json::from_slice(
        &fs::read(fixture_path).map_err(|error| format!("cannot read output fixture: {error}"))?,
    )
    .map_err(|error| format!("malformed output fixture: {error}"))?;
    let config = &fixture.configuration;
    if fixture.schema_version != 1
        || fixture.semantic != "qwen3_8_flash_next_accumulated_decoder_final_mixer_logits"
        || fixture.model != MODEL
        || fixture.reference.implementation
            != "source_derived_and_official_huggingface_transformers_qwen4_exp"
        || fixture.reference.transformers_version != "5.16.1"
        || fixture.reference.source
            != "Qwen4ExpTextModel.forward; Qwen4ExpTextGatedResidual.forward; Qwen4ExpForCausalLM.forward"
        || config.hidden_size != HIDDEN
        || config.hc_count != HC_COUNT
        || config.hc_lowrank != HC_LOWRANK
        || config.vocab_size != VOCAB
        || config.rms_norm_eps.to_bits() != 1.0e-6_f32.to_bits()
        || config.tie_word_embeddings
        || config.boundary_dtype != "BF16"
        || fixture.tensors.len() != 4
        || fixture.steps.len() != 2
        || sha256_file(model_lock_path)? != fixture.reference.model_lock_sha256
        || sha256_file(&checkpoint_dir.join("config.json"))? != fixture.reference.config_sha256
        || sha256_file(&checkpoint_dir.join("model.safetensors.index.json"))?
            != fixture.reference.tensor_index_sha256
        || sha256_file(decoder_fixture_path)? != fixture.reference.decoder_fixture_sha256
    {
        return Err("text output fixture identity or configuration is unsupported".to_owned());
    }
    let lock: ModelLock =
        serde_json::from_slice(&fs::read(model_lock_path).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    if lock.model != fixture.model || lock.revision != fixture.revision {
        return Err("text output model lock mismatch".to_owned());
    }

    let (parent, decoder_outputs): (AccumulatedLayers4Through47VerificationReport, _) =
        verify_accumulated_layers4_through47_fixture_with_outputs(
            checkpoint_dir,
            model_lock_path,
            ngram_fixture_path,
            ngram_row_fixture_path,
            layer0_hyper_fixture_path,
            layer0_deltanet_fixture_path,
            layer0_attention_fixture_path,
            layer0_sparse_moe_fixture_path,
            layer0_fixture_path,
            ple_fixture_path,
            layer1_attention_fixture_path,
            layer1_fixture_path,
            layers01_fixture_path,
            layer2_fixture_path,
            layer3_fixture_path,
            decoder_fixture_path,
        )?;
    let output = verify_output_core(
        checkpoint_dir,
        &lock,
        &fixture.tensors,
        &fixture.steps,
        &decoder_outputs,
    )?;
    Ok(TextOutputVerificationReport {
        schema_version: 1,
        semantic: "qwen3_8_flash_next_accumulated_decoder_final_mixer_logits_verification",
        model: fixture.model,
        revision: fixture.revision,
        decoder_layers_verified: 48,
        steps_verified: 2,
        vocab_size: VOCAB,
        exact_mixer_capture_hashes: 2 * 9,
        exact_full_logit_hashes: 2,
        exact_ranked_logit_entries: 2 * 20,
        parent_verified_payload_bytes: parent.total_verified_payload_bytes,
        output_verified_payload_bytes: output.output_verified_payload_bytes,
        total_verified_payload_bytes: parent.total_verified_payload_bytes
            + output.output_verified_payload_bytes,
        top20_token_ids_by_step: output.top20_token_ids_by_step,
        top20_cutoff_tie_counts_by_step: output.top20_cutoff_tie_counts_by_step,
        accepted_tokens: 0,
        performance_claim: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn committed_fixture_is_hash_only_and_has_two_full_vocab_steps() {
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../fixtures/accumulated/qwen3_8_flash_next_final_mixer_logits.json"
        ))
        .unwrap();
        assert_eq!(fixture.steps.len(), 2);
        assert!(fixture.steps.iter().all(|step| step.captures.len() == 10));
        assert!(
            fixture
                .steps
                .iter()
                .all(|step| step.top20_token_ids.len() == 20)
        );
        assert_eq!(fixture.steps[0].strictly_above_cutoff_token_ids.len(), 19);
        assert_eq!(fixture.steps[0].cutoff_tie_token_ids.len(), 5);
        assert_eq!(fixture.steps[1].cutoff_tie_token_ids.len(), 1);
    }
}
