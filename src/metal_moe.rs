use crate::expert::{
    InputSpec, bf16_hash, from_bf16, linear_bf16, make_hidden, read_expert_slice, swiglu_bf16,
    to_bf16,
};
use crate::host_safety::{
    HostSafetyMonitor, HostSafetyPolicy, HostSafetySnapshot, PersistentResidencyDeclaration,
};
use crate::verify_mixture_fixture;
use metal::{Buffer, CompileOptions, Device, MTLCommandBufferStatus, MTLResourceOptions, MTLSize};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;
use std::time::Instant;

const HIDDEN: usize = 2560;
const INTERMEDIATE: usize = 640;
const EXPERTS: usize = 512;
const TOP_K: usize = 10;
const MEASUREMENTS: usize = 30;
const CONTROL_MEASUREMENTS: usize = 5;
const WARMUPS: usize = 3;

#[derive(Deserialize)]
struct Fixture {
    case: Case,
}

#[derive(Deserialize)]
struct Case {
    layer: usize,
    input_spec: InputSpec,
    input_bf16_sha256: String,
    expert_execution_order: Vec<usize>,
    gate_up: TensorBank,
    down: TensorBank,
    experts: Vec<FixtureExpert>,
    mixture_bf16_sha256: String,
}

#[derive(Deserialize)]
struct TensorBank {
    tensor: String,
    shard: String,
}

#[derive(Deserialize)]
struct FixtureExpert {
    expert: usize,
    route_weight_bf16: f32,
    gate_up_payload_sha256: String,
    down_payload_sha256: String,
    weighted_down_bf16_sha256: String,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GemvShape {
    rows: u32,
    columns: u32,
}

struct InstalledExpert {
    expert: usize,
    route_weight_bf16: f32,
    expected_weighted_hash: String,
    gate_up_weight: Buffer,
    down_weight: Buffer,
    gate_up: Buffer,
    activated: Buffer,
    down: Buffer,
}

struct Runtime {
    queue: metal::CommandQueue,
    pipeline: metal::ComputePipelineState,
    gate_shape: Buffer,
    down_shape: Buffer,
    hidden: Buffer,
    installed: Vec<InstalledExpert>,
    device_name: String,
    compile_wall_time_ns: u128,
    install_wall_time_ns: u128,
}

impl Runtime {
    fn compile_and_install(
        kernel_path: &Path,
        checkpoint_dir: &Path,
        case: &Case,
        hidden: &[u16],
    ) -> Result<Self, String> {
        let source = fs::read_to_string(kernel_path)
            .map_err(|error| format!("cannot read {}: {error}", kernel_path.display()))?;
        let device = Device::system_default().ok_or("no Metal device is available")?;
        if device.max_threads_per_threadgroup().width < 32 {
            return Err("Metal device cannot dispatch 32-lane GEMV".to_owned());
        }
        let options = CompileOptions::new();
        options.set_fast_math_enabled(false);
        let compile_started = Instant::now();
        let library = device
            .new_library_with_source(&source, &options)
            .map_err(|error| format!("Metal compilation failed: {error}"))?;
        let function = library
            .get_function("firewing_bf16_gemv_exact", None)
            .map_err(|error| format!("Metal function lookup failed: {error}"))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|error| format!("Metal pipeline creation failed: {error}"))?;
        let compile_wall_time_ns = compile_started.elapsed().as_nanos();
        let shared = MTLResourceOptions::StorageModeShared;
        let buffer_value = |shape: &GemvShape| {
            device.new_buffer_with_data(
                (shape as *const GemvShape).cast(),
                std::mem::size_of::<GemvShape>() as u64,
                shared,
            )
        };
        let gate_shape = buffer_value(&GemvShape {
            rows: (INTERMEDIATE * 2) as u32,
            columns: HIDDEN as u32,
        });
        let down_shape = buffer_value(&GemvShape {
            rows: HIDDEN as u32,
            columns: INTERMEDIATE as u32,
        });
        let hidden_buffer = device.new_buffer_with_data(
            hidden.as_ptr().cast(),
            std::mem::size_of_val(hidden) as u64,
            shared,
        );

        let install_started = Instant::now();
        let mut installed = Vec::with_capacity(TOP_K);
        for entry in &case.experts {
            let gate_up = read_expert_slice(
                &checkpoint_dir.join(&case.gate_up.shard),
                &case.gate_up.tensor,
                entry.expert,
                EXPERTS,
                INTERMEDIATE * 2,
                HIDDEN,
            )?;
            let down = read_expert_slice(
                &checkpoint_dir.join(&case.down.shard),
                &case.down.tensor,
                entry.expert,
                EXPERTS,
                HIDDEN,
                INTERMEDIATE,
            )?;
            if bf16_hash(&gate_up) != entry.gate_up_payload_sha256
                || bf16_hash(&down) != entry.down_payload_sha256
            {
                return Err(format!(
                    "Metal MoE expert {} payload mismatch",
                    entry.expert
                ));
            }
            installed.push(InstalledExpert {
                expert: entry.expert,
                route_weight_bf16: entry.route_weight_bf16,
                expected_weighted_hash: entry.weighted_down_bf16_sha256.clone(),
                gate_up_weight: device.new_buffer_with_data(
                    gate_up.as_ptr().cast(),
                    std::mem::size_of_val(gate_up.as_slice()) as u64,
                    shared,
                ),
                down_weight: device.new_buffer_with_data(
                    down.as_ptr().cast(),
                    std::mem::size_of_val(down.as_slice()) as u64,
                    shared,
                ),
                gate_up: device.new_buffer((INTERMEDIATE * 2 * 2) as u64, shared),
                activated: device.new_buffer((INTERMEDIATE * 2) as u64, shared),
                down: device.new_buffer((HIDDEN * 2) as u64, shared),
            });
        }
        let install_wall_time_ns = install_started.elapsed().as_nanos();
        Ok(Self {
            queue: device.new_command_queue(),
            pipeline,
            gate_shape,
            down_shape,
            hidden: hidden_buffer,
            installed,
            device_name: device.name().to_owned(),
            compile_wall_time_ns,
            install_wall_time_ns,
        })
    }

    fn dispatch_projection(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        weights: &Buffer,
        input: &Buffer,
        output: &Buffer,
        shape: &Buffer,
        rows: usize,
    ) {
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(weights), 0);
        encoder.set_buffer(1, Some(input), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_buffer(3, Some(shape), 0);
        encoder.set_threadgroup_memory_length(0, 32 * std::mem::size_of::<f32>() as u64);
        encoder.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(32, 1, 1));
    }

    fn execute(&self) -> Result<MixtureCaptures, String> {
        let gate_command = self.queue.new_command_buffer();
        let gate_encoder = gate_command.new_compute_command_encoder();
        for expert in &self.installed {
            self.dispatch_projection(
                gate_encoder,
                &expert.gate_up_weight,
                &self.hidden,
                &expert.gate_up,
                &self.gate_shape,
                INTERMEDIATE * 2,
            );
        }
        gate_encoder.end_encoding();
        gate_command.commit();
        gate_command.wait_until_completed();
        if gate_command.status() != MTLCommandBufferStatus::Completed {
            return Err(format!(
                "Metal MoE gate transaction failed: {:?}",
                gate_command.status()
            ));
        }

        for expert in &self.installed {
            // SAFETY: completion above initializes exactly 2 * INTERMEDIATE shared BF16 values.
            let gate_up = unsafe {
                std::slice::from_raw_parts(
                    expert.gate_up.contents().cast::<u16>(),
                    INTERMEDIATE * 2,
                )
            };
            let activated = swiglu_bf16(&gate_up[..INTERMEDIATE], &gate_up[INTERMEDIATE..]);
            // SAFETY: the destination is a shared Metal allocation of INTERMEDIATE BF16 values.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    activated.as_ptr(),
                    expert.activated.contents().cast::<u16>(),
                    INTERMEDIATE,
                );
            }
        }

        let down_command = self.queue.new_command_buffer();
        let down_encoder = down_command.new_compute_command_encoder();
        for expert in &self.installed {
            self.dispatch_projection(
                down_encoder,
                &expert.down_weight,
                &expert.activated,
                &expert.down,
                &self.down_shape,
                HIDDEN,
            );
        }
        down_encoder.end_encoding();
        down_command.commit();
        down_command.wait_until_completed();
        if down_command.status() != MTLCommandBufferStatus::Completed {
            return Err(format!(
                "Metal MoE down transaction failed: {:?}",
                down_command.status()
            ));
        }

        let mut mixture = vec![to_bf16(0.0); HIDDEN];
        let mut weighted_hashes = Vec::with_capacity(TOP_K);
        for expert in &self.installed {
            // SAFETY: completion above initializes exactly HIDDEN shared BF16 values.
            let down =
                unsafe { std::slice::from_raw_parts(expert.down.contents().cast::<u16>(), HIDDEN) };
            let weighted = down
                .iter()
                .map(|value| to_bf16(from_bf16(*value) * expert.route_weight_bf16))
                .collect::<Vec<_>>();
            weighted_hashes.push((expert.expert, bf16_hash(&weighted)));
            for (output, contribution) in mixture.iter_mut().zip(weighted) {
                *output = to_bf16(from_bf16(*output) + from_bf16(contribution));
            }
        }
        Ok(MixtureCaptures {
            weighted_hashes,
            mixture,
        })
    }
}

struct MixtureCaptures {
    weighted_hashes: Vec<(usize, String)>,
    mixture: Vec<u16>,
}

pub(crate) struct ExactResidentTop10Runner {
    runtime: Runtime,
    mixture_hash: String,
}

impl ExactResidentTop10Runner {
    pub(crate) fn install(
        checkpoint_dir: &Path,
        mixture_fixture_path: &Path,
        kernel_path: &Path,
    ) -> Result<Self, String> {
        let fixture: Fixture = serde_json::from_slice(
            &fs::read(mixture_fixture_path).map_err(|error| error.to_string())?,
        )
        .map_err(|error| format!("malformed mixture fixture: {error}"))?;
        let case = fixture.case;
        if case.layer != 0
            || case.experts.len() != TOP_K
            || case.expert_execution_order.len() != TOP_K
            || case
                .experts
                .iter()
                .map(|entry| entry.expert)
                .collect::<Vec<_>>()
                != case.expert_execution_order
        {
            return Err("Metal MoE fixture route or layer is unsupported".to_owned());
        }
        let hidden = make_hidden(HIDDEN, &case.input_spec)?;
        if bf16_hash(&hidden) != case.input_bf16_sha256 {
            return Err("Metal MoE input authority mismatch".to_owned());
        }
        let mixture_hash = case.mixture_bf16_sha256.clone();
        let runtime = Runtime::compile_and_install(kernel_path, checkpoint_dir, &case, &hidden)?;
        Ok(Self {
            runtime,
            mixture_hash,
        })
    }

    pub(crate) fn execute_exact(&self) -> Result<(), String> {
        let captures = self.runtime.execute()?;
        require_exact(&self.runtime, &captures, &self.mixture_hash)
    }

    pub(crate) fn device_name(&self) -> &str {
        &self.runtime.device_name
    }
}

fn execute_control(runtime: &Runtime) -> MixtureCaptures {
    let hidden =
        unsafe { std::slice::from_raw_parts(runtime.hidden.contents().cast::<u16>(), HIDDEN) };
    let mut mixture = vec![to_bf16(0.0); HIDDEN];
    let mut weighted_hashes = Vec::with_capacity(TOP_K);
    for expert in &runtime.installed {
        let gate_weight = unsafe {
            std::slice::from_raw_parts(
                expert.gate_up_weight.contents().cast::<u16>(),
                INTERMEDIATE * 2 * HIDDEN,
            )
        };
        let down_weight = unsafe {
            std::slice::from_raw_parts(
                expert.down_weight.contents().cast::<u16>(),
                HIDDEN * INTERMEDIATE,
            )
        };
        let gate_up = linear_bf16(gate_weight, hidden, INTERMEDIATE * 2, HIDDEN);
        let activated = swiglu_bf16(&gate_up[..INTERMEDIATE], &gate_up[INTERMEDIATE..]);
        let down = linear_bf16(down_weight, &activated, HIDDEN, INTERMEDIATE);
        let weighted = down
            .iter()
            .map(|value| to_bf16(from_bf16(*value) * expert.route_weight_bf16))
            .collect::<Vec<_>>();
        weighted_hashes.push((expert.expert, bf16_hash(&weighted)));
        for (output, contribution) in mixture.iter_mut().zip(weighted) {
            *output = to_bf16(from_bf16(*output) + from_bf16(contribution));
        }
    }
    MixtureCaptures {
        weighted_hashes,
        mixture,
    }
}

fn require_exact(
    runtime: &Runtime,
    captures: &MixtureCaptures,
    mixture_hash: &str,
) -> Result<(), String> {
    if captures.weighted_hashes.len() != runtime.installed.len() {
        return Err("Metal MoE weighted capture count mismatch".to_owned());
    }
    for ((expert, actual), expected) in captures.weighted_hashes.iter().zip(&runtime.installed) {
        if *expert != expected.expert || actual != &expected.expected_weighted_hash {
            return Err(format!(
                "Metal MoE expert {expert} weighted output mismatch"
            ));
        }
    }
    let actual = bf16_hash(&captures.mixture);
    if actual != mixture_hash {
        return Err(format!(
            "Metal MoE mixture mismatch: expected {mixture_hash}, got {actual}"
        ));
    }
    Ok(())
}

fn quantile(values: &[u128], numerator: usize, denominator: usize) -> u128 {
    let mut ordered = values.to_vec();
    ordered.sort_unstable();
    ordered[((ordered.len() - 1) * numerator + denominator / 2) / denominator]
}

#[derive(Debug, Serialize)]
pub struct MetalTop10MoeReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub implementation_commit: String,
    pub kernel_sha256: String,
    pub device_name: String,
    pub layer: usize,
    pub expert_execution_order: Vec<usize>,
    pub unique_experts: usize,
    pub exact_weighted_expert_hashes_per_execution: usize,
    pub exact_mixture_hashes_per_execution: usize,
    pub exact_candidate_measurements: usize,
    pub logical_resident_source_bytes: usize,
    pub persistent_metal_buffer_bytes: usize,
    pub command_buffers_per_candidate: usize,
    pub compile_wall_time_ns: u128,
    pub authority_verification_wall_time_ns: u128,
    pub install_and_hash_wall_time_ns: u128,
    pub control_wall_times_ns: Vec<u128>,
    pub candidate_wall_times_ns: Vec<u128>,
    pub control_median_wall_time_ns: u128,
    pub candidate_p10_wall_time_ns: u128,
    pub candidate_median_wall_time_ns: u128,
    pub candidate_p90_wall_time_ns: u128,
    pub median_speedup: String,
    pub projected_48_routed_layers_median_ns: u128,
    pub projected_routed_only_tps: String,
    pub four_tps_routed_compute_budget_ns: u128,
    pub cache_state: &'static str,
    pub batch_size: usize,
    pub concurrency: usize,
    pub host_safety_policy: HostSafetyPolicy,
    pub host_safety_snapshots: Vec<HostSafetySnapshot>,
    pub accepted_tokens: usize,
    pub performance_claim: Option<String>,
}

#[allow(clippy::too_many_arguments)]
pub fn benchmark_metal_top10_moe(
    checkpoint_dir: &Path,
    model_lock_path: &Path,
    router_fixture_path: &Path,
    expert_fixture_path: &Path,
    mixture_fixture_path: &Path,
    kernel_path: &Path,
    implementation_commit: &str,
) -> Result<MetalTop10MoeReport, String> {
    if implementation_commit.len() != 40
        || !implementation_commit
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        return Err("implementation commit must be a full hexadecimal Git hash".to_owned());
    }
    let mut safety = HostSafetyMonitor::start_normative(vec![PersistentResidencyDeclaration {
        object: "top10_real_bf16_expert_weights_and_metal_working_buffers".to_owned(),
        maximum_bytes: 98_398_736,
        lifetime: "benchmark_measurement_series".to_owned(),
        eviction_order: 1,
    }])?;
    let authority_started = Instant::now();
    let authority = verify_mixture_fixture(
        checkpoint_dir,
        model_lock_path,
        router_fixture_path,
        expert_fixture_path,
        mixture_fixture_path,
    )?;
    let authority_verification_wall_time_ns = authority_started.elapsed().as_nanos();
    safety.checkpoint("mixture_authority_complete", true)?;

    let fixture: Fixture =
        serde_json::from_slice(&fs::read(mixture_fixture_path).map_err(|error| error.to_string())?)
            .map_err(|error| format!("malformed mixture fixture: {error}"))?;
    let case = fixture.case;
    if case.layer != 0
        || case.experts.len() != TOP_K
        || case.expert_execution_order.len() != TOP_K
        || case
            .experts
            .iter()
            .map(|entry| entry.expert)
            .collect::<Vec<_>>()
            != case.expert_execution_order
    {
        return Err("Metal MoE fixture route or layer is unsupported".to_owned());
    }
    let hidden = make_hidden(HIDDEN, &case.input_spec)?;
    if bf16_hash(&hidden) != case.input_bf16_sha256 {
        return Err("Metal MoE input authority mismatch".to_owned());
    }
    let runtime = Runtime::compile_and_install(kernel_path, checkpoint_dir, &case, &hidden)?;
    safety.checkpoint("metal_install_complete", false)?;

    for _ in 0..WARMUPS {
        require_exact(&runtime, &runtime.execute()?, &case.mixture_bf16_sha256)?;
    }
    safety.checkpoint("warmups_complete", false)?;

    let mut control_wall_times_ns = Vec::with_capacity(CONTROL_MEASUREMENTS);
    let mut candidate_wall_times_ns = Vec::with_capacity(MEASUREMENTS);
    for _ in 0..CONTROL_MEASUREMENTS {
        let started = Instant::now();
        let control = execute_control(&runtime);
        control_wall_times_ns.push(started.elapsed().as_nanos());
        require_exact(&runtime, &control, &case.mixture_bf16_sha256)?;
        for _ in 0..(MEASUREMENTS / CONTROL_MEASUREMENTS) {
            let started = Instant::now();
            let candidate = runtime.execute()?;
            candidate_wall_times_ns.push(started.elapsed().as_nanos());
            require_exact(&runtime, &candidate, &case.mixture_bf16_sha256)?;
        }
    }
    safety.checkpoint("measurements_complete", false)?;
    let control_median = quantile(&control_wall_times_ns, 1, 2);
    let candidate_p10 = quantile(&candidate_wall_times_ns, 1, 10);
    let candidate_median = quantile(&candidate_wall_times_ns, 1, 2);
    let candidate_p90 = quantile(&candidate_wall_times_ns, 9, 10);
    let projected = candidate_median * 48;
    let device_name = runtime.device_name.clone();
    let compile_wall_time_ns = runtime.compile_wall_time_ns;
    let install_and_hash_wall_time_ns = runtime.install_wall_time_ns;
    drop(runtime);
    let (host_safety_policy, host_safety_snapshots) = safety.finish()?;
    Ok(MetalTop10MoeReport {
        schema_version: 1,
        semantic: "qwen3_8_flash_next_real_top10_moe_exact_resident_metal_two_transaction",
        implementation_commit: implementation_commit.to_owned(),
        kernel_sha256: format!(
            "{:x}",
            Sha256::digest(fs::read(kernel_path).map_err(|error| error.to_string())?)
        ),
        device_name,
        layer: authority.layer,
        expert_execution_order: case.expert_execution_order,
        unique_experts: TOP_K,
        exact_weighted_expert_hashes_per_execution: TOP_K,
        exact_mixture_hashes_per_execution: 1,
        exact_candidate_measurements: MEASUREMENTS + WARMUPS,
        logical_resident_source_bytes: 98_304_000,
        persistent_metal_buffer_bytes: 98_398_736,
        command_buffers_per_candidate: 2,
        compile_wall_time_ns,
        authority_verification_wall_time_ns,
        install_and_hash_wall_time_ns,
        control_wall_times_ns,
        candidate_wall_times_ns,
        control_median_wall_time_ns: control_median,
        candidate_p10_wall_time_ns: candidate_p10,
        candidate_median_wall_time_ns: candidate_median,
        candidate_p90_wall_time_ns: candidate_p90,
        median_speedup: format!("{:.6}", control_median as f64 / candidate_median as f64),
        projected_48_routed_layers_median_ns: projected,
        projected_routed_only_tps: format!("{:.6}", 1_000_000_000_f64 / projected as f64),
        four_tps_routed_compute_budget_ns: 250_000_000,
        cache_state: "warm_application_exact_top10_bf16_weights_persistent_in_shared_metal_buffers_install_excluded",
        batch_size: 1,
        concurrency: 1,
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
    fn quantiles_use_nearest_rank_index() {
        let values = (0..30).map(|value| value * 10).collect::<Vec<_>>();
        assert_eq!(quantile(&values, 1, 10), 30);
        assert_eq!(quantile(&values, 1, 2), 150);
        assert_eq!(quantile(&values, 9, 10), 260);
    }
}
