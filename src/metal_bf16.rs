use crate::expert::{
    InputSpec, bf16_hash, from_bf16, linear_bf16, make_hidden, read_expert_slice, swiglu_bf16,
    to_bf16,
};
use crate::host_safety::{
    HostSafetyMonitor, HostSafetyPolicy, HostSafetySnapshot, PersistentResidencyDeclaration,
};
use crate::verify_expert_fixture;
use metal::{CompileOptions, Device, MTLCommandBufferStatus, MTLResourceOptions, MTLSize};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;
use std::time::Instant;

const HIDDEN: usize = 2560;
const INTERMEDIATE: usize = 640;
const EXPERTS: usize = 512;
const MEASUREMENTS: usize = 30;
const CONTROL_MEASUREMENTS: usize = 5;
const WARMUPS: usize = 3;

#[derive(Deserialize)]
struct Fixture {
    case: Case,
}

#[derive(Deserialize)]
struct Case {
    expert: usize,
    route_weight_bf16: f32,
    input_spec: InputSpec,
    input_bf16_sha256: String,
    gate_up: TensorSlice,
    down: TensorSlice,
    expected_bf16_sha256: ExpectedHashes,
}

#[derive(Deserialize)]
struct TensorSlice {
    tensor: String,
    shard: String,
    expert_payload_sha256: String,
}

#[derive(Deserialize)]
struct ExpectedHashes {
    gate_up: String,
    swiglu: String,
    down: String,
    weighted_down: String,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GemvShape {
    rows: u32,
    columns: u32,
}

struct Runtime {
    device: Device,
    queue: metal::CommandQueue,
    pipeline: metal::ComputePipelineState,
    device_name: String,
    compile_wall_time_ns: u128,
}

impl Runtime {
    fn compile(kernel_path: &Path) -> Result<Self, String> {
        let source = fs::read_to_string(kernel_path)
            .map_err(|error| format!("cannot read {}: {error}", kernel_path.display()))?;
        let device = Device::system_default().ok_or("no Metal device is available")?;
        if device.max_threads_per_threadgroup().width < 32 {
            return Err("Metal device cannot dispatch 32-lane GEMV".to_owned());
        }
        let options = CompileOptions::new();
        options.set_fast_math_enabled(false);
        let started = Instant::now();
        let library = device
            .new_library_with_source(&source, &options)
            .map_err(|error| format!("Metal compilation failed: {error}"))?;
        let function = library
            .get_function("firewing_bf16_gemv_exact", None)
            .map_err(|error| format!("Metal function lookup failed: {error}"))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|error| format!("Metal pipeline creation failed: {error}"))?;
        let compile_wall_time_ns = started.elapsed().as_nanos();
        let queue = device.new_command_queue();
        let device_name = device.name().to_owned();
        Ok(Self {
            device,
            queue,
            pipeline,
            device_name,
            compile_wall_time_ns,
        })
    }

    fn linear(&self, weights: &[u16], input: &[u16], rows: usize) -> Result<Vec<u16>, String> {
        if rows == 0 || input.is_empty() || weights.len() != rows * input.len() {
            return Err("Metal BF16 GEMV shape mismatch".to_owned());
        }
        let shared = MTLResourceOptions::StorageModeShared;
        let weight_buffer = self.device.new_buffer_with_data(
            weights.as_ptr().cast(),
            std::mem::size_of_val(weights) as u64,
            shared,
        );
        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr().cast(),
            std::mem::size_of_val(input) as u64,
            shared,
        );
        let output_buffer = self.device.new_buffer((rows * 2) as u64, shared);
        let shape = GemvShape {
            rows: rows as u32,
            columns: input.len() as u32,
        };
        let shape_buffer = self.device.new_buffer_with_data(
            (&shape as *const GemvShape).cast(),
            std::mem::size_of::<GemvShape>() as u64,
            shared,
        );
        let command = self.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(&weight_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_buffer(3, Some(&shape_buffer), 0);
        encoder.set_threadgroup_memory_length(0, 32 * std::mem::size_of::<f32>() as u64);
        encoder.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(32, 1, 1));
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(format!("Metal BF16 GEMV failed: {:?}", command.status()));
        }
        // SAFETY: the shared output buffer owns `rows * 2` initialized bytes
        // through this copy and Metal completion is synchronous above.
        let output = unsafe {
            std::slice::from_raw_parts(output_buffer.contents().cast::<u16>(), rows).to_vec()
        };
        Ok(output)
    }
}

#[derive(Debug, Eq, PartialEq, Serialize)]
pub struct MetalBf16GemvReport {
    pub schema_version: u32,
    pub semantic: &'static str,
    pub implementation_commit: String,
    pub kernel_sha256: String,
    pub device_name: String,
    pub layer: usize,
    pub expert: usize,
    pub exact_capture_hashes: usize,
    pub exact_candidate_measurements: usize,
    pub logical_source_bytes_per_expert: usize,
    pub compile_wall_time_ns: u128,
    pub authority_verification_wall_time_ns: u128,
    pub warm_source_load_and_hash_wall_time_ns: u128,
    pub control_wall_times_ns: Vec<u128>,
    pub candidate_wall_times_ns: Vec<u128>,
    pub control_median_wall_time_ns: u128,
    pub candidate_p10_wall_time_ns: u128,
    pub candidate_median_wall_time_ns: u128,
    pub candidate_p90_wall_time_ns: u128,
    pub median_speedup: String,
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

struct ExpertCaptures {
    gate_up: Vec<u16>,
    swiglu: Vec<u16>,
    down: Vec<u16>,
    weighted_down: Vec<u16>,
}

fn execute_expert(
    linear: impl Fn(&[u16], &[u16], usize) -> Result<Vec<u16>, String>,
    gate_up_weight: &[u16],
    down_weight: &[u16],
    hidden: &[u16],
    route_weight: f32,
) -> Result<ExpertCaptures, String> {
    let gate_up = linear(gate_up_weight, hidden, INTERMEDIATE * 2)?;
    let activated = swiglu_bf16(&gate_up[..INTERMEDIATE], &gate_up[INTERMEDIATE..]);
    let down = linear(down_weight, &activated, HIDDEN)?;
    let weighted = down
        .iter()
        .map(|value| to_bf16(from_bf16(*value) * route_weight))
        .collect();
    Ok(ExpertCaptures {
        gate_up,
        swiglu: activated,
        down,
        weighted_down: weighted,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn benchmark_metal_bf16_gemv(
    checkpoint_dir: &Path,
    model_lock_path: &Path,
    router_fixture_path: &Path,
    expert_fixture_path: &Path,
    kernel_path: &Path,
    implementation_commit: &str,
) -> Result<MetalBf16GemvReport, String> {
    if implementation_commit.len() != 40
        || !implementation_commit
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        return Err("implementation commit must be a full hexadecimal Git hash".to_owned());
    }
    let mut safety = HostSafetyMonitor::start_normative(vec![PersistentResidencyDeclaration {
        object: "one_real_bf16_expert_gate_up_and_down_payload".to_owned(),
        maximum_bytes: 9_830_400,
        lifetime: "benchmark_measurement_series".to_owned(),
        eviction_order: 1,
    }])?;
    let authority_started = Instant::now();
    let authority = verify_expert_fixture(
        checkpoint_dir,
        model_lock_path,
        router_fixture_path,
        expert_fixture_path,
    )?;
    let authority_verification_wall_time_ns = authority_started.elapsed().as_nanos();
    let fixture: Fixture =
        serde_json::from_slice(&fs::read(expert_fixture_path).map_err(|error| error.to_string())?)
            .map_err(|error| format!("malformed expert fixture: {error}"))?;
    let case = fixture.case;
    let hidden = make_hidden(HIDDEN, &case.input_spec)?;
    if bf16_hash(&hidden) != case.input_bf16_sha256 {
        return Err("Metal benchmark input authority mismatch".to_owned());
    }
    let source_started = Instant::now();
    let gate_up_weight = read_expert_slice(
        &checkpoint_dir.join(&case.gate_up.shard),
        &case.gate_up.tensor,
        case.expert,
        EXPERTS,
        INTERMEDIATE * 2,
        HIDDEN,
    )?;
    let down_weight = read_expert_slice(
        &checkpoint_dir.join(&case.down.shard),
        &case.down.tensor,
        case.expert,
        EXPERTS,
        HIDDEN,
        INTERMEDIATE,
    )?;
    if bf16_hash(&gate_up_weight) != case.gate_up.expert_payload_sha256
        || bf16_hash(&down_weight) != case.down.expert_payload_sha256
    {
        return Err("Metal benchmark source payload mismatch".to_owned());
    }
    let warm_source_load_and_hash_wall_time_ns = source_started.elapsed().as_nanos();
    safety.checkpoint("source_authority_complete", true)?;
    let runtime = Runtime::compile(kernel_path)?;
    safety.checkpoint("metal_compile_complete", true)?;

    let expected = &case.expected_bf16_sha256;
    let require_exact = |captures: &ExpertCaptures| {
        for (name, actual, expected) in [
            ("gate_up", &captures.gate_up, &expected.gate_up),
            ("swiglu", &captures.swiglu, &expected.swiglu),
            ("down", &captures.down, &expected.down),
            (
                "weighted_down",
                &captures.weighted_down,
                &expected.weighted_down,
            ),
        ] {
            let actual = bf16_hash(actual);
            if &actual != expected {
                return Err(format!(
                    "Metal expert {name} mismatch: expected {expected}, got {actual}"
                ));
            }
        }
        Ok(())
    };
    let metal = |weights: &[u16], input: &[u16], rows| runtime.linear(weights, input, rows);
    for _ in 0..WARMUPS {
        require_exact(&execute_expert(
            metal,
            &gate_up_weight,
            &down_weight,
            &hidden,
            case.route_weight_bf16,
        )?)?;
    }
    safety.checkpoint("warmups_complete", true)?;

    let mut control_wall_times_ns = Vec::with_capacity(CONTROL_MEASUREMENTS);
    let mut candidate_wall_times_ns = Vec::with_capacity(MEASUREMENTS);
    for _ in 0..CONTROL_MEASUREMENTS {
        let started = Instant::now();
        let control = execute_expert(
            |weights, input, rows| Ok(linear_bf16(weights, input, rows, input.len())),
            &gate_up_weight,
            &down_weight,
            &hidden,
            case.route_weight_bf16,
        )?;
        control_wall_times_ns.push(started.elapsed().as_nanos());
        require_exact(&control)?;
        for _ in 0..(MEASUREMENTS / CONTROL_MEASUREMENTS) {
            let started = Instant::now();
            let candidate = execute_expert(
                metal,
                &gate_up_weight,
                &down_weight,
                &hidden,
                case.route_weight_bf16,
            )?;
            candidate_wall_times_ns.push(started.elapsed().as_nanos());
            require_exact(&candidate)?;
        }
    }
    safety.checkpoint("measurements_complete", true)?;
    let (host_safety_policy, host_safety_snapshots) = safety.finish()?;
    let control_median = quantile(&control_wall_times_ns, 1, 2);
    let candidate_p10 = quantile(&candidate_wall_times_ns, 1, 10);
    let candidate_median = quantile(&candidate_wall_times_ns, 1, 2);
    let candidate_p90 = quantile(&candidate_wall_times_ns, 9, 10);
    Ok(MetalBf16GemvReport {
        schema_version: 1,
        semantic: "qwen3_8_flash_next_real_expert_exact_metal_bf16_gemv",
        implementation_commit: implementation_commit.to_owned(),
        kernel_sha256: format!(
            "{:x}",
            Sha256::digest(fs::read(kernel_path).map_err(|error| error.to_string())?)
        ),
        device_name: runtime.device_name,
        layer: authority.layer,
        expert: authority.expert,
        exact_capture_hashes: 4,
        exact_candidate_measurements: MEASUREMENTS + WARMUPS,
        logical_source_bytes_per_expert: gate_up_weight.len() * 2 + down_weight.len() * 2,
        compile_wall_time_ns: runtime.compile_wall_time_ns,
        authority_verification_wall_time_ns,
        warm_source_load_and_hash_wall_time_ns,
        control_wall_times_ns,
        candidate_wall_times_ns,
        control_median_wall_time_ns: control_median,
        candidate_p10_wall_time_ns: candidate_p10,
        candidate_median_wall_time_ns: candidate_median,
        candidate_p90_wall_time_ns: candidate_p90,
        median_speedup: format!("{:.6}", control_median as f64 / candidate_median as f64),
        cache_state: "warm_application_bf16_expert_weights_copied_into_bounded_metal_buffers_each_projection",
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
