use serde::Serialize;
use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PersistentResidencyDeclaration {
    pub object: String,
    pub maximum_bytes: u64,
    pub lifetime: String,
    pub eviction_order: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct HostSafetyPolicy {
    pub minimum_system_memory_free_percent: u64,
    pub maximum_process_physical_footprint_bytes: u64,
    pub maximum_post_phase_physical_footprint_bytes: u64,
    pub maximum_swap_growth_bytes: u64,
    pub maximum_new_throttled_pages: u64,
    pub pressure_event_monitor_required_above_bytes: u64,
    pub protected_service_names: Vec<String>,
    pub declared_persistent_residency: Vec<PersistentResidencyDeclaration>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct HostSafetySnapshot {
    pub phase: String,
    pub release_boundary: bool,
    pub system_memory_free_percent: u64,
    pub swap_used_bytes: u64,
    pub swap_growth_bytes: u64,
    pub throttled_pages: u64,
    pub new_throttled_pages: u64,
    pub process_resident_bytes: u64,
    pub process_physical_footprint_bytes: u64,
    pub process_peak_resident_bytes: u64,
    pub process_disk_bytes_read: u64,
    pub process_disk_bytes_read_growth: u64,
    pub process_disk_bytes_written: u64,
    pub process_disk_bytes_written_growth: u64,
    pub malloc_pressure_relief_bytes: u64,
    pub allocator_relief_state: String,
    pub declared_persistent_residency: Vec<PersistentResidencyDeclaration>,
    pub evictions: Vec<String>,
    pub memory_pressure_events: Vec<String>,
    pub pressure_event_observer_state: String,
    pub protected_service_pids: BTreeMap<String, Vec<u32>>,
}

pub(crate) struct HostSafetyMonitor {
    policy: HostSafetyPolicy,
    baseline_swap_bytes: u64,
    baseline_throttled_pages: u64,
    baseline_disk_bytes_read: u64,
    baseline_disk_bytes_written: u64,
    baseline_services: BTreeMap<String, Vec<u32>>,
    snapshots: Vec<HostSafetySnapshot>,
}

#[repr(C)]
#[derive(Default)]
struct RusageInfoV2 {
    uuid: [u8; 16],
    user_time: u64,
    system_time: u64,
    pkg_idle_wkups: u64,
    interrupt_wkups: u64,
    pageins: u64,
    wired_size: u64,
    resident_size: u64,
    phys_footprint: u64,
    proc_start_abstime: u64,
    proc_exit_abstime: u64,
    child_user_time: u64,
    child_system_time: u64,
    child_pkg_idle_wkups: u64,
    child_interrupt_wkups: u64,
    child_pageins: u64,
    child_elapsed_abstime: u64,
    diskio_bytesread: u64,
    diskio_byteswritten: u64,
}

#[cfg(target_os = "macos")]
#[link(name = "proc")]
unsafe extern "C" {
    fn proc_pid_rusage(
        pid: libc::c_int,
        flavor: libc::c_int,
        buffer: *mut libc::c_void,
    ) -> libc::c_int;
    fn malloc_zone_pressure_relief(zone: *mut libc::c_void, goal: usize) -> usize;
}

fn command_output(program: &str, arguments: &[&str]) -> Result<String, String> {
    let output = Command::new(program)
        .args(arguments)
        .output()
        .map_err(|error| format!("{program}: {error}"))?;
    if !output.status.success() {
        return Err(format!("{program} exited {}", output.status));
    }
    String::from_utf8(output.stdout).map_err(|error| format!("{program}: {error}"))
}

fn system_memory_free_percent() -> Result<u64, String> {
    command_output("/usr/bin/memory_pressure", &["-Q"])?
        .lines()
        .find_map(|line| {
            line.trim()
                .strip_prefix("System-wide memory free percentage:")
        })
        .and_then(|value| value.trim().strip_suffix('%'))
        .and_then(|value| value.parse().ok())
        .ok_or("memory_pressure output lacks free percentage".to_owned())
}

fn swap_used_bytes() -> Result<u64, String> {
    let output = command_output("/usr/sbin/sysctl", &["-n", "vm.swapusage"])?;
    let used = output
        .split_whitespace()
        .collect::<Vec<_>>()
        .windows(3)
        .find_map(|fields| (fields[0] == "used" && fields[1] == "=").then_some(fields[2]))
        .ok_or("vm.swapusage output lacks used value")?;
    let (number, multiplier) = if let Some(value) = used.strip_suffix('M') {
        (value, 1024.0_f64 * 1024.0)
    } else if let Some(value) = used.strip_suffix('G') {
        (value, 1024.0_f64 * 1024.0 * 1024.0)
    } else {
        return Err("vm.swapusage used value has unknown unit".to_owned());
    };
    let bytes = number
        .parse::<f64>()
        .map_err(|error| format!("vm.swapusage used value: {error}"))?
        * multiplier;
    if !bytes.is_finite() || bytes < 0.0 || bytes > u64::MAX as f64 {
        return Err("vm.swapusage used value is invalid".to_owned());
    }
    Ok(bytes.round() as u64)
}

fn throttled_pages() -> Result<u64, String> {
    command_output("/usr/bin/vm_stat", &[])?
        .lines()
        .find_map(|line| line.trim().strip_prefix("Pages throttled:"))
        .and_then(|value| value.trim().strip_suffix('.'))
        .and_then(|value| value.trim().parse().ok())
        .ok_or("vm_stat output lacks throttled pages".to_owned())
}

#[cfg(target_os = "macos")]
fn process_usage() -> Result<RusageInfoV2, String> {
    let mut usage = RusageInfoV2::default();
    // SAFETY: `usage` has Darwin's rusage_info_v2 layout and is exclusively borrowed.
    let result = unsafe {
        proc_pid_rusage(
            std::process::id() as libc::c_int,
            2,
            (&mut usage as *mut RusageInfoV2).cast(),
        )
    };
    if result != 0 {
        return Err(format!("proc_pid_rusage failed with {result}"));
    }
    Ok(usage)
}

#[cfg(not(target_os = "macos"))]
fn process_usage() -> Result<RusageInfoV2, String> {
    Err("Darwin host-safety counters are required".to_owned())
}

#[cfg(target_os = "macos")]
fn peak_resident_bytes() -> Result<u64, String> {
    // SAFETY: Darwin initializes the complete rusage structure for RUSAGE_SELF.
    let mut usage = unsafe { std::mem::zeroed::<libc::rusage>() };
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) } != 0 {
        return Err("getrusage(RUSAGE_SELF) failed".to_owned());
    }
    u64::try_from(usage.ru_maxrss).map_err(|_| "negative peak resident set".to_owned())
}

#[cfg(not(target_os = "macos"))]
fn peak_resident_bytes() -> Result<u64, String> {
    Err("Darwin peak-resident counter is required".to_owned())
}

#[cfg(target_os = "macos")]
fn pressure_relief() -> u64 {
    // SAFETY: null visits all malloc zones; zero requests all releasable bytes.
    unsafe { malloc_zone_pressure_relief(std::ptr::null_mut(), 0) as u64 }
}

#[cfg(not(target_os = "macos"))]
fn pressure_relief() -> u64 {
    0
}

fn protected_service_pids(names: &[String]) -> Result<BTreeMap<String, Vec<u32>>, String> {
    let output = command_output("/bin/ps", &["-axo", "pid=,comm="])?;
    let mut result = names
        .iter()
        .cloned()
        .map(|name| (name, Vec::new()))
        .collect::<BTreeMap<_, _>>();
    result.insert("firewing-self".to_owned(), vec![std::process::id()]);
    for line in output.lines() {
        let mut fields = line.trim().splitn(2, char::is_whitespace);
        let Some(pid) = fields.next().and_then(|value| value.parse::<u32>().ok()) else {
            continue;
        };
        let Some(command) = fields.next().map(str::trim) else {
            continue;
        };
        let basename = Path::new(command)
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or(command);
        for (name, pids) in &mut result {
            if basename == name {
                pids.push(pid);
            }
        }
    }
    Ok(result)
}

fn require_services(
    baseline: &BTreeMap<String, Vec<u32>>,
    current: &BTreeMap<String, Vec<u32>>,
    phase: &str,
) -> Result<(), String> {
    for name in baseline.keys() {
        if current.get(name).is_none_or(Vec::is_empty) {
            return Err(format!(
                "safety stop at {phase}: resident service {name} disappeared"
            ));
        }
    }
    Ok(())
}

impl HostSafetyMonitor {
    pub(crate) fn start_normative(
        declared_persistent_residency: Vec<PersistentResidencyDeclaration>,
    ) -> Result<Self, String> {
        let mut policy = normative_policy();
        policy.declared_persistent_residency = declared_persistent_residency;
        let baseline_swap_bytes = swap_used_bytes()?;
        let baseline_throttled_pages = throttled_pages()?;
        let baseline_usage = process_usage()?;
        let services = protected_service_pids(&policy.protected_service_names)?;
        let baseline_services = services
            .into_iter()
            .filter(|(_, pids)| !pids.is_empty())
            .collect();
        let mut monitor = Self {
            policy,
            baseline_swap_bytes,
            baseline_throttled_pages,
            baseline_disk_bytes_read: baseline_usage.diskio_bytesread,
            baseline_disk_bytes_written: baseline_usage.diskio_byteswritten,
            baseline_services,
            snapshots: Vec::new(),
        };
        monitor.checkpoint("process_start", false)?;
        Ok(monitor)
    }

    pub(crate) fn checkpoint(&mut self, phase: &str, release_boundary: bool) -> Result<(), String> {
        let relief = if release_boundary {
            pressure_relief()
        } else {
            0
        };
        let memory_free = system_memory_free_percent()?;
        let swap = swap_used_bytes()?;
        let throttled = throttled_pages()?;
        let usage = process_usage()?;
        let peak = peak_resident_bytes()?;
        let services = protected_service_pids(&self.policy.protected_service_names)?;
        let snapshot = HostSafetySnapshot {
            phase: phase.to_owned(),
            release_boundary,
            system_memory_free_percent: memory_free,
            swap_used_bytes: swap,
            swap_growth_bytes: swap.saturating_sub(self.baseline_swap_bytes),
            throttled_pages: throttled,
            new_throttled_pages: throttled.saturating_sub(self.baseline_throttled_pages),
            process_resident_bytes: usage.resident_size,
            process_physical_footprint_bytes: usage.phys_footprint,
            process_peak_resident_bytes: peak,
            process_disk_bytes_read: usage.diskio_bytesread,
            process_disk_bytes_read_growth: usage
                .diskio_bytesread
                .saturating_sub(self.baseline_disk_bytes_read),
            process_disk_bytes_written: usage.diskio_byteswritten,
            process_disk_bytes_written_growth: usage
                .diskio_byteswritten
                .saturating_sub(self.baseline_disk_bytes_written),
            malloc_pressure_relief_bytes: relief,
            allocator_relief_state: if release_boundary {
                "phase_buffers_dropped_then_malloc_relief_requested".to_owned()
            } else {
                "not_requested".to_owned()
            },
            declared_persistent_residency: self.policy.declared_persistent_residency.clone(),
            evictions: if release_boundary {
                vec!["phase_scoped_model_buffers_dropped_before_checkpoint".to_owned()]
            } else {
                Vec::new()
            },
            memory_pressure_events: Vec::new(),
            pressure_event_observer_state: "not_required_process_cap_does_not_exceed_10_gib"
                .to_owned(),
            protected_service_pids: services.clone(),
        };
        let evidence = serde_json::to_string(&snapshot).map_err(|error| error.to_string())?;
        self.snapshots.push(snapshot.clone());
        if snapshot.system_memory_free_percent < self.policy.minimum_system_memory_free_percent {
            return Err(format!("safety stop: memory floor; snapshot={evidence}"));
        }
        if snapshot.process_physical_footprint_bytes
            > self.policy.maximum_process_physical_footprint_bytes
            || snapshot.process_peak_resident_bytes
                > self.policy.maximum_process_physical_footprint_bytes
        {
            return Err(format!(
                "safety stop: process footprint; snapshot={evidence}"
            ));
        }
        if release_boundary
            && snapshot.process_physical_footprint_bytes
                > self.policy.maximum_post_phase_physical_footprint_bytes
        {
            return Err(format!(
                "safety stop: post-phase footprint; snapshot={evidence}"
            ));
        }
        if snapshot.swap_growth_bytes > self.policy.maximum_swap_growth_bytes {
            return Err(format!("safety stop: swap growth; snapshot={evidence}"));
        }
        if snapshot.new_throttled_pages > self.policy.maximum_new_throttled_pages {
            return Err(format!("safety stop: VM throttling; snapshot={evidence}"));
        }
        require_services(&self.baseline_services, &services, phase)
            .map_err(|error| format!("{error}; snapshot={evidence}"))
    }

    pub(crate) fn finish(mut self) -> Result<(HostSafetyPolicy, Vec<HostSafetySnapshot>), String> {
        self.checkpoint("buffer_release", true)?;
        Ok((self.policy, self.snapshots))
    }
}

fn normative_policy() -> HostSafetyPolicy {
    HostSafetyPolicy {
        minimum_system_memory_free_percent: 10,
        // Stay at or below the threshold that would require a pressure-event
        // observer and declared eviction plan under TARGET.md.
        maximum_process_physical_footprint_bytes: 10 * 1024 * 1024 * 1024,
        maximum_post_phase_physical_footprint_bytes: 4 * 1024 * 1024 * 1024,
        maximum_swap_growth_bytes: 0,
        maximum_new_throttled_pages: 0,
        pressure_event_monitor_required_above_bytes: 10 * 1024 * 1024 * 1024,
        protected_service_names: vec![
            "ChatGPT".to_owned(),
            "Codex".to_owned(),
            "WindowServer".to_owned(),
            "syncthing".to_owned(),
            "bird".to_owned(),
            "cloudd".to_owned(),
        ],
        declared_persistent_residency: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disappeared_baseline_service_fails_closed() {
        let baseline = BTreeMap::from([("WindowServer".to_owned(), vec![1])]);
        let current = BTreeMap::from([("WindowServer".to_owned(), vec![])]);
        assert!(require_services(&baseline, &current, "layer_1").is_err());
    }

    #[test]
    fn normative_policy_never_enters_unobserved_pressure_event_mode() {
        let policy = normative_policy();
        assert_eq!(policy.minimum_system_memory_free_percent, 10);
        assert_eq!(policy.maximum_swap_growth_bytes, 0);
        assert_eq!(policy.maximum_new_throttled_pages, 0);
        assert!(policy.declared_persistent_residency.is_empty());
        assert!(
            policy.maximum_process_physical_footprint_bytes
                <= policy.pressure_event_monitor_required_above_bytes
        );
    }
}
