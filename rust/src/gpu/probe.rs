//! Dependency-free CUDA environment probe.
//!
//! P0 answers one question: *can this host run the GPU bypass at all?* Every
//! fact here is discovered by shelling out to `nvidia-smi` and by looking for
//! the CUDA runtime libraries on disk, so the module compiles and runs with
//! the crate's existing dependency set and stays usable while the `cudarc`
//! dependency is still unavailable (P0-a). The binding-level assertions —
//! `cudarc` loading the driver, and NVRTC actually compiling a kernel — land
//! in P0-b.
//!
//! The probe deliberately never panics and never errors. A host without CUDA
//! is the expected outcome on the project's Windows development machines, so
//! "no GPU" is reported as data rather than raised as a failure; the test
//! harness decides whether that is acceptable via `NATAL_GPU_REQUIRE`.
//!
//! ## Why the free-memory number matters
//!
//! The target server shares one RTX 5090 across tenants, so the nominal
//! device size is not what a run may allocate. [`GpuDevice::memory_free_mib`]
//! reports what is actually available and is the input the §D5 budget guard
//! must use.

use std::path::{Path, PathBuf};
use std::process::Command;

/// A single CUDA device as reported by `nvidia-smi`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuDevice {
    /// Zero-based device index.
    pub index: u32,
    /// Marketing name, e.g. `NVIDIA GeForce RTX 5090 D V2`.
    pub name: String,
    /// Compute capability string, e.g. `12.0` for sm_120. `None` when the
    /// installed driver is too old to report `compute_cap`.
    pub compute_capability: Option<String>,
    /// Total device memory in MiB.
    pub memory_total_mib: u64,
    /// Device memory currently held, by this container *and* by other tenants.
    pub memory_used_mib: u64,
}

impl GpuDevice {
    /// Device memory not held by any tenant, in MiB.
    ///
    /// ## Returns
    /// Total minus used, saturating at zero.
    pub fn memory_free_mib(&self) -> u64 {
        self.memory_total_mib.saturating_sub(self.memory_used_mib)
    }
}

/// Everything P0 discovers about the host's CUDA environment.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CudaProbe {
    /// Resolved path of the `nvidia-smi` executable, when it is on `PATH`.
    pub nvidia_smi: Option<PathBuf>,
    /// Driver version reported by `nvidia-smi --query-gpu=driver_version`.
    pub driver_version: Option<String>,
    /// Maximum CUDA version the installed driver supports.
    pub cuda_version: Option<String>,
    /// Devices reported by the driver.
    pub devices: Vec<GpuDevice>,
    /// Path of the NVRTC shared library, which runtime kernel compilation
    /// needs. `None` here is the one finding that would force the AOT
    /// (compile-time `nvcc`) route instead.
    pub nvrtc: Option<PathBuf>,
    /// CUDA toolkit installation directories that were found.
    pub toolkit_roots: Vec<PathBuf>,
}

impl CudaProbe {
    /// Scan the host for a CUDA environment.
    ///
    /// ## Returns
    /// A probe whose fields are `None`/empty wherever discovery failed. The
    /// call performs no allocation beyond the report itself, never blocks on
    /// the device, and never panics.
    pub fn scan() -> Self {
        let nvidia_smi = find_in_path("nvidia-smi");
        let mut probe = Self {
            nvidia_smi: nvidia_smi.clone(),
            ..Self::default()
        };
        if let Some(smi) = nvidia_smi.as_deref() {
            probe.driver_version = query_first(smi, "driver_version");
            probe.cuda_version = parse_cuda_version(smi);
            probe.devices = query_devices(smi);
        }
        probe.toolkit_roots = cuda_toolkit_roots();
        probe.nvrtc = search_dirs(&nvrtc_search_dirs(&probe.toolkit_roots), is_nvrtc_name);
        probe
    }

    /// Whether the host can host the GPU bypass at all.
    ///
    /// ## Returns
    /// `true` when at least one device is visible *and* NVRTC is present.
    /// Memory sufficiency is a separate, per-run question (see
    /// [`GpuDevice::memory_free_mib`]).
    pub fn is_usable(&self) -> bool {
        !self.devices.is_empty() && self.nvrtc.is_some()
    }

    /// Render the probe as a multi-line human-readable report.
    ///
    /// ## Returns
    /// One line per discovered fact, ending with the overall usability verdict.
    pub fn report(&self) -> String {
        let mut lines = Vec::new();
        match &self.nvidia_smi {
            Some(path) => lines.push(format!("nvidia-smi      : {}", path.display())),
            None => lines.push("nvidia-smi      : NOT FOUND".to_owned()),
        }
        lines.push(format!(
            "driver version  : {}",
            self.driver_version.as_deref().unwrap_or("<unknown>")
        ));
        lines.push(format!(
            "CUDA version    : {}",
            self.cuda_version.as_deref().unwrap_or("<unknown>")
        ));
        match &self.nvrtc {
            Some(path) => lines.push(format!("NVRTC           : {}", path.display())),
            None => lines.push(
                "NVRTC           : NOT FOUND (runtime kernel compilation unavailable)".to_owned(),
            ),
        }
        if self.toolkit_roots.is_empty() {
            lines.push("toolkit roots   : <none>".to_owned());
        } else {
            let roots: Vec<String> = self
                .toolkit_roots
                .iter()
                .map(|root| root.display().to_string())
                .collect();
            lines.push(format!("toolkit roots   : {}", roots.join(", ")));
        }
        if self.devices.is_empty() {
            lines.push("devices         : <none>".to_owned());
        }
        for device in &self.devices {
            lines.push(format!(
                "device {}        : {} | cc={} | {} MiB total, {} MiB used, {} MiB free",
                device.index,
                device.name,
                device.compute_capability.as_deref().unwrap_or("<unknown>"),
                device.memory_total_mib,
                device.memory_used_mib,
                device.memory_free_mib(),
            ));
        }
        lines.push(format!("usable          : {}", self.is_usable()));
        lines.join("\n")
    }
}

/// Marker that precedes the supported CUDA version in `nvidia-smi` output.
const CUDA_VERSION_MARKER: &str = "CUDA Version:";

/// Locate an executable on `PATH`, trying the `.exe` suffix on Windows.
///
/// ## Parameters
/// - `binary`: Executable name without a platform suffix.
///
/// ## Returns
/// The first existing path, or `None` when `PATH` is unset or nothing matches.
fn find_in_path(binary: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    let with_exe = format!("{binary}.exe");
    std::env::split_paths(&path).find_map(|dir| {
        let plain = dir.join(binary);
        if plain.is_file() {
            return Some(plain);
        }
        let exe = dir.join(&with_exe);
        exe.is_file().then_some(exe)
    })
}

/// Run `nvidia-smi` with `args` and capture stdout.
///
/// ## Parameters
/// - `smi`: Resolved `nvidia-smi` path.
/// - `args`: Arguments to pass.
///
/// ## Returns
/// Stdout as text when the command ran and exited zero, otherwise `None`.
fn run_smi(smi: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new(smi).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Read the first non-empty line of a single-field `nvidia-smi` query.
///
/// ## Parameters
/// - `smi`: Resolved `nvidia-smi` path.
/// - `field`: `--query-gpu` field name, e.g. `driver_version`.
///
/// ## Returns
/// The trimmed value, or `None` when the query failed or produced nothing.
fn query_first(smi: &Path, field: &str) -> Option<String> {
    let query = format!("--query-gpu={field}");
    let text = run_smi(smi, &[&query, "--format=csv,noheader"])?;
    let value = text.lines().next()?.trim();
    (!value.is_empty()).then(|| value.to_owned())
}

/// Extract the driver's maximum supported CUDA version from `nvidia-smi`.
///
/// ## Parameters
/// - `smi`: Resolved `nvidia-smi` path.
///
/// ## Returns
/// A dotted version string such as `13.2`, or `None` when the banner does not
/// carry one.
fn parse_cuda_version(smi: &Path) -> Option<String> {
    let text = run_smi(smi, &[])?;
    let start = text.find(CUDA_VERSION_MARKER)? + CUDA_VERSION_MARKER.len();
    let version: String = text[start..]
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    (!version.is_empty()).then_some(version)
}

/// Query the device list, degrading gracefully on drivers without `compute_cap`.
///
/// ## Parameters
/// - `smi`: Resolved `nvidia-smi` path.
///
/// ## Returns
/// One entry per visible device, empty when the queries are unsupported.
fn query_devices(smi: &Path) -> Vec<GpuDevice> {
    // `compute_cap` is absent on older drivers, which makes the whole query
    // fail rather than returning a placeholder, so retry with fewer fields.
    for query in [
        "--query-gpu=index,name,compute_cap,memory.total,memory.used",
        "--query-gpu=index,name,memory.total,memory.used",
    ] {
        let Some(text) = run_smi(smi, &[query, "--format=csv,noheader,nounits"]) else {
            continue;
        };
        let devices: Vec<GpuDevice> = text.lines().filter_map(parse_device_line).collect();
        if !devices.is_empty() {
            return devices;
        }
    }
    Vec::new()
}

/// Parse one CSV row of a device query.
///
/// ## Parameters
/// - `line`: A single `--format=csv,noheader` row, with or without the
///   `compute_cap` column.
///
/// ## Returns
/// The parsed device, or `None` when the row has an unexpected shape or a
/// numeric field fails to parse.
fn parse_device_line(line: &str) -> Option<GpuDevice> {
    let fields: Vec<&str> = line.split(',').map(str::trim).collect();
    let (index, name, capability, total, used) = match fields.as_slice() {
        [index, name, capability, total, used] => (index, name, Some(*capability), total, used),
        [index, name, total, used] => (index, name, None, total, used),
        _ => return None,
    };
    Some(GpuDevice {
        index: index.parse().ok()?,
        name: name.to_string(),
        compute_capability: capability
            .filter(|cap| !cap.is_empty() && *cap != "N/A")
            .map(str::to_string),
        memory_total_mib: total.parse().ok()?,
        memory_used_mib: used.parse().ok()?,
    })
}

/// Collect candidate CUDA toolkit installation directories.
///
/// ## Returns
/// `CUDA_PATH` when set, plus the `/usr/local/cuda` symlink and its versioned
/// siblings on Linux. Sorted and deduplicated so the report is stable.
fn cuda_toolkit_roots() -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Some(root) = std::env::var_os("CUDA_PATH") {
        roots.push(PathBuf::from(root));
    }
    if let Ok(entries) = std::fs::read_dir("/usr/local") {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name == "cuda" || name.starts_with("cuda-") {
                roots.push(entry.path());
            }
        }
    }
    roots.sort();
    roots.dedup();
    roots
}

/// Expand toolkit roots into the directories that would hold NVRTC.
///
/// ## Parameters
/// - `roots`: Toolkit installation directories.
///
/// ## Returns
/// Candidate library directories, including the distro-packaged CUDA
/// locations on Linux.
fn nvrtc_search_dirs(roots: &[PathBuf]) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    for root in roots {
        if cfg!(windows) {
            dirs.push(root.join("bin"));
        } else {
            dirs.push(root.join("lib64"));
            dirs.push(root.join("lib"));
        }
    }
    // NVRTC is loaded by name at run time, so the directories the dynamic
    // loader searches are as authoritative as the toolkit layout. On this
    // project's server the library ships inside the Python `nvidia` packages
    // (e.g. `.../site-packages/nvidia/cu13/lib`), which the operator exposes
    // through `LD_LIBRARY_PATH`; `cudarc` needs the same variable, so the
    // probe must not report "NOT FOUND" in a configuration that works.
    if let Some(path) = std::env::var_os(loader_path_var()) {
        dirs.extend(std::env::split_paths(&path));
    }
    if !cfg!(windows) {
        dirs.push(PathBuf::from("/usr/lib/x86_64-linux-gnu"));
        dirs.push(PathBuf::from("/usr/lib64"));
        dirs.push(PathBuf::from("/usr/lib"));
    }
    dirs
}

/// Name of the environment variable the dynamic loader searches on this
/// platform.
///
/// ## Returns
/// `LD_LIBRARY_PATH` on Linux (and other ELF targets), `PATH` on Windows.
fn loader_path_var() -> &'static str {
    if cfg!(windows) {
        "PATH"
    } else {
        "LD_LIBRARY_PATH"
    }
}

/// Test whether a file name is an NVRTC shared library.
///
/// ## Parameters
/// - `name`: Bare file name, not a path.
///
/// ## Returns
/// `true` for `libnvrtc.so*` on Linux and `nvrtc64_*.dll` on Windows.
fn is_nvrtc_name(name: &str) -> bool {
    if cfg!(windows) {
        name.starts_with("nvrtc64_") && name.ends_with(".dll")
    } else {
        name.starts_with("libnvrtc.so")
    }
}

/// Return the first entry across `dirs` whose name satisfies `matches`.
///
/// ## Parameters
/// - `dirs`: Directories to scan; unreadable ones are skipped.
/// - `matches`: Predicate over the bare file name.
///
/// ## Returns
/// The full path of the first match, or `None`.
fn search_dirs<P>(dirs: &[PathBuf], matches: P) -> Option<PathBuf>
where
    P: Fn(&str) -> bool,
{
    for dir in dirs {
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            if matches(&name.to_string_lossy()) {
                return Some(entry.path());
            }
        }
    }
    None
}

#[cfg(test)]
#[path = "../../tests/unit/gpu/probe.rs"]
mod tests;
