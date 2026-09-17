//! The device executor: uploaded state, kernel launches, and downloads.
//!
//! For P2 the executor owns the full individual/sperm state on the device and
//! can run the aging stage. Later phases add the other deterministic kernels
//! and the sampling kernels, at which point [`GpuExecutor`] becomes the object
//! a session drives for a whole tick.
//!
//! ## Batch-minor state
//!
//! Uploads transpose to the device layout once ([`super::layout`]); every
//! kernel then sees the batch axis innermost, and a download transposes back
//! once. `ind` is `(2, A, Z, B)` and `sperm` is `(A, Z, Z, B)`.
//!
//! ## Memory budget (§D5)
//!
//! [`GpuExecutor::new`] checks the *measured* free device memory against the
//! double-buffered state size before allocating. An over-budget run fails
//! explicitly; the executor never falls back to the CPU, because silently
//! switching engines mid-run would produce a trajectory neither engine can
//! reproduce.

use std::mem::size_of;
use std::sync::Arc;

use cudarc::driver::CudaStream;

use crate::gpu::buffers::DeviceBuffer;
use crate::gpu::context::GpuContext;
use crate::gpu::kernels::{DensityBuffers, Kernels};
use crate::gpu::layout::{batch_to_inner, batch_to_outer};
use crate::model::blueprint::Blueprint;
use crate::model::ecology::EcologyParams;

/// Owns the device state for one batched model instance.
pub struct GpuExecutor {
    /// The context and default stream every operation is ordered on.
    context: GpuContext,
    /// NVRTC-compiled kernels, loaded once.
    kernels: Kernels,
    /// Number of batch elements (demes or replicates).
    n_batch: usize,
    /// Length of the age axis.
    n_ages: usize,
    /// Length of the zygote-type axis.
    n_ztypes: usize,
    /// Batch-minor individual counts, `(2, A, Z, B)`.
    ind: DeviceBuffer<f32>,
    /// Aging scratch, swapped into `ind` after each aging launch.
    ind_scratch: DeviceBuffer<f32>,
    /// Batch-minor stored sperm, `(A, Z, Z, B)`.
    sperm: DeviceBuffer<f32>,
    /// Aging scratch, swapped into `sperm` after each aging launch.
    sperm_scratch: DeviceBuffer<f32>,
}

impl GpuExecutor {
    /// Upload the initial state and load the kernels.
    ///
    /// ## Parameters
    /// - `context`: An owned device context.
    /// - `n_batch`, `n_ages`, `n_ztypes`: Model dimensions.
    /// - `ind_host`: Batch-major individual counts, `(B, 2, A, Z)`.
    /// - `sperm_host`: Batch-major stored sperm, `(B, A, Z, Z)`.
    ///
    /// ## Returns
    /// A ready executor, or a description of the first failure.
    ///
    /// ## Errors
    /// Returns a description when the host slices have the wrong length, the
    /// measured free memory is below the double-buffered state size, or any
    /// upload/compile step fails.
    pub fn new(
        context: GpuContext,
        n_batch: usize,
        n_ages: usize,
        n_ztypes: usize,
        ind_host: &[f32],
        sperm_host: &[f32],
    ) -> Result<Self, String> {
        let expected_ind = 2 * n_ages * n_ztypes * n_batch;
        let expected_sperm = n_ages * n_ztypes * n_ztypes * n_batch;
        if ind_host.len() != expected_ind {
            return Err(format!(
                "GpuExecutor individual state needs {expected_ind} elements, got {}",
                ind_host.len()
            ));
        }
        if sperm_host.len() != expected_sperm {
            return Err(format!(
                "GpuExecutor sperm state needs {expected_sperm} elements, got {}",
                sperm_host.len()
            ));
        }
        // Double-buffered state: each plane needs a scratch for out-of-place
        // aging. Check before allocating so an over-budget run fails cleanly.
        let required = 2 * Self::state_bytes(n_batch, n_ages, n_ztypes);
        let (free, _total) = context.memory_info()?;
        if required > free {
            return Err(format!(
                "GPU memory budget exceeded: the state needs {} MiB ({required} B), \
                 but only {} MiB is free; reduce the batch size or free the device",
                required / (1024 * 1024),
                free / (1024 * 1024)
            ));
        }
        let ind_axes = [2usize, n_ages, n_ztypes];
        let sperm_axes = [n_ages, n_ztypes, n_ztypes];
        let ind_inner = batch_to_inner(ind_host, &ind_axes, n_batch)?;
        let sperm_inner = batch_to_inner(sperm_host, &sperm_axes, n_batch)?;
        let stream = context.stream();
        let ind = DeviceBuffer::from_host(&stream, &ind_inner)?;
        let ind_scratch = DeviceBuffer::from_host(&stream, &vec![0.0f32; ind_inner.len()])?;
        let sperm = DeviceBuffer::from_host(&stream, &sperm_inner)?;
        let sperm_scratch = DeviceBuffer::from_host(&stream, &vec![0.0f32; sperm_inner.len()])?;
        let kernels = Kernels::load(&context.context())?;
        Ok(Self {
            context,
            kernels,
            n_batch,
            n_ages,
            n_ztypes,
            ind,
            ind_scratch,
            sperm,
            sperm_scratch,
        })
    }

    /// Bytes needed for one individual + sperm state copy, without scratch.
    ///
    /// ## Parameters
    /// - `n_batch`, `n_ages`, `n_ztypes`: Model dimensions.
    ///
    /// ## Returns
    /// `(2·A·Z + A·Z²) · B · 4` bytes, the §D5 budget input.
    pub fn state_bytes(n_batch: usize, n_ages: usize, n_ztypes: usize) -> usize {
        let ind = 2 * n_ages * n_ztypes;
        let sperm = n_ages * n_ztypes * n_ztypes;
        (ind + sperm) * n_batch * size_of::<f32>()
    }

    /// Run one aging stage on the device.
    ///
    /// Every individual and sperm age class shifts down one slot and age 0 is
    /// zeroed, exactly as `kernels::age_structured::aging` does on the host.
    /// The scratch buffers make the shift race-free; each plane's scratch is
    /// swapped in on success.
    ///
    /// ## Returns
    /// `Ok(())` once both planes are enqueued on the stream.
    ///
    /// ## Errors
    /// Returns a description when a launch is rejected.
    pub fn age_tick(&mut self) -> Result<(), String> {
        let stream = self.context.stream();
        let ind_stride = self.n_ztypes * self.n_batch;
        self.kernels.age_shift(
            &stream,
            self.ind.slice(),
            self.ind_scratch.slice_mut(),
            self.ind.len(),
            self.n_ages,
            ind_stride,
        )?;
        std::mem::swap(&mut self.ind, &mut self.ind_scratch);
        let sperm_stride = self.n_ztypes * self.n_ztypes * self.n_batch;
        self.kernels.age_shift(
            &stream,
            self.sperm.slice(),
            self.sperm_scratch.slice_mut(),
            self.sperm.len(),
            self.n_ages,
            sperm_stride,
        )?;
        std::mem::swap(&mut self.sperm, &mut self.sperm_scratch);
        Ok(())
    }

    /// Compute the per-batch density-regulation scaling factor.
    ///
    /// This is the first half of the survival stage: it evaluates the
    /// equilibrium metrics and the growth curve on the device and returns the
    /// `B` scaling factors. It does not mutate the state, so it can be
    /// cross-checked against the host reference in isolation.
    ///
    /// ## Parameters
    /// - `blueprint`: Dimensions and `new_adult_age`.
    /// - `ecology`: Per-deme ecology columns; `n_demes` must equal the batch.
    ///
    /// ## Returns
    /// One scaling factor per batch element (as `f32`).
    ///
    /// ## Errors
    /// Returns a description when the dimensions or column lengths disagree,
    /// when the growth mode has a custom curve (`>= 5`, not portable), or when
    /// a launch fails.
    pub fn density_scaling(
        &self,
        blueprint: &Blueprint,
        ecology: &EcologyParams,
    ) -> Result<Vec<f32>, String> {
        let n_batch = self.n_batch;
        let n_ages = self.n_ages;
        let n_ztypes = self.n_ztypes;
        if blueprint.n_ages != n_ages || blueprint.n_ztypes != n_ztypes {
            return Err(format!(
                "executor dimensions (A={n_ages}, Z={n_ztypes}) disagree with the blueprint \
                 (A={}, Z={})",
                blueprint.n_ages, blueprint.n_ztypes
            ));
        }
        if ecology.n_demes != n_batch {
            return Err(format!(
                "executor batch is {n_batch} but the ecology carries {} deme columns",
                ecology.n_demes
            ));
        }
        expect_len(
            "survival_rates",
            ecology.survival_rates.len(),
            n_batch * 2 * n_ages,
        )?;
        expect_len(
            "reproduction_rates",
            ecology.reproduction_rates.len(),
            n_batch * n_ages,
        )?;
        expect_len("fertility", ecology.fertility.len(), n_batch * n_ages)?;
        expect_len(
            "competition_weights",
            ecology.competition_weights.len(),
            n_batch * n_ages,
        )?;
        expect_len(
            "equilibrium_declared",
            ecology.equilibrium_declared.len(),
            n_batch,
        )?;
        if !ecology.equilibrium_distribution.is_empty() {
            expect_len(
                "equilibrium_distribution",
                ecology.equilibrium_distribution.len(),
                n_batch * 2 * n_ages,
            )?;
        }
        if let Some((deme, mode)) = ecology
            .growth_mode
            .iter()
            .enumerate()
            .find(|(_, mode)| **mode >= 5)
        {
            return Err(format!(
                "growth mode {mode} at deme {deme} is a custom curve and is not portable to the device"
            ));
        }
        let declared: Vec<i32> = ecology
            .equilibrium_declared
            .iter()
            .map(|&flag| i32::from(flag))
            .collect();
        let mut distribution = vec![0.0f32; n_batch * 2 * n_ages];
        for (dst, src) in distribution
            .iter_mut()
            .zip(&ecology.equilibrium_distribution)
        {
            *dst = *src as f32;
        }
        let growth: Vec<i32> = ecology
            .growth_mode
            .iter()
            .map(|mode| *mode as i32)
            .collect();
        let stream = self.context.stream();
        let survival = upload_f64(&stream, &ecology.survival_rates)?;
        let reproduction = upload_f64(&stream, &ecology.reproduction_rates)?;
        let fertility = upload_f64(&stream, &ecology.fertility)?;
        let competition = upload_f64(&stream, &ecology.competition_weights)?;
        let declared_buf = DeviceBuffer::from_host(&stream, &declared)?;
        let distribution_buf = DeviceBuffer::from_host(&stream, &distribution)?;
        let external = upload_f64(&stream, &ecology.external_expected_eggs)?;
        let carrying = upload_f64(&stream, &ecology.carrying_capacity)?;
        let eggs = upload_f64(&stream, &ecology.eggs_per_female)?;
        let sex_ratio = upload_f64(&stream, &ecology.sex_ratio)?;
        let growth_buf = DeviceBuffer::from_host(&stream, &growth)?;
        let low_density = upload_f64(&stream, &ecology.low_density_growth_rate)?;
        let mut scaling = DeviceBuffer::from_host(&stream, &vec![0.0f32; n_batch])?;
        self.kernels.density_scaling(
            &stream,
            &mut DensityBuffers {
                ind: self.ind.slice(),
                survival_rates: survival.slice(),
                reproduction_rates: reproduction.slice(),
                fertility: fertility.slice(),
                competition_weights: competition.slice(),
                equilibrium_declared: declared_buf.slice(),
                equilibrium_distribution: distribution_buf.slice(),
                external_expected_eggs: external.slice(),
                carrying_capacity: carrying.slice(),
                eggs_per_female: eggs.slice(),
                sex_ratio: sex_ratio.slice(),
                growth_mode: growth_buf.slice(),
                low_density_growth_rate: low_density.slice(),
                scaling_out: scaling.slice_mut(),
            },
            n_batch,
            n_ages,
            n_ztypes,
            blueprint.new_adult_age,
        )?;
        scaling.to_host(&stream)
    }

    /// Copy the individual state back in the batch-major CPU layout.
    ///
    /// ## Returns
    /// `(B, 2, A, Z)` individual counts.
    pub fn download_ind(&self) -> Result<Vec<f32>, String> {
        let stream = self.context.stream();
        let inner = self.ind.to_host(&stream)?;
        batch_to_outer(&inner, &[2, self.n_ages, self.n_ztypes], self.n_batch)
    }

    /// Copy the stored sperm back in the batch-major CPU layout.
    ///
    /// ## Returns
    /// `(B, A, Z, Z)` stored sperm.
    pub fn download_sperm(&self) -> Result<Vec<f32>, String> {
        let stream = self.context.stream();
        let inner = self.sperm.to_host(&stream)?;
        batch_to_outer(
            &inner,
            &[self.n_ages, self.n_ztypes, self.n_ztypes],
            self.n_batch,
        )
    }

    /// The executor's context, for later stages and diagnostics.
    ///
    /// ## Returns
    /// The device context the executor was built with.
    pub fn context(&self) -> &GpuContext {
        &self.context
    }

    /// Number of batch elements.
    ///
    /// ## Returns
    /// The batch axis length.
    pub fn n_batch(&self) -> usize {
        self.n_batch
    }
}

/// Upload a host `f64` slice as an `f32` device buffer (`f64 → f32` per §D2).
///
/// ## Parameters
/// - `stream`: Stream the transfer is ordered on.
/// - `values`: Host values to narrow and upload.
///
/// ## Returns
/// The device buffer, or a description of the transfer failure.
fn upload_f64(stream: &Arc<CudaStream>, values: &[f64]) -> Result<DeviceBuffer<f32>, String> {
    let host: Vec<f32> = values.iter().map(|value| *value as f32).collect();
    DeviceBuffer::from_host(stream, &host)
}

/// Validate an ecology column length.
///
/// ## Parameters
/// - `name`: Column name for the error message.
/// - `got`: Observed length.
/// - `expected`: Required length.
///
/// ## Errors
/// Returns a readable description when the lengths differ.
fn expect_len(name: &str, got: usize, expected: usize) -> Result<(), String> {
    if got != expected {
        return Err(format!(
            "ecology.{name}: expected {expected} elements, got {got}"
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../tests/unit/gpu/executor.rs"]
mod tests;
