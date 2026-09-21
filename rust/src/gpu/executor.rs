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
use crate::gpu::kernels::{DensityBuffers, Kernels, ReproductionBuffers, MAX_CSR_ROW};
use crate::gpu::layout::{batch_to_inner, batch_to_outer};
use crate::model::blueprint::Blueprint;
use crate::model::ecology::EcologyParams;
use crate::model::genetics::GeneticsTensors;

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
    /// Sampling seed for the counter-based device RNG.
    seed: u64,
    /// Number of device ticks completed, used to vary draw sites per tick.
    tick: u64,
    /// Cached static migration buffers for the current CSR, if any.
    migration_cache: Option<MigrationCache>,
    /// Device-staged observation history rows for the current run, if any.
    history: Option<DeviceHistory>,
}

/// Device-resident static buffers for one migration CSR.
///
/// The migration graph is frozen in the blueprint, so its forward arrays and
/// the reverse CSR never change between ticks. Building them once removes the
/// per-tick host rebuild and the repeated host-to-device uploads; the
/// tick-varying `migration_rate` column is still transferred every call.
///
/// The stochastic scratch (`fwd_*`) is safe to reuse because the prepare
/// kernel writes every entry's slot for every age and zygote type on each
/// launch, including explicit zeros, so no stale sample can be read.
struct MigrationCache {
    /// Hash of the CSR contents the cache was built from.
    fingerprint: u64,
    /// Forward row pointers, `n_batch + 1`.
    indptr: DeviceBuffer<i32>,
    /// Reverse row pointers, `n_batch + 1`.
    rev_indptr: DeviceBuffer<i32>,
    /// Deterministic reverse source indices, one per CSR entry.
    rev_src: DeviceBuffer<i32>,
    /// Deterministic reverse weights, one per CSR entry.
    rev_weight: DeviceBuffer<f32>,
    /// Deterministic source-row weight sums, one per source.
    row_sum: DeviceBuffer<f32>,
    /// Forward destinations for the stochastic pass, one per CSR entry.
    dest: DeviceBuffer<i32>,
    /// Forward weights for the stochastic pass, one per CSR entry.
    weights: DeviceBuffer<f32>,
    /// Stochastic reverse CSR entry indices, one per CSR entry.
    rev_entry: DeviceBuffer<i32>,
    /// Stochastic female forward scratch, `nnz · A · Z`.
    fwd_f: DeviceBuffer<f32>,
    /// Stochastic sperm forward scratch, `nnz · A · Z · Z`.
    fwd_s: DeviceBuffer<f32>,
    /// Stochastic male forward scratch, `nnz · A · Z`.
    fwd_m: DeviceBuffer<f32>,
}

/// Host description of one device-side history projection window.
///
/// The session builds this from the bound [`crate::output::history::HistoryData`]
/// configuration so the device rows match the host observation projection.
pub struct HistorySpec {
    /// Number of record rows this window may hold.
    pub capacity: usize,
    /// Values per row, excluding the host-added tick column.
    pub width: usize,
    /// Population dimensions `[n_demes, n_sexes, n_ages, n_ztypes]`.
    pub dims: [usize; 4],
    /// Observation weights, `groups · S · A · Z`.
    pub mask: Vec<f64>,
    /// Deme indices retained by the projection.
    pub selected: Vec<usize>,
    /// Collapse the age axis to one output column.
    pub collapse: bool,
    /// Aggregate all selected demes into one output column.
    pub aggregate: bool,
    /// Stage raw state rows (batch-major on the host) instead of observation
    /// projections. Raw rows hold the device-native `ind`/`sperm` planes and are
    /// transposed on flush.
    pub raw: bool,
    /// Raw rows include the stored-sperm plane. Discrete demes carry none.
    pub raw_sperm: bool,
    /// Treat `capacity` as a bounded ring. Set when the window was clamped by
    /// `max_rows`, so exceeding it overwrites the oldest row instead of
    /// failing. Exact windows keep the hard "full" guard.
    pub wrap: bool,
}

/// Device-resident history rows for the current run.
///
/// Observation rows are projected on the device and raw rows copy the state
/// planes; either way they are downloaded once, so a long run does not
/// synchronize host state at every record tick. `wrap` rows live in a ring
/// whose oldest entry is overwritten once a `max_rows`-clamped window fills.
struct DeviceHistory {
    /// Values per row, excluding the tick column.
    width: usize,
    /// Allocated row capacity.
    capacity: usize,
    /// Valid rows currently held (`<= capacity`).
    rows: usize,
    /// Ring index of the oldest valid row.
    head: usize,
    /// Total rows ever recorded; `written - rows` were overwritten.
    written: usize,
    /// Stage raw state rows instead of observation projections.
    raw: bool,
    /// Raw rows include the stored-sperm plane.
    raw_sperm: bool,
    /// Overwrite the oldest row once full instead of erroring.
    wrap: bool,
    /// Number of observation groups.
    n_groups: usize,
    /// `n_sexes · n_ages · n_ztypes`.
    plane: usize,
    /// Number of selected demes.
    n_selected: usize,
    /// Sex-axis length.
    n_sexes: usize,
    /// Age-axis length.
    n_ages: usize,
    /// Zygote-type axis length.
    n_ztypes: usize,
    /// Output deme extent (`1` when aggregating).
    out_d: usize,
    /// Output age extent (`1` when collapsing).
    out_a: usize,
    /// Age-axis collapse flag.
    collapse: bool,
    /// Deme aggregation flag.
    aggregate: bool,
    /// Observation weights uploaded as `f32` (empty for raw windows).
    mask: DeviceBuffer<f32>,
    /// Selected deme indices uploaded as `i32` (empty for raw windows).
    selected: DeviceBuffer<i32>,
    /// Row-major row storage, `capacity · width`.
    buffer: DeviceBuffer<f32>,
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
        ensure_memory_budget(free, required)?;
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
            seed: 0,
            tick: 0,
            migration_cache: None,
            history: None,
        })
    }

    /// Set the sampling seed for the counter-based device RNG.
    ///
    /// ## Parameters
    /// - `seed`: Session seed; the two Philox key words are its halves.
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
    }

    /// Build an executor running `n_replicates` independent copies of one
    /// panmictic model.
    ///
    /// Every replicate is a batch element; the counter-based RNG gives each a
    /// disjoint stream, so the replicates are statistically independent yet
    /// reproducible from `seed`.
    ///
    /// ## Parameters
    /// - `context`: An owned device context.
    /// - `n_replicates`: Number of independent trajectories.
    /// - `n_ages`, `n_ztypes`: Model dimensions.
    /// - `ind_one`, `sperm_one`: One replicate's initial state (batch-major).
    /// - `seed`: Ensemble seed.
    ///
    /// ## Returns
    /// A ready executor with `n_batch == n_replicates`.
    ///
    /// ## Errors
    /// Returns a description when the single-replicate slices are the wrong
    /// size or the underlying upload fails.
    pub fn ensemble(
        context: GpuContext,
        n_replicates: usize,
        n_ages: usize,
        n_ztypes: usize,
        ind_one: &[f32],
        sperm_one: &[f32],
        seed: u64,
    ) -> Result<Self, String> {
        if ind_one.len() != 2 * n_ages * n_ztypes {
            return Err(format!(
                "ensemble individual state needs {} elements per replicate, got {}",
                2 * n_ages * n_ztypes,
                ind_one.len()
            ));
        }
        if sperm_one.len() != n_ages * n_ztypes * n_ztypes {
            return Err(format!(
                "ensemble sperm state needs {} elements per replicate, got {}",
                n_ages * n_ztypes * n_ztypes,
                sperm_one.len()
            ));
        }
        let ind_all: Vec<f32> = (0..n_replicates)
            .flat_map(|_| ind_one.iter().copied())
            .collect();
        let sperm_all: Vec<f32> = (0..n_replicates)
            .flat_map(|_| sperm_one.iter().copied())
            .collect();
        let mut executor = Self::new(
            context,
            n_replicates,
            n_ages,
            n_ztypes,
            &ind_all,
            &sperm_all,
        )?;
        executor.set_seed(seed);
        Ok(executor)
    }

    /// The two Philox key words derived from the executor seed.
    ///
    /// ## Returns
    /// `(low, high)` 32-bit words.
    fn rng_key(&self) -> (u32, u32) {
        (self.seed as u32, (self.seed >> 32) as u32)
    }

    /// Draw-site counter for a given stage within the current tick.
    ///
    /// ## Parameters
    /// - `stage`: Small per-stage index that must be stable across ticks.
    ///
    /// ## Returns
    /// A 32-bit site that changes every tick.
    fn rng_site(&self, stage: u32) -> u32 {
        (self.tick as u32).wrapping_mul(16).wrapping_add(stage)
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
        self.density_scaling_impl(blueprint, ecology, false)
    }

    /// Device-resident density scaling (no host round-trip).
    ///
    /// `survival_tick`/`discrete_survival_tick` use this so a deterministic
    /// tick never synchronizes on the scaling factor. The host-returning
    /// [`GpuExecutor::density_scaling`] wrapper remains for callers and tests
    /// that need the values.
    ///
    /// ## Parameters
    /// - `blueprint`, `ecology`: Live contracts.
    /// - `discrete_actual`: When `true`, compare the total age-0 count (the
    ///   discrete-generation rule) instead of the competition-weighted
    ///   age-structured juvenile count.
    ///
    /// ## Errors
    /// As [`GpuExecutor::density_scaling`].
    pub(crate) fn density_scaling_device(
        &self,
        blueprint: &Blueprint,
        ecology: &EcologyParams,
        discrete_actual: bool,
    ) -> Result<DeviceBuffer<f32>, String> {
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
        expect_len(
            "external_expected_eggs",
            ecology.external_expected_eggs.len(),
            n_batch,
        )?;
        expect_len(
            "carrying_capacity",
            ecology.carrying_capacity.len(),
            n_batch,
        )?;
        expect_len("eggs_per_female", ecology.eggs_per_female.len(), n_batch)?;
        expect_len("sex_ratio", ecology.sex_ratio.len(), n_batch)?;
        expect_len(
            "low_density_growth_rate",
            ecology.low_density_growth_rate.len(),
            n_batch,
        )?;
        expect_len("growth_mode", ecology.growth_mode.len(), n_batch)?;
        if let Some((deme, mode)) = ecology
            .growth_mode
            .iter()
            .enumerate()
            .find(|(_, mode)| !(0..=4).contains(*mode))
        {
            return Err(format!(
                "growth mode {mode} at deme {deme} is outside the portable range 0..=4 \
                 (custom curves are not supported on the device)"
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
            discrete_actual,
        )?;
        Ok(scaling)
    }

    /// Host copy of the device density scaling.
    ///
    /// ## Parameters
    /// - `blueprint`, `ecology`: Live contracts.
    /// - `discrete_actual`: Discrete vs age-structured `actual` rule.
    ///
    /// ## Errors
    /// As [`GpuExecutor::density_scaling`].
    pub(crate) fn density_scaling_impl(
        &self,
        blueprint: &Blueprint,
        ecology: &EcologyParams,
        discrete_actual: bool,
    ) -> Result<Vec<f32>, String> {
        let scaling = self.density_scaling_device(blueprint, ecology, discrete_actual)?;
        let stream = self.context.stream();
        scaling.to_host(&stream)
    }

    /// Run one deterministic survival stage (density regulation + recruitment
    /// + combined-rate scaling) on the device.
    ///
    /// Mirrors the host `survival` pipeline for `stochastic == false`:
    /// [`GpuExecutor::density_scaling`] produces the factor, age-0 counts are
    /// rescaled, and every individual and sperm category is multiplied by
    /// `age_survival × viability`.
    ///
    /// ## Parameters
    /// - `blueprint`: Dimensions and `new_adult_age`.
    /// - `ecology`: Per-deme ecology columns (`n_demes == n_batch`).
    /// - `variants`: Shared genetics variant bank.
    /// - `deme_variants`: Per-batch index into `variants`.
    ///
    /// ## Returns
    /// `Ok(())` after the state is rescaled in place on the device.
    ///
    /// ## Errors
    /// Returns a description when dimensions disagree, a variant is missing,
    /// or a launch fails.
    pub fn survival_tick(
        &mut self,
        blueprint: &Blueprint,
        ecology: &EcologyParams,
        variants: &[GeneticsTensors],
        deme_variants: &[usize],
    ) -> Result<(), String> {
        let n_batch = self.n_batch;
        let n_ages = self.n_ages;
        let n_ztypes = self.n_ztypes;
        if deme_variants.len() != n_batch {
            return Err(format!(
                "survival_tick needs {n_batch} deme-variant ids, got {}",
                deme_variants.len()
            ));
        }
        let scaling = self.density_scaling_device(blueprint, ecology, false)?;
        let variant_stride = 2 * n_ages * n_ztypes;
        let mut viability = vec![0.0f32; n_batch * variant_stride];
        for (batch, &variant_id) in deme_variants.iter().enumerate() {
            let genetics = variants.get(variant_id).ok_or_else(|| {
                format!("deme {batch} references missing genetics variant {variant_id}")
            })?;
            if genetics.viability_fitness.len() != variant_stride {
                return Err(format!(
                    "variant {variant_id} viability_fitness has {} elements, expected {variant_stride}",
                    genetics.viability_fitness.len()
                ));
            }
            let dst = &mut viability[batch * variant_stride..(batch + 1) * variant_stride];
            for (slot, value) in dst.iter_mut().zip(&genetics.viability_fitness) {
                *slot = *value as f32;
            }
        }
        let stream = self.context.stream();
        let survival_rates = upload_f64(&stream, &ecology.survival_rates)?;
        let viability_buf = DeviceBuffer::from_host(&stream, &viability)?;
        if blueprint.stochastic {
            let (key0, key1) = self.rng_key();
            let recruit_site = self.rng_site(0);
            let survival_site = self.rng_site(1);
            self.kernels.recruit_stochastic(
                &stream,
                self.ind.slice_mut(),
                scaling.slice(),
                n_batch,
                n_ages,
                n_ztypes,
                blueprint.continuous_sampling,
                key0,
                key1,
                recruit_site,
            )?;
            let mut violation = DeviceBuffer::from_host(&stream, &[0i32])?;
            self.kernels.survival_stochastic(
                &stream,
                self.ind.slice_mut(),
                self.sperm.slice_mut(),
                survival_rates.slice(),
                viability_buf.slice(),
                n_batch,
                n_ages,
                n_ztypes,
                blueprint.new_adult_age,
                blueprint.continuous_sampling,
                violation.slice_mut(),
                key0,
                key1,
                survival_site,
            )?;
            // Surface a meaningfully negative virgin count instead of
            // silently clamping it, matching the host's explicit error.
            if violation.to_host(&stream)?.first().copied().unwrap_or(0) != 0 {
                return Err("Invalid state: n_virgins < 0 in GPU stochastic survival".to_owned());
            }
            return Ok(());
        }
        let mut factor = DeviceBuffer::from_host(&stream, &vec![0.0f32; n_batch])?;
        self.kernels.recruit_factor(
            &stream,
            self.ind.slice(),
            scaling.slice(),
            factor.slice_mut(),
            n_batch,
            n_ages,
            n_ztypes,
        )?;
        self.kernels.survival_scale_ind(
            &stream,
            self.ind.slice_mut(),
            factor.slice(),
            survival_rates.slice(),
            viability_buf.slice(),
            n_batch,
            n_ages,
            n_ztypes,
            blueprint.new_adult_age,
        )?;
        self.kernels.survival_scale_sperm(
            &stream,
            self.sperm.slice_mut(),
            survival_rates.slice(),
            viability_buf.slice(),
            n_batch,
            n_ages,
            n_ztypes,
            blueprint.new_adult_age,
        )?;
        Ok(())
    }

    /// Run one deterministic reproduction stage on the device.
    ///
    /// Mirrors the host `reproduction` pipeline for `stochastic == false`:
    /// effective adult males, the mating matrix, sperm displacement, and
    /// fertilization from the offspring tensor, with zygote viability applied
    /// to the newborn age class.
    ///
    /// ## Parameters
    /// - `blueprint`: Dimensions, `new_adult_age`, and sex-chromosome flags.
    /// - `ecology`: Per-deme ecology columns (`n_demes == n_batch`).
    /// - `variants`: Shared genetics variant bank.
    /// - `deme_variants`: Per-batch index into `variants`.
    ///
    /// ## Returns
    /// `Ok(())` after the state is updated in place on the device.
    ///
    /// ## Errors
    /// Returns a description when dimensions disagree, a variant is missing or
    /// malformed, or a launch fails.
    pub fn reproduction_tick(
        &mut self,
        blueprint: &Blueprint,
        ecology: &EcologyParams,
        variants: &[GeneticsTensors],
        deme_variants: &[usize],
    ) -> Result<(), String> {
        self.reproduction_impl(blueprint, ecology, variants, deme_variants, false)
    }

    /// Run one discrete-generation reproduction stage on the device.
    ///
    /// Mirrors `kernels::discrete_generation::reproduction`: adult-female
    /// matings are distributed over adult males and fertilized into age-0
    /// offspring, with discrete or continuous sampling.
    ///
    /// ## Errors
    /// As [`GpuExecutor::reproduction_tick`].
    pub fn discrete_reproduction_tick(
        &mut self,
        blueprint: &Blueprint,
        ecology: &EcologyParams,
        variants: &[GeneticsTensors],
        deme_variants: &[usize],
    ) -> Result<(), String> {
        self.reproduction_impl(blueprint, ecology, variants, deme_variants, true)
    }

    /// Shared reproduction body; `discrete` selects the lifecycle kernel.
    #[allow(clippy::too_many_arguments)]
    fn reproduction_impl(
        &mut self,
        blueprint: &Blueprint,
        ecology: &EcologyParams,
        variants: &[GeneticsTensors],
        deme_variants: &[usize],
        discrete: bool,
    ) -> Result<(), String> {
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
        if deme_variants.len() != n_batch {
            return Err(format!(
                "reproduction_tick needs {n_batch} deme-variant ids, got {}",
                deme_variants.len()
            ));
        }
        expect_len(
            "mating_rates",
            ecology.mating_rates.len(),
            n_batch * 2 * n_ages,
        )?;
        expect_len(
            "sperm_displacement_rate",
            ecology.sperm_displacement_rate.len(),
            n_batch,
        )?;
        expect_len(
            "reproduction_rates",
            ecology.reproduction_rates.len(),
            n_batch * n_ages,
        )?;
        expect_len("fertility", ecology.fertility.len(), n_batch * n_ages)?;
        expect_len("eggs_per_female", ecology.eggs_per_female.len(), n_batch)?;
        expect_len("sex_ratio", ecology.sex_ratio.len(), n_batch)?;
        let z2 = n_ztypes * n_ztypes;
        let z3 = z2 * n_ztypes;
        let mut fecundity = vec![0.0f32; n_batch * 2 * n_ztypes];
        let mut sexual_selection = vec![0.0f32; n_batch * z2];
        let mut offspring = vec![0.0f32; n_batch * z3];
        let mut zygote = vec![0.0f32; n_batch * 2 * n_ztypes];
        let mut female_compat = vec![0.0f32; n_batch * n_ztypes];
        let mut male_compat = vec![0.0f32; n_batch * n_ztypes];
        for (batch, &variant_id) in deme_variants.iter().enumerate() {
            let genetics = variants.get(variant_id).ok_or_else(|| {
                format!("deme {batch} references missing genetics variant {variant_id}")
            })?;
            let base2 = batch * 2 * n_ztypes;
            copy_narrow(
                &mut fecundity[base2..base2 + 2 * n_ztypes],
                &genetics.fecundity_fitness,
                "fecundity_fitness",
                variant_id,
            )?;
            copy_narrow(
                &mut zygote[base2..base2 + 2 * n_ztypes],
                &genetics.zygote_viability_fitness,
                "zygote_viability_fitness",
                variant_id,
            )?;
            let basez = batch * n_ztypes;
            copy_narrow(
                &mut female_compat[basez..basez + n_ztypes],
                &genetics.female_ztype_compatibility,
                "female_ztype_compatibility",
                variant_id,
            )?;
            copy_narrow(
                &mut male_compat[basez..basez + n_ztypes],
                &genetics.male_ztype_compatibility,
                "male_ztype_compatibility",
                variant_id,
            )?;
            let basezz = batch * z2;
            copy_narrow(
                &mut sexual_selection[basezz..basezz + z2],
                &genetics.sexual_selection_fitness,
                "sexual_selection_fitness",
                variant_id,
            )?;
            let basezzz = batch * z3;
            copy_narrow(
                &mut offspring[basezzz..basezzz + z3],
                &genetics.offspring_tensor,
                "offspring_tensor",
                variant_id,
            )?;
        }
        if blueprint.female_only_by_sex_chrom.len() != n_ztypes
            || blueprint.male_only_by_sex_chrom.len() != n_ztypes
        {
            return Err(format!(
                "reproduction_tick needs {n_ztypes} sex-chromosome flags per sex, got {} female and {} male",
                blueprint.female_only_by_sex_chrom.len(),
                blueprint.male_only_by_sex_chrom.len()
            ));
        }
        let female_only: Vec<i32> = blueprint
            .female_only_by_sex_chrom
            .iter()
            .map(|&flag| i32::from(flag))
            .collect();
        let male_only: Vec<i32> = blueprint
            .male_only_by_sex_chrom
            .iter()
            .map(|&flag| i32::from(flag))
            .collect();
        let stream = self.context.stream();
        let mating_rates = upload_f64(&stream, &ecology.mating_rates)?;
        let displacement = upload_f64(&stream, &ecology.sperm_displacement_rate)?;
        let reproduction_rates = upload_f64(&stream, &ecology.reproduction_rates)?;
        let fertility = upload_f64(&stream, &ecology.fertility)?;
        let eggs = upload_f64(&stream, &ecology.eggs_per_female)?;
        let sex_ratio = upload_f64(&stream, &ecology.sex_ratio)?;
        let female_only_buf = DeviceBuffer::from_host(&stream, &female_only)?;
        let male_only_buf = DeviceBuffer::from_host(&stream, &male_only)?;
        let fecundity_buf = DeviceBuffer::from_host(&stream, &fecundity)?;
        let sexual_selection_buf = DeviceBuffer::from_host(&stream, &sexual_selection)?;
        let offspring_buf = DeviceBuffer::from_host(&stream, &offspring)?;
        let zygote_buf = DeviceBuffer::from_host(&stream, &zygote)?;
        let female_compat_buf = DeviceBuffer::from_host(&stream, &female_compat)?;
        let male_compat_buf = DeviceBuffer::from_host(&stream, &male_compat)?;
        let reproduction_buffers = ReproductionBuffers {
            mating_rates: mating_rates.slice(),
            sperm_displacement_rate: displacement.slice(),
            reproduction_rates: reproduction_rates.slice(),
            fertility: fertility.slice(),
            eggs_per_female: eggs.slice(),
            sex_ratio: sex_ratio.slice(),
            female_only: female_only_buf.slice(),
            male_only: male_only_buf.slice(),
            fecundity: fecundity_buf.slice(),
            sexual_selection: sexual_selection_buf.slice(),
            offspring: offspring_buf.slice(),
            zygote_viability: zygote_buf.slice(),
            female_compat: female_compat_buf.slice(),
            male_compat: male_compat_buf.slice(),
        };
        if discrete {
            let (key0, key1) = self.rng_key();
            let site = self.rng_site(2);
            self.kernels.discrete_reproduction(
                &stream,
                self.ind.slice_mut(),
                &reproduction_buffers,
                n_batch,
                n_ages,
                n_ztypes,
                blueprint.stochastic,
                blueprint.continuous_sampling,
                blueprint.has_sex_chromosomes,
                key0,
                key1,
                site,
            )?;
        } else if blueprint.stochastic {
            let (key0, key1) = self.rng_key();
            let site = self.rng_site(2);
            self.kernels.reproduction_stochastic(
                &stream,
                self.ind.slice_mut(),
                self.sperm.slice_mut(),
                &reproduction_buffers,
                n_batch,
                n_ages,
                n_ztypes,
                blueprint.new_adult_age,
                blueprint.has_sex_chromosomes,
                blueprint.fixed_egg_count,
                blueprint.continuous_sampling,
                key0,
                key1,
                site,
            )?;
        } else {
            self.kernels.reproduction(
                &stream,
                self.ind.slice_mut(),
                self.sperm.slice_mut(),
                &reproduction_buffers,
                n_batch,
                n_ages,
                n_ztypes,
                blueprint.new_adult_age,
                blueprint.has_sex_chromosomes,
            )?;
        }
        Ok(())
    }

    /// Run one full deterministic age-structured tick on the device.
    ///
    /// The stage order mirrors `kernels::age_structured::run_tick` for a
    /// hook-free, deterministic model: reproduction → survival → aging. No
    /// data crosses the host boundary between stages.
    ///
    /// ## Parameters
    /// - `blueprint`: Dimensions, `new_adult_age`, and sex-chromosome flags.
    /// - `ecology`: Per-deme ecology columns (`n_demes == n_batch`).
    /// - `variants`: Shared genetics variant bank.
    /// - `deme_variants`: Per-batch index into `variants`.
    ///
    /// ## Returns
    /// `Ok(())` after the whole tick is enqueued on the stream.
    ///
    /// ## Errors
    /// Returns a description from the first failing stage.
    pub fn tick(
        &mut self,
        blueprint: &Blueprint,
        ecology: &EcologyParams,
        variants: &[GeneticsTensors],
        deme_variants: &[usize],
    ) -> Result<(), String> {
        self.reproduction_tick(blueprint, ecology, variants, deme_variants)?;
        self.survival_tick(blueprint, ecology, variants, deme_variants)?;
        self.age_tick()?;
        self.tick = self.tick.wrapping_add(1);
        Ok(())
    }

    /// Run one discrete-generation survival stage on the device.
    ///
    /// Mirrors `kernels::discrete_generation::survival`: density regulation
    /// (the discrete total age-0 rule) followed by age-0 viability, with
    /// discrete or continuous sampling.
    ///
    /// ## Parameters
    /// - `blueprint`: Dimensions and sampling flags (`n_ages == 2`).
    /// - `ecology`: Per-deme ecology columns (`n_demes == n_batch`).
    /// - `variants`: Shared genetics variant bank.
    /// - `deme_variants`: Per-batch index into `variants`.
    ///
    /// ## Errors
    /// As [`GpuExecutor::survival_tick`].
    pub fn discrete_survival_tick(
        &mut self,
        blueprint: &Blueprint,
        ecology: &EcologyParams,
        variants: &[GeneticsTensors],
        deme_variants: &[usize],
    ) -> Result<(), String> {
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
        if deme_variants.len() != n_batch {
            return Err(format!(
                "discrete_survival_tick needs {n_batch} deme-variant ids, got {}",
                deme_variants.len()
            ));
        }
        expect_len(
            "survival_rates",
            ecology.survival_rates.len(),
            n_batch * 2 * n_ages,
        )?;
        let scaling = self.density_scaling_device(blueprint, ecology, true)?;
        let variant_stride = 2 * n_ages * n_ztypes;
        let mut viability = vec![0.0f32; n_batch * variant_stride];
        for (batch, &variant_id) in deme_variants.iter().enumerate() {
            let genetics = variants.get(variant_id).ok_or_else(|| {
                format!("deme {batch} references missing genetics variant {variant_id}")
            })?;
            if genetics.viability_fitness.len() != variant_stride {
                return Err(format!(
                    "variant {variant_id} viability_fitness has {} elements, expected {variant_stride}",
                    genetics.viability_fitness.len()
                ));
            }
            let dst = &mut viability[batch * variant_stride..(batch + 1) * variant_stride];
            for (slot, value) in dst.iter_mut().zip(&genetics.viability_fitness) {
                *slot = *value as f32;
            }
        }
        let stream = self.context.stream();
        let survival_rates = upload_f64(&stream, &ecology.survival_rates)?;
        let viability_buf = DeviceBuffer::from_host(&stream, &viability)?;
        let (key0, key1) = self.rng_key();
        let site = self.rng_site(1);
        self.kernels.discrete_survival(
            &stream,
            self.ind.slice_mut(),
            n_batch,
            n_ages,
            n_ztypes,
            blueprint.stochastic,
            blueprint.continuous_sampling,
            scaling.slice(),
            survival_rates.slice(),
            viability_buf.slice(),
            key0,
            key1,
            site,
        )
    }

    /// Run one full discrete-generation tick (reproduction → survival → aging).
    ///
    /// ## Parameters
    /// - `blueprint`, `ecology`, `variants`, `deme_variants`: Live contracts.
    ///
    /// ## Errors
    /// Returns the first device-stage error.
    pub fn discrete_tick(
        &mut self,
        blueprint: &Blueprint,
        ecology: &EcologyParams,
        variants: &[GeneticsTensors],
        deme_variants: &[usize],
    ) -> Result<(), String> {
        self.discrete_reproduction_tick(blueprint, ecology, variants, deme_variants)?;
        self.discrete_survival_tick(blueprint, ecology, variants, deme_variants)?;
        self.age_tick()?;
        self.tick = self.tick.wrapping_add(1);
        Ok(())
    }

    /// Run one deterministic CSR migration step across the batch (demes).
    ///
    /// Mirrors `kernels::spatial::migrate_csr_deterministic`: each destination
    /// gathers from the host-built reverse CSR in a fixed order, so the result
    /// is deterministic (no atomics). A batch whose `migration_rate` column is
    /// empty is left untouched, matching the session's "no migration declared"
    /// short circuit.
    ///
    /// ## Parameters
    /// - `blueprint`: Frozen CSR (row pointers, destinations, weights).
    /// - `ecology`: Per-deme ecology; `migration_rate` is `(B, 2, A)` or empty.
    /// - `stay_after`: Whether unmoved mass stays at the source (row sums).
    ///
    /// ## Returns
    /// `Ok(())` after the state is replaced by the migrated planes.
    ///
    /// ## Errors
    /// Returns a description when the CSR or ecology lengths are inconsistent.
    pub fn migrate_tick(
        &mut self,
        blueprint: &Blueprint,
        ecology: &EcologyParams,
        stay_after: bool,
    ) -> Result<(), String> {
        let n_batch = self.n_batch;
        let n_ages = self.n_ages;
        let n_ztypes = self.n_ztypes;
        if blueprint.migration_indptr.len() != n_batch + 1 {
            return Err(format!(
                "migrate_tick needs {} CSR row pointers, got {}",
                n_batch + 1,
                blueprint.migration_indptr.len()
            ));
        }
        if ecology.migration_rate.is_empty() {
            return Ok(());
        }
        expect_len(
            "migration_rate",
            ecology.migration_rate.len(),
            n_batch * 2 * n_ages,
        )?;
        self.ensure_migration_cache(blueprint)?;
        let cache = self
            .migration_cache
            .take()
            .expect("migration cache ensured above");
        let result = (|| -> Result<(), String> {
            let stream = self.context.stream();
            let rate = upload_f64(&stream, &ecology.migration_rate)?;
            self.kernels.migration(
                &stream,
                self.ind.slice(),
                self.sperm.slice(),
                self.ind_scratch.slice_mut(),
                self.sperm_scratch.slice_mut(),
                rate.slice(),
                cache.indptr.slice(),
                cache.rev_indptr.slice(),
                cache.rev_src.slice(),
                cache.rev_weight.slice(),
                cache.row_sum.slice(),
                stay_after,
                n_batch,
                n_ages,
                n_ztypes,
            )
        })();
        self.migration_cache = Some(cache);
        result?;
        std::mem::swap(&mut self.ind, &mut self.ind_scratch);
        std::mem::swap(&mut self.sperm, &mut self.sperm_scratch);
        Ok(())
    }

    /// Run one stochastic CSR migration step across the batch (demes).
    ///
    /// Mirrors `kernels::spatial::migrate_csr_stochastic_rngs` for both
    /// discrete and continuous sampling: pass 1 samples each source's outbound
    /// and multinomial split into entry-indexed scratch, pass 2 gathers into
    /// each destination. A two-pass gather keeps the result reproducible
    /// (no atomics). Empty `migration_rate` is a no-op.
    ///
    /// ## Parameters
    /// - `blueprint`: Frozen CSR.
    /// - `ecology`: Per-deme ecology; `migration_rate` is `(B, 2, A)` or empty.
    ///
    /// ## Errors
    /// Returns a description when the CSR or ecology lengths are inconsistent.
    pub fn migrate_tick_stochastic(
        &mut self,
        blueprint: &Blueprint,
        ecology: &EcologyParams,
    ) -> Result<(), String> {
        let n_batch = self.n_batch;
        let n_ages = self.n_ages;
        let n_ztypes = self.n_ztypes;
        if blueprint.migration_indptr.len() != n_batch + 1 {
            return Err(format!(
                "migrate_tick_stochastic needs {} CSR row pointers, got {}",
                n_batch + 1,
                blueprint.migration_indptr.len()
            ));
        }
        if ecology.migration_rate.is_empty() {
            return Ok(());
        }
        expect_len(
            "migration_rate",
            ecology.migration_rate.len(),
            n_batch * 2 * n_ages,
        )?;
        let indptr: Vec<i32> = blueprint
            .migration_indptr
            .iter()
            .map(|value| *value as i32)
            .collect();
        if let Some(row_len) = indptr.windows(2).map(|pair| pair[1] - pair[0]).max() {
            if row_len as usize > MAX_CSR_ROW {
                return Err(format!(
                    "stochastic migration CSR row length {row_len} exceeds the device scratch \
                     limit of {MAX_CSR_ROW}"
                ));
            }
        }
        self.ensure_migration_cache(blueprint)?;
        let mut cache = self
            .migration_cache
            .take()
            .expect("migration cache ensured above");
        let (key0, key1) = self.rng_key();
        let site = self.rng_site(3);
        let result = (|| -> Result<(), String> {
            let stream = self.context.stream();
            let rate = upload_f64(&stream, &ecology.migration_rate)?;
            self.kernels.migration_stochastic(
                &stream,
                self.ind.slice(),
                self.sperm.slice(),
                self.ind_scratch.slice_mut(),
                self.sperm_scratch.slice_mut(),
                rate.slice(),
                cache.indptr.slice(),
                cache.dest.slice(),
                cache.weights.slice(),
                cache.fwd_f.slice_mut(),
                cache.fwd_s.slice_mut(),
                cache.fwd_m.slice_mut(),
                cache.rev_indptr.slice(),
                cache.rev_entry.slice(),
                n_batch,
                n_ages,
                n_ztypes,
                blueprint.continuous_sampling,
                key0,
                key1,
                site,
            )
        })();
        self.migration_cache = Some(cache);
        result?;
        std::mem::swap(&mut self.ind, &mut self.ind_scratch);
        std::mem::swap(&mut self.sperm, &mut self.sperm_scratch);
        Ok(())
    }

    /// Build the static device migration buffers for `blueprint`.
    ///
    /// ## Parameters
    /// - `blueprint`: Frozen CSR (row pointers, destinations, weights).
    /// - `fingerprint`: Hash of the CSR contents, stored for invalidation.
    ///
    /// ## Returns
    /// The populated cache, or a description of the first invalid destination
    /// or failed transfer.
    fn build_migration_cache(
        &self,
        blueprint: &Blueprint,
        fingerprint: u64,
    ) -> Result<MigrationCache, String> {
        let n_batch = self.n_batch;
        let n_ages = self.n_ages;
        let n_ztypes = self.n_ztypes;
        let indptr: Vec<i32> = blueprint
            .migration_indptr
            .iter()
            .map(|value| *value as i32)
            .collect();
        let dest: Vec<i32> = blueprint
            .migration_dest_idx
            .iter()
            .map(|value| *value as i32)
            .collect();
        let weights: Vec<f32> = blueprint
            .migration_weights
            .iter()
            .map(|value| *value as f32)
            .collect();
        let nnz = dest.len();
        let mut rev_indptr = vec![0i32; n_batch + 1];
        for &d in &dest {
            if d < 0 || d as usize >= n_batch {
                return Err(format!("migration destination {d} out of range"));
            }
            rev_indptr[d as usize + 1] += 1;
        }
        for i in 1..=n_batch {
            rev_indptr[i] += rev_indptr[i - 1];
        }
        // Reverse CSR for the deterministic gather: incoming (source, weight)
        // lists per destination, plus each source row's weight sum.
        let mut rev_src = vec![0i32; nnz];
        let mut rev_weight = vec![0f32; nnz];
        let mut row_sum = vec![0f32; n_batch];
        let mut cursor = rev_indptr.clone();
        for src in 0..n_batch {
            let start = indptr[src] as usize;
            let end = indptr[src + 1] as usize;
            let mut sum = 0.0f32;
            for entry in start..end {
                let d = dest[entry] as usize;
                let position = cursor[d] as usize;
                rev_src[position] = src as i32;
                rev_weight[position] = weights[entry];
                cursor[d] += 1;
                sum += weights[entry];
            }
            row_sum[src] = sum;
        }
        // Reverse CSR for the stochastic gather stores the CSR entry index so
        // pass 2 can read the entry-indexed forward scratch.
        let mut rev_entry = vec![0i32; nnz];
        let mut cursor = rev_indptr.clone();
        for pair in indptr.windows(2) {
            let start = pair[0] as usize;
            let count = (pair[1] - pair[0]) as usize;
            for (entry, &d) in dest.iter().enumerate().skip(start).take(count) {
                let d = d as usize;
                let position = cursor[d] as usize;
                rev_entry[position] = entry as i32;
                cursor[d] += 1;
            }
        }
        let stream = self.context.stream();
        Ok(MigrationCache {
            fingerprint,
            indptr: DeviceBuffer::from_host(&stream, &indptr)?,
            rev_indptr: DeviceBuffer::from_host(&stream, &rev_indptr)?,
            rev_src: DeviceBuffer::from_host(&stream, &rev_src)?,
            rev_weight: DeviceBuffer::from_host(&stream, &rev_weight)?,
            row_sum: DeviceBuffer::from_host(&stream, &row_sum)?,
            dest: DeviceBuffer::from_host(&stream, &dest)?,
            weights: DeviceBuffer::from_host(&stream, &weights)?,
            rev_entry: DeviceBuffer::from_host(&stream, &rev_entry)?,
            fwd_f: DeviceBuffer::from_host(&stream, &vec![0.0f32; nnz * n_ages * n_ztypes])?,
            fwd_s: DeviceBuffer::from_host(
                &stream,
                &vec![0.0f32; nnz * n_ages * n_ztypes * n_ztypes],
            )?,
            fwd_m: DeviceBuffer::from_host(&stream, &vec![0.0f32; nnz * n_ages * n_ztypes])?,
        })
    }

    /// Ensure the static migration cache matches `blueprint`.
    ///
    /// The cache is rebuilt only when the CSR contents change, so a session's
    /// frozen blueprint builds it once and reuses it for every tick.
    ///
    /// ## Parameters
    /// - `blueprint`: Frozen CSR to compare against the cached fingerprint.
    ///
    /// ## Errors
    /// Returns a description when the cache must be rebuilt and the rebuild
    /// fails.
    fn ensure_migration_cache(&mut self, blueprint: &Blueprint) -> Result<(), String> {
        let fingerprint = csr_fingerprint(blueprint);
        let stale = self
            .migration_cache
            .as_ref()
            .is_none_or(|cache| cache.fingerprint != fingerprint);
        if stale {
            let cache = self.build_migration_cache(blueprint, fingerprint)?;
            self.migration_cache = Some(cache);
        }
        Ok(())
    }

    /// Reserve device memory for the lazily-built migration cache.
    ///
    /// Called at enable time so a CSR whose cache cannot fit fails before any
    /// tick, rather than as a bare allocation error on the first migration.
    /// A blueprint whose CSR does not match the batch is skipped (no migration
    /// plan is used).
    ///
    /// ## Parameters
    /// - `blueprint`: Frozen CSR the executor would cache.
    ///
    /// ## Returns
    /// `Ok(())` when the cache fits the measured free memory.
    ///
    /// ## Errors
    /// Returns a description when the cache size overflows or exceeds the free
    /// memory.
    pub fn ensure_migration_budget(&self, blueprint: &Blueprint) -> Result<(), String> {
        if blueprint.migration_indptr.len() != self.n_batch + 1 {
            return Ok(());
        }
        let required = migration_cache_bytes(
            self.n_batch,
            self.n_ages,
            self.n_ztypes,
            blueprint.migration_dest_idx.len(),
        )?;
        let (free, _total) = self.context.memory_info()?;
        ensure_memory_budget(free, required)
    }

    /// Allocate the device history window described by `spec`.
    ///
    /// The projection dimensions must match the executor, and the whole window
    /// (rows plus mask/selection) must fit in the measured free memory. The
    /// session treats an error as "device staging unavailable" and keeps the
    /// host per-record path, so no engine fallback is involved.
    ///
    /// ## Parameters
    /// - `spec`: Projection configuration and row capacity.
    ///
    /// ## Returns
    /// `Ok(())` once the window is resident.
    ///
    /// ## Errors
    /// Returns a description when the dimensions disagree, the width does not
    /// match the mask, or the window does not fit.
    pub fn configure_history(&mut self, spec: &HistorySpec) -> Result<(), String> {
        let [d, s, a, z] = spec.dims;
        let plane = s * a * z;
        if plane == 0 || d != self.n_batch || s != 2 || a != self.n_ages || z != self.n_ztypes {
            return Err(format!(
                "history projection dims {:?} do not match the executor (A={}, Z={})",
                spec.dims, self.n_ages, self.n_ztypes
            ));
        }
        if spec.capacity == 0 {
            return Err("history spec capacity must be positive".to_owned());
        }
        // Raw rows carry the state planes; observation rows carry a projection.
        let (groups, out_d, out_a, width) = if spec.raw {
            let ind = d * s * a * z;
            let sperm = if spec.raw_sperm { d * a * z * z } else { 0 };
            let width = ind
                .checked_add(sperm)
                .ok_or_else(|| "history raw width overflow".to_owned())?;
            (0, 0, 0, width)
        } else {
            if spec.mask.is_empty() || spec.mask.len() % plane != 0 {
                return Err(format!(
                    "history mask length {} is not a positive multiple of the state plane {plane}",
                    spec.mask.len()
                ));
            }
            if spec.selected.is_empty() || spec.selected.iter().any(|index| *index >= d) {
                return Err("history deme selection is empty or out of range".to_owned());
            }
            let groups = spec.mask.len() / plane;
            let out_d = if spec.aggregate {
                1
            } else {
                spec.selected.len()
            };
            let out_a = if spec.collapse { 1 } else { a };
            (groups, out_d, out_a, groups * out_d * s * out_a)
        };
        if spec.width != width {
            return Err(format!(
                "history spec width mismatch: computed {width}, got {}",
                spec.width
            ));
        }
        let rows = spec
            .capacity
            .checked_mul(width)
            .ok_or_else(|| "history window size overflow".to_owned())?;
        let row_bytes = rows
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| "history window size overflow".to_owned())?;
        let mask_bytes = spec
            .mask
            .len()
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| "history mask size overflow".to_owned())?;
        let selected_bytes = spec
            .selected
            .len()
            .checked_mul(size_of::<i32>())
            .ok_or_else(|| "history selection size overflow".to_owned())?;
        let required = row_bytes
            .checked_add(mask_bytes)
            .and_then(|value| value.checked_add(selected_bytes))
            .ok_or_else(|| "history window size overflow".to_owned())?;
        let (free, _total) = self.context.memory_info()?;
        ensure_memory_budget(free, required)?;
        let stream = self.context.stream();
        let mask: Vec<f32> = spec.mask.iter().map(|value| *value as f32).collect();
        let selected: Vec<i32> = spec.selected.iter().map(|value| *value as i32).collect();
        self.history = Some(DeviceHistory {
            width,
            capacity: spec.capacity,
            rows: 0,
            head: 0,
            written: 0,
            raw: spec.raw,
            raw_sperm: spec.raw_sperm,
            wrap: spec.wrap,
            n_groups: groups,
            plane,
            n_selected: spec.selected.len(),
            n_sexes: s,
            n_ages: a,
            n_ztypes: z,
            out_d,
            out_a,
            collapse: spec.collapse,
            aggregate: spec.aggregate,
            mask: DeviceBuffer::from_host(&stream, &mask)?,
            selected: DeviceBuffer::from_host(&stream, &selected)?,
            buffer: DeviceBuffer::from_host(&stream, &vec![0.0f32; rows])?,
        });
        Ok(())
    }

    /// Project the current device state into the next history row.
    ///
    /// ## Returns
    /// `Ok(())` after the row is enqueued and the row count advances.
    ///
    /// ## Errors
    /// Returns a description when no window is configured, it is full, or the
    /// launch fails.
    pub fn record_history_row(&mut self) -> Result<(), String> {
        let mut history = self
            .history
            .take()
            .ok_or_else(|| "device history is not configured".to_owned())?;
        if history.rows >= history.capacity && !history.wrap {
            self.history = Some(history);
            return Err("device history window is full".to_owned());
        }
        // A ring window overwrites its oldest row once full; an exact window
        // only ever fills up to `capacity` and is guarded above.
        let position = if history.rows < history.capacity {
            (history.head + history.rows) % history.capacity
        } else {
            history.head
        };
        let row_offset = position * history.width;
        let stream = self.context.stream();
        let result = if history.raw {
            self.copy_raw_history_row(&mut history, row_offset)
        } else {
            self.kernels.observation_project(
                &stream,
                self.ind.slice(),
                history.mask.slice(),
                history.selected.slice(),
                history.n_groups,
                history.plane,
                history.n_selected,
                history.n_sexes,
                history.n_ages,
                history.n_ztypes,
                self.n_batch,
                history.collapse,
                history.aggregate,
                history.out_d,
                history.out_a,
                row_offset,
                history.buffer.slice_mut(),
            )
        };
        if result.is_ok() {
            history.written += 1;
            if history.rows < history.capacity {
                history.rows += 1;
            } else {
                history.head = (history.head + 1) % history.capacity;
            }
        }
        self.history = Some(history);
        result
    }

    /// Copy the current state planes into a raw history row (device to device).
    ///
    /// The row keeps the device-native batch-minor layout; the session
    /// transposes it back when it flushes the window.
    ///
    /// ## Parameters
    /// - `history`: The taken device window being written.
    /// - `row_offset`: Element offset of the row inside the window buffer.
    ///
    /// ## Errors
    /// Returns a description when the row lies outside the allocation or a copy
    /// fails.
    fn copy_raw_history_row(
        &self,
        history: &mut DeviceHistory,
        row_offset: usize,
    ) -> Result<(), String> {
        let stream = self.context.stream();
        let ind_len = 2 * self.n_ages * self.n_ztypes * self.n_batch;
        let sperm_start = row_offset
            .checked_add(ind_len)
            .ok_or_else(|| "history row offset overflow".to_owned())?;
        let end = row_offset
            .checked_add(history.width)
            .ok_or_else(|| "history row offset overflow".to_owned())?;
        if end > history.buffer.len() || sperm_start > end {
            return Err("history row offset out of range".to_owned());
        }
        {
            let mut view = history
                .buffer
                .slice_mut()
                .try_slice_mut(row_offset..sperm_start)
                .ok_or_else(|| "history ind slice out of range".to_owned())?;
            stream
                .memcpy_dtod(self.ind.slice(), &mut view)
                .map_err(|err| format!("history ind copy failed: {err}"))?;
        }
        if history.raw_sperm {
            let mut view = history
                .buffer
                .slice_mut()
                .try_slice_mut(sperm_start..end)
                .ok_or_else(|| "history sperm slice out of range".to_owned())?;
            stream
                .memcpy_dtod(self.sperm.slice(), &mut view)
                .map_err(|err| format!("history sperm copy failed: {err}"))?;
        }
        Ok(())
    }

    /// Copy the staged history rows back, flushing them from the device.
    ///
    /// ## Returns
    /// Row-major `rows · width` values in chronological order (the ring's
    /// oldest row first), or an empty vector when no window is configured.
    ///
    /// ## Errors
    /// Returns a description when the device-to-host copy fails.
    pub fn download_history_rows(&mut self) -> Result<Vec<f32>, String> {
        let Some(history) = self.history.as_ref() else {
            return Ok(Vec::new());
        };
        if history.rows == 0 {
            return Ok(Vec::new());
        }
        let stream = self.context.stream();
        let all = history.buffer.to_host(&stream)?;
        let mut rows = Vec::with_capacity(history.rows * history.width);
        for index in 0..history.rows {
            let position = (history.head + index) % history.capacity;
            let start = position * history.width;
            rows.extend_from_slice(&all[start..start + history.width]);
        }
        Ok(rows)
    }

    /// Values per staged history row.
    ///
    /// ## Returns
    /// The row width, or `None` when no window is configured.
    pub fn history_width(&self) -> Option<usize> {
        self.history.as_ref().map(|history| history.width)
    }

    /// Whether the staged window holds raw state rows instead of projections.
    ///
    /// ## Returns
    /// `true` for a raw window, `false` when no window is configured or it
    /// holds observation projections.
    pub fn history_is_raw(&self) -> bool {
        self.history.as_ref().is_some_and(|history| history.raw)
    }

    /// How many oldest rows a clamped ring window has overwritten.
    ///
    /// ## Returns
    /// `written - rows`, or `0` when no window is configured.
    pub fn history_dropped(&self) -> usize {
        self.history
            .as_ref()
            .map_or(0, |history| history.written - history.rows)
    }

    /// Drop the device history window, releasing its memory.
    pub fn clear_history(&mut self) {
        self.history = None;
    }

    /// Replace the device state and tick counter from a host checkpoint.
    ///
    /// Rolls a GPU-backed session back to a recorded checkpoint so the device
    /// state matches the restored host state. The counter-based RNG is keyed
    /// by the restored tick, so the device reproduces the checkpointed
    /// trajectory from that point. The migration cache is graph-static and
    /// stays valid.
    ///
    /// ## Parameters
    /// - `ind_host`: Batch-major individual counts, `(B, 2, A, Z)`.
    /// - `sperm_host`: Batch-major stored sperm, `(B, A, Z, Z)`.
    /// - `tick`: Device tick counter to restore.
    ///
    /// ## Returns
    /// `Ok(())` after both planes are resident again.
    ///
    /// ## Errors
    /// Returns a description when the host slices have the wrong length or an
    /// upload fails.
    pub fn restore_state(
        &mut self,
        ind_host: &[f32],
        sperm_host: &[f32],
        tick: u64,
    ) -> Result<(), String> {
        let n_batch = self.n_batch;
        let n_ages = self.n_ages;
        let n_ztypes = self.n_ztypes;
        let expected_ind = 2 * n_ages * n_ztypes * n_batch;
        let expected_sperm = n_ages * n_ztypes * n_ztypes * n_batch;
        if ind_host.len() != expected_ind {
            return Err(format!(
                "restore_state individual state needs {expected_ind} elements, got {}",
                ind_host.len()
            ));
        }
        if sperm_host.len() != expected_sperm {
            return Err(format!(
                "restore_state sperm state needs {expected_sperm} elements, got {}",
                sperm_host.len()
            ));
        }
        let stream = self.context.stream();
        let ind_inner = batch_to_inner(ind_host, &[2, n_ages, n_ztypes], n_batch)?;
        let sperm_inner = batch_to_inner(sperm_host, &[n_ages, n_ztypes, n_ztypes], n_batch)?;
        self.ind = DeviceBuffer::from_host(&stream, &ind_inner)?;
        self.sperm = DeviceBuffer::from_host(&stream, &sperm_inner)?;
        self.tick = tick;
        Ok(())
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

/// Hash the frozen migration CSR so the device cache can detect a change.
///
/// Weights are hashed by their bit pattern because `f64` is not `Hash`.
///
/// ## Parameters
/// - `blueprint`: Frozen model carrying the CSR arrays.
///
/// ## Returns
/// A stable-within-process hash of the row pointers, destinations, and
/// weights.
fn csr_fingerprint(blueprint: &Blueprint) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    blueprint.migration_indptr.hash(&mut hasher);
    blueprint.migration_dest_idx.hash(&mut hasher);
    for weight in &blueprint.migration_weights {
        hasher.write_u64(weight.to_bits());
    }
    hasher.finish()
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

/// Copy a host `f64` genetics table into a preallocated `f32` slot.
///
/// ## Parameters
/// - `dst`: Destination slot, already sliced to the table's extent.
/// - `src`: Source table.
/// - `name`: Table name for the error message.
/// - `variant_id`: Variant index for the error message.
///
/// ## Errors
/// Returns a readable description when the lengths differ.
fn copy_narrow(dst: &mut [f32], src: &[f64], name: &str, variant_id: usize) -> Result<(), String> {
    if dst.len() != src.len() {
        return Err(format!(
            "variant {variant_id} {name}: expected {} elements, got {}",
            dst.len(),
            src.len()
        ));
    }
    for (slot, value) in dst.iter_mut().zip(src) {
        *slot = *value as f32;
    }
    Ok(())
}

/// Fail explicitly when the state would not fit in the measured free memory.
///
/// ## Parameters
/// - `free`: Free device memory in bytes, as measured.
/// - `required`: Bytes the executor needs.
///
/// ## Errors
/// Returns a readable description when `required > free`; the caller never
/// falls back to the CPU.
fn ensure_memory_budget(free: usize, required: usize) -> Result<(), String> {
    if required > free {
        return Err(format!(
            "GPU memory budget exceeded: the state needs {} MiB ({required} B), \
             but only {} MiB is free; reduce the batch size or free the device",
            required / (1024 * 1024),
            free / (1024 * 1024)
        ));
    }
    Ok(())
}

/// Bytes the migration cache holds for a CSR with `nnz` non-zeros.
///
/// The cache is built lazily on the first migration, so its footprint must be
/// reserved by the enable-time budget guard instead of surfacing as a bare
/// allocation failure mid-run.
///
/// ## Parameters
/// - `n_batch`, `n_ages`, `n_ztypes`: Model dimensions.
/// - `nnz`: CSR non-zero count (`migration_dest_idx.len()`).
///
/// ## Returns
/// The cache size in bytes.
///
/// ## Errors
/// Returns a description when the arithmetic overflows `usize`.
fn migration_cache_bytes(
    n_batch: usize,
    n_ages: usize,
    n_ztypes: usize,
    nnz: usize,
) -> Result<usize, String> {
    let i32_bytes = size_of::<i32>();
    let f32_bytes = size_of::<f32>();
    let overflow = || "migration cache size overflow".to_owned();
    let rows = n_batch.checked_add(1).ok_or_else(overflow)?;
    // indptr + rev_indptr (i32 each).
    let mut total = rows
        .checked_mul(i32_bytes)
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(overflow)?;
    // rev_src + dest + rev_entry (i32) and rev_weight + weights (f32).
    let entry_i32 = nnz
        .checked_mul(i32_bytes)
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(overflow)?;
    let entry_f32 = nnz
        .checked_mul(f32_bytes)
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(overflow)?;
    total = total
        .checked_add(entry_i32)
        .and_then(|value| value.checked_add(entry_f32))
        .ok_or_else(overflow)?;
    // row_sum (f32, one per source).
    total = total
        .checked_add(n_batch.checked_mul(f32_bytes).ok_or_else(overflow)?)
        .ok_or_else(overflow)?;
    // fwd_f + fwd_m (nnz·A·Z f32) and fwd_s (nnz·A·Z·Z f32).
    let az = n_ages.checked_mul(n_ztypes).ok_or_else(overflow)?;
    let fwd_fm = nnz
        .checked_mul(az)
        .and_then(|value| value.checked_mul(f32_bytes))
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(overflow)?;
    let fwd_s = nnz
        .checked_mul(az)
        .and_then(|value| value.checked_mul(n_ztypes))
        .and_then(|value| value.checked_mul(f32_bytes))
        .ok_or_else(overflow)?;
    total
        .checked_add(fwd_fm)
        .and_then(|value| value.checked_add(fwd_s))
        .ok_or_else(overflow)
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
