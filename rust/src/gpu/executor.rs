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
use crate::gpu::kernels::{DensityBuffers, Kernels, ReproductionBuffers};
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
        })
    }

    /// Set the sampling seed for the counter-based device RNG.
    ///
    /// ## Parameters
    /// - `seed`: Session seed; the two Philox key words are its halves.
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
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
        )?;
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
        let scaling_host = self.density_scaling(blueprint, ecology)?;
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
        let scaling = DeviceBuffer::from_host(&stream, &scaling_host)?;
        let survival_rates = upload_f64(&stream, &ecology.survival_rates)?;
        let viability_buf = DeviceBuffer::from_host(&stream, &viability)?;
        if blueprint.stochastic {
            if blueprint.continuous_sampling {
                return Err("GPU stochastic survival requires continuous_sampling=false".to_owned());
            }
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
                key0,
                key1,
                recruit_site,
            )?;
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
                key0,
                key1,
                survival_site,
            )?;
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
        if blueprint.stochastic {
            if blueprint.continuous_sampling {
                return Err(
                    "GPU stochastic reproduction requires continuous_sampling=false".to_owned(),
                );
            }
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
        // Reverse CSR: incoming (source, weight) lists per destination.
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
        let mut rev_src = vec![0i32; nnz];
        let mut rev_weight = vec![0f32; nnz];
        let mut cursor: Vec<i32> = rev_indptr.clone();
        let mut row_sum_w = vec![0f32; n_batch];
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
            row_sum_w[src] = sum;
        }
        let stream = self.context.stream();
        let rate = upload_f64(&stream, &ecology.migration_rate)?;
        let indptr_buf = DeviceBuffer::from_host(&stream, &indptr)?;
        let rev_indptr_buf = DeviceBuffer::from_host(&stream, &rev_indptr)?;
        let rev_src_buf = DeviceBuffer::from_host(&stream, &rev_src)?;
        let rev_weight_buf = DeviceBuffer::from_host(&stream, &rev_weight)?;
        let row_sum_buf = DeviceBuffer::from_host(&stream, &row_sum_w)?;
        self.kernels.migration(
            &stream,
            self.ind.slice(),
            self.sperm.slice(),
            self.ind_scratch.slice_mut(),
            self.sperm_scratch.slice_mut(),
            rate.slice(),
            indptr_buf.slice(),
            rev_indptr_buf.slice(),
            rev_src_buf.slice(),
            rev_weight_buf.slice(),
            row_sum_buf.slice(),
            stay_after,
            n_batch,
            n_ages,
            n_ztypes,
        )?;
        std::mem::swap(&mut self.ind, &mut self.ind_scratch);
        std::mem::swap(&mut self.sperm, &mut self.sperm_scratch);
        Ok(())
    }

    /// Run one stochastic CSR migration step across the batch (demes).
    ///
    /// Mirrors `kernels::spatial::migrate_csr_stochastic_rngs` for
    /// `continuous_sampling == false`: pass 1 samples each source's outbound
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
        // Reverse CSR stores the CSR entry index so pass 2 can read `fwd_*`.
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
        let rate = upload_f64(&stream, &ecology.migration_rate)?;
        let indptr_buf = DeviceBuffer::from_host(&stream, &indptr)?;
        let dest_buf = DeviceBuffer::from_host(&stream, &dest)?;
        let weights_buf = DeviceBuffer::from_host(&stream, &weights)?;
        let rev_indptr_buf = DeviceBuffer::from_host(&stream, &rev_indptr)?;
        let rev_entry_buf = DeviceBuffer::from_host(&stream, &rev_entry)?;
        let mut fwd_f = DeviceBuffer::from_host(&stream, &vec![0.0f32; nnz * n_ages * n_ztypes])?;
        let mut fwd_s =
            DeviceBuffer::from_host(&stream, &vec![0.0f32; nnz * n_ages * n_ztypes * n_ztypes])?;
        let mut fwd_m = DeviceBuffer::from_host(&stream, &vec![0.0f32; nnz * n_ages * n_ztypes])?;
        let (key0, key1) = self.rng_key();
        let site = self.rng_site(3);
        self.kernels.migration_stochastic(
            &stream,
            self.ind.slice(),
            self.sperm.slice(),
            self.ind_scratch.slice_mut(),
            self.sperm_scratch.slice_mut(),
            rate.slice(),
            indptr_buf.slice(),
            dest_buf.slice(),
            weights_buf.slice(),
            fwd_f.slice_mut(),
            fwd_s.slice_mut(),
            fwd_m.slice_mut(),
            rev_indptr_buf.slice(),
            rev_entry_buf.slice(),
            n_batch,
            n_ages,
            n_ztypes,
            key0,
            key1,
            site,
        )?;
        std::mem::swap(&mut self.ind, &mut self.ind_scratch);
        std::mem::swap(&mut self.sperm, &mut self.sperm_scratch);
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
