//! P2 kernel-loader tests.
//!
//! Loading exercises NVRTC compilation end to end and asserts by default (the
//! GPU server is the primary runtime); `NATAL_GPU_REQUIRE=0` disables it.

use crate::gpu::context::GpuContext;
use crate::gpu::hardware_required;

use super::Kernels;

#[test]
fn aging_kernel_compiles_and_loads() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let context = GpuContext::new(0).expect("device 0 context");
    Kernels::load(&context.context()).expect("aging kernel compiles and loads");
}

#[test]
fn kernel_launchers_reject_invalid_shapes() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let context = GpuContext::new(0).expect("device 0 context");
    let kernels = Kernels::load(&context.context()).expect("kernels load");
    let stream = context.stream();
    let four = stream.alloc_zeros::<f32>(4).expect("alloc");
    let zero_f = stream.alloc_zeros::<f32>(0).expect("alloc");
    let zero_i = stream.alloc_zeros::<i32>(0).expect("alloc");

    // age_shift: length mismatch, empty plane, and zero stride.
    let mut three = stream.alloc_zeros::<f32>(3).expect("alloc");
    assert!(kernels
        .age_shift(&stream, &four, &mut three, 4, 4, 1)
        .is_err());
    let mut empty_dst = stream.alloc_zeros::<f32>(0).expect("alloc");
    assert!(kernels
        .age_shift(&stream, &zero_f, &mut empty_dst, 0, 4, 1)
        .is_ok());
    let mut four_dst = stream.alloc_zeros::<f32>(4).expect("alloc");
    assert!(kernels
        .age_shift(&stream, &four, &mut four_dst, 4, 4, 0)
        .is_err());

    // density_scaling: zero batch and zero new_adult_age are rejected.
    let mut out = stream.alloc_zeros::<f32>(1).expect("alloc");
    let mut buffers = super::DensityBuffers {
        ind: &four,
        survival_rates: &four,
        reproduction_rates: &four,
        fertility: &four,
        competition_weights: &four,
        equilibrium_declared: &zero_i,
        equilibrium_distribution: &four,
        external_expected_eggs: &four,
        carrying_capacity: &four,
        eggs_per_female: &four,
        sex_ratio: &four,
        growth_mode: &zero_i,
        low_density_growth_rate: &four,
        scaling_out: &mut out,
    };
    assert!(kernels
        .density_scaling(&stream, &mut buffers, 0, 4, 2, 1)
        .is_err());
    assert!(kernels
        .density_scaling(&stream, &mut buffers, 1, 4, 2, 0)
        .is_err());

    // Zero-batch / empty-plane short circuits.
    let mut factor = stream.alloc_zeros::<f32>(1).expect("alloc");
    assert!(kernels
        .recruit_factor(&stream, &four, &four, &mut factor, 0, 4, 2)
        .is_ok());
    let mut ind0 = stream.alloc_zeros::<f32>(0).expect("alloc");
    let mut sperm0 = stream.alloc_zeros::<f32>(0).expect("alloc");
    assert!(kernels
        .survival_scale_ind(&stream, &mut ind0, &four, &four, &four, 1, 4, 2, 1)
        .is_ok());
    assert!(kernels
        .survival_scale_sperm(&stream, &mut sperm0, &four, &four, 1, 4, 2, 1)
        .is_ok());
    let mut ind1 = stream.alloc_zeros::<f32>(0).expect("alloc");
    let mut sperm1 = stream.alloc_zeros::<f32>(0).expect("alloc");
    let reproduction_buffers = super::ReproductionBuffers {
        mating_rates: &four,
        sperm_displacement_rate: &four,
        reproduction_rates: &four,
        fertility: &four,
        eggs_per_female: &four,
        sex_ratio: &four,
        female_only: &zero_i,
        male_only: &zero_i,
        fecundity: &four,
        sexual_selection: &four,
        offspring: &four,
        zygote_viability: &four,
        female_compat: &four,
        male_compat: &four,
    };
    assert!(kernels
        .reproduction(
            &stream,
            &mut ind1,
            &mut sperm1,
            &reproduction_buffers,
            0,
            4,
            2,
            1,
            false
        )
        .is_ok());
}
