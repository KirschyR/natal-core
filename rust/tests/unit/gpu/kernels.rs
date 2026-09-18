//! P2 kernel-loader tests.
//!
//! Loading exercises NVRTC compilation end to end and asserts by default (the
//! GPU server is the primary runtime); `NATAL_GPU_REQUIRE=0` disables it.

use crate::gpu::context::GpuContext;
use crate::gpu::hardware_required;
use cudarc::driver::CudaStream;
use std::sync::Arc;

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

#[test]
fn migration_kernel_short_circuits_empty_batch() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let context = GpuContext::new(0).expect("device 0 context");
    let kernels = Kernels::load(&context.context()).expect("kernels load");
    let stream = context.stream();
    let zero_f = stream.alloc_zeros::<f32>(0).expect("alloc");
    let zero_i = stream.alloc_zeros::<i32>(0).expect("alloc");
    let ind_in = stream.alloc_zeros::<f32>(0).expect("alloc");
    let sperm_in = stream.alloc_zeros::<f32>(0).expect("alloc");
    let mut ind_out = stream.alloc_zeros::<f32>(0).expect("alloc");
    let mut sperm_out = stream.alloc_zeros::<f32>(0).expect("alloc");
    assert!(kernels
        .migration(
            &stream,
            &ind_in,
            &sperm_in,
            &mut ind_out,
            &mut sperm_out,
            &zero_f,
            &zero_i,
            &zero_i,
            &zero_i,
            &zero_f,
            &zero_f,
            false,
            0,
            4,
            2,
        )
        .is_ok());
}

/// Host Philox4x32-10 reference (matches the CUDA `philox4x32_10`).
///
/// ## Parameters
/// - `a`, `b`: 32-bit operands.
///
/// ## Returns
/// High 32 bits of the 64-bit product.
fn philox_mulhi(a: u32, b: u32) -> u32 {
    (((a as u64) * (b as u64)) >> 32) as u32
}

/// Host Philox4x32-10 round function.
///
/// ## Parameters
/// - `counter`: Four counter words.
/// - `key`: Two key words.
///
/// ## Returns
/// The four output words.
fn philox4x32_10(mut counter: [u32; 4], mut key: [u32; 2]) -> [u32; 4] {
    const M0: u32 = 0xD2511F53;
    const M1: u32 = 0xCD9E8D57;
    const W0: u32 = 0x9E3779B9;
    const W1: u32 = 0xBB67AE85;
    for _ in 0..10 {
        let h0 = philox_mulhi(M0, counter[0]);
        let l0 = M0.wrapping_mul(counter[0]);
        let h1 = philox_mulhi(M1, counter[2]);
        let l1 = M1.wrapping_mul(counter[2]);
        counter = [h1 ^ counter[1] ^ key[0], l1, h0 ^ counter[3] ^ key[1], l0];
        key[0] = key[0].wrapping_add(W0);
        key[1] = key[1].wrapping_add(W1);
    }
    counter
}

/// Host uniform in `[0, 1)` for draw `i`.
///
/// ## Parameters
/// - `i`: Draw index.
/// - `key0`, `key1`: Key words.
/// - `site`: Draw-site counter.
///
/// ## Returns
/// The 24-bit uniform.
fn uniform_host(i: u64, key0: u32, key1: u32, site: u32) -> f32 {
    let out = philox4x32_10(
        [(i & 0xFFFF_FFFF) as u32, (i >> 32) as u32, site, 0],
        [key0, key1],
    );
    ((out[0] >> 8) as f32) * (1.0f32 / 16777216.0f32)
}

#[test]
fn fill_uniform_matches_host_philox_and_reproduces() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let context = GpuContext::new(0).expect("device 0 context");
    let kernels = Kernels::load(&context.context()).expect("kernels load");
    let stream = context.stream();
    let (key0, key1, site) = (0x1234_5678u32, 0x9ABC_DEF0u32, 7u32);
    let n = 1000usize;
    let mut out = stream.alloc_zeros::<f32>(n).expect("alloc");
    kernels
        .fill_uniform(&stream, &mut out, key0, key1, site)
        .expect("fill");
    let first = stream.clone_dtoh(&out).expect("download");
    for (i, value) in first.iter().enumerate() {
        let want = uniform_host(i as u64, key0, key1, site);
        assert_eq!(value.to_bits(), want.to_bits(), "index {i}");
    }

    let mut again = stream.alloc_zeros::<f32>(n).expect("alloc");
    kernels
        .fill_uniform(&stream, &mut again, key0, key1, site)
        .expect("fill");
    let second = stream.clone_dtoh(&again).expect("download");
    assert_eq!(
        first.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        second.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );

    let n_big = 100_000usize;
    let mut big = stream.alloc_zeros::<f32>(n_big).expect("alloc");
    kernels
        .fill_uniform(&stream, &mut big, key0, key1, site)
        .expect("fill");
    let samples = stream.clone_dtoh(&big).expect("download");
    let mean = samples.iter().sum::<f32>() / n_big as f32;
    let variance = samples
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f32>()
        / n_big as f32;
    assert!((mean - 0.5).abs() < 0.01, "uniform mean {mean}");
    assert!(
        (variance - 1.0 / 12.0).abs() < 0.005,
        "uniform variance {variance}"
    );
}

/// Draw `n` samples from one sampler and copy them back to the host.
///
/// ## Parameters
/// - `kernels`: Loaded kernels.
/// - `stream`: Stream to launch on.
/// - `kind`, `a`, `b`: Sampler selector and parameters.
/// - `n`: Number of samples.
///
/// ## Returns
/// The host samples.
fn draw(
    kernels: &Kernels,
    stream: &Arc<CudaStream>,
    kind: u32,
    a: f32,
    b: f32,
    n: usize,
) -> Vec<f32> {
    let mut out = stream.alloc_zeros::<f32>(n).expect("alloc");
    kernels
        .sample_into(stream, &mut out, kind, a, b, 0x1234_5678, 0x9ABC_DEF0, 3)
        .expect("sample");
    stream.clone_dtoh(&out).expect("download")
}

#[test]
fn samplers_match_theoretical_moments() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let context = GpuContext::new(0).expect("device 0 context");
    let kernels = Kernels::load(&context.context()).expect("kernels load");
    let stream = context.stream();
    let n = 1_000_000usize;
    let cases: [(&str, u32, f32, f32, f64, f64); 6] = [
        ("binomial n=10 p=0.5", 1, 10.0, 0.5, 5.0, 2.5),
        ("binomial n=100 p=0.3", 1, 100.0, 0.3, 30.0, 21.0),
        ("binomial n=10000 p=0.3", 1, 10_000.0, 0.3, 3000.0, 2100.0),
        ("poisson lambda=20", 2, 20.0, 0.0, 20.0, 20.0),
        ("poisson lambda=200", 2, 200.0, 0.0, 200.0, 200.0),
        ("gamma shape=2 scale=3", 3, 2.0, 3.0, 6.0, 18.0),
    ];
    for (label, kind, a, b, mean, variance) in cases {
        let values = draw(&kernels, &stream, kind, a, b, n);
        let got_mean = values.iter().map(|x| *x as f64).sum::<f64>() / n as f64;
        let got_var = values
            .iter()
            .map(|x| {
                let d = *x as f64 - got_mean;
                d * d
            })
            .sum::<f64>()
            / n as f64;
        let mean_tol = 4.0 * (variance / n as f64).sqrt();
        assert!(
            (got_mean - mean).abs() <= mean_tol,
            "{label}: mean {got_mean} vs {mean}"
        );
        let var_tol = 6.0 * variance * (2.0 / n as f64).sqrt();
        assert!(
            (got_var - variance).abs() <= var_tol,
            "{label}: var {got_var} vs {variance}"
        );
    }

    let normal = draw(&kernels, &stream, 4, 0.0, 0.0, n);
    let mean = normal.iter().map(|x| *x as f64).sum::<f64>() / n as f64;
    let variance = normal
        .iter()
        .map(|x| {
            let d = *x as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n as f64;
    assert!(mean.abs() < 0.03, "normal mean {mean}");
    assert!((variance - 1.0).abs() < 0.05, "normal variance {variance}");
}

#[test]
fn samplers_reproduce_for_the_same_key() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let context = GpuContext::new(0).expect("device 0 context");
    let kernels = Kernels::load(&context.context()).expect("kernels load");
    let stream = context.stream();
    let first = draw(&kernels, &stream, 1, 100.0, 0.3, 4096);
    let second = draw(&kernels, &stream, 1, 100.0, 0.3, 4096);
    assert_eq!(
        first.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        second.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
fn multinomial_conserves_totals_and_proportions() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let context = GpuContext::new(0).expect("device 0 context");
    let kernels = Kernels::load(&context.context()).expect("kernels load");
    let stream = context.stream();
    let rows = 50_000usize;
    let k = 4usize;
    let total = 100.0f32;
    let weights = [0.1f32, 0.2, 0.3, 0.4];
    let probs_host: Vec<f32> = (0..rows * k).map(|i| weights[i % k]).collect();
    let totals_host = vec![total; rows];
    let probs = stream.clone_htod(&probs_host).expect("upload");
    let totals = stream.clone_htod(&totals_host).expect("upload");
    let mut counts = stream.alloc_zeros::<f32>(rows * k).expect("alloc");
    kernels
        .multinomial_seq(&stream, &probs, &totals, &mut counts, rows, k, 11, 22, 33)
        .expect("multinomial");
    let host = stream.clone_dtoh(&counts).expect("download");
    for row in 0..rows {
        let sum: f32 = (0..k).map(|j| host[row * k + j]).sum();
        assert!(
            (sum - total).abs() < 1e-3,
            "row {row} total {sum} vs {total}"
        );
    }
    for (j, &p) in weights.iter().enumerate() {
        let got = (0..rows).map(|r| host[r * k + j] as f64).sum::<f64>() / rows as f64;
        let want = total as f64 * p as f64;
        let tolerance = 4.0 * (total as f64 * p as f64 * (1.0 - p as f64) / rows as f64).sqrt();
        assert!(
            (got - want).abs() <= tolerance.max(0.2),
            "category {j}: mean {got} vs {want}"
        );
    }
}
