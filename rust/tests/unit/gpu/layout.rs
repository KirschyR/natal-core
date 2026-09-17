//! L1 tests for the batch-axis layout transpose.
//!
//! These are host-only and run on every machine: the transpose is a pure
//! permutation, so its correctness is fully checked without a GPU. A round
//! trip must preserve every element bit for bit — the acceptance criterion for
//! integer state.

use super::{batch_to_inner, batch_to_outer};

/// Collect the raw bit patterns of an `f32` slice, for exact comparison.
///
/// ## Parameters
/// - `values`: Floating-point values to canonicalize.
///
/// ## Returns
/// One `u32` per value.
fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// Build `1..=n` as exactly representable `f32` values.
///
/// ## Parameters
/// - `n`: Number of elements.
///
/// ## Returns
/// The sequence `1.0, 2.0, …`.
fn integer_sequence(n: usize) -> Vec<f32> {
    (1..=n).map(|value| value as f32).collect()
}

#[test]
fn known_transpose_matches_hand_computation() {
    // (B=2, d0=2, d1=2): batch 0 is 0..3, batch 1 is 10..13.
    let outer: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0];
    let inner = batch_to_inner(&outer, &[2, 2], 2).expect("shape is valid");
    // (d0, d1, B): cell c gets [outer[c], outer[4 + c]].
    assert_eq!(inner, vec![0.0, 10.0, 1.0, 11.0, 2.0, 12.0, 3.0, 13.0]);
    let back = batch_to_outer(&inner, &[2, 2], 2).expect("shape is valid");
    assert_eq!(bits(&back), bits(&outer));
}

#[test]
fn round_trip_is_bit_identical_for_representative_dims() {
    // A=8, Z=9 with 7 batch elements, matching the spatial-model hot shape.
    let axes = [2usize, 8, 9];
    let n_batch = 7;
    let outer = integer_sequence(axes.iter().product::<usize>() * n_batch);
    let inner = batch_to_inner(&outer, &axes, n_batch).expect("shape is valid");
    let back = batch_to_outer(&inner, &axes, n_batch).expect("shape is valid");
    assert_eq!(bits(&back), bits(&outer));
}

#[test]
fn empty_axes_behave_as_a_scalar_per_batch() {
    let outer = vec![1.0f32, 2.0, 3.0];
    let inner = batch_to_inner(&outer, &[], 3).expect("shape is valid");
    assert_eq!(inner, outer);
    let back = batch_to_outer(&inner, &[], 3).expect("shape is valid");
    assert_eq!(bits(&back), bits(&outer));
}

#[test]
fn zero_batch_yields_empty_slices() {
    assert!(batch_to_inner::<f32>(&[], &[2, 3], 0)
        .expect("empty batch is valid")
        .is_empty());
    assert!(batch_to_outer::<f32>(&[], &[2, 3], 0)
        .expect("empty batch is valid")
        .is_empty());
}

#[test]
fn mismatched_lengths_are_rejected() {
    assert!(batch_to_inner(&[0.0f32; 5], &[2, 3], 1).is_err());
    assert!(batch_to_outer(&[0.0f32; 7], &[2, 3], 1).is_err());
}

#[test]
fn transpose_preserves_the_multiset_of_values() {
    let axes = [2usize, 3, 4];
    let n_batch = 5;
    let outer = integer_sequence(axes.iter().product::<usize>() * n_batch);
    let mut inner = batch_to_inner(&outer, &axes, n_batch).expect("shape is valid");
    let mut sorted_outer = outer.clone();
    sorted_outer.sort_by(f32::total_cmp);
    inner.sort_by(f32::total_cmp);
    assert_eq!(bits(&inner), bits(&sorted_outer));
}
