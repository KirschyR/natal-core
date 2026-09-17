//! Batch-axis layout transpose between the CPU and device conventions.
//!
//! The CPU stacks one contiguous block per batch element — `(B, d0, d1, …, dk)`
//! in row-major order, where `B` is the batch axis (a deme in spatial models, a
//! replicate in ensemble runs). That layout suits `chunks_mut` on the host, but
//! it is the worst possible one for a GPU: a thread indexed by its batch
//! element would stride by the product of every trailing axis, so neighbouring
//! threads touch memory `2·A·Z` elements apart (144 for `A=8, Z=9`) and no
//! access can be coalesced.
//!
//! The device therefore stores `(d0, d1, …, dk, B)`, with the batch axis
//! innermost so that neighbouring threads read neighbouring addresses. The
//! transpose happens only at the upload/download boundary; the CPU layout and
//! every existing kernel are untouched.
//!
//! Because both functions are pure permutations, a round trip preserves every
//! element bit for bit — the property the L1 gate checks on integer state.

#![allow(clippy::needless_range_loop)] // Index loops express the (batch, cell) transpose directly.

/// Number of cells per batch element, i.e. the product of the trailing axes.
///
/// ## Parameters
/// - `axes`: Lengths of the non-batch axes, outermost first.
///
/// ## Returns
/// `1` for an empty axis list (a scalar per batch element).
fn cells_per_batch(axes: &[usize]) -> usize {
    axes.iter().copied().product()
}

/// Validate the element count shared by both transpose directions.
///
/// ## Parameters
/// - `len`: Length of the slice handed in.
/// - `axes`: Lengths of the non-batch axes.
/// - `n_batch`: Number of batch elements.
///
/// ## Returns
/// `(cells, total)` — cells per batch element and the required slice length.
///
/// ## Errors
/// Returns a description when `len` does not match `cells × n_batch`, or when
/// that product overflows `usize`.
fn checked_shape(len: usize, axes: &[usize], n_batch: usize) -> Result<(usize, usize), String> {
    let cells = cells_per_batch(axes);
    let total = cells
        .checked_mul(n_batch)
        .ok_or_else(|| format!("layout {axes:?} x {n_batch} batches overflows usize"))?;
    if len != total {
        return Err(format!(
            "layout {axes:?} x {n_batch} batches needs {total} elements, got {len}"
        ));
    }
    Ok((cells, total))
}

/// Reorder a batch-major CPU slice into the batch-minor device layout.
///
/// Input index `b · C + c` (batch `b`, cell `c`, `C = product(axes)`) becomes
/// output index `c · B + b`; every other element position is a plain copy of
/// one input element.
///
/// ## Parameters
/// - `outer`: Batch-major slice, length `product(axes) · n_batch`.
/// - `axes`: Lengths of the non-batch axes, outermost first.
/// - `n_batch`: Number of batch elements.
///
/// ## Returns
/// The transposed slice in `(d0, …, dk, B)` order.
///
/// ## Errors
/// Returns a description when `outer` has the wrong length or the shape
/// overflows `usize`.
pub fn batch_to_inner<T: Copy>(
    outer: &[T],
    axes: &[usize],
    n_batch: usize,
) -> Result<Vec<T>, String> {
    let (cells, total) = checked_shape(outer.len(), axes, n_batch)?;
    // The nested push order is the cell-major output order; the simple
    // index arithmetic is exact because both sides are permutations.
    let mut inner = Vec::with_capacity(total);
    for cell in 0..cells {
        for batch in 0..n_batch {
            inner.push(outer[batch * cells + cell]);
        }
    }
    Ok(inner)
}

/// Reorder a batch-minor device slice back into the batch-major CPU layout.
///
/// The inverse of [`batch_to_inner`]: input index `c · B + b` becomes output
/// index `b · C + c`.
///
/// ## Parameters
/// - `inner`: Batch-minor slice, length `product(axes) · n_batch`.
/// - `axes`: Lengths of the non-batch axes, outermost first.
/// - `n_batch`: Number of batch elements.
///
/// ## Returns
/// The transposed slice in `(B, d0, …, dk)` order.
///
/// ## Errors
/// Returns a description when `inner` has the wrong length or the shape
/// overflows `usize`.
pub fn batch_to_outer<T: Copy>(
    inner: &[T],
    axes: &[usize],
    n_batch: usize,
) -> Result<Vec<T>, String> {
    let (cells, total) = checked_shape(inner.len(), axes, n_batch)?;
    let mut outer = Vec::with_capacity(total);
    for batch in 0..n_batch {
        for cell in 0..cells {
            outer.push(inner[cell * n_batch + batch]);
        }
    }
    Ok(outer)
}

#[cfg(test)]
#[path = "../../tests/unit/gpu/layout.rs"]
mod tests;
