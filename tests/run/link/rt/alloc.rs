//! The runtime of `alloc.runic`, with `std`: it allocates the vector the object fills, and sums
//! and frees it.

unsafe extern "C" {
    fn fill_squares(n: i32) -> i64;
}

/// A vector of `n` zeros on the heap, given to the object as a pointer to its first element.
#[unsafe(no_mangle)]
pub extern "C" fn make_vec(n: i32) -> *mut i64 {
    let elements = vec![0i64; n as usize].into_boxed_slice();
    Box::into_raw(elements) as *mut i64
}

/// The sum of the `n` elements of a vector `make_vec` made, which it frees.
#[unsafe(no_mangle)]
pub extern "C" fn sum_vec(elements: *mut i64, n: i32) -> i64 {
    let elements =
        unsafe { Box::from_raw(core::ptr::slice_from_raw_parts_mut(elements, n as usize)) };
    elements.iter().sum()
}

/// What the object computes for `n`.
#[unsafe(no_mangle)]
pub extern "C" fn squares(n: i32) -> i64 {
    unsafe { fill_squares(n) }
}
