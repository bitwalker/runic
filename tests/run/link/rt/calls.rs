//! The runtime of `calls.runic`: it calls `compiled_mix`, which the object defines, and defines
//! `rust_mix`, which the object calls. Built with `--cfg mismatch`, it declares `compiled_mix`
//! with an `f32` where the object has an `f64`, and the link must fail.

#![no_std]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}

unsafe extern "C" {
    #[cfg(not(mismatch))]
    fn compiled_mix(cell: *mut i32, b: i8, x: i64, f: f64) -> i8;
    #[cfg(mismatch)]
    fn compiled_mix(cell: *mut i32, b: i8, x: i64, f: f32) -> i8;
}

/// Doubles the cell and returns `b + x + f` (`f` truncated toward zero) as an `i8`, wrapping.
#[unsafe(no_mangle)]
pub extern "C" fn rust_mix(cell: *mut i32, b: i8, x: i64, f: f64) -> i8 {
    unsafe { *cell *= 2 };
    (b as i64).wrapping_add(x).wrapping_add(f as i64) as i8
}

/// Calls the object with a cell holding 5, and returns its `i8` result times 1000 plus the cell.
#[unsafe(no_mangle)]
pub extern "C" fn round_trip(b: i32, x: i64, f: f64) -> i32 {
    let mut cell = 5;
    #[cfg(mismatch)]
    let f = f as f32;
    let r = unsafe { compiled_mix(&mut cell, b as i8, x, f) };
    (r as i32) * 1000 + cell
}
