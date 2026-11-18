//! The runtime of `stack.runic`: `fill` writes through the address of a slot in the object's
//! frame, and calls back into the object while its own frame holds an array.

#![no_std]

use core::hint::black_box;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}

unsafe extern "C" {
    fn framed(n: i32) -> i32;
    fn slot_address() -> usize;
}

/// Writes `10 n` through `slot`, and for `n > 0` returns `framed(n - 1)`. An array of its own,
/// on the stack the object shares, must hold `n` in every element after the call; if it does
/// not, the result is -1000000.
#[unsafe(no_mangle)]
pub extern "C" fn fill(slot: *mut i32, n: i32) -> i32 {
    let mut mine = [n; 8];
    black_box(&mut mine);
    unsafe { *slot = n * 10 };
    let r = if n > 0 { unsafe { framed(n - 1) } } else { 0 };
    if black_box(&mine).iter().all(|&v| v == n) { r } else { -1_000_000 }
}

/// `framed(n)` twice in a row: the second call finds the stack where the first left it.
#[unsafe(no_mangle)]
pub extern "C" fn twice(n: i32) -> i32 {
    unsafe { framed(n) + framed(n) }
}

/// `framed(n)`, which recurses through `fill` `n` times.
#[unsafe(no_mangle)]
pub extern "C" fn depth(n: i32) -> i32 {
    unsafe { framed(n) }
}

/// 1 when the slot the object allocates is below this function's own array, within 1 KiB of
/// it, and at the same address on a second call; 0 otherwise.
#[unsafe(no_mangle)]
pub extern "C" fn layout() -> i32 {
    let mine = [0u8; 64];
    let here = black_box(&mine).as_ptr() as usize;
    let first = unsafe { slot_address() };
    let second = unsafe { slot_address() };
    (first == second && first < here && here - first < 1024) as i32
}
