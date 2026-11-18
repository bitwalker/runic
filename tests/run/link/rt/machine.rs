//! The runtime of `machine.runic`: it enters the object's state machine through its
//! `extern("C")` wrapper.

#![no_std]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}

unsafe extern "C" {
    fn run_machine(k: i32) -> i64;
}

/// The machine run for `k` steps.
#[unsafe(no_mangle)]
pub extern "C" fn machine(k: i32) -> i64 {
    unsafe { run_machine(k) }
}
