//! The runtime of `host.runic`: it calls the object, which calls the host and `helper`. Built
//! with `--cfg bare`, it does not define `helper`, and nothing else does.

#![no_std]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}

unsafe extern "C" {
    fn use_host(x: i32) -> i32;
}

/// Adds 1000, for the object.
#[cfg(not(bare))]
#[unsafe(no_mangle)]
pub extern "C" fn helper(x: i32) -> i32 {
    x + 1000
}

/// What the object computes for `x`.
#[unsafe(no_mangle)]
pub extern "C" fn hosted(x: i32) -> i32 {
    unsafe { use_host(x) }
}
