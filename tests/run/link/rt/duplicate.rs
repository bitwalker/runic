//! The runtime of `duplicate-export.runic`: it exports a function under the name `shown`, which
//! the object exports one of its own functions under too.

#![no_std]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}

/// Exported as `shown`, as Rust exports a `#[no_mangle]` function of a `cdylib`.
#[unsafe(no_mangle)]
pub extern "C" fn shown(x: i32) -> i32 {
    x * 100
}
