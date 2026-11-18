//! The runtime of `export-global.runic`: it exports `raise`, which calls the object.

#![no_std]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}

unsafe extern "C" {
    fn raise_level(by: i32);
}

/// Adds `by` to the object's global `level`.
#[unsafe(no_mangle)]
pub extern "C" fn raise(by: i32) {
    unsafe { raise_level(by) }
}
