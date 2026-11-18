//! The runtime of `memory.runic`: two statics the object reads and writes, and a global of the
//! object that Rust reads and writes.

#![no_std]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}

/// Read and written by the object's `bump_rust`.
#[unsafe(no_mangle)]
pub static mut RUST_COUNTER: i32 = 7;

/// The object's data holds the address of its third element.
#[unsafe(no_mangle)]
pub static RUST_TABLE: [i32; 4] = [11, 22, 33, 44];

unsafe extern "C" {
    /// Defined by the object, with 40 in it.
    static mut SHARED: i64;
    fn bump_rust() -> i32;
    fn read_shared() -> i64;
    fn read_third() -> i32;
}

/// Sets `RUST_COUNTER` to 100, lets the object add 1 to it twice, and returns what the first
/// call returned times 1000 plus what Rust then reads.
#[unsafe(no_mangle)]
pub extern "C" fn counter() -> i32 {
    unsafe {
        RUST_COUNTER = 100;
        let first = bump_rust();
        bump_rust();
        first * 1000 + RUST_COUNTER
    }
}

/// Adds 2 to the object's `SHARED` and returns what the object then reads from it.
#[unsafe(no_mangle)]
pub extern "C" fn shared() -> i64 {
    unsafe {
        SHARED += 2;
        read_shared()
    }
}

/// What the object finds behind the address in its data.
#[unsafe(no_mangle)]
pub extern "C" fn third() -> i32 {
    unsafe { read_third() }
}
