//! The runtime of `heap.runic`: it compares the addresses the object reads with the ones Rust
//! sees, and with its own data.

#![no_std]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}

unsafe extern "C" {
    static __heap_base: u8;
    static __data_end: u8;
    fn heap_base() -> usize;
    fn data_end() -> usize;
}

/// A static of Rust's, which the linker places among the data, below `__data_end`.
#[unsafe(no_mangle)]
pub static mut RUST_DATA: [u32; 4] = [1, 2, 3, 4];

/// 1 when the object reads the `__heap_base` Rust sees.
#[unsafe(no_mangle)]
pub extern "C" fn same_heap_base() -> i32 {
    (unsafe { heap_base() } == &raw const __heap_base as usize) as i32
}

/// 1 when the object reads the `__data_end` Rust sees.
#[unsafe(no_mangle)]
pub extern "C" fn same_data_end() -> i32 {
    (unsafe { data_end() } == &raw const __data_end as usize) as i32
}

/// 1 when the data lies after the 1 MiB stack, Rust's static is within it, and the heap starts
/// at or after its end.
#[unsafe(no_mangle)]
pub extern "C" fn ordered() -> i32 {
    let data = &raw const RUST_DATA as usize;
    let (base, end) = unsafe { (heap_base(), data_end()) };
    (1 << 20 <= data && data < end && end <= base) as i32
}
