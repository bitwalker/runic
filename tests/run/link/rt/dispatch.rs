//! The runtime of `dispatch.runic` and `dispatch-more.runic`: it finds the entries of the data
//! section `dispatch` between the bounds the linker defines, and calls through them.

#![no_std]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}

/// An entry as the objects write it, a `struct<i32, i32, (i64) -> i64>`.
#[repr(C)]
pub struct Entry {
    id: i32,
    arity: i32,
    function: extern "C" fn(i64) -> i64,
}

unsafe extern "C" {
    static __start_dispatch: Entry;
    static __stop_dispatch: Entry;
}

fn entries() -> &'static [Entry] {
    unsafe {
        let start = &raw const __start_dispatch;
        let stop = &raw const __stop_dispatch;
        core::slice::from_raw_parts(start, stop.offset_from(start) as usize)
    }
}

/// The number of entries in the section.
#[unsafe(no_mangle)]
pub extern "C" fn count() -> i32 {
    entries().len() as i32
}

/// The sum of the entries' arities.
#[unsafe(no_mangle)]
pub extern "C" fn arities() -> i32 {
    entries().iter().map(|entry| entry.arity).sum()
}

/// The function of the entry `id` called with `x`, or -1 when no entry has that id.
#[unsafe(no_mangle)]
pub extern "C" fn call(id: i32, x: i64) -> i64 {
    match entries().iter().find(|entry| entry.id == id) {
        Some(entry) => (entry.function)(x),
        None => -1,
    }
}
