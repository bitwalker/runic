//! The runtime of `section-rust.runic`: it contributes one entry of its own to the data section
//! `tally`, which the object contributes to too, and walks the section between the bounds the
//! linker defines. Built with `--cfg pointer`, it also tries to contribute an entry that holds a
//! function pointer, which `rustc` refuses on the Wasm targets.

#![no_std]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}

/// An entry as the object writes it, a `struct<i32, i32>`: an id and a value, with no pointer.
#[repr(C)]
pub struct Entry {
    id: i32,
    value: i32,
}

/// Rust's own entry of the section: plain bytes, which a Rust static in a named section may be.
#[used]
#[unsafe(link_section = "tally")]
static RUST_ENTRY: Entry = Entry { id: 3, value: 300 };

/// An entry that holds a function pointer, which needs a relocation: `rustc` refuses it.
#[cfg(pointer)]
#[repr(C)]
pub struct Callable {
    id: i32,
    function: extern "C" fn(i32) -> i32,
}

#[cfg(pointer)]
extern "C" fn rust_double(x: i32) -> i32 {
    x * 2
}

#[cfg(pointer)]
#[used]
#[unsafe(link_section = "tally")]
static RUST_CALLABLE: Callable = Callable {
    id: 4,
    function: rust_double,
};

unsafe extern "C" {
    static __start_tally: Entry;
    static __stop_tally: Entry;
    fn compiled_count() -> i32;
}

fn entries() -> &'static [Entry] {
    unsafe {
        let start = &raw const __start_tally;
        let stop = &raw const __stop_tally;
        core::slice::from_raw_parts(start, stop.offset_from(start) as usize)
    }
}

/// The number of entries in the section, as Rust counts them.
#[unsafe(no_mangle)]
pub extern "C" fn count() -> i32 {
    entries().len() as i32
}

/// The number of entries in the section, as compiled code counts them.
#[unsafe(no_mangle)]
pub extern "C" fn count_compiled() -> i32 {
    unsafe { compiled_count() }
}

/// The value of the entry `id`, or -1 when no entry has that id.
#[unsafe(no_mangle)]
pub extern "C" fn value(id: i32) -> i32 {
    entries().iter().find(|entry| entry.id == id).map_or(-1, |entry| entry.value)
}
