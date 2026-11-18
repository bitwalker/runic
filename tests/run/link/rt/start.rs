//! The runtime of `start.runic`: a constructor of its own, which counts its runs, and an `init`
//! that runs every constructor of the link once. Built with `--cfg wrapped`, `init` does
//! nothing and nothing names `__wasm_call_ctors`, so the linker wraps every export to run the
//! constructors first.

#![no_std]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}

unsafe extern "C" {
    /// The object's count of the runs of its start function.
    static mut OBJECT_STARTS: i32;
    #[cfg(not(wrapped))]
    fn __wasm_call_ctors();
}

/// The count of the runs of `rust_start`.
static mut RUST_STARTS: i32 = 0;

extern "C" fn rust_start() {
    unsafe { RUST_STARTS += 1 };
}

/// Rust's constructor: a function pointer in `.init_array`, which LLVM makes an init function
/// of the object, as the object's start function is one of its own.
#[used]
#[unsafe(link_section = ".init_array")]
static RUST_CONSTRUCTOR: extern "C" fn() = rust_start;

/// Runs the constructors of the link, the object's and Rust's, once. On the WASI targets the
/// export `_initialize` that wasi-libc's `crt1-reactor.o` defines does this.
#[unsafe(no_mangle)]
pub extern "C" fn init() {
    #[cfg(not(wrapped))]
    unsafe {
        __wasm_call_ctors()
    };
}

/// The runs of the object's start function times 10, plus the runs of Rust's constructor.
#[unsafe(no_mangle)]
pub extern "C" fn counts() -> i32 {
    unsafe { OBJECT_STARTS * 10 + RUST_STARTS }
}
