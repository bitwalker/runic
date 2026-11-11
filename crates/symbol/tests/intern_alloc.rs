//! Interning a string that is already interned must not allocate.
//!
//! This lives in its own integration test binary because it installs a counting global allocator.

use std::{
    alloc::{GlobalAlloc, Layout, System},
    borrow::Cow,
    cell::Cell,
};

use runic_symbol::Symbol;

thread_local! {
    /// The number of allocations made by the current thread.
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // `try_with`: the allocator can be called while this thread's locals are being torn down.
        let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Returns the number of allocations this thread makes while running `f`.
fn allocations_in<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let before = ALLOCATIONS.with(Cell::get);
    let value = f();
    (value, ALLOCATIONS.with(Cell::get) - before)
}

/// Long enough that no small-string optimisation could hold it inline.
const PATH: &str =
    "crates/some/deeply/nested/directory/containing/a/source/file/named/module.runic";

#[test]
fn interning_an_interned_string_does_not_allocate() {
    let (first, first_allocations) = allocations_in(|| Symbol::intern(PATH));
    assert!(first_allocations > 0, "a new string has to be copied into the interner");
    assert_eq!(first.as_str(), PATH);

    let (again, allocations) = allocations_in(|| Symbol::intern(PATH));
    assert_eq!(again, first);
    assert_eq!(allocations, 0, "`intern(&str)` allocated for an interned string");

    let owned = String::from(PATH);
    let (again, allocations) = allocations_in(|| Symbol::intern(&owned));
    assert_eq!(again, first);
    assert_eq!(allocations, 0, "`intern(&String)` allocated for an interned string");

    let (again, allocations) = allocations_in(|| Symbol::intern_owned(owned));
    assert_eq!(again, first);
    assert_eq!(allocations, 0, "`intern_owned` allocated for an interned string");

    let boxed: Box<str> = Box::from(PATH);
    let (again, allocations) = allocations_in(|| Symbol::from(boxed));
    assert_eq!(again, first);
    assert_eq!(allocations, 0, "`From<Box<str>>` allocated for an interned string");

    let (again, allocations) = allocations_in(|| Symbol::from(Cow::Borrowed(PATH)));
    assert_eq!(again, first);
    assert_eq!(allocations, 0, "`From<Cow<'static, str>>` allocated for an interned string");

    assert_eq!(Symbol::lookup(PATH), Some(first));
    assert_eq!(Symbol::lookup("a string that was never interned by this test binary"), None);
}

#[test]
fn interning_a_new_owned_string_reuses_its_allocation() {
    let owned =
        String::from("crates/another/long/path/that/only/this/test/interns/once/unique.runic");
    let pointer = owned.as_ptr();
    let capacity_is_exact = owned.capacity() == owned.len();
    let symbol = Symbol::intern_owned(owned);
    if capacity_is_exact {
        assert_eq!(symbol.as_str().as_ptr(), pointer, "`intern_owned` copied a new string");
    }
    assert_eq!(Symbol::intern(symbol.as_str()), symbol);
}
