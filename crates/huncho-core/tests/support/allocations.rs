use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    static TRACK: Cell<bool> = const { Cell::new(false) };
    static CALLS: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
}

struct Meter;
fn record(bytes: usize) {
    let _ = TRACK.try_with(|track| {
        if track.get() {
            let _ = CALLS.try_with(|calls| {
                let (count, requested) = calls.get();
                calls.set((count.saturating_add(1), requested.saturating_add(bytes)));
            });
        }
    });
}

// SAFETY: every allocation/deallocation delegates unchanged to System. Only
// allocation-free thread-local counters inspect successful alloc/realloc calls.
unsafe impl GlobalAlloc for Meter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let result = unsafe { System.alloc(layout) };
        if !result.is_null() {
            record(layout.size());
        }
        result
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let result = unsafe { System.alloc_zeroed(layout) };
        if !result.is_null() {
            record(layout.size());
        }
        result
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, bytes: usize) -> *mut u8 {
        let result = unsafe { System.realloc(ptr, layout, bytes) };
        if !result.is_null() {
            record(bytes);
        }
        result
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
    }
}
#[global_allocator]
static ALLOCATOR: Meter = Meter;

pub fn measure<T>(f: impl FnOnce() -> T) -> (T, (usize, usize)) {
    struct Disable;
    impl Drop for Disable {
        fn drop(&mut self) {
            TRACK.with(|track| track.set(false));
        }
    }
    CALLS.with(|calls| calls.set((0, 0)));
    TRACK.with(|track| track.set(true));
    let disable = Disable;
    let result = f();
    drop(disable);
    (result, CALLS.with(|calls| calls.get()))
}
