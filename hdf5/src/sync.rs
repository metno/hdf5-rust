use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::LazyLock;

pub(crate) use hdf5_sys::LOCK;

thread_local! {
    pub static SILENCED: AtomicBool = AtomicBool::new(false);
}

pub(crate) static LIBRARY_INIT: LazyLock<()> = LazyLock::new(|| {
    let _guard = hdf5_sys::LOCK.lock();
    unsafe {
        // Ensure hdf5 does not invalidate handles which might
        // still be live on other threads on program exit
        ::hdf5_sys::h5::H5dont_atexit();
        ::hdf5_sys::h5::H5open();
        // Ignore errors on stdout
        crate::error::silence_errors_no_sync(true);
        // Register filters lzf/blosc if available
        crate::hl::filters::register_filters();
    }
});

/// Guards the execution of the provided closure with a recursive static mutex.
pub fn sync<T, F>(func: F) -> T
where
    F: FnOnce() -> T,
{
    let _ = LazyLock::force(&LIBRARY_INIT);
    SILENCED.with(|silence| {
        let is_silenced = silence.load(Ordering::Acquire);
        if !is_silenced {
            let _guard = LOCK.lock();
            unsafe {
                crate::error::silence_errors_no_sync(true);
            }
            silence.store(true, Ordering::Release);
        }
    });
    let _guard = LOCK.lock();
    func()
}

#[cfg(test)]
mod tests {
    use parking_lot::ReentrantMutex;
    use std::sync::LazyLock;

    #[test]
    pub fn test_reentrant_mutex() {
        static LOCK: LazyLock<ReentrantMutex<()>> = LazyLock::new(|| ReentrantMutex::new(()));
        let g1 = LOCK.try_lock();
        assert!(g1.is_some());
        let g2 = LOCK.lock();
        assert_eq!(*g2, ());
        let g3 = LOCK.try_lock();
        assert!(g3.is_some());
        let g4 = LOCK.lock();
        assert_eq!(*g4, ());
    }

    #[test]
    // Test for locking behaviour on initialisation
    pub fn lock_part1() {
        let _ = *crate::globals::H5P_ROOT;
    }

    #[test]
    // Test for locking behaviour on initialisation
    pub fn lock_part2() {
        let _ = h5call!(*crate::globals::H5P_ROOT);
    }
}
