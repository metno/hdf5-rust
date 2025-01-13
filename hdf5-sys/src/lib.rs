//! Rust bindings to the `hdf5` library for reading and writing data to and from storage
#![allow(non_camel_case_types, non_snake_case, dead_code, deprecated)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::cognitive_complexity)]
#![allow(clippy::upper_case_acronyms)]
#![allow(clippy::wildcard_imports)]
#![allow(clippy::module_name_repetitions)]
#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg(feature = "static")]
extern crate hdf5_src;
#[cfg(feature = "zlib")]
extern crate libz_sys;
#[cfg(feature = "mpio")]
extern crate mpi_sys;

macro_rules! extern_static {
    ($dest:ident, $src:ident) => {
        extern "C" {
            static $src: id_t;
        }
        pub static $dest: &'static id_t = unsafe { &$src };
    };
}

#[cfg(all(feature = "mpio", not(feature = "have-parallel")))]
compile_error!("Enabling \"mpio\" feature requires HDF5 library built with MPI support");

#[cfg(all(feature = "mpio", feature = "static"))]
compile_error!("\"mpio\" and \"static\" are incompatible features");

pub mod h5;
pub mod h5a;
pub mod h5ac;
pub mod h5c;
pub mod h5d;
pub mod h5e;
pub mod h5f;
pub mod h5fd;
pub mod h5g;
pub mod h5i;
pub mod h5l;
pub mod h5mm;
pub mod h5o;
pub mod h5p;
pub mod h5r;
pub mod h5s;
pub mod h5t;
pub mod h5vl;
pub mod h5z;

#[cfg(feature = "1.8.15")]
pub mod h5pl;

#[cfg(feature = "1.14.0")]
pub mod h5es;

#[allow(non_camel_case_types)]
mod internal_prelude {
    pub use crate::h5::{
        haddr_t, hbool_t, herr_t, hsize_t, hssize_t, htri_t, H5_ih_info_t, H5_index_t,
        H5_iter_order_t,
    };
    pub use crate::h5i::hid_t;
    pub use crate::h5t::H5T_cset_t;
    pub use libc::{int64_t, off_t, size_t, ssize_t, time_t, uint32_t, uint64_t, FILE};
    #[allow(unused_imports)]
    pub use std::os::raw::{
        c_char, c_double, c_float, c_int, c_long, c_longlong, c_uchar, c_uint, c_ulong,
        c_ulonglong, c_void,
    };
}

use parking_lot::ReentrantMutex;
/// Lock which can be used to serialise access to the hdf5 library
pub static LOCK: ReentrantMutex<()> = ReentrantMutex::new(());

include!(concat!(env!("OUT_DIR"), "/version.rs"));

#[cfg(test)]
mod tests {
    use super::h5::H5get_libversion;
    use super::h5::H5open;
    use super::h5p::H5P_CLS_ROOT;
    use super::{Version, HDF5_VERSION, LOCK};

    #[test]
    fn version_test() {
        let _lock = LOCK.lock();
        let (mut major, mut minor, mut micro) = (0, 0, 0);
        unsafe { H5get_libversion(&mut major, &mut minor, &mut micro) };
        let runtime_version = Version { major: major as _, minor: minor as _, micro: micro as _ };

        assert_eq!(runtime_version, HDF5_VERSION);
    }

    #[test]
    pub fn test_smoke() {
        let _lock = LOCK.lock();
        unsafe {
            H5open();
            assert!(*H5P_CLS_ROOT > 0);
        }
    }
}
