use std::env;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Version {
    pub major: u8,
    pub minor: u8,
    pub micro: u8,
}

impl Version {
    pub const fn new(major: u8, minor: u8, micro: u8) -> Self {
        Self { major, minor, micro }
    }
}

fn known_hdf5_versions() -> Vec<Version> {
    // Keep up to date with known_hdf5_versions in hdf5-sys
    let mut vs = Vec::new();
    vs.push(Version::new(2, 0, 0)); // 2.0.0
    vs.extend((5..=21).map(|v| Version::new(1, 8, v))); // 1.8.[5-23]
    vs.extend((0..=8).map(|v| Version::new(1, 10, v))); // 1.10.[0-10]
    vs.extend((0..=2).map(|v| Version::new(1, 12, v))); // 1.12.[0-2]
    vs.extend((0..=6).map(|v| Version::new(1, 14, v))); // 1.14.[0-6]
    vs
}

fn main() {
    for version in known_hdf5_versions() {
        println!(
            "cargo::rustc-check-cfg=cfg(feature, values(\"{}.{}.{}\"))",
            version.major, version.minor, version.micro
        );
    }
    for feature in ["have-direct", "have-parallel", "have-threadsafe", "have-filter-deflate"] {
        println!("cargo::rustc-check-cfg=cfg(feature, values(\"{feature}\"))");
    }
    println!("cargo::rustc-check-cfg=cfg(msvc_dll_indirection)");

    let print_feature = |key: &str| println!("cargo::rustc-cfg=feature=\"{key}\"");
    let print_cfg = |key: &str| println!("cargo::rustc-cfg={key}");
    for (key, _) in env::vars() {
        match key.as_str() {
            // public features
            "DEP_HDF5_HAVE_DIRECT" => print_feature("have-direct"),
            "DEP_HDF5_HAVE_PARALLEL" => print_feature("have-parallel"),
            "DEP_HDF5_HAVE_THREADSAFE" => print_feature("have-threadsafe"),
            "DEP_HDF5_HAVE_FILTER_DEFLATE" => print_feature("have-filter-deflate"),
            // internal config flags
            "DEP_HDF5_MSVC_DLL_INDIRECTION" => print_cfg("msvc_dll_indirection"),
            // public version features
            key if key.starts_with("DEP_HDF5_VERSION_") => {
                print_feature(&key.trim_start_matches("DEP_HDF5_VERSION_").replace('_', "."));
            }
            _ => continue,
        }
    }
}
