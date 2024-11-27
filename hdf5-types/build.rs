fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rustc-check-cfg=cfg(windows_dll)");
    if std::env::var_os("DEP_HDF5_MSVC_DLL_INDIRECTION").is_some() {
        println!("cargo::rustc-cfg=windows_dll");
    }

    // Declare the known HDF5 versions we might feature flag on
    // in this crate.
    println!("cargo::rustc-check-cfg=cfg(feature, values(\"1.12.0\"))");

    for (key, _) in std::env::vars() {
        if key.starts_with("DEP_HDF5_VERSION_") {
            let version = key.trim_start_matches("DEP_HDF5_VERSION_").replace("_", ".");
            println!("cargo::rustc-cfg=feature=\"{version}\"");
        }
    }
}
