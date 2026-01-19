use std::env;

fn feature_enabled(feature: &str) -> bool {
    env::var(format!("CARGO_FEATURE_{}", feature)).is_ok()
}

// Fix for: CMake Error: TRY_RUN() invoked in cross-compiling mode, please set the following cache variables appropriately
fn configure_cmake_cross_run_advanced_cache_vars(cfg: &mut cmake::Config) {
    for &option in &[
        "TEST_LFS_WORKS_RUN",
        "H5_PRINTF_LL_TEST_RUN",
        "H5_PRINTF_LL_TEST_RUN__TRYRUN_OUTPUT",
        "H5_LDOUBLE_TO_LONG_SPECIAL_RUN",
        "H5_LDOUBLE_TO_LONG_SPECIAL_RUN__TRYRUN_OUTPUT",
        "H5_LONG_TO_LDOUBLE_SPECIAL_RUN",
        "H5_LONG_TO_LDOUBLE_SPECIAL_RUN__TRYRUN_OUTPUT",
        "H5_LDOUBLE_TO_LLONG_ACCURATE_RUN",
        "H5_LDOUBLE_TO_LLONG_ACCURATE_RUN__TRYRUN_OUTPUT",
        "H5_LLONG_TO_LDOUBLE_CORRECT_RUN",
        "H5_LLONG_TO_LDOUBLE_CORRECT_RUN__TRYRUN_OUTPUT",
        "H5_DISABLE_SOME_LDOUBLE_CONV_RUN",
        "H5_DISABLE_SOME_LDOUBLE_CONV_RUN__TRYRUN_OUTPUT",
        "H5_NO_ALIGNMENT_RESTRICTIONS_RUN",
        "H5_NO_ALIGNMENT_RESTRICTIONS_RUN__TRYRUN_OUTPUT",
    ] {
        println!("cargo::rerun-if-env-changed={option}");
        let value = env::var(option).unwrap_or_else(|_| "OFF".to_string());
        cfg.define(option, value);
    }
}

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    let mut cfg = cmake::Config::new("ext/hdf5");

    if cfg!(target_env = "msvc") {
        cfg.define("CMAKE_POLICY_DEFAULT_CMP0091", "NEW");
        if let Ok(var) = env::var("CMAKE_MSVC_RUNTIME_LIBRARY") {
            cfg.define("CMAKE_MSVC_RUNTIME_LIBRARY", var);
        }
    }

    // only build the static c library, disable everything else
    cfg.define("HDF5_NO_PACKAGES", "ON");
    for option in &[
        "BUILD_SHARED_LIBS",
        "BUILD_TESTING",
        "HDF5_BUILD_TOOLS",
        "HDF5_BUILD_EXAMPLES",
        "HDF5_BUILD_JAVA",
        "HDF5_BUILD_FORTRAN",
        "HDF5_BUILD_CPP_LIB",
        "HDF5_BUILD_UTILS",
        "HDF5_ENABLE_PARALLEL",
        "HDF5_ENABLE_NONSTANDARD_FEATURES",
    ] {
        cfg.define(option, "OFF");
    }

    // disable these by default, can be enabled via features
    for option in &[
        "HDF5_ENABLE_DEPRECATED_SYMBOLS",
        "HDF5_ENABLE_THREADSAFE",
        "ALLOW_UNSUPPORTED",
        "HDF5_BUILD_HL_LIB",
        "HDF5_ENABLE_NONSTANDARD_FEATURE_FLOAT16",
        "HDF5_ENABLE_SZIP_SUPPORT",
    ] {
        cfg.define(option, "OFF");
    }

    if feature_enabled("ZLIB") {
        let zlib_include_dir = env::var_os("DEP_Z_INCLUDE").unwrap();
        let mut zlib_header = env::split_paths(&zlib_include_dir).next().unwrap();
        zlib_header.push("zlib.h");
        let zlib_lib = "z";
        cfg.define("HDF5_ENABLE_ZLIB_SUPPORT", "ON")
            .define("H5_ZLIB_HEADER", &zlib_header)
            .define("ZLIB_STATIC_LIBRARY", zlib_lib);
        println!("cargo::metadata=zlib_header={}", zlib_header.to_str().unwrap());
        println!("cargo::metadata=zlib={}", zlib_lib);
    } else {
        cfg.define("HDF5_ENABLE_Z_LIB_SUPPORT", "OFF");
    }

    if feature_enabled("DEPRECATED") {
        cfg.define("HDF5_ENABLE_DEPRECATED_SYMBOLS", "ON");
    }

    if feature_enabled("THREADSAFE") {
        cfg.define("HDF5_ENABLE_THREADSAFE", "ON");
        if feature_enabled("HL") {
            println!("cargo::warning=Unsupported HDF5 options: hl with threadsafe.");
            cfg.define("ALLOW_UNSUPPORTED", "ON");
        }
    }

    let targeting_windows = env::var("CARGO_CFG_TARGET_OS").unwrap() == "windows";

    if feature_enabled("HL") {
        cfg.define("HDF5_BUILD_HL_LIB", "ON");
        let hdf5_hl_lib =
            if cfg!(target_env = "msvc") { "libhdf5_hl" } else { "hdf5_hl" }.to_owned();
        println!("cargo::metadata=hl_library={}", hdf5_hl_lib);
    }

    if cfg!(unix) && targeting_windows {
        let wine_exec =
            if env::var("CARGO_CFG_TARGET_ARCH").unwrap() == "x86_64" { "wine64" } else { "wine" };
        // when cross-compiling to windows, use Wine to run code generation programs
        cfg.define("CMAKE_CROSSCOMPILING_EMULATOR", wine_exec);
    }

    configure_cmake_cross_run_advanced_cache_vars(&mut cfg);
    let dst = cfg.build();
    println!("cargo::metadata=root={}", dst.display());

    let hdf5_incdir = format!("{}/include", dst.display());
    println!("cargo::metadata=include={}", hdf5_incdir);

    let hdf5_lib = if cfg!(target_env = "msvc") { "libhdf5" } else { "hdf5" }.to_owned();

    println!("cargo::metadata=library={}", hdf5_lib);
}
