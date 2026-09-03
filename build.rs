use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=native/sleef_shim.c");
    println!("cargo:rerun-if-changed=native/topk_shim.cpp");
    println!("cargo:rerun-if-env-changed=SLEEF_ROOT");

    let root = env::var_os("SLEEF_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/opt/homebrew/opt/sleef"));
    let include = root.join("include");
    let library = root.join("lib");
    if !include.join("sleef.h").is_file() || !library.join("libsleef.dylib").is_file() {
        panic!("SLEEF is required; install it with `brew install sleef` or set SLEEF_ROOT");
    }

    cc::Build::new()
        .file("native/sleef_shim.c")
        .include(include)
        .define("ACCELERATE_NEW_LAPACK", None)
        .define("ACCELERATE_LAPACK_ILP64", None)
        .flag_if_supported("-std=c11")
        .compile("firewing_sleef_shim");
    cc::Build::new()
        .cpp(true)
        .file("native/topk_shim.cpp")
        .flag_if_supported("-std=c++17")
        .compile("firewing_topk_shim");
    println!("cargo:rustc-link-search=native={}", library.display());
    println!("cargo:rustc-link-lib=dylib=sleef");
    println!("cargo:rustc-link-lib=framework=Accelerate");
}
