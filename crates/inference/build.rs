//! When the user has CUDA Tookit, remind to enable CUDA feature for better performance

use std::env;
use std::process::Command;

fn main() {
    if has_cuda_toolkit() && !has_feature("cuda") && is_cuda_engine() {
        println!("cargo:warning=CUDA not enabled, re-run with `--features cuda`");
    }
    if is_mac() && !has_feature("metal") {
        println!("cargo:warning=Metal not enabled, re-run with `--features metal`");
    }
}

fn has_feature(s: &str) -> bool {
    env::var(format!("CARGO_FEATURE_{}", s.to_uppercase())).is_ok()
}

fn has_cuda_toolkit() -> bool {
    if let Ok(output) = Command::new("nvcc").arg("--version").output() {
        output.status.success()
    } else {
        false
    }
}

fn is_cuda_engine() -> bool {
    has_feature("mistralrs") || has_feature("llamacpp")
}

#[cfg(target_os = "macos")]
fn is_mac() -> bool {
    true
}

#[cfg(not(target_os = "macos"))]
fn is_mac() -> bool {
    false
}
