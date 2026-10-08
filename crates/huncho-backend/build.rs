// Record the CPU arithmetic kernels selected at compilation, rather than the
// features available on whichever host later runs this binary. Candle's packed
// dot products choose AVX2/NEON/WASM implementations with target_feature cfgs.
fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_FEATURE");
    println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_ARCH");
    let features = std::env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    let mut arithmetic: Vec<_> = features
        .split(',')
        .filter(|feature| {
            matches!(
                *feature,
                "avx" | "avx2" | "f16c" | "fma" | "neon" | "dotprod" | "simd128"
            ) || feature.starts_with("avx512")
        })
        .collect();
    arithmetic.sort_unstable();
    arithmetic.dedup();
    let profile = if arithmetic.is_empty() {
        String::new()
    } else {
        format!(
            "{}:{}",
            std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default(),
            arithmetic.join(",")
        )
    };
    println!("cargo:rustc-env=HUNCHO_COMPILED_CPU_KERNELS={profile}");
}
