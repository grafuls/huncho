//! Static CPU kernel identity. This neither enables instructions nor performs
//! runtime dispatch; a nonportable build must run on compatible hardware.

pub(crate) fn record(extra: &mut std::collections::BTreeMap<String, String>) {
    let profile = env!("HUNCHO_COMPILED_CPU_KERNELS");
    if !profile.is_empty() {
        extra.insert("cpu_kernel_build".into(), profile.into());
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn identity_tracks_compiled_kernels_instead_of_host_detection() {
        let mut extra = Default::default();
        super::record(&mut extra);
        let profile = extra.get("cpu_kernel_build");
        assert_eq!(
            profile.is_some(),
            !env!("HUNCHO_COMPILED_CPU_KERNELS").is_empty()
        );
        for (feature, enabled) in [
            ("avx2", cfg!(target_feature = "avx2")),
            ("f16c", cfg!(target_feature = "f16c")),
            ("fma", cfg!(target_feature = "fma")),
            ("neon", cfg!(target_feature = "neon")),
            ("simd128", cfg!(target_feature = "simd128")),
        ] {
            assert_eq!(
                profile.is_some_and(|profile| profile
                    .split_once(':')
                    .unwrap()
                    .1
                    .split(',')
                    .any(|item| item == feature)),
                enabled,
                "{feature}"
            );
        }
    }
}
