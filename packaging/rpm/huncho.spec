%global crate_name huncho
%global debug_package %{nil}

Name:           huncho
Version:        0.1.0
Release:        4%{?dist}
Summary:        Portable serving engine for System One decision models

License:        Apache-2.0
URL:            https://github.com/grafuls/huncho
Source0:        %{name}-%{version}.tar.gz
Source1:        huncho.env
Source2:        huncho.service
Source3:        huncho.1
Source4:        huncho-sysusers.conf

# Rust toolchain + C compiler. `pkgconf-pkg-config` and OpenSSL are needed by
# the `hf` TLS provider (`reqwest` native-tls -> openssl-sys) and the ONNX
# Runtime build (ort).
BuildRequires:  cargo >= 1.80
BuildRequires:  rust >= 1.80
BuildRequires:  gcc
BuildRequires:  gcc-c++
BuildRequires:  openssl-devel
BuildRequires:  pkgconf-pkg-config
BuildRequires:  systemd-rpm-macros

# On RHEL/EPEL derivatives we build against the official ONNX Runtime Linux
# release instead of `ort-sys`'s prebuilt binary, so we need a download/extract
# toolchain in %build.
%if 0%{?rhel}
BuildRequires:  curl
BuildRequires:  tar
BuildRequires:  gzip
%endif

# rpm auto-detects the shared-library dependencies (libstdc++, libgcc_s, libm,
# libc) from the ELF. ca-certificates provides the CA roots the HF Hub TLS
# resolution needs at runtime and is not a linked library.
Requires:       ca-certificates

# The sysusers.d file below (Source4) makes rpm auto-generate
# `Provides: user(huncho), group(huncho)`, so dnf can resolve the package
# before the user exists. The `%pre` scriptlet calls systemd-sysusers to create
# the user; we declare the binary as a requirement so it is installed first
# (it lives in a separate systemd-standalone-sysusers package).
Requires(pre):  /usr/bin/systemd-sysusers

%description
Huncho is a portable serving engine for Jev-style System One decision models.
It takes a state and a set of typed questions and returns calibrated
probabilities — no text generation. It implements the Jev wire contract, so an
unmodified Python SDK works against Huncho with only a base-URL change.

This package builds the single `huncho` binary with the full feature set (ONNX
Runtime, the Candle backend that loads Hugging Face F1/ModernBERT weights
directly, Hugging Face Hub resolution by repo id, and official Hugging Face
tokenization). It also ships a systemd unit, an environment file, and the
built-in mock model package so the service runs out of the box.

%prep
%setup -q

%build
# Full feature set: ONNX Runtime (fetches a prebuilt runtime at build time, needs
# network), the Candle backend (primary F1/ModernBERT path, loads HF safetensors
# directly, no extra system libs — built with default-features=false), Hugging
# Face Hub resolution, and HF tokenizers.
#
# RHEL/EPEL 9 ships glibc 2.34 and GCC 11. The prebuilt ONNX Runtime that
# `ort-sys` downloads by default is built against glibc >= 2.38 / GCC 13
# libstdc++ (it references `__isoc23_strtol*` and `_M_replace_cold`), which is
# ABI-incompatible with el9. Use the official ONNX Runtime 1.28.0 Linux release
# instead — it is built for manylinux (glibc 2.17 baseline; max GLIBC_2.27,
# GLIBCXX_3.4.21) — and dynamic-link it.
%if 0%{?rhel}
mkdir -p %{_builddir}/onnxruntime
curl -sSL -o %{_builddir}/onnxruntime/onnxruntime.tgz \
    https://github.com/microsoft/onnxruntime/releases/download/v1.28.0/onnxruntime-linux-x64-1.28.0.tgz
tar -xzf %{_builddir}/onnxruntime/onnxruntime.tgz -C %{_builddir}/onnxruntime
export ORT_LIB_LOCATION=%{_builddir}/onnxruntime/onnxruntime-linux-x64-1.28.0/lib
export ORT_PREFER_DYNAMIC_LINK=1
%endif

cargo build --release --locked --features onnx,hf,tokenizers,candle --bin huncho
# Cargo's `strip = true` in [profile.release] should do this, but it was not
# applied under the rpmbuild environment; strip deterministically here.
strip --strip-unneeded target/release/huncho

%install
install -Dm0755 target/release/huncho %{buildroot}%{_bindir}/huncho
install -Dm0644 %{SOURCE2} %{buildroot}%{_unitdir}/huncho.service
install -Dm0644 %{SOURCE1} %{buildroot}%{_sysconfdir}/huncho/huncho.env
install -Dm0644 %{SOURCE4} %{buildroot}%{_sysusersdir}/huncho.conf

# Man page and package documentation.
install -Dm0644 %{SOURCE3} %{buildroot}%{_mandir}/man1/huncho.1
install -Dm0644 docs/operations.md %{buildroot}%{_docdir}/%{name}/operations.md
install -Dm0644 docs/model-package.md %{buildroot}%{_docdir}/%{name}/model-package.md

# Bundle the built-in mock model package so operators can test
# `huncho serve --manifest ...` without fetching weights.
install -Dm0644 examples/mock-model/huncho-model.json \
    %{buildroot}%{_datadir}/huncho/examples/mock-model/huncho-model.json
install -Dm0644 examples/mock-model/mock-model.onnx \
    %{buildroot}%{_datadir}/huncho/examples/mock-model/mock-model.onnx

# Ship the dynamically-linked ONNX Runtime shared library for el9. It is built
# for manylinux and is el9-compatible; the `huncho` binary carries a DT_NEEDED
# on libonnxruntime.so.1, which the loader resolves from %{_libdir}.
%if 0%{?rhel}
install -Dm0755 %{_builddir}/onnxruntime/onnxruntime-linux-x64-1.28.0/lib/libonnxruntime.so.1.28.0 \
    %{buildroot}%{_libdir}/libonnxruntime.so.1.28.0
ln -sf libonnxruntime.so.1.28.0 %{buildroot}%{_libdir}/libonnxruntime.so.1
ln -sf libonnxruntime.so.1 %{buildroot}%{_libdir}/libonnxruntime.so
install -Dm0755 %{_builddir}/onnxruntime/onnxruntime-linux-x64-1.28.0/lib/libonnxruntime_providers_shared.so \
    %{buildroot}%{_libdir}/libonnxruntime_providers_shared.so
%endif
install -Dm0644 examples/mock-model/golden.json \
    %{buildroot}%{_datadir}/huncho/examples/mock-model/golden.json
install -Dm0644 examples/mock-model/README.md \
    %{buildroot}%{_datadir}/huncho/examples/mock-model/README.md

# Model cache dir (also seeds HF resolution); owned by the service user.
mkdir -p %{buildroot}%{_localstatedir}/lib/huncho

%pre
# Declare the user via systemd-sysusers (idempotent). rpm auto-adds the
# `Requires(pre): /usr/bin/systemd-sysusers` and `Provides: user/group(huncho)`.
%sysusers_create_package %{name} %{SOURCE4}

%post
%systemd_post huncho.service

%preun
%systemd_preun huncho.service

%postun
%systemd_postun_with_restart huncho.service

%files
%{_bindir}/huncho
%{_unitdir}/huncho.service
%config(noreplace) %attr(0644,root,root) %{_sysconfdir}/huncho/huncho.env
%{_sysusersdir}/huncho.conf
%dir %attr(0755,huncho,huncho) %{_localstatedir}/lib/huncho
%doc README.md
# brp-compress gzips man pages, so match either form.
%{_mandir}/man1/huncho.1*
%{_docdir}/%{name}/operations.md
%{_docdir}/%{name}/model-package.md
%{_datadir}/huncho/examples/mock-model/huncho-model.json
%{_datadir}/huncho/examples/mock-model/mock-model.onnx
%{_datadir}/huncho/examples/mock-model/golden.json
%{_datadir}/huncho/examples/mock-model/README.md

# The el9 build dynamic-links ONNX Runtime; ship the shared library with it.
%if 0%{?rhel}
%{_libdir}/libonnxruntime.so.1.28.0
%{_libdir}/libonnxruntime.so.1
%{_libdir}/libonnxruntime.so
%{_libdir}/libonnxruntime_providers_shared.so
%endif

%changelog
* Mon Oct 05 2026 grafuls <grafuls@users.noreply.github.com> - 0.1.0-4
- Enable EPEL 9 / RHEL 9 derivatives. The prebuilt ONNX Runtime `ort-sys`
  downloads is built against glibc >= 2.38 and fails to link on el9's glibc
  2.34 / GCC 11; use the official manylinux ONNX Runtime 1.28.0 release,
  dynamic-link it, and ship libonnxruntime.so.1 in the package.
* Thu Oct 01 2026 grafuls <grafuls@users.noreply.github.com> - 0.1.0-3
- Correct the operations docs: `laya`-style F1 packages declare a `candle`
  artifact (no ONNX), so example commands use `--backend candle`.

* Thu Oct 01 2026 grafuls <grafuls@users.noreply.github.com> - 0.1.0-2
- Enable the `candle` backend (HF safetensors F1/ModernBERT, e.g.
  `convaiinnovations/laya`), which is the primary real-model path.

* Thu Oct 01 2026 grafuls <grafuls@users.noreply.github.com> - 0.1.0-1
- Initial RPM packaging: single huncho binary (onnx+hf+tokenizers),
  systemd unit, env file, and bundled mock model package.
