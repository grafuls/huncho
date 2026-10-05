# One RPM and public command, with isolated CPU and CUDA executables. NVIDIA
# libraries remain optional at install time; the launcher probes CUDA safely.
# Builders without the NVIDIA toolkit can explicitly use --without cuda.
%bcond_without cuda
%global _features onnx,hf,tokenizers,candle,clef

# Only NVIDIA dependencies are optional. Keep all other ELF requirements,
# including the C/C++ runtime and the EL9 ONNX Runtime dependency.
%global __requires_exclude ^lib(cuda|cudart|cublas|cublasLt|curand|nvrtc|nvrtc-builtins)[.]so.*$

%global crate_name huncho
%global debug_package %{nil}

Name:           huncho
Version:        0.1.0
Release:        7%{?dist}
Summary:        Portable serving engine for System One decision models

License:        Apache-2.0
URL:            https://github.com/grafuls/huncho
Source0:        huncho-%{version}.tar.gz
Source1:        huncho.env
Source2:        huncho.service
Source3:        huncho.1
Source4:        huncho-sysusers.conf
Source5:        huncho-launcher.sh

# Rust toolchain + C compiler. `pkgconf-pkg-config` and OpenSSL are needed by
# the `hf` TLS provider (`reqwest` native-tls -> openssl-sys) and the ONNX
# Runtime build (ort).
BuildRequires:  cargo >= 1.80
BuildRequires:  rust >= 1.80
BuildRequires:  gcc
BuildRequires:  gcc-c++
BuildRequires:  binutils
BuildRequires:  python3
BuildRequires:  openssl-devel
BuildRequires:  pkgconf-pkg-config
BuildRequires:  systemd-rpm-macros

# The toolkit is needed only on the builder, never on CPU-only installations.
%if %{with cuda}
BuildRequires:  cuda-toolkit
Obsoletes:      huncho-cuda < %{version}-%{release}
%endif

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
#
Requires(pre):  /usr/bin/systemd-sysusers

%description
Huncho is a portable serving engine for Jev-style System One decision models.
It takes a state and a set of typed questions and returns calibrated
probabilities — no text generation. It implements the Jev wire contract, so an
unmodified Python SDK works against Huncho with only a base-URL change.

The huncho command automatically uses CUDA for Clef when a compatible NVIDIA
GPU, driver and CUDA runtime are available, and otherwise uses CPU. Set
HUNCHO_CLEF_DEVICE=cpu to force CPU or cuda[:N] to require a GPU. Other backends
retain their existing device support. Both runtimes include ONNX Runtime,
Candle, Hugging Face Hub resolution, Clef, and Hugging Face tokenization.

The package includes a systemd unit, environment file and mock model package.
%if %{without cuda}
This build omits the optional CUDA runtime and runs on CPU only.
%endif

%prep
%setup -q -n huncho-%{version}

%build
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

# Save the portable executable before enabling CUDA; it must never link NVIDIA
# libraries. Cargo rebuilds the affected crates when the feature set changes.
cargo build --release --locked --features %{_features} --bin huncho
install -m0755 target/release/huncho huncho-cpu
strip --strip-unneeded huncho-cpu
if readelf -d huncho-cpu | grep -E 'NEEDED.*lib(cuda|cudart|cublas|cublasLt|curand|nvrtc)'; then
    echo "CPU executable must not depend on NVIDIA libraries" >&2
    exit 1
fi

%if %{with cuda}
# cudarc's build script runs `nvcc` to detect the CUDA version (it panics if
# `nvcc --version` fails) and links the CUDA dylibs (`-lcuda -lcudart -lcublas
# -lcublasLt -lnvrtc -lcurand`). The toolkit installs under
# `/usr/local/cuda-<ver>` (the `nvcc` is NOT on the default PATH), so prepend
# its `bin/` to PATH and its `lib64/` to LIBRARY_PATH/LD_LIBRARY_PATH so both
# the compile and link steps succeed.
for d in /usr/local/cuda*; do
  if [ -x "$d/bin/nvcc" ]; then
    export PATH="$d/bin:$PATH"
    for libdir in "$d/lib64" "$d/lib"; do
      if [ -d "$libdir" ]; then
        export LIBRARY_PATH="$libdir:$LIBRARY_PATH"
        export LD_LIBRARY_PATH="$libdir:$LD_LIBRARY_PATH"
      fi
    done
  fi
done

# candle-kernels/cudaforge detects the GPU compute capability by running
# `nvidia-smi`, which is absent in the buildroot. Pin it so the kernels are
# built once. sm_80 (Ampere, CUDA 13's baseline is Turing/sm_75) enables bf16
# WMMA and runs on Ampere/Ada/Hopper/Blackwell via driver PTX JIT.
export CUDA_COMPUTE_CAP=80

cargo build --release --locked --features %{_features},cuda --bin huncho
install -m0755 target/release/huncho huncho-cuda
strip --strip-unneeded huncho-cuda
%endif

%check
python3 -m unittest discover -s packaging/rpm/tests -v
./huncho-cpu --version
./huncho-cpu bench --backend mock --iterations 1

%install
install -Dm0755 huncho-cpu %{buildroot}%{_libexecdir}/huncho/huncho-cpu
install -Dm0755 %{SOURCE5} %{buildroot}%{_bindir}/huncho
sed -i 's|@LIBEXECDIR@|%{_libexecdir}|g' %{buildroot}%{_bindir}/huncho
%if %{with cuda}
install -Dm0755 huncho-cuda %{buildroot}%{_libexecdir}/huncho/huncho-cuda
%endif
install -Dm0644 %{SOURCE2} %{buildroot}%{_unitdir}/huncho.service
install -Dm0644 %{SOURCE1} %{buildroot}%{_sysconfdir}/huncho/huncho.env
install -Dm0644 %{SOURCE4} %{buildroot}%{_sysusersdir}/huncho.conf

install -Dm0644 %{SOURCE3} %{buildroot}%{_mandir}/man1/huncho.1
install -Dm0644 docs/operations.md %{buildroot}%{_docdir}/%{name}/operations.md
install -Dm0644 docs/model-package.md %{buildroot}%{_docdir}/%{name}/model-package.md

# Bundle the built-in mock model package for offline smoke tests.
install -Dm0644 examples/mock-model/huncho-model.json \
    %{buildroot}%{_datadir}/huncho/examples/mock-model/huncho-model.json
install -Dm0644 examples/mock-model/mock-model.onnx \
    %{buildroot}%{_datadir}/huncho/examples/mock-model/mock-model.onnx
install -Dm0644 examples/mock-model/golden.json \
    %{buildroot}%{_datadir}/huncho/examples/mock-model/golden.json
install -Dm0644 examples/mock-model/README.md \
    %{buildroot}%{_datadir}/huncho/examples/mock-model/README.md

# Both internal executables use this runtime on EL9.
%if 0%{?rhel}
install -Dm0755 %{_builddir}/onnxruntime/onnxruntime-linux-x64-1.28.0/lib/libonnxruntime.so.1.28.0 \
    %{buildroot}%{_libdir}/libonnxruntime.so.1.28.0
ln -sf libonnxruntime.so.1.28.0 %{buildroot}%{_libdir}/libonnxruntime.so.1
ln -sf libonnxruntime.so.1 %{buildroot}%{_libdir}/libonnxruntime.so
install -Dm0755 %{_builddir}/onnxruntime/onnxruntime-linux-x64-1.28.0/lib/libonnxruntime_providers_shared.so \
    %{buildroot}%{_libdir}/libonnxruntime_providers_shared.so
%endif

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
%doc README.md
%{_bindir}/huncho
%dir %{_libexecdir}/huncho
%{_libexecdir}/huncho/huncho-cpu
%if %{with cuda}
%{_libexecdir}/huncho/huncho-cuda
%endif
%{_unitdir}/huncho.service
%config(noreplace) %attr(0644,root,root) %{_sysconfdir}/huncho/huncho.env
%{_sysusersdir}/huncho.conf
%dir %attr(0755,huncho,huncho) %{_localstatedir}/lib/huncho
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
* Mon Oct 05 2026 grafuls <grafuls@users.noreply.github.com> - 0.1.0-7
- Consolidate CPU and CUDA runtimes into huncho with automatic device probing
  and CPU fallback, optional NVIDIA libraries, and one public huncho command.

* Mon Oct 05 2026 grafuls <grafuls@users.noreply.github.com> - 0.1.0-6
- Add an optional `with_cuda` build mode that produces a standalone `huncho-cuda`
  package with Candle's CUDA backend enabled, so the Clef backend
  (Cloudflare/clef) runs on an NVIDIA GPU (`HUNCHO_CLEF_DEVICE=cuda`). The
  portable `huncho` package is unchanged.

* Mon Oct 05 2026 grafuls <grafuls@users.noreply.github.com> - 0.1.0-5
- Enable the `clef` model backend (Cloudflare/clef, served via Candle) so
  `huncho serve --model Cloudflare/clef` works out of the box. It is pure Rust
  (Candle + tokenizers) and adds no new system build dependencies.
* Thu Oct 01 2026 grafuls <grafuls@users.noreply.github.com> - 0.1.0-3
- Correct the operations docs: `laya`-style F1 packages declare a `candle`
  artifact (no ONNX), so example commands use `--backend candle`.

* Thu Oct 01 2026 grafuls <grafuls@users.noreply.github.com> - 0.1.0-2
- Enable the `candle` backend (HF safetensors F1/ModernBERT, e.g.
  `convaiinnovations/laya`), which is the primary real-model path.

* Thu Oct 01 2026 grafuls <grafuls@users.noreply.github.com> - 0.1.0-1
- Initial RPM packaging: single huncho binary (onnx+hf+tokenizers),
  systemd unit, env file, and bundled mock model package.
