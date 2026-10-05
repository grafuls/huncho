# RPM packaging for huncho

Build a redistributable RPM (binary + source) of the `huncho` serving engine on
any RPM-based distro (Fedora, RHEL, Rocky, etc.).

Install one package, `huncho`. Its `huncho` command automatically uses CUDA for
Clef when a compatible NVIDIA GPU, driver, and runtime are usable; otherwise
it runs on CPU. CPU machines need no NVIDIA packages.

The package contains private CPU and CUDA executables, both built with
`onnx,hf,tokenizers,candle,clef` (plus `cuda` for the GPU executable). The launcher
runs a small CUDA kernel probe before selecting a runtime, so missing shared
libraries, unavailable devices, and incompatible kernels can fall back safely.
Model-loading and inference errors, including GPU out-of-memory errors, are
reported without retrying the workload on CPU. Other backends keep their
existing device support. The RPM also ships:

- a **systemd unit** (`huncho.service`) that runs `huncho serve --mock` out of
  the box;
- an **environment file** at `/etc/huncho/huncho.env`;
- a **man page** (`man huncho`);
- the **mock model package** at
  `/usr/share/huncho/examples/mock-model/` for testing
  `huncho serve --manifest ...`;
- the `huncho` system user and the `/var/lib/huncho` model cache directory
  (also used by HF resolution).

## Build

Requires `rpm-build`, `git`, the Rust toolchain, Python 3 for packaging tests,
and the NVIDIA CUDA toolkit. The toolkit is required on the builder, including
builders with no GPU. The `onnx` feature fetches a prebuilt ONNX Runtime at
build time, so network access is needed.

```sh
packaging/rpm/build-rpm.sh
```

Output:

- `~/rpmbuild/RPMS/x86_64/huncho-0.1.0-9.fc44.x86_64.rpm`
- `~/rpmbuild/SRPMS/huncho-0.1.0-9.fc44.src.rpm`

Run the packaging regression tests on both Fedora and EL9 when editing the spec:

```sh
python3 -m unittest discover -s packaging/rpm/tests -v
```

The EL9 runtime test parses the actual spec with `rpmspec` and runs its check
script against a dynamically linked executable. Use EL9's own RPM tools too;
Fedora's newer parser does not expose all compatibility failures.

## RHEL / EPEL 9 derivatives

On Fedora the `onnx` feature uses `ort-sys`'s prebuilt ONNX Runtime, which is
statically linked and self-contained. That prebuilt is built against glibc
>= 2.38 / GCC 13 libstdc++ and **cannot link on RHEL 9 / EPEL 9** (glibc 2.34,
GCC 11) — it references `__isoc23_strtol*` and `_M_replace_cold`, which do not
exist there.

When building for a RHEL-derived distro (`%{?rhel}` is set, e.g. `el9`), the
spec instead downloads the **official ONNX Runtime 1.28.0 Linux release**, which
is built for manylinux (glibc 2.17 baseline, max GLIBC_2.27),
**dynamically links** it (`ORT_PREFER_DYNAMIC_LINK=1`), and ships
`libonnxruntime.so.1` in the package at `%{_libdir}` for both private runtime
executables. This keeps the build lightweight (no CMake/protobuf toolchain) while
producing a fully functional ONNX-capable binary on EL9.

Each runtime executable records a `DT_NEEDED` on `libonnxruntime.so.1`, so the
license of that bundled runtime is MIT; the rest of the package is
Apache-2.0.

The RPM `%check` stage uses the ONNX library staged under the package buildroot
via `LD_LIBRARY_PATH`. This is needed before installation, including for
`huncho-cpu --version`; it does not alter the installed executable's search path.

## CUDA selection and CPU fallback

Users run the same command on either kind of host:

```sh
huncho serve --model Cloudflare/clef
HUNCHO_CLEF_DEVICE=cpu huncho serve --model Cloudflare/clef
HUNCHO_CLEF_DEVICE=cuda:1 huncho serve --model Cloudflare/clef
```

`auto` (the default) probes GPU 0. `cpu` skips CUDA entirely. `cuda` and
`cuda:N` require the chosen GPU and report errors instead of falling back.
On CUDA Clef defaults to BF16; on CPU it defaults to FP16
backbone weights and an FP32 head. Explicit `--dtype` overrides are preserved.

The CPU executable has no NVIDIA library dependencies. The CUDA executable
links NVIDIA libraries, but those dependencies are excluded from RPM's hard
requirements so installation on a CPU host works. Other ELF requirements are
still generated normally. GPU users must install a compatible driver and CUDA
runtime matching the toolkit used to build the RPM. See [GPU setup](../../docs/gpu-setup.md).

Kernels are built for `sm_80` (Ampere) and require a driver capable of loading
the toolkit's PTX. Older GPUs or drivers that fail the probe use CPU in auto
mode. The probe runs without downloading or loading a model; sufficient VRAM
for the chosen model is still required.

The default spec builds both executables. For builders without NVIDIA's
repository/toolkit, explicitly omit CUDA (the resulting `huncho` runs only on CPU):

```sh
packaging/rpm/build-rpm.sh --without cuda
```

There is one canonical spec, `packaging/rpm/huncho.spec`. The unified package
obsoletes older `huncho-cuda` RPMs. `huncho` is the only public command and
package name; update existing scripts to use it. Set `HUNCHO_CLEF_DEVICE=cuda`
to require CUDA. CPU-only builds do not replace the former CUDA package.

## Install

```sh
sudo dnf install ~/rpmbuild/RPMS/x86_64/huncho-0.1.0-9.fc44.x86_64.rpm
sudo systemctl enable --now huncho
curl -s http://127.0.0.1:8080/v1/models
```

To serve a real model package, edit `ExecStart` in
`/usr/lib/systemd/system/huncho.service` to use `--manifest <path>` or
`--model owner/repo`, then run
`sudo systemctl daemon-reload && sudo systemctl restart huncho`.

The service defaults to `HUNCHO_BACKEND=auto` and selects a compatible runtime
for each model. Set it explicitly in `/etc/huncho/huncho.env` only to override
that selection.

## Publish to Fedora COPR

COPR (Cool Other Package Repository) is the standard place to publish a
third-party RPM for Fedora/RHEL users. It builds the package on Fedora builders
from the git source and serves a dnf repository, so users can install and
automatically update `huncho` with `dnf`.

The repo is already wired for COPR's **make srpm** SCM build method. COPR
clones the git repo and invokes the `srpm` target of `.copr/Makefile` (in the
repo root), passing `outdir` and `spec` as make variables. That file assembles
the source tarball and auxiliary files and runs `rpmbuild -bs` into `outdir`.
The root `Makefile`'s `srpm` target just delegates to the same file.

1. **Create a project** at <https://copr.fedorainfracloud.org> (sign in with
   your Fedora account). This project's COPR is `quadsdev/huncho`.
2. **Add a package** with the **SCM** source type and the **make srpm**
   method: point it at this repo's git URL and the `main` branch, and set the
   spec file to `packaging/rpm/huncho.spec`. COPR runs `.copr/Makefile`'s
   `srpm` target to build the source RPM.
3. **Configure builders**: enable the desired chroots and add the NVIDIA CUDA
   repository appropriate to each distro so `cuda-toolkit` is available. The
   unified build needs that toolkit in every chroot; a GPU is not required.
   Chroots without a supported toolkit must explicitly build `--without cuda`.
   Enable network access for Cargo/ONNX Runtime downloads.
4. **Use one package entry**: when migrating an older project, remove the
   separate `huncho-cuda` COPR entry (which referenced `huncho-cuda.spec`). Keep
   the `huncho` entry pointing at the canonical spec. The `quadsdev/huncho`
   project already uses this layout. Repository settings are configured in
   COPR separately from the spec.
5. **Build**: trigger a build in the web UI, via `copr-cli build`, or enable the
   GitHub webhook so a push to `main` rebuilds automatically.
6. **Configure auto-rebuild** (optional but recommended): in the COPR web UI go
   to the project's **Settings → Integrations** and copy the webhook URL
   (`https://copr.fedorainfracloud.org/webhooks/<forge>/<project-id>/<secret>/`).
   Then, in the GitHub repo, add a **Webhook** (Settings → Webhooks → Add
   webhook) with that Payload URL, content type **application/json**, and event
   **Push**. Every push to `main` now triggers a COPR rebuild of the `huncho`
   package. Equivalently, this repo has a `push` webhook registered against the
   `huncho` package already.

The configured x86_64 targets use these NVIDIA repositories:

| COPR chroot | NVIDIA repository |
|---|---|
| `fedora-43-x86_64` | `https://developer.download.nvidia.com/compute/cuda/repos/fedora43/x86_64/` |
| `fedora-44-x86_64` | `https://developer.download.nvidia.com/compute/cuda/repos/fedora44/x86_64/` |
| `fedora-rawhide-x86_64` | `https://developer.download.nvidia.com/compute/cuda/repos/fedora44/x86_64/` |
| `epel-9-x86_64` | `https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/` |

Set these in each chroot's **Additional repositories**, or with
`copr-cli edit-chroot <owner>/huncho/<chroot> --repos <repository-url>`.
Repositories configured for one chroot do not apply to the others. Omitting
one causes dependency installation to fail with `No match for argument:
cuda-toolkit`, before the Rust build starts.

NVIDIA does not publish a Rawhide repository. This target uses the Fedora 44
toolkit; check [NVIDIA's compiler compatibility table](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#host-compiler-support-policy)
when Rawhide advances its GCC major version, and validate it with a COPR build.

Check the live configuration after changing build targets or repository settings:

```sh
python3 .copr/check-cuda-repos.py quadsdev/huncho
```

The check reports missing per-chroot repositories or disabled CUDA builds and
exits nonzero if either would prevent shipping the unified package.

Verify locally that the COPR entry point works:

```sh
make srpm   # writes huncho-<version>.src.rpm into the repo root
# equivalently, exactly what COPR runs:
make -f .copr/Makefile srpm outdir=. spec=packaging/rpm/huncho.spec
```

Then point users at the repo:

```sh
sudo dnf copr enable <owner>/huncho
sudo dnf install huncho
```

## Files

| Path | Purpose |
|------|---------|
| `huncho.spec` | RPM spec (builds from source, defines the package). |
| `huncho-launcher.sh` | Public command; probes CUDA and selects a private runtime. |
| `tests/test_launcher.py` | Launcher regression tests, including missing-library fallback. |
| `huncho.env` | Default service environment (local/dev-safe). |
| `huncho.service` | Packaged systemd unit (`/usr/bin/huncho`). |
| `huncho-sysusers.conf` | systemd-sysusers definition for the `huncho` service user. |
| `huncho.1` | Roff man page. |
| `build-rpm.sh` | Assembles `~/rpmbuild` and runs `rpmbuild -ba` (binary + source RPM). |
| `.copr/Makefile` | COPR `make srpm` entry point: builds the source RPM into COPR's `outdir`. |

## Service user

The `huncho` user/group are declared in `huncho-sysusers.conf` and installed to
`/usr/lib/sysusers.d/huncho.conf`. Because it is a sysusers file, rpm
auto-generates `Provides: user(huncho), group(huncho)`. This is required: without
it, `dnf install` fails with `nothing provides user(huncho)` because the
`Requires(pre)` is resolved *before* `%pre` runs and no other package provides
the user. `%pre` runs `systemd-sysusers` (via `%sysusers_create_package`) to
actually create the account, and `/var/lib/huncho` is owned by that user.

## Notes

- `strip --strip-unneeded` is applied to both executables in `%build`: `cargo`'s `strip = true`
  profile setting did not take effect under the `rpmbuild` environment, and
  without an explicit strip the ELF keeps its `.symtab`/debug sections.
- `brp-compress` gzips the man page during packaging, so `%files` lists
  `%{_mandir}/man1/huncho.1*`.

Source tarballs include tracked working-tree changes and non-ignored new files;
tracked deletions are omitted. Review `git status` before building a release.
