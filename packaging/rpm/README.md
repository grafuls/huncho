# RPM packaging for huncho

Build a redistributable RPM (binary + source) of the `huncho` serving engine on
any RPM-based distro (Fedora, RHEL, Rocky, etc.).

The packaged binary is built with the full feature set used by the project's
Dockerfile: `onnx,hf,tokenizers,candle` — ONNX Runtime, the Candle backend
(HF `safetensors` F1/ModernBERT models, the primary real-model path), Hugging
Face Hub resolution, and the official HF tokenizer. The RPM also ships:

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

Requires `rpm-build`, `git`, and the Rust toolchain. The `onnx` feature fetches
a prebuilt ONNX Runtime at build time, so network access is needed.

```sh
packaging/rpm/build-rpm.sh
```

Output:

- `~/rpmbuild/RPMS/x86_64/huncho-0.1.0-1.fc44.x86_64.rpm`
- `~/rpmbuild/SRPMS/huncho-0.1.0-1.fc44.src.rpm`

## Install

```sh
sudo dnf install ~/rpmbuild/RPMS/x86_64/huncho-0.1.0-1.fc44.x86_64.rpm
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
   your Fedora account), e.g. `grafuls/huncho`.
2. **Add a package** with the **SCM** source type and the **make srpm**
   method: point it at this repo's git URL and the `main` branch, and set the
   spec file to `packaging/rpm/huncho.spec`. COPR runs `.copr/Makefile`'s
   `srpm` target to build the source RPM.
3. **Enable chroots**, e.g. `fedora-rawhide-x86_64`, `fedora-42-x86_64`, and
   optionally `epel-9-x86_64`.
4. **Build**: trigger a build in the web UI, via `copr-cli build`, or enable the
   GitHub webhook so a push to `main` rebuilds automatically.
5. **Configure auto-rebuild** (optional but recommended): in the COPR web UI go
   to the project's **Settings → Integrations** and copy the webhook URL
   (`https://copr.fedorainfracloud.org/webhooks/<forge>/<project-id>/<secret>/`).
   Then, in the GitHub repo, add a **Webhook** (Settings → Webhooks → Add
   webhook) with that Payload URL, content type **application/json**, and event
   **Push**. Every push to `main` now triggers a COPR rebuild of the `huncho`
   package. Equivalently, this repo has a `push` webhook registered against the
   `huncho` package already.

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

- `strip --strip-unneeded` is applied in `%build`: `cargo`'s `strip = true`
  profile setting did not take effect under the `rpmbuild` environment, and
  without an explicit strip the ELF keeps its `.symtab`/debug sections.
- `brp-compress` gzips the man page during packaging, so `%files` lists
  `%{_mandir}/man1/huncho.1*`.
