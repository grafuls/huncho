#!/usr/bin/env bash
# Assemble a source RPM (SRPM) for Fedora COPR.
#
# COPR's SCM build method clones the git repo and runs `make srpm` from the
# checkout root, then looks for the resulting `.src.rpm` in that directory.
# This script builds the SRPM into the current working directory so COPR can
# pick it up. It mirrors the source-assembly in `build-rpm.sh` but only
# produces the SRPM (COPR compiles the binary on its own builders).
#
# Usage:
#   packaging/rpm/build-srpm.sh
set -euo pipefail

NAME="huncho"
VERSION="0.1.0"

REPO_ROOT="$(git rev-parse --show-toplevel)"
PKGDIR="${REPO_ROOT}/packaging/rpm"
TARBALL="${NAME}-${VERSION}.tar.gz"

# Throwaway rpmbuild tree. Build the SRPM into $PWD (the git checkout root).
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
mkdir -p \
    "${WORK}"/BUILD \
    "${WORK}"/BUILDROOT \
    "${WORK}"/RPMS \
    "${WORK}"/SOURCES \
    "${WORK}"/SPECS \
    "${WORK}"/SRPMS

# Source tarball from all tracked files under a `<name>-<version>/` top-level
# dir, matching `%setup -q` and the source type used by the spec. Untracked
# files (target/, brand/, cruft) are intentionally excluded.
git -C "${REPO_ROOT}" ls-files -z \
    | tar -C "${REPO_ROOT}" --null --files-from=- \
        --transform="s|^|${NAME}-${VERSION}/|" \
        --create --gzip --file "${WORK}/SOURCES/${TARBALL}"

# Auxiliary package files referenced by Source1..Source4.
cp "${PKGDIR}/${NAME}.env"           "${WORK}/SOURCES/"
cp "${PKGDIR}/${NAME}.service"       "${WORK}/SOURCES/"
cp "${PKGDIR}/${NAME}.1"             "${WORK}/SOURCES/"
cp "${PKGDIR}/${NAME}-sysusers.conf" "${WORK}/SOURCES/"

cp "${PKGDIR}/${NAME}.spec" "${WORK}/SPECS/"

rpmbuild -bs \
    --define "_topdir ${WORK}" \
    --define "_srcrpmdir $PWD" \
    "${WORK}/SPECS/${NAME}.spec"

echo
echo "== SRPMs =="
find "$PWD" -maxdepth 1 -name '*.src.rpm' -printf '%p\n'
