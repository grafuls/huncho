#!/usr/bin/env bash
# Build the huncho RPM from the current tree.
#
# Usage:
#   packaging/rpm/build-rpm.sh            # build binary + source RPMs
#   packaging/rpm/build-rpm.sh --nodeps   # pass extra args to rpmbuild
#
# Requires `rpm-build`, `git`, and the Rust toolchain (network needed during the
# %build step to fetch the prebuilt ONNX Runtime). Output is written to
# ~/rpmbuild/RPMS/ and ~/rpmbuild/SRPMS/.
set -euo pipefail

NAME="huncho"
VERSION="0.1.0"

REPO_ROOT="$(git rev-parse --show-toplevel)"
RPMBUILD="${RPMBUILD:-$HOME/rpmbuild}"

TARBALL="${NAME}-${VERSION}.tar.gz"

# Recreate the rpmbuild tree (clean each run).
rm -rf "${RPMBUILD:?}"/{BUILD,BUILDROOT,RPMS,SOURCES,SPECS,SRPMS}
mkdir -p "${RPMBUILD}"/{BUILD,BUILDROOT,RPMS,SOURCES,SPECS,SRPMS}

# Source tarball: all tracked files (including any uncommitted working-tree
# edits) under a <name>-<version>/ top-level dir. Untracked files (target/,
# brand/, local cruft) are intentionally excluded so `git archive HEAD` is not
# needed; this lets an in-progress source change land in the package.
git -C "${REPO_ROOT}" ls-files -z \
    | tar -C "${REPO_ROOT}" --null --files-from=- \
        --transform="s|^|${NAME}-${VERSION}/|" \
        --create --gzip --file "${RPMBUILD}/SOURCES/${TARBALL}"

# Spec + auxiliary package files.
cp "${REPO_ROOT}/packaging/rpm/${NAME}.spec"       "${RPMBUILD}/SPECS/"
cp "${REPO_ROOT}/packaging/rpm/${NAME}.env"        "${RPMBUILD}/SOURCES/"
cp "${REPO_ROOT}/packaging/rpm/${NAME}.service"    "${RPMBUILD}/SOURCES/"
cp "${REPO_ROOT}/packaging/rpm/${NAME}.1"          "${RPMBUILD}/SOURCES/"
cp "${REPO_ROOT}/packaging/rpm/${NAME}-sysusers.conf" "${RPMBUILD}/SOURCES/"

echo "Building ${NAME}-${VERSION} ..."
rpmbuild -ba "${RPMBUILD}/SPECS/${NAME}.spec" "$@"

echo
echo "== RPMs =="
find "${RPMBUILD}/RPMS" -type f -name '*.rpm' -printf '%p\n'
echo "== SRPMs =="
find "${RPMBUILD}/SRPMS" -type f -name '*.rpm' -printf '%p\n'
