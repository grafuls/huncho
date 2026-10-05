#!/usr/bin/env python3
"""Check the live COPR prerequisites for the unified CPU/CUDA RPM.

Run before enabling builds on a new target or after editing project settings.
Requires an authenticated copr-cli; this check does not change configuration.
"""

import argparse
import json
import subprocess
import sys


CUDA_REPOSITORIES = {
    "fedora-43-x86_64": "fedora43",
    "fedora-44-x86_64": "fedora44",
    # NVIDIA has no Rawhide repo. Track the newest validated Fedora toolkit
    # and verify the host compiler remains within NVCC's supported range.
    "fedora-rawhide-x86_64": "fedora44",
    "epel-9-x86_64": "rhel9",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", nargs="?", default="quadsdev/huncho")
    args = parser.parse_args()
    failed = False
    for chroot, distro in CUDA_REPOSITORIES.items():
        config = json.loads(subprocess.check_output([
            "copr-cli", "get-chroot", "--output-format", "json",
            f"{args.project}/{chroot}",
        ], text=True))
        expected = f"https://developer.download.nvidia.com/compute/cuda/repos/{distro}/x86_64"
        repos = {repo.rstrip("/") for repo in config["additional_repos"]}
        problems = []
        if expected not in repos:
            problems.append(f"missing NVIDIA repository {expected}/")
        if "cuda" in config["without_opts"]:
            problems.append("CUDA disabled by a per-chroot --without option")
        if problems:
            failed = True
            print(f"FAIL {chroot}: {'; '.join(problems)}", flush=True)
        else:
            print(f"OK   {chroot}", flush=True)
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
