#!/bin/bash
set -eu
root=/var/tmp/huncho-optimization-20261007
trap 'printf "%s\n" "$?" > "$root/stage5-cpu-thread-pilot.exit"' EXIT
source /var/tmp/huncho-t4-build/environment.sh
python3 "$root/huncho-cpu-thread-pilot.py"
