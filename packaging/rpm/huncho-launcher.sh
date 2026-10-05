#!/bin/sh
# The CUDA executable has mandatory NVIDIA shared-library dependencies. Probe
# it in a separate process so even a dynamic-loader failure can fall back to
# the CPU executable. rpmbuild substitutes the private executable directory.
runtime_dir='@LIBEXECDIR@/huncho'

case "${HUNCHO_CLEF_DEVICE-auto}" in
    auto)
        if [ -x "$runtime_dir/huncho-cuda" ] &&
            "$runtime_dir/huncho-cuda" __check-cuda >/dev/null 2>&1; then
            exec "$runtime_dir/huncho-cuda" "$@"
        fi
        ;;
    cpu)
        ;;
    cuda|cuda:*)
        if [ ! -x "$runtime_dir/huncho-cuda" ]; then
            echo 'huncho: this package was built without CUDA; install a CUDA-enabled huncho package or set HUNCHO_CLEF_DEVICE=cpu' >&2
            exit 1
        fi
        # Explicit CUDA requests must report errors rather than fall back.
        exec "$runtime_dir/huncho-cuda" "$@"
        ;;
    *)
        echo 'huncho: invalid HUNCHO_CLEF_DEVICE; use auto, cpu, cuda, or cuda:N' >&2
        exit 2
        ;;
esac

# exec preserves arguments, signals and exit status (including under systemd).
exec "$runtime_dir/huncho-cpu" "$@"
