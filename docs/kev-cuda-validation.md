# Kev CUDA validation — 2026-10-06

Kev-4B successfully loaded and answered HTTP requests on a Tesla T4 using the installed
`huncho-0.1.0-11.el9.t4.x86_64` RPM. Automatic device selection chose CUDA
and FP16 without `LD_LIBRARY_PATH` or an explicit dtype override.

The node runs RHEL 9.4, NVIDIA driver 580.178.04, and CUDA 12.8.93. Huncho's
kernels target `sm_75`; its CUDA executable links the CUDA 12.8 libraries.
The T4 reports 15,360 MiB of memory. The restored driver loaded without a reboot.

The adapter snapshot was `6cfce5c2fa4b4bd64026336ab649c5ca78857d52` from
`jaredpalmer/kev-4b`; the base snapshot was
`1001bb4d826a52d1f399e183466143f4da7b741b` from `Qwen/Qwen3.5-4B-Base`.

Validation passed:

- CPU and CUDA upstream reference tests for Kev and Clef: seven tests.
- BF16 checkpoint conversion and FP16 CPU/GPU probability agreement on the T4.
- CLI prompt/answer reference checks, automatic dtype choice, and overrides.
- RPM packaging checks, CUDA probe, CPU fallback smoke test, and `rpm -V huncho`.
- Real Kev-4B HTTP requests containing choice, noul, and score questions.
- The installed launcher selected CUDA automatically and returned exactly the
  same answers as the directly tested CUDA executable.

For the short state `The sky is blue.`, each HTTP request asked all three
question types. Two requests were measured per process:

| Runtime | Load | First request | Second request | Peak observed GPU memory |
| --- | ---: | ---: | ---: | ---: |
| Installed launcher, automatic CUDA/FP16 | 43.08 s | 0.843 s | 0.659 s | 8,197 MiB |
| CPU/FP16 comparison | 41.06 s | 18.522 s | 18.458 s | 0 MiB |

The largest absolute difference across numeric CPU/GPU answer fields was
0.001032. These are short smoke-test observations, not a throughput benchmark
or a measurement of maximum-context memory use. The CPU comparison ran while
the RPM build was active.

Run on this node with:

```sh
huncho serve --model jaredpalmer/kev-4b
# Require CUDA and report an error if unavailable:
HUNCHO_DEVICE=cuda huncho serve --model jaredpalmer/kev-4b
```

The temporary test servers were stopped; `huncho.service` remains inactive.
Build logs, response JSON, and the binary/source RPMs are under
`/var/tmp/huncho-t4-build` on the node. Local RPM copies were also saved under
`/tmp` on the development machine.

Binary RPM SHA-256:
`4b0df0240f8a651f1d4f14a5f6567421edecfff88448f49fec2dffa8b91f957c`.

Source RPM SHA-256:
`9650982507f6cd473286e0723983be2613083609d66ad59af61d2161129e90e7`.
