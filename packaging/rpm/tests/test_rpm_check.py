"""Exercise the expanded RPM check script with EL9-style dynamic linking."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(shutil.which("cc") and shutil.which("rpmspec"), "requires cc and rpmspec")
class RpmCheckTests(unittest.TestCase):
    def test_el9_check_finds_bundled_runtime_before_installation(self):
        with tempfile.TemporaryDirectory(prefix="huncho-rpm-check-") as tmp:
            root = Path(tmp)
            buildroot = root / "buildroot"
            libdir = buildroot / "usr/lib64"
            libdir.mkdir(parents=True)
            # This ELF has the same loader requirement as the EL9 executable,
            # without a download, GPU, ONNX dependency or embedded search path.
            subprocess.run([
                "cc", "-shared", "-fPIC", "-Wl,-soname,libonnxruntime.so.1",
                "-x", "c", "-", "-o", str(libdir / "libonnxruntime.so.1"),
            ], input="int test_onnx_runtime(void) { return 0; }", text=True, check=True)
            subprocess.run([
                "cc", "-x", "c", "-", "-L", str(libdir),
                "-l:libonnxruntime.so.1", "-o", str(root / "huncho-cpu"),
            ], input="int test_onnx_runtime(void); int main(void) { return test_onnx_runtime(); }",
                text=True, check=True)

            spec = Path(__file__).resolve().parents[1] / "huncho.spec"
            expanded = subprocess.check_output([
                "rpmspec", "--parse", "--undefine", "fedora",
                "--define", "rhel 9", "--define", "dist .el9",
                "--define", "_libdir /usr/lib64", str(spec),
            ], text=True)
            check_script = expanded.split("%check\n", 1)[1].split("\n%", 1)[0]

            # The outer unittest process already runs the packaging suite.
            # Stub only its recursive invocation; run the spec's smoke commands
            # and library-path setup unchanged against the real linked ELF.
            bindir = root / "bin"
            bindir.mkdir()
            python = bindir / "python3"
            python.write_text("#!/bin/sh\nexit 0\n")
            python.chmod(0o755)
            env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}",
                       RPM_BUILD_ROOT=str(buildroot))
            env.pop("LD_LIBRARY_PATH", None)
            result = subprocess.run(
                ["/bin/sh", "-ec", check_script], cwd=root, env=env,
                capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
