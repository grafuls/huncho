"""Run the installed launcher against controlled CPU/CUDA processes, no GPU needed."""

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="huncho launcher ")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.runtime = self.root / "libexec" / "huncho"
        self.runtime.mkdir(parents=True)
        self.launcher = self.root / "huncho"
        template = Path(__file__).resolve().parents[1] / "huncho-launcher.sh"
        self.launcher.write_text(
            template.read_text().replace("@LIBEXECDIR@", str(self.runtime.parent))
        )
        self.launcher.chmod(0o755)
        self.env = dict(os.environ)
        self.env.pop("HUNCHO_CLEF_DEVICE", None)
        self.env["PROBE_LOG"] = str(self.root / "probes")
        self.stub("cpu")
        self.stub("cuda")

    def stub(self, kind, probe_status=0, run_status=0):
        script = self.runtime / f"huncho-{kind}"
        script.write_text(
            f"#!{sys.executable}\n"
            "import json, os, signal, sys\n"
            "if sys.argv[1:] == ['__check-cuda']:\n"
            "    with open(os.environ['PROBE_LOG'], 'a') as f: f.write('probe\\n')\n"
            f"    sys.exit({probe_status})\n"
            "if sys.argv[1:] == ['--signal']:\n"
            "    os.kill(os.getpid(), signal.SIGTERM)\n"
            f"print(json.dumps([{kind!r}, sys.argv[1:], "
            "os.environ.get('HUNCHO_CLEF_DEVICE'), os.getpid()]))\n"
            f"sys.exit({run_status})\n"
        )
        script.chmod(0o755)

    def run_launcher(self, device=None, args=("serve",), **kwargs):
        if device is not None:
            self.env["HUNCHO_CLEF_DEVICE"] = device
        return subprocess.run(
            [self.launcher, *args], env=self.env, capture_output=True,
            text=True, timeout=10, **kwargs
        )

    def test_default_selects_usable_cuda_and_preserves_arguments(self):
        args = ["serve", "--model", "a path with spaces", "--token", "literal$`token", ""]
        result = self.run_launcher(args=args)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)[:2], ["cuda", args])
        self.assertEqual((self.root / "probes").read_text(), "probe\n")

    def test_failed_probe_falls_back_to_cpu(self):
        for status in (1, 127):  # initialization failure or dynamic-loader failure
            with self.subTest(status=status):
                self.stub("cuda", probe_status=status)
                result = self.run_launcher("auto")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout)[0], "cpu")
                self.assertEqual(result.stderr, "")

    def test_missing_cuda_executable_falls_back(self):
        (self.runtime / "huncho-cuda").unlink()
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)[0], "cpu")

    def test_cpu_override_does_not_probe_cuda(self):
        result = self.run_launcher("cpu")
        self.assertEqual(json.loads(result.stdout)[0], "cpu")
        self.assertFalse((self.root / "probes").exists())

    def test_explicit_cuda_keeps_ordinal_and_exit_status(self):
        self.stub("cuda", probe_status=1, run_status=42)
        result = self.run_launcher("cuda:2")
        self.assertEqual(result.returncode, 42)
        data = json.loads(result.stdout)
        self.assertEqual(data[0], "cuda")
        self.assertEqual(data[2], "cuda:2")
        self.assertFalse((self.root / "probes").exists())

    def test_explicit_cuda_without_executable_fails(self):
        (self.runtime / "huncho-cuda").unlink()
        result = self.run_launcher("cuda")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("built without CUDA", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_inference_failure_does_not_retry_on_cpu(self):
        self.stub("cuda", run_status=42)
        result = self.run_launcher()
        self.assertEqual(result.returncode, 42)
        self.assertEqual(json.loads(result.stdout)[0], "cuda")

    def test_invalid_device_is_rejected(self):
        for device in ("", "gpu"):
            with self.subTest(device=device):
                result = self.run_launcher(device)
                self.assertEqual(result.returncode, 2)
                self.assertIn("invalid HUNCHO_CLEF_DEVICE", result.stderr)

    def test_exec_preserves_pid_and_signals(self):
        for device in ("cpu", "auto"):
            self.env["HUNCHO_CLEF_DEVICE"] = device
            with subprocess.Popen(
                [self.launcher, "serve"], env=self.env,
                stdout=subprocess.PIPE, text=True
            ) as proc:
                stdout, _ = proc.communicate(timeout=10)
                self.assertEqual(json.loads(stdout)[3], proc.pid)
            result = self.run_launcher(device, args=["--signal"])
            self.assertEqual(result.returncode, -signal.SIGTERM)

    @unittest.skipUnless(shutil.which("cc"), "C compiler needed for ELF loader test")
    def test_real_missing_shared_library_falls_back(self):
        # Exercise an actual loader failure, before the CUDA program can run.
        lib = self.root / "libhuncho_probe_fixture.so"
        subprocess.run([
            "cc", "-shared", "-fPIC", "-x", "c", "-", "-o", lib
        ], input="int probe(void) { return 0; }", text=True, check=True)
        subprocess.run([
            "cc", "-x", "c", "-", "-L", str(self.root),
            f"-Wl,-rpath,{self.root}", "-lhuncho_probe_fixture",
            "-o", self.runtime / "huncho-cuda"
        ], input="int probe(void); int main(void) { return probe(); }", text=True, check=True)
        lib.unlink()
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)[0], "cpu")
        result = self.run_launcher("cuda")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("libhuncho_probe_fixture.so", result.stderr)


if __name__ == "__main__":
    unittest.main()
