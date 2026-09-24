import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import baseline


class RecorderTests(unittest.TestCase):
    def test_file_hash_streaming(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data"
            path.write_bytes(b"abc")
            self.assertEqual(baseline.sha256(path),
                             "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")

    def test_missing_tool_is_reported(self):
        with patch("baseline.subprocess.run", side_effect=FileNotFoundError("missing")):
            self.assertIsNone(baseline.capture(["missing"])["returncode"])

    def run_benchmark(self, exit_code=0, stdout=None, split=False):
        if stdout is None:
            stdout = json.dumps([{"samples_ns": [100] * 5, "samples_ts": [10.0] * 5}]).encode()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = root / ("model-00001-of-00002.gguf" if split else "model with spaces.gguf")
            model.write_bytes(b"GGUFfixture")
            binary = root / "fake bench.exe"
            binary.write_bytes(b"fixture")
            cache = root / "CMakeCache.txt"
            cache.write_text("GGML_CUDA:BOOL=OFF")
            output = root / "result"
            argv = ["baseline", "bench", "--binary", str(binary), "--model", str(model),
                    "--cmake-cache", str(cache), "--arm", "p100-baseline", "--threads", "2",
                    "--output", str(output)]

            def fake_run(command, **kwargs):
                self.assertEqual(command[2], str(model.resolve()))
                self.assertIn("256", command)
                self.assertNotIn("--no-warmup", command)
                kwargs["stdout"].write(stdout)
                return subprocess.CompletedProcess(command, exit_code)

            with patch.object(sys, "argv", argv), patch("baseline.inventory", return_value={}), \
                    patch("baseline.subprocess.run", side_effect=fake_run):
                if split:
                    with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                        baseline.main()
                    self.assertFalse(output.exists())
                    return
                code = baseline.main()
                report = json.loads((output / "metadata.json").read_text())
                self.assertEqual(report["model"]["sha256"], baseline.sha256(model))
                self.assertEqual((output / "stdout.json").read_bytes(), stdout)
                with self.assertRaises(FileExistsError):
                    baseline.main()
                return code, report

    def test_success_and_no_overwrite(self):
        code, report = self.run_benchmark()
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "completed")

    def test_nonzero_exit_preserves_evidence(self):
        code, report = self.run_benchmark(exit_code=7, stdout=b"partial output")
        self.assertEqual(code, 7)
        self.assertEqual(report["status"], "failed")

    def test_invalid_json_is_failure(self):
        code, report = self.run_benchmark(stdout=b"not json")
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "failed")

    def test_split_model_rejected_before_running(self):
        self.run_benchmark(split=True)

    def test_missing_samples_are_failure(self):
        code, report = self.run_benchmark(stdout=b'[{}]')
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "failed")


if __name__ == "__main__":
    unittest.main()
