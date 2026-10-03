"""CPU/local-only contract tests; never contact a remote or launch GPU code."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("pull_results_until_deadline.py")
spec = importlib.util.spec_from_file_location("backup", SCRIPT)
backup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backup)


class BackupTests(unittest.TestCase):
    def test_deadline_timezone_required(self):
        with self.assertRaises(ValueError):
            backup.parse_deadline("2026-10-04T07:27:00")
        self.assertEqual(backup.parse_deadline("2026-10-04T07:27:00Z"),
                         backup.parse_deadline("2026-10-04T03:27:00-04:00"))

    def test_final_window_and_timeout_are_bounded(self):
        self.assertEqual(backup.schedule(700, 30, 600, 5, 45), (30, 45))
        self.assertEqual(backup.schedule(300, 30, 600, 5, 45), (5, 45))
        self.assertEqual(backup.schedule(.2, 30, 600, 5, 45), (.2, .2))

    def test_unsafe_source_rejected(self):
        for host, root, paths in [("-oProxyCommand=bad", "/work/job", ["reports"]),
                                  ("host", "/", ["reports"]),
                                  ("host", "/work/../etc", ["reports"]),
                                  ("host", "/work/job", ["../secrets"]),
                                  ("host;bad", "/work/job", ["reports"])]:
            with self.assertRaises(ValueError):
                backup.validate(host, root, paths)
        backup.validate("user@gpu-host", "/work/job", ["reports"])

    def test_command_preserves_data_and_ssh_identity(self):
        config = {"host": "user@host", "remote_root": "/work/job", "paths": ["reports"],
                  "known_hosts": "/tmp/key path/known_hosts", "identity_file": None}
        cmd = backup.command(config, Path("/tmp/output"), "one")
        for forbidden in ("--delete", "--inplace", "--remove-source-files"):
            self.assertNotIn(forbidden, cmd)
        self.assertIn("--backup", cmd)
        self.assertIn("--partial-dir=.rsync-partial", cmd)
        self.assertIn("--include=/reports/***", cmd)
        self.assertIn("StrictHostKeyChecking=yes", cmd[cmd.index("-e") + 1])
        self.assertEqual(cmd[-2:], ["user@host:/work/job/", "/tmp/output/mirror/"])

    def test_transfer_timeout(self):
        with tempfile.TemporaryFile() as stream:
            self.assertEqual(backup.run_transfer([sys.executable, "-c", "import time; time.sleep(30)"],
                                                 stream, .03), 124)

    @unittest.skipUnless(shutil.which("rsync"), "local rsync unavailable")
    def test_real_local_rsync_history_and_empty_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root / "source"; output = root / "output"
            (source / "reports").mkdir(parents=True); (source / "venv").mkdir()
            (output / "mirror").mkdir(parents=True)
            original = source / "reports/raw.json"; original.write_text("first result")
            (source / "venv/secret").write_text("must not copy")
            config = {"host": "unused", "remote_root": "/unused", "paths": ["reports"],
                      "known_hosts": "/unused", "identity_file": None}
            def sync(attempt):
                args = backup.command(config, output, attempt)
                index = args.index("-e"); del args[index:index + 2]
                args[-2] = str(source) + "/"
                result = subprocess.run(args, capture_output=True, timeout=5)
                self.assertEqual(result.returncode, 0, result.stderr.decode())
            sync("first")
            self.assertFalse((output / "mirror/venv").exists())
            original.write_text("second and longer result")
            sync("second")
            self.assertEqual((output / "history/second/reports/raw.json").read_text(), "first result")
            shutil.rmtree(source / "reports")
            sync("empty_remote")
            self.assertEqual((output / "mirror/reports/raw.json").read_text(), "second and longer result")

    def test_empty_remote_failure_and_resume_never_delete_local(self):
        # A fake rsync models success once, then an empty/dead remote; no network.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bindir = root / "bin"; bindir.mkdir()
            fake = bindir / "rsync"
            fake.write_text("#!" + sys.executable + "\n" + '''import os,sys,pathlib
p=pathlib.Path(sys.argv[-1]);f=p/'reports'/'kept.json'
if not f.exists():
 f.parent.mkdir(parents=True,exist_ok=True);f.write_text('{"raw":"preserved"}');sys.exit(0)
sys.exit(255 if os.environ.get('FAKE_FAIL') else 0)
''')
            fake.chmod(0o755)
            key = root / "known_hosts"; key.write_text("test-only")
            out = root / "backup"
            env = dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ["PATH"])
            args = [sys.executable, str(SCRIPT), "--host", "test-host", "--remote-root", "/work/job",
                    "--output", str(out), "--known-hosts", str(key), "--remaining-minutes", ".004",
                    "--interval", ".03", "--final-interval", ".03", "--transfer-timeout", ".1"]
            one = subprocess.run(args, env=env, capture_output=True, timeout=5)
            self.assertEqual(one.returncode, 0, one.stderr.decode())
            kept = out / "mirror/reports/kept.json"; original = kept.read_bytes()
            self.assertTrue((out / "STOP_STARTING_GPU_JOBS").exists())
            two = subprocess.run(args + ["--resume"], env=dict(env, FAKE_FAIL="1"), capture_output=True, timeout=5)
            self.assertEqual(two.returncode, 1)
            self.assertEqual(kept.read_bytes(), original)
            self.assertTrue(list((out / "logs").glob("*_previous_stop_marker.txt")))
            status = json.loads((out / "status.json").read_text())
            self.assertFalse(status["running"])
            self.assertEqual(status["successes"], 0)
            self.assertGreater(status["failures"], 0)
            changed = list(args); changed[changed.index("test-host")] = "different-host"
            three = subprocess.run(changed + ["--resume"], env=env, capture_output=True, timeout=5)
            self.assertEqual(three.returncode, 2)
            self.assertEqual(kept.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
