#!/usr/bin/env python3
"""Bounded local rsync supervisor. Never deletes local data or changes the host.

Start this BEFORE a remote experiment. It copies only selected artifact trees,
not a venv or the whole remote home. A successful sync is not a passed experiment.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

DEFAULT_PATHS = ("reports", "archives", "state_reference_banks", "artifacts")


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def parse_deadline(value):
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("deadline must include Z or a UTC offset")
    return dt.timestamp()


def schedule(remaining, interval, final_window, final_interval, timeout):
    return (min(final_interval if remaining <= final_window else interval, remaining),
            max(0.0, min(timeout, remaining)))


def validate(host, remote_root, paths):
    if not re.fullmatch(r"(?:[A-Za-z0-9_.-]+@)?[A-Za-z0-9][A-Za-z0-9_.-]*", host):
        raise ValueError("host must be an SSH alias or user@hostname; no shell/options")
    remote = PurePosixPath(remote_root)
    if not remote.is_absolute() or str(remote) == "/" or ".." in remote.parts:
        raise ValueError("remote-root must be an absolute experiment directory")
    for item in paths:
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", item):
            raise ValueError("each --path must be one top-level artifact directory name")


def command(config, output, attempt, checksum=False):
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
           "-o", "StrictHostKeyChecking=yes", "-o", "ServerAliveInterval=5",
           "-o", "ServerAliveCountMax=2",
           "-o", "UserKnownHostsFile=" + config["known_hosts"]]
    if config.get("identity_file"):
        ssh += ["-i", config["identity_file"], "-o", "IdentitiesOnly=yes"]
    args = ["rsync", "-a", "--safe-links", "--protect-args", "--timeout=15",
            "--partial", "--partial-dir=.rsync-partial", "--backup",
            "--backup-dir=" + str(output / "history" / attempt),
            "--stats", "-e", shlex.join(ssh)]
    if checksum:
        args.append("--checksum")
    for name in config["paths"]:
        args.append("--include=/" + name + "/***")
    args += ["--exclude=*", config["host"] + ":" + config["remote_root"].rstrip("/") + "/",
             str(output / "mirror") + "/"]
    # No --delete, --inplace, --remove-source-files, or arbitrary remote command.
    return args


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def run_transfer(cmd, stream, timeout):
    process = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        return process.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
        # Kill only this supervisor's transfer group, including its SSH child.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        if isinstance(exc, KeyboardInterrupt):
            raise
        return 124


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", required=True)
    ap.add_argument("--remote-root", required=True)
    ap.add_argument("--output", required=True, type=Path, help="New dedicated LOCAL backup directory")
    ap.add_argument("--known-hosts", required=True, type=Path, help="Existing verified host key file")
    ap.add_argument("--identity-file", type=Path)
    when = ap.add_mutually_exclusive_group(required=True)
    when.add_argument("--deadline", help="Assignment end, e.g. 2026-10-04T07:27:00Z")
    when.add_argument("--remaining-minutes", type=float)
    ap.add_argument("--path", action="append", dest="paths")
    ap.add_argument("--interval", type=float, default=30)
    ap.add_argument("--final-window", type=float, default=600)
    ap.add_argument("--final-interval", type=float, default=5)
    ap.add_argument("--stop-new-jobs-before", type=float, default=300)
    ap.add_argument("--transfer-timeout", type=float, default=45)
    ap.add_argument("--checksum-last", action="store_true", help="After the stop-new-jobs marker, checksum if time permits; can be costly")
    ap.add_argument("--resume", action="store_true", help="Resume only an identically configured backup directory")
    ap.add_argument("--dry-run", action="store_true", help="Print one rsync argv without network or file changes")
    args = ap.parse_args()
    paths = args.paths or list(DEFAULT_PATHS)
    validate(args.host, args.remote_root, paths)
    for value in (args.interval, args.final_window, args.final_interval,
                  args.stop_new_jobs_before, args.transfer_timeout):
        if value <= 0:
            ap.error("all intervals/windows/timeouts must be positive")
    end = parse_deadline(args.deadline) if args.deadline else time.time() + 60 * args.remaining_minutes
    remaining = end - time.time()
    if remaining <= 0:
        ap.error("deadline has passed; no remote requests made")
    # Monotonic deadline avoids NTP/local wall-clock adjustments extending the run.
    end_mono = time.monotonic() + remaining
    output = args.output.expanduser().resolve()
    config = {"schema": "deadline_rsync_v1", "host": args.host,
              "remote_root": str(PurePosixPath(args.remote_root)),
              "paths": sorted(set(paths)), "known_hosts": str(args.known_hosts.expanduser().resolve()),
              "identity_file": str(args.identity_file.expanduser().resolve()) if args.identity_file else None}
    if args.dry_run:
        print(json.dumps(command(config, output, "example"), indent=2))
        return 0
    for binary in ("rsync", "ssh"):
        if not shutil.which(binary):
            ap.error(f"{binary} unavailable; script never installs host packages")
    for key in ("known_hosts", "identity_file"):
        if config[key] and not Path(config[key]).is_file():
            ap.error(f"missing {key}; use an existing verified SSH configuration")
    if output.exists():
        if not args.resume or json.loads((output / "config.json").read_text()) != config:
            ap.error("existing output needs --resume with identical source/SSH/path configuration")
    else:
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "config.json", config)
    (output / "mirror").mkdir(exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    old_marker = output / "STOP_STARTING_GPU_JOBS"
    if old_marker.exists():
        old_marker.rename(output / "logs" / (run_id + "_previous_stop_marker.txt"))
    atomic_json(output / ("session_" + run_id + ".json"),
                {"start": timestamp(), "deadline": datetime.fromtimestamp(end, timezone.utc).isoformat(),
                 "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                 "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    successes = failures = 0
    checksum_done = False
    last = None
    interrupted = False
    try:
        while (remaining := end_mono - time.monotonic()) > 0:
            if remaining < min(1.0, args.transfer_timeout):
                break  # Do not start an almost-certainly interrupted last transfer.
            marker = output / "STOP_STARTING_GPU_JOBS"
            if remaining <= args.stop_new_jobs_before and not marker.exists():
                marker.write_text("Reserve remaining assignment time for completion, pulling and verification.\n")
            pause, timeout = schedule(remaining, args.interval, args.final_window,
                                      args.final_interval, args.transfer_timeout)
            attempt = run_id + f"_{successes + failures:05d}"
            do_checksum = bool(args.checksum_last and marker.exists() and not checksum_done)
            cmd = command(config, output, attempt, do_checksum)
            started = time.monotonic()
            with (output / "logs" / (attempt + ".log")).open("xb") as stream:
                code = run_transfer(cmd, stream, timeout)
            ok = code == 0
            successes += int(ok); failures += int(not ok)
            checksum_done |= bool(ok and do_checksum)
            last = {"time": timestamp(), "attempt": attempt, "exit_code": code,
                    "seconds": time.monotonic() - started, "checksum": do_checksum,
                    "remaining_seconds": max(0, end_mono - time.monotonic())}
            with (output / "attempts.jsonl").open("a") as stream:
                stream.write(json.dumps(last) + "\n")
            atomic_json(output / "status.json", {"running": True, "successes": successes,
                         "failures": failures, "last": last, "checksum_done": checksum_done})
            print(json.dumps(last), flush=True)
            # A failed/disappeared remote can never erase a previous local mirror.
            time.sleep(max(0, min(pause - last["seconds"], end_mono - time.monotonic())))
    except KeyboardInterrupt:
        interrupted = True
    finally:
        atomic_json(output / "status.json", {"running": False, "interrupted": interrupted,
                    "successes": successes, "failures": failures, "last": last,
                    "checksum_done": checksum_done, "finished": timestamp(),
                    "note": "A mirror is not proof of a complete/passed experiment. Inspect job_result, source manifest and raw reports."})
    return 0 if successes and last and last["exit_code"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
