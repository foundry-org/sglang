import argparse

from sglang.cli.utils import get_git_commit_hash
from sglang.version import __version__


def _startup_trace(label):
    import os as _os

    if _os.environ.get("SGLANG_STARTUP_TRACE") != "1":
        return
    try:
        import sys as _sys

        with open("/proc/self/stat") as f:
            ticks = int(f.read().split(")")[-1].split()[19])
        with open("/proc/uptime") as f:
            up = float(f.read().split()[0])
        print(
            f"[startup-trace] {up - ticks / _os.sysconf('SC_CLK_TCK'):6.2f}s {label}",
            file=_sys.stderr,
            flush=True,
        )
    except Exception:
        pass


def version(args, extra_argv):
    print(f"sglang version: {__version__}")
    print(f"git revision: {get_git_commit_hash()[:7]}")


def main():
    parser = argparse.ArgumentParser()

    # complex sub commands
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    subparsers.add_parser(
        "serve",
        help="Launch an SGLang server.",
        add_help=False,
    )
    subparsers.add_parser(
        "generate",
        help="Run inference on a multimodal model.",
        add_help=False,
    )

    # simple commands
    version_parser = subparsers.add_parser(
        "version",
        help="Show the version information.",
    )
    version_parser.set_defaults(func=version)

    _startup_trace("cli main: args parsed (sglang.cli.main imported)")
    args, extra_argv = parser.parse_known_args()

    if args.subcommand == "serve":
        from sglang.srt.utils.early_forkserver import start_early

        start_early()  # no-op unless SGLANG_EARLY_FORKSERVER=1
        from sglang.cli.serve import serve

        serve(args, extra_argv)
    elif args.subcommand == "generate":
        from sglang.cli.generate import generate

        generate(args, extra_argv)
    elif args.subcommand == "version":
        version(args, extra_argv)
