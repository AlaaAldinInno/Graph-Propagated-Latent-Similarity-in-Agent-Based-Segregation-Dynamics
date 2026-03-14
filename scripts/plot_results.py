#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _exe_candidates(build: Path, name: str, config: str) -> list[Path]:
    exe_name = f"{name}.exe" if os.name == "nt" else name
    return [
        build / exe_name,
        build / config / exe_name,
        build / "Debug" / exe_name,
        build / "Release" / exe_name,
        build / "RelWithDebInfo" / exe_name,
        build / "MinSizeRel" / exe_name,
    ]


def _resolve_runner(build: Path, config: str) -> Path:
    for candidate in _exe_candidates(build, "sgnn_runner", config):
        if candidate.exists():
            return candidate
    checked = "\n".join(f"  - {p}" for p in _exe_candidates(build, "sgnn_runner", config))
    raise FileNotFoundError(f"Could not find sgnn_runner executable. Checked:\n{checked}")


def summarize() -> None:
    parser = argparse.ArgumentParser(description="Summarize SGNN size-sweep outputs")
    parser.add_argument(
        "--config",
        default=("Release" if os.name == "nt" else ""),
        help="Build configuration for multi-config generators (e.g., Debug/Release on Windows).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    build = root / "build"
    build.mkdir(exist_ok=True)
    subprocess.run(["cmake", "-S", str(root), "-B", str(build)], check=True)

    build_cmd = ["cmake", "--build", str(build)]
    if args.config:
        build_cmd.extend(["--config", args.config])
    if os.name != "nt":
        build_cmd.extend(["-j", str(os.cpu_count() or 1)])
    subprocess.run(build_cmd, check=True)

    runner = _resolve_runner(build, args.config or "")
    subprocess.run([str(runner), "summarize"], check=True)


if __name__ == "__main__":
    try:
        summarize()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        raise
