#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _exe_candidates(build: Path, name: str, config: str) -> list[Path]:
    exe_name = f"{name}.exe" if os.name == "nt" else name
    candidates = [
        build / exe_name,
        build / config / exe_name,
        build / "Debug" / exe_name,
        build / "Release" / exe_name,
        build / "RelWithDebInfo" / exe_name,
        build / "MinSizeRel" / exe_name,
    ]
    # keep order, remove duplicates
    out: list[Path] = []
    seen: set[Path] = set()
    for c in candidates:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _resolve_runner(build: Path, config: str) -> Path:
    for candidate in _exe_candidates(build, "sgnn_runner", config):
        if candidate.exists():
            return candidate
    checked = "\n".join(f"  - {p}" for p in _exe_candidates(build, "sgnn_runner", config))
    raise FileNotFoundError(f"Could not find sgnn_runner executable. Checked:\n{checked}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and run C++ SGNN experiments")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min-size", type=int, default=20)
    parser.add_argument("--max-size", type=int, default=140)
    parser.add_argument("--step-size", type=int, default=20)
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
    subprocess.run([
        str(runner),
        "run",
        "--workers", str(args.workers),
        "--min-size", str(args.min_size),
        "--max-size", str(args.max_size),
        "--step-size", str(args.step_size),
    ], check=True)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        raise
