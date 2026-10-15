#!/usr/bin/env python3
"""Exercise the built wheel as an actual user would, outside the source tree.

No hosted model or private corpus is required. This intentionally fails on the
2026-10-15 base because the packaged wheel lacks its fixture data.

Run: python scripts/smoke_wheel.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str, cwd: Path, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        list(args),
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    if result.returncode:
        raise RuntimeError(
            f"command exited {result.returncode}: {' '.join(args)}\n"
            f"stdout:\n{result.stdout[-2500:]}\n"
            f"stderr:\n{result.stderr[-4000:]}"
        )
    return result.stdout


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="grounded-agent-wheel-smoke-") as temporary:
        temp = Path(temporary)
        wheelhouse = temp / "wheels"
        wheelhouse.mkdir()
        print("Building the actual wheel...", flush=True)
        run(
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--wheel-dir",
            str(wheelhouse),
            str(ROOT),
            cwd=temp,
        )
        wheels = list(wheelhouse.glob("grounded_agent_lab-*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"expected one built wheel; observed {len(wheels)}")
        venv_dir = temp / "isolated-install"
        venv.EnvBuilder(with_pip=True).create(venv_dir)
        scripts = "Scripts" if os.name == "nt" else "bin"
        pip = venv_dir / scripts / ("pip.exe" if os.name == "nt" else "pip")
        cli = venv_dir / scripts / (
            "grounded-agent.exe" if os.name == "nt" else "grounded-agent"
        )
        print("Installing wheel into a new environment...", flush=True)
        run(str(pip), "install", "--disable-pip-version-check", str(wheels[0]), cwd=temp)
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["PYTHONNOUSERSITE"] = "1"
        print("Invoking installed CLI from outside the checkout...", flush=True)
        raw = run(str(cli), "demo", "--json", cwd=temp, env=env)
        payload = json.loads(raw)
        if not payload.get("ok") or payload.get("n") != 7:
            raise AssertionError(f"unexpected installed demo result: {payload}")
        print(f"Installed-wheel smoke PASS: {payload['passed']}/{payload['n']} demo cases")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, RuntimeError, subprocess.TimeoutExpired) as exc:
        sys.stderr.write(f"Installed-wheel smoke FAIL: {exc}\n")
        raise SystemExit(1) from exc
