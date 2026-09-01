#!/usr/bin/env python3
"""Freeze the runtime dependency closure of this project into ``requirements.lock``.

Why this exists
---------------
``pyproject.toml`` declares ranges (``torch>=2.0.0,<3.0.0``), which is correct for a
library but means two installs a week apart resolve to different code.  Every
performance and retrieval number this project reports was measured against one
specific resolution and nothing recorded which.

Why not ``pip freeze``
----------------------
``pip freeze`` dumps the whole environment - here that includes ``twine``,
``keyring``, ``pytest`` and their trees, none of which the framework imports at
runtime.  This walks ``Requires-Dist`` from the declared runtime roots instead,
evaluating environment markers, so the lock is the *runtime* closure and nothing
more.

Usage
-----
    python scripts/lock_requirements.py            # write requirements.lock
    python scripts/lock_requirements.py --check     # exit 1 if the file is stale

The lock is platform- and Python-specific by construction (marker evaluation and
wheel availability both depend on them); the header records which.
"""

from __future__ import annotations

import argparse
import platform
import re
import sys
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Dict, List, Set

try:  # 3.11+
    import tomllib
except ModuleNotFoundError:  # 3.10 has no tomllib
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "reading pyproject.toml needs a TOML parser: python 3.11+ ships tomllib, "
            "otherwise run `pip install tomli`. It is deliberately not a runtime "
            "dependency of this project - only this maintenance script needs it."
        ) from None

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
LOCKFILE = ROOT / "requirements.lock"

_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(spec: str) -> str:
    match = _NAME_RE.match(spec)
    return match.group(1) if match else ""


def runtime_roots() -> List[str]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return [requirement_name(dep) for dep in data["project"]["dependencies"]]


def marker_holds(spec: str) -> bool:
    """True when *spec*'s environment marker applies here (and needs no extra)."""
    if ";" not in spec:
        return True
    marker = spec.split(";", 1)[1].strip()
    if "extra" in marker:
        # Requirements gated on an extra belong to that extra, not the runtime set.
        return False
    try:
        from packaging.markers import Marker

        return bool(Marker(marker).evaluate())
    except Exception:
        # No packaging available, or a marker we cannot parse: keep the dependency
        # rather than silently dropping something the runtime may need.
        return True


def closure(roots: List[str]) -> Dict[str, str]:
    """Resolve *roots* and everything they require to {canonical name: version}."""
    resolved: Dict[str, str] = {}
    missing: Set[str] = set()
    queue = list(roots)
    seen: Set[str] = set()

    while queue:
        name = queue.pop()
        key = canonical(name)
        if not key or key in seen:
            continue
        seen.add(key)
        try:
            dist = distribution(name)
        except PackageNotFoundError:
            missing.add(name)
            continue
        resolved[canonical(dist.metadata["Name"])] = dist.version
        for spec in dist.requires or []:
            if marker_holds(spec):
                queue.append(requirement_name(spec))

    if missing:
        raise SystemExit(
            f"not installed, cannot lock: {', '.join(sorted(missing))}\n"
            "Install the runtime dependencies first: pip install -e ."
        )
    return resolved


def render(resolved: Dict[str, str]) -> str:
    header = [
        "# requirements.lock - fully pinned runtime closure of pyproject.toml [project.dependencies].",
        "#",
        "# Generated, do not edit by hand:  python scripts/lock_requirements.py",
        "# Check for staleness:             python scripts/lock_requirements.py --check",
        "#",
        f"# python   : {platform.python_version()}",
        f"# platform : {sys.platform} / {platform.machine()}",
        "#",
        "# Runtime dependencies only - test, lint and build extras are deliberately absent.",
        "# Model weights are pinned separately, by commit, in docureason/tripath/model_pins.py.",
        "#",
        "# Platform scope: markers were evaluated on the platform above, so this is that",
        "# platform's closure. Installing it elsewhere still pins every package listed;",
        "# packages that only another platform requires (the CUDA runtime wheels torch",
        "# pulls in on linux, for instance) are resolved by pip at install time and are",
        "# not pinned here. Regenerate on a target platform to pin those too.",
        "",
    ]
    body = [f"{name}=={version}" for name, version in sorted(resolved.items())]
    return "\n".join(header + body) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if requirements.lock is stale")
    args = parser.parse_args()

    rendered = render(closure(runtime_roots()))

    if args.check:
        current = LOCKFILE.read_text(encoding="utf-8") if LOCKFILE.exists() else ""

        # The header records the generating interpreter and platform, so compare the
        # pins only - a lock generated on Linux is not stale when checked on Windows.
        def pins(text: str) -> List[str]:
            return [line for line in text.splitlines() if line and not line.startswith("#")]

        if pins(current) != pins(rendered):
            print("requirements.lock is stale - run: python scripts/lock_requirements.py")
            return 1
        print("requirements.lock is up to date")
        return 0

    LOCKFILE.write_text(rendered, encoding="utf-8")
    print(f"wrote {LOCKFILE.relative_to(ROOT)} ({len(pins_of(rendered))} pinned packages)")
    return 0


def pins_of(text: str) -> List[str]:
    return [line for line in text.splitlines() if line and not line.startswith("#")]


if __name__ == "__main__":
    raise SystemExit(main())
