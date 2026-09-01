"""test_version_consistency.py — the version must be declared in exactly one place.

The version was previously a literal in nine locations (banner, REST response, two
package ``__init__`` files, the YAML header, three scripts and a test assertion), so
bumps drifted.  Runtime code now reads :mod:`docureason.tripath.version`; ``pyproject.toml``
keeps its own literal because build backends read it statically.  This test asserts
the two agree.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import docureason
import docureason.tripath as tripath
from docureason.tripath.version import __version__

pytestmark = pytest.mark.unit

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _pyproject_version() -> str:
    text = _PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match, "no version field found in pyproject.toml"
    return match.group(1)


def test_pyproject_matches_runtime_version():
    assert _pyproject_version() == __version__


def test_packages_expose_the_same_version():
    assert docureason.__version__ == __version__
    assert tripath.__version__ == __version__
