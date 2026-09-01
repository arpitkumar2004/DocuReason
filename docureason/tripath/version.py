"""version.py — single runtime source of truth for the framework version.

Every runtime component reads the version from here.  It was previously
hard-coded as a literal in nine places (``config.py`` banner, ``serving/main.py``
response, two ``__init__`` files, ``configs/config.yaml`` header, three scripts
and a test assertion), which is why version bumps drifted.

``pyproject.toml`` keeps its own literal because build backends read it
statically; ``tests/test_version_consistency.py`` asserts the two agree.
"""

from __future__ import annotations

__version__ = "1.1.4"
