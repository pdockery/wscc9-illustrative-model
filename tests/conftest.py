"""Import path for the test suite.

The library modules live in ``lib/`` at the repository root, a copy of the
shared 9-bus library; ``lib/MANIFEST.md`` records each
module's SHA-256 and the date of the copy. Inserting
``lib/`` here, before any test module is imported, makes collection
independent of the order in which pytest reaches the files, and ``tests/``
beside it lets ``test_golden`` import ``golden_runner``.

    python -m pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "lib", _ROOT / "tests"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)


def pytest_configure(config: pytest.Config) -> None:
    """Register the ``slides`` marker, which the deck notebooks' slide checks carry."""
    config.addinivalue_line(
        "markers", "slides: executes a deck notebook and checks its clears against the slides"
    )
