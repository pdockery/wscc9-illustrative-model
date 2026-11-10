"""The libraries must reproduce the frozen numbers exactly.

A fresh copy of lib/ must reproduce tests/golden/golden.json, so any diff here
is a regression until it is explained.

    pytest tests/test_golden.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import golden_runner as gr  # noqa: E402


@pytest.fixture(scope="module")
def golden() -> dict:
    if not gr.GOLDEN.exists():
        pytest.fail(f"no golden at {gr.GOLDEN}; it is copied with lib/")
    return json.loads(gr.GOLDEN.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def current() -> dict:
    return gr.run_all()


def test_no_scenario_errors(current):
    errored = {k: v["__error__"] for k, v in current.items() if "__error__" in v}
    assert not errored, f"scenarios raised: {errored}"


def test_scenario_set_unchanged(current, golden):
    assert set(current) == set(golden), (
        "scenario set changed -- re-freeze deliberately with --write, "
        "do not let it drift"
    )


@pytest.mark.parametrize("scenario", sorted(gr.SCENARIOS))
def test_scenario_matches_golden(scenario, current, golden):
    if scenario not in golden:
        pytest.skip("not in golden")
    cur, ref = current[scenario], golden[scenario]
    diffs = [f"{k}: golden={ref.get(k, '<absent>')!r} now={cur.get(k, '<absent>')!r}"
             for k in sorted(set(cur) | set(ref)) if cur.get(k) != ref.get(k)]
    assert not diffs, "\n".join(diffs[:25])
