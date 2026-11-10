"""Structural guards on the library.

The shared modules carry no knowledge of the 9-bus case, so another case
can share them; every guard below is a hard gate.
The Colab guard holds the modules a notebook's bootstrap fetches from
``main/lib/`` to the 14 modules ``lib/`` carries and ``lib/MANIFEST.md``
records.

    python -m pytest tests/test_structure.py -q
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: The library folder at the repository root, beside the notebooks.
LIB = ROOT / "lib"

#: Modules that must become case-agnostic. ``wscc9_model`` and
#: ``ieee9_network`` are the case itself and are deliberately excluded.
SHARED = ["seams_engine.py", "footprints.py", "atc.py",
          "revenue_allocation.py", "risk.py", "bilateral.py",
          "wscc9_figures.py", "nodal_plot.py", "etag.py"]

#: The flat modules a notebook's Colab bootstrap fetches by name from
#: ``https://raw.githubusercontent.com/pdockery/wscc9-illustrative-model/main/lib/``.
#: The modules import one another, so every bootstrap fetches all 14, and
#: the set is exactly what ``lib/`` carries.
COLAB_MODULES = {"atc.py", "bilateral.py", "etag.py", "footprints.py",
                 "ieee9_network.py", "intertie.py", "lab_types.py",
                 "nodal_plot.py", "revenue_allocation.py", "risk.py",
                 "seam_views.py", "seams_engine.py", "wscc9_figures.py",
                 "wscc9_model.py"}


def _src(name: str) -> str:
    return (LIB / name).read_text(encoding="utf-8")


def test_shared_modules_exist():
    missing = [m for m in SHARED if not (LIB / m).exists()]
    assert not missing, missing


def test_colab_bootstrap_matches_disk():
    """Every module the notebooks fetch must exist, and every shared module
    must be fetched. This is what makes adding a new top-level module a
    breaking change -- the reason the Case registry lives in seams_engine.py."""
    on_disk = {p.name for p in LIB.glob("*.py")}
    assert on_disk == COLAB_MODULES, (
        f"missing on disk: {sorted(COLAB_MODULES - on_disk)}; "
        f"not fetched: {sorted(on_disk - COLAB_MODULES)}")
    assert set(SHARED) <= COLAB_MODULES, (
        "a shared module is not in the Colab fetch list")


def test_manifest_lists_the_carried_modules():
    """``lib/MANIFEST.md`` records every carried module by name and SHA-256."""
    text = (LIB / "MANIFEST.md").read_text(encoding="utf-8")
    listed = set(re.findall(r"^\|\s*`([\w]+\.py)`\s*\|\s*`[0-9a-f]{64}`",
                            text, re.M))
    assert listed == COLAB_MODULES, (
        f"not in the manifest: {sorted(COLAB_MODULES - listed)}; "
        f"not carried: {sorted(listed - COLAB_MODULES)}")


def test_no_carried_module_imports_an_uncarried_one():
    """A module-scope or lazy import of a sibling module names a carried one,
    so a Colab bootstrap that fetches the 14 is complete."""
    carried = {m[:-3] for m in COLAB_MODULES}
    third_party = {"numpy", "pandas", "scipy", "matplotlib", "pypsa",
                   "pandapower", "pycirclize", "IPython", "typing_extensions",
                   "mpl_toolkits"}
    stdlib = set(sys.stdlib_module_names)
    bad = []
    for name in sorted(COLAB_MODULES):
        for node in ast.walk(ast.parse(_src(name))):
            if isinstance(node, ast.Import):
                tops = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                tops = [node.module.split(".")[0]]
            else:
                continue
            bad += [f"{name}:{node.lineno}: {top}" for top in tops
                    if top not in carried | third_party | stdlib]
    assert not bad, bad


@pytest.mark.parametrize("module", SHARED)
def test_no_module_scope_wscc9_import(module):
    """No shared module may import the case at MODULE scope.  [HARD since phase 2]

    Column-0 only: an indented ``import wscc9_model`` inside a function or a
    ``__main__`` smoke block is lazy, costs nothing when the case is absent,
    and is acceptable.
    """
    src = _src(module)
    hits = [ln for ln in src.splitlines()
            if re.match(r"^(import wscc9_model|from wscc9_model)", ln)]
    assert not hits, hits


@pytest.mark.parametrize("module", ["revenue_allocation.py", "risk.py",
                                    "bilateral.py"])
def test_no_wscc9_defaults(module):
    """The ``wm.DEFAULT_LOADS if loads is None`` pattern is the single largest
    silent-wrong-answer risk when a second case shares these libraries.
    [HARD since phase 2 — defaults now resolve through seams_engine.active()]"""
    src = _src(module)
    hits = [ln.strip() for ln in src.splitlines() if "wm.DEFAULT_" in ln]
    assert not hits, hits[:10]


def test_no_hardcoded_slack_bus():
    """``slack_bus="1"`` is the second silent-wrong-answer hazard: it is a
    9-bus bus name that a WECC network does not have.  [HARD since phase 3 —
    ``compute_ptdf`` now defaults to the first sorted bus, and every library
    call site resolves through the case.]

    Scope: LIBRARY code only. A ``__main__`` smoke block explicitly builds the
    9-bus case, so an explicit matching slack there is correct, and doc markup
    (`` `` ``) may cite the old form when explaining the change.
    """
    hits = []
    for module in SHARED:
        src = _src(module).split('if __name__ == "__main__":')[0]
        for i, ln in enumerate(src.splitlines(), 1):
            if "``" in ln:
                continue
            if re.search(r'slack_bus\s*=\s*["\']1["\']', ln):
                hits.append(f"{module}:{i}: {ln.strip()}")
    assert not hits, hits


def test_library_imports_without_the_case():
    """The invariant that proves the refactor worked: every shared module must
    import cleanly when ``wscc9_model`` does not exist.  [HARD since phase 2]"""
    # NB: must use the find_spec protocol -- Python 3.12 ignores the legacy
    # find_module/load_module pair, which would silently let the import through
    # and turn this guard into a false pass.
    prog = textwrap.dedent(f"""
        import sys, importlib
        BLOCKED = ('wscc9_model', 'ieee9_network')
        class _Block:
            def find_spec(self, name, path=None, target=None):
                if name in BLOCKED:
                    raise ImportError(f'{{name}} is blocked')
                return None
        sys.meta_path.insert(0, _Block())
        sys.path.insert(0, r"{LIB}")
        for m in {[s[:-3] for s in SHARED]!r}:
            importlib.import_module(m)
        print("OK")
    """)
    r = subprocess.run([sys.executable, "-c", prog], capture_output=True,
                       text=True, cwd=str(LIB))
    assert "OK" in r.stdout, r.stderr[-2000:]
