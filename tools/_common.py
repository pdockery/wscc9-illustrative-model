"""Shared helpers for nb_execute.py and nb_staleness.py (standard library only)."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import subprocess
import sys
import tokenize
import typing
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, NamedTuple

#: The library folder, the shared 9-bus library; lib/MANIFEST.md records each module's
#: SHA-256 and the date of the copy.
LIB_DIR = "lib"
#: The notebooks sit at the repository root.
NOTEBOOK_GLOB = "*.ipynb"
#: The record of the copy: its date and a SHA-256 per module.
MANIFEST = "lib/MANIFEST.md"
STAMP_KEY = "lab_execution"
#: The ignored folder, under the repository root, that holds the tools' caches and ledgers.
CACHE_DIR = "tools/.cache"


def git(root: str, *args: str, text: bool = True) -> str:
    r = subprocess.run(
        ["git", "-C", root, *args],
        capture_output=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
    )
    out = r.stdout
    if r.returncode != 0:
        return "" if text else b""  # type: ignore[return-value]  # git returns bytes when text is False
    return out.decode("utf-8", "replace") if text else out  # type: ignore[return-value]  # git returns bytes when text is False


def repo_root(path: str) -> str:
    start = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
    out = git(start, "rev-parse", "--show-toplevel").strip()
    if not out:
        raise RuntimeError("not inside a git repository: %s" % path)
    return os.path.normpath(out)


def lib_dirs() -> list[str]:
    """The library directories every notebook depends on, repository-relative."""
    return [LIB_DIR]


def lib_state(root: str, dirs: list[str]) -> tuple[bool, str]:
    """Return (dirty, sha256 of `git diff HEAD`, the porcelain status and the untracked files over dirs).

    The status lists every untracked file, and the sha256 of each one's content enters the hash,
    so a change to an untracked file, such as a fresh copy of lib/, changes the hash.
    """
    diff = git(root, "diff", "HEAD", "--", *dirs, text=False)
    status = git(root, "status", "--porcelain", "--untracked-files=all", "--", *dirs)
    dirty = bool(diff.strip()) or bool(status.strip())
    h = hashlib.sha256(diff + status.encode("utf-8"))  # type: ignore[arg-type, operator]  # diff is bytes here (text=False), typed str
    for line in status.splitlines():
        if not line.startswith("?? "):
            continue
        rel = line[3:].strip().strip('"')
        p = Path(root) / rel
        if p.is_file():
            h.update(rel.encode("utf-8"))
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return dirty, h.hexdigest()


def manifest_hash(root: str) -> tuple[str, str] | tuple[None, None]:
    """The path and sha256 of lib/MANIFEST.md, or (None, None) when the file is absent."""
    p = Path(root) / MANIFEST
    if not p.is_file():
        return None, None
    return MANIFEST, hashlib.sha256(p.read_bytes()).hexdigest()


def cache_file(root: str, name: str) -> Path:
    """The path of one cache or ledger file of the repository at root, under CACHE_DIR."""
    return Path(root) / CACHE_DIR / name


def write_atomic(path: Path, data: bytes) -> None:
    """Write bytes to path through a temporary sibling and a rename, so a reader never sees a partial file.

    The bytes are written as given, so a caller that encodes LF text keeps LF on every platform.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_bytes(data)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def read_stamp(path: str, tail_bytes: int = 1 << 20) -> dict[str, Any] | None:
    """Return the lab_execution stamp of a notebook, or None. The tail of the file is read first,
    where nbformat writes the key-sorted top-level metadata block."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        f.seek(max(0, size - tail_bytes))
        tail = f.read().decode("utf-8", "replace")
    marker = '"%s"' % STAMP_KEY
    i = tail.rfind(marker)
    if i >= 0:
        j = tail.find(":", i + len(marker))
        try:
            val, _ = json.JSONDecoder().raw_decode(tail[j + 1 :].lstrip())
            return val
        except ValueError:
            pass
    if '"nbformat_minor"' in tail and '"metadata"' in tail:
        return None
    with open(path, encoding="utf-8") as f:  # metadata not in the tail: full read
        return (json.load(f).get("metadata") or {}).get(STAMP_KEY)


#: The bump that invalidates tools/.cache/fingerprints, needed whenever code_fingerprint or
#: normalize_docstring changes what it hashes. The interpreter release sits beside it in the
#: key, because ast.dump prints differently under different Python releases.
FINGERPRINT_RULE = 2
#: The bump for the runtime form of the fingerprint, needed whenever strip_hints changes what it
#: removes. The form's cache key carries it beside FINGERPRINT_RULE.
RUNTIME_RULE = 2
_DOCSTRING_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
_FINGERPRINTS: dict[str, str | None] = {}  # cache key -> fingerprint, for one process
#: The whitespace the formatter trims, Rust's char::is_whitespace. Python's str.strip also takes
#: \x1c to \x1f, which the formatter keeps as text.
_TRIMMED = (
    "\t\n\x0b\x0c\r \x85\xa0 " + "".join(map(chr, range(0x2000, 0x200B))) + "    　"
)
_TAB_WIDTH = 8
#: The `typing` names a run under Python 3.10, the oldest the repository supports, can import.
#: A newer name raises ImportError there, so its import is code and stays in the comparison.
_TYPING_NAMES = frozenset(typing.__all__) - {
    "LiteralString",
    "Never",
    "NoDefault",
    "NotRequired",
    "ReadOnly",
    "Required",
    "Self",
    "TypeAliasType",
    "TypeIs",
    "TypeVarTuple",
    "Unpack",
    "assert_never",
    "assert_type",
    "clear_overloads",
    "dataclass_transform",
    "evaluate_forward_ref",
    "get_overloads",
    "get_protocol_members",
    "is_protocol",
    "override",
    "reveal_type",
}
#: The run-time objects whose annotation changes what a class body declares, by import path.
_DECLARED = {
    "dataclasses.InitVar": "InitVar",
    "dataclasses.KW_ONLY": "KW_ONLY",
    "typing.ClassVar": "ClassVar",
}
_DECLARED_NAMES = frozenset(_DECLARED.values())
#: The calls that take a class and cannot rewrite it, so a class passed to one stays plain.
_PLAIN_CALLS = frozenset(
    {
        "getattr",
        "hasattr",
        "id",
        "isinstance",
        "issubclass",
        "len",
        "print",
        "repr",
        "setattr",
        "str",
        "super",
        "type",
    }
)
_FUNCTIONS = (ast.AsyncFunctionDef, ast.FunctionDef)
#: The nodes whose names belong to a scope of their own.
_NEW_SCOPES = (
    *_FUNCTIONS,
    ast.ClassDef,
    ast.DictComp,
    ast.GeneratorExp,
    ast.Lambda,
    ast.ListComp,
    ast.SetComp,
)


def _indentation(prefix: str) -> tuple:
    """The width and the removable length the formatter gives a docstring line's indentation.

    The rule is measured on ruff 0.15.5, and it is not `expandtabs`. Spaces alone count one each.
    Spaces then tabs align to a tab stop, and tabs then spaces count eight and one, and in both
    forms any whitespace after them adds nothing. A run that goes on from those forms with a
    space or a tab is mixed. Its first spaces and tabs count in full, a later tab moves to the
    next stop, and any later whitespace counts its UTF-8 length.

    Parameters
    ----------
    prefix : str
        The whitespace at the start of a line.

    Returns
    -------
    tuple of int
        The width in columns, and the number of characters the form holds.

    """
    spaces = len(prefix) - len(prefix.lstrip(" "))
    tabs = len(prefix) - spaces - len(prefix[spaces:].lstrip("\t"))
    rest = prefix[spaces + tabs :]
    if not tabs:
        return spaces, spaces
    inner = 0
    if not spaces:
        inner = len(rest) - len(rest.lstrip(" "))
        rest = rest[inner:]
        if not rest.startswith("\t"):
            return _TAB_WIDTH * tabs + inner, tabs + inner
    elif rest[:1] not in (" ", "\t"):
        return spaces + _TAB_WIDTH * tabs - spaces % _TAB_WIDTH, spaces + tabs
    width = spaces + _TAB_WIDTH * tabs + inner
    for char in rest:
        width += (
            _TAB_WIDTH - width % _TAB_WIDTH
            if char == "\t"
            else len(char.encode("utf-8"))
        )
    return width, len(prefix)


def normalize_docstring(value: str) -> str:
    """The docstring value with the whitespace the formatter rewrites removed.

    The rule is the one `ruff format` applies, measured against ruff 0.15.5. Trailing whitespace
    goes from every line, the indentation counts by _indentation, the margin of the narrowest
    non-blank line after the first is removed, and blank lines at either end go. Indentation
    beyond the margin stays, because the formatter keeps it. A tab after the first non-blank
    character stays, because the formatter keeps that too. The formatter removes a margin that
    holds a tab from a line indented with spaces alone by its character count and not its width,
    so 8 spaces beside a one-tab margin keep 7, and the rule does the same.

    Parameters
    ----------
    value : str
        The value of the docstring constant, with its escapes already resolved.

    Returns
    -------
    str
        The normalized text, which is equal across a reformat.

    """
    rows = []
    for line in value.split("\n"):
        body = line.strip(_TRIMMED)
        prefix = line[: len(line) - len(line.lstrip(_TRIMMED))]
        rows.append((prefix, _indentation(prefix), body))
    filled = [indentation for _, indentation, body in rows[1:] if body]
    width, length = min(filled, key=lambda indentation: indentation[0], default=(0, 0))
    out = [rows[0][2]]
    for prefix, (columns, _), body in rows[1:]:
        cut = width if prefix.strip(" ") else length
        out.append(" " * (columns - cut) + body if body else "")
    return "\n".join(out).strip(_TRIMMED)


def decode_source(data: bytes) -> str | None:
    """A module's bytes read as Python reads them, under its BOM or PEP 263 cookie.

    Parameters
    ----------
    data : bytes
        The file's bytes.

    Returns
    -------
    str or None
        The text, or None when the declared encoding is unknown or does not decode the bytes.

    """
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
        return data.decode(encoding)
    except (SyntaxError, UnicodeDecodeError, LookupError):
        return None


def docstring_nodes(tree: ast.AST) -> list:
    """The constant nodes of a tree that the formatter treats as docstrings.

    Parameters
    ----------
    tree : ast.AST
        A parsed module.

    Returns
    -------
    list of ast.Constant
        Each first statement of a module, class or function that is a plain string.

    """
    found = []
    for node in ast.walk(tree):
        if isinstance(node, _DOCSTRING_OWNERS) and node.body:
            first = node.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                found.append(first.value)
    return found


def code_fingerprint(text: str) -> str | None:
    """The sha256 of a module's parsed code, or None when it does not parse.

    The hash is taken over `ast.dump` of the parsed module, which excludes positions, with three
    forms the formatter changes brought to one reading. A docstring passes through
    normalize_docstring. The `kind` of a constant is cleared, because the formatter drops a `u`
    prefix. A `del` of one parenthesized tuple is read as a `del` of its elements, because the
    formatter parenthesizes a long `del a, b`. None is never equal to anything, including None,
    so a caller compares through `a is not None and a == b`.

    The fingerprint does not see comments, layout, parentheses, trailing commas, quoting, string
    prefixes, the spelling of a number, the splitting of a semicolon statement, line endings, a
    byte order mark, positions, or the whitespace normalize_docstring removes. The hash states
    that the parsed code is equal, and three limits sit outside that statement.

    Layout reaches the compiled instructions, though not what they compute. Under CPython 3.12 a
    multi-line return adds NOPs and a line break can invert a jump, so `co_code` differs between
    two texts of equal parsed code.

    Positions reach values at run time, because `f_lineno`, `co_firstlineno`, a warning's line
    number and a count of the file's own lines all move under a pass. A module that branches on
    one of those is not protected by this hash.

    The docstring whitespace reaches `__doc__`. A help text prints with different indentation,
    and a read of `__doc__` by column, a doctest or a count of the value can change result.
    A read from another module is not seen. Every word of a docstring still counts here. Non-breaking and other Unicode spaces at
    the ends of a docstring line count as whitespace, because the formatter removes them.

    A docstring with a newline written as an escape can read as a code change when the
    formatter moves its margin, because the value's lines are then not the source's lines. The
    error is a refusal, never a pass, and no module on the list on 2026-11-03 hits it. The hash
    depends on the interpreter's `ast.dump`, so a fingerprint is compared within one Python
    release and is not stored without the release in its key.

    Parameters
    ----------
    text : str
        The module source, decoded. Line endings and a leading byte order mark do not matter. A
        coding cookie has no effect on a str, so a caller holding bytes goes through
        fingerprint_bytes, which honours it.

    Returns
    -------
    str or None
        The hexadecimal digest, or None for a module `ast.parse` rejects.

    """
    return _digest(normalized_tree(text))


def normalized_tree(
    text: str,
    *,
    runtime: bool = False,
    keep: frozenset[str] = frozenset(),
    regions: list | None = None,
) -> ast.Module | None:
    """A module parsed and brought to the reading the fingerprints hash.

    Parameters
    ----------
    text : str
        The module source, decoded.
    runtime : bool
        True to pass the tree through strip_hints as well. The docstrings are normalized
        first, so that a string a stripped statement leaves in first place stays a string.
    keep, regions
        Passed to strip_hints.

    Returns
    -------
    ast.Module or None
        The tree, or None for a module `ast.parse` rejects.

    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", SyntaxWarning
            )  # an invalid escape is not this tool's report
            tree = ast.parse(text.lstrip("\ufeff"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant):
                node.kind = None
            elif (
                isinstance(node, ast.Delete)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Tuple)
            ):
                node.targets = list(node.targets[0].elts)
        for node in docstring_nodes(tree):
            node.value = normalize_docstring(node.value)
        return strip_hints(tree, keep=keep, regions=regions) if runtime else tree
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None


def _digest(tree: ast.Module | None) -> str | None:
    """The sha256 of `ast.dump` of a tree, or None when there is no tree to dump."""
    if tree is None:
        return None
    try:
        dumped = ast.dump(tree)
    except (ValueError, RecursionError, MemoryError):
        return None
    return hashlib.sha256(dumped.encode("utf-8", "backslashreplace")).hexdigest()


def runtime_fingerprint(text: str) -> str | None:
    """The code fingerprint with type hints removed by strip_hints.

    A hint moves no stored result, so nb_staleness.py compares this form. The plain form stays
    the test of a touched module, because a hinted module is reformatted.

    Parameters
    ----------
    text : str
        The module source, decoded.

    Returns
    -------
    str or None
        The hexadecimal digest, or None for a module `ast.parse` rejects.

    """
    return _digest(normalized_tree(text, runtime=True))


class _Scope(NamedTuple):
    """The statement list a rule is reading.

    Attributes
    ----------
    owner : ast.AST
        The module, class or function that holds the statements.
    bound : set of str
        The names the owning function binds, empty outside a function.
    parameters : set of str
        The owning function's parameters that it never deletes, empty outside a function.

    """

    owner: ast.AST
    bound: set[str]
    parameters: set[str]


class _Strip(NamedTuple):
    """What the rules read about one module, gathered before any stripping.

    Attributes
    ----------
    flags : dict
        The line of the import behind each spelling of `TYPE_CHECKING`, from _flag_spellings.
    imported : dict
        The import path and line of each name a top-level import binds, from _top_imports.
    binders : collections.Counter
        The number of bindings of each name in any scope, from binder_names.
    live : set of str
        The names code that runs reads, from _live_reads, with the caller's `keep`.
    passed : set of str
        The names passed to a call that can rewrite a class, from _passed_names.
    regions : list or None
        The list that collects what was stripped, or None when no caller wants it.

    """

    flags: dict[str, int]
    imported: dict[str, tuple[str, int]]
    binders: Counter
    live: set[str]
    passed: set[str]
    regions: list | None


class HintSite(NamedTuple):
    """One annotation a module writes.

    Attributes
    ----------
    key : tuple
        The dotted owner, the parameter, return or target, and the count among equal places,
        which two versions of a module share.
    annotation : ast.expr
        The annotation.
    runs : bool
        True where Python evaluates the annotation absent the future import, which is every
        place outside a function body's own statements.
    holder : tuple or None
        The dotted class whose body writes the annotation, else None.

    """

    key: tuple
    annotation: ast.expr
    runs: bool
    holder: tuple | None


def _dotted(node: ast.expr) -> str:
    """The dotted text of a name or an attribute chain on a name, else empty."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ""
    return ".".join([node.id, *reversed(parts)])


def _names_bound_by(node: ast.AST) -> list[str]:
    """The names one node binds in its scope."""
    if isinstance(node, (*_FUNCTIONS, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.Import):
        return [alias.asname or alias.name.split(".")[0] for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        return [alias.asname or alias.name for alias in node.names]
    if isinstance(node, ast.arg):
        return [node.arg]
    if isinstance(node, ast.Name):
        return [] if isinstance(node.ctx, ast.Load) else [node.id]
    if isinstance(node, ast.ExceptHandler | ast.MatchAs | ast.MatchStar):
        return [node.name] if node.name else []
    if isinstance(node, ast.MatchMapping):
        return [node.rest] if node.rest else []
    if isinstance(node, ast.Global | ast.Nonlocal):
        return list(node.names)
    return []


def binder_names(tree: ast.AST) -> Counter:
    """The count of bindings of each name in a tree, in every scope.

    Parameters
    ----------
    tree : ast.AST
        The parsed module.

    Returns
    -------
    collections.Counter
        Each name with the number of statements, parameters and targets that bind it.

    """
    return Counter(name for node in ast.walk(tree) for name in _names_bound_by(node))


def _flag_spellings(tree: ast.Module, binders: Counter) -> dict[str, int]:
    """The spellings of `TYPE_CHECKING` that mean the typing constant, with their lines.

    A spelling counts only from an unconditional import at the top of the module, because
    a block above the import raises NameError, and only when nothing else in the module binds
    its name, because the block then holds what the module put there. typing_extensions does
    not count, because a local module of that name can define the flag as anything.

    Parameters
    ----------
    tree : ast.Module
        The module, before any stripping.
    binders : collections.Counter
        From binder_names.

    Returns
    -------
    dict
        Each spelling, `TYPE_CHECKING` or `<alias>.TYPE_CHECKING`, with the line of its import.

    """
    if any(
        isinstance(node, ast.Attribute)
        and node.attr == "TYPE_CHECKING"
        and not isinstance(node.ctx, ast.Load)
        for node in ast.walk(tree)
    ):
        return {}
    flags: dict[str, int] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            spellings = [
                (alias.asname or alias.name, ".TYPE_CHECKING")
                for alias in node.names
                if alias.name == "typing"
            ]
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module == "typing"
            and not node.level
        ):
            spellings = [
                ("TYPE_CHECKING", "")
                for alias in node.names
                if alias.name == "TYPE_CHECKING"
                and alias.asname in (None, "TYPE_CHECKING")
            ]
        else:
            continue
        for name, rest in spellings:
            if binders[name] == 1:
                flags[name + rest] = node.lineno
    return flags


def _is_dead(node: ast.If, owner: ast.AST, flags: dict[str, int]) -> bool:
    """True for an `if TYPE_CHECKING:` statement whose body has no effect at all.

    The compiler reads a dead block for the scope and the kind of the code around it, so a
    block in a function or a class can make a name local or a function a generator. Only a
    block at the module level is dead in that sense.

    Parameters
    ----------
    node : ast.If
        The statement.
    owner : ast.AST
        The module, class or function that holds it.
    flags : dict
        From _flag_spellings.

    Returns
    -------
    bool
        True when the statement's body never runs and the compiler takes nothing from it.

    """
    return (
        isinstance(owner, ast.Module)
        and flags.get(_dotted(node.test), node.lineno) < node.lineno
    )


def _top_imports(tree: ast.Module, binders: Counter) -> dict[str, tuple[str, int]]:
    """The names a top-level import binds and nothing else does.

    Parameters
    ----------
    tree : ast.Module
        The module.
    binders : collections.Counter
        From binder_names.

    Returns
    -------
    dict
        Each name with its import path and the line of the import.

    """
    found: dict[str, tuple[str, int]] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            rows = [
                (alias.asname or alias.name.split(".")[0], alias.name)
                for alias in node.names
            ]
        elif isinstance(node, ast.ImportFrom) and not node.level:
            rows = [
                (alias.asname or alias.name, f"{node.module}.{alias.name}")
                for alias in node.names
            ]
        else:
            continue
        for name, path in rows:
            if binders[name] == 1:
                found[name] = (path, node.lineno)
    return found


def _live_reads(tree: ast.Module, flags: dict[str, int]) -> set[str]:
    """The names read by code that runs, outside every annotation and every dead block.

    A name in a module's `__all__` counts, because `from module import *` reads it.

    Parameters
    ----------
    tree : ast.Module
        The module, before any stripping.
    flags : dict
        From _flag_spellings.

    Returns
    -------
    set of str
        The names.

    """
    names: set[str] = set()

    def visit(node: ast.AST, owner: ast.AST) -> None:
        if isinstance(node, ast.If) and _is_dead(node, owner, flags):
            for item in node.orelse:
                visit(item, owner)
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            names.add(node.id)
        elif isinstance(node, ast.Assign | ast.AugAssign) and "__all__" in {
            _dotted(target)
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
        }:
            names.update(
                item.value
                for item in ast.walk(node.value)
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            )
        inner = node if isinstance(node, _NEW_SCOPES) else owner
        for field, value in ast.iter_fields(node):
            if field in ("annotation", "returns"):
                continue
            for item in value if isinstance(value, list) else [value]:
                if isinstance(item, ast.AST):
                    visit(item, inner)

    visit(tree, tree)
    return names


def _passed_names(tree: ast.Module) -> set[str]:
    """The names passed to a call other than those that cannot rewrite a class.

    Parameters
    ----------
    tree : ast.Module
        The module.

    Returns
    -------
    set of str
        The names, of which a class name marks a class that `dataclass(C)` can read.

    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _dotted(node.func) not in _PLAIN_CALLS:
            for argument in (*node.args, *(item.value for item in node.keywords)):
                names.update(
                    item.id for item in ast.walk(argument) if isinstance(item, ast.Name)
                )
    return names


def future_annotations(tree: ast.Module) -> bool:
    """True for a module with `from __future__ import annotations`.

    Parameters
    ----------
    tree : ast.Module
        The module, parsed with its hints.

    Returns
    -------
    bool
        True when Python leaves every annotation of the module unevaluated.

    """
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
        for node in tree.body
    )


def hint_sites(tree: ast.Module) -> list[HintSite]:
    """The annotations of a module that can run, with a key two versions share.

    An annotation inside a module-level dead block has no effect, so it is not a site.

    Parameters
    ----------
    tree : ast.Module
        The module, parsed with its hints.

    Returns
    -------
    list of HintSite
        The annotations in file order.

    """
    flags = _flag_spellings(tree, binder_names(tree))
    found: list[HintSite] = []
    seen: Counter = Counter()

    def add(
        path: tuple, place: str, annotation: ast.expr, runs: bool, holder: tuple | None
    ) -> None:
        count = seen[path, place]
        seen[path, place] += 1
        found.append(HintSite((path, place, count), annotation, runs, holder))

    def visit(body: list, owner: ast.AST, path: tuple) -> None:
        for node in body:
            if isinstance(node, ast.If) and _is_dead(node, owner, flags):
                visit(node.orelse, owner, path)
            elif isinstance(node, _FUNCTIONS):
                inner = (*path, node.name)
                for argument in ast.walk(node.args):
                    if isinstance(argument, ast.arg) and argument.annotation:
                        add(inner, argument.arg, argument.annotation, True, None)
                if node.returns:
                    add(inner, "<return>", node.returns, True, None)
                visit(node.body, node, inner)
            elif isinstance(node, ast.ClassDef):
                visit(node.body, node, (*path, node.name))
            elif isinstance(node, ast.AnnAssign):
                holder = path if isinstance(owner, ast.ClassDef) else None
                runs = not isinstance(owner, _FUNCTIONS)
                add(path, ast.dump(node.target), node.annotation, runs, holder)
            else:
                for _, value in ast.iter_fields(node):
                    if not isinstance(value, list) or not value:
                        continue
                    if isinstance(value[0], ast.stmt):
                        visit(value, owner, path)
                    elif isinstance(value[0], ast.ExceptHandler | ast.match_case):
                        for item in value:
                            visit(item.body, owner, path)

    visit(tree.body, tree, ())
    return found


def evaluated_names(tree: ast.Module) -> frozenset[str]:
    """The names the annotations Python evaluates read, empty under the future import.

    Parameters
    ----------
    tree : ast.Module
        The module, parsed with its hints.

    Returns
    -------
    frozenset of str
        The names, which strip_hints takes as `keep`, because such a name must stay bound.

    """
    if future_annotations(tree):
        return frozenset()
    return frozenset(
        node.id
        for site in hint_sites(tree)
        if site.runs
        for node in ast.walk(site.annotation)
        if isinstance(node, ast.Name)
    )


def _hide(
    state: _Strip, node: ast.stmt | ast.expr | ast.alias, *, lead: bool = False
) -> None:
    """Record what a rule stripped, by its span in the source.

    Parameters
    ----------
    state : _Strip
        The module's readings.
    node : ast.stmt, ast.expr or ast.alias
        The stripped node.
    lead : bool
        True when the token before the node goes with it: the colon of an annotation or the
        arrow of a return.

    """
    if state.regions is not None:
        state.regions.append(
            (node.lineno, node.col_offset, node.end_lineno, node.end_col_offset, lead)
        )


def _declared_kind(annotation: ast.expr, line: int, state: _Strip) -> str | None:
    """What an annotation declares in a class that reads its fields.

    A dataclass reads ClassVar, InitVar and KW_ONLY through the module's own bindings, so the
    head of the annotation is resolved through the top-level imports, whatever it is called
    there, and the import must stand above the line. A head spelled like one of the three
    that does not resolve is reported as unresolved, which compares as itself and never as a
    field or as a plain assignment.

    Parameters
    ----------
    annotation : ast.expr
        The annotation, quoted or not.
    line : int
        The line of the statement.
    state : _Strip
        The module's readings.

    Returns
    -------
    str or None
        `ClassVar`, `InitVar` or `KW_ONLY`, `unresolved` with one of those names, or None for
        any other annotation.

    """
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        try:
            annotation = ast.parse(annotation.value.strip(), mode="eval").body
        except (SyntaxError, ValueError):
            return None
    head = annotation.value if isinstance(annotation, ast.Subscript) else annotation
    dotted = _dotted(head)
    first, _, rest = dotted.partition(".")
    path, where = state.imported.get(first, (first, line))
    kind = _DECLARED.get(f"{path}.{rest}" if rest else path)
    if kind is not None and where < line:
        return kind
    last = dotted.rpartition(".")[2]
    return f"unresolved {last}" if last in _DECLARED_NAMES else None


def _reads_fields(owner: ast.ClassDef, state: _Strip) -> bool:
    """True for a class whose decorator, base, metaclass or a call can read its annotations."""
    return bool(
        owner.decorator_list
        or owner.bases
        or owner.keywords
        or owner.name in state.passed
    )


def _mark(node: ast.AnnAssign, label: str) -> ast.stmt:
    """The statement with its annotation replaced by a label no source can write.

    The form stays an annotated statement, because a statement that reads the target differs
    from an annotation that binds or reads nothing.

    Parameters
    ----------
    node : ast.AnnAssign
        The statement.
    label : str
        What the annotation does at run time.

    Returns
    -------
    ast.stmt
        The statement to compare in place of it.

    """
    return ast.copy_location(
        ast.AnnAssign(
            target=node.target,
            annotation=ast.Name(id=f"<{label}>", ctx=ast.Load()),
            value=node.value,
            simple=node.simple,
        ),
        node,
    )


def _class_field(node: ast.AnnAssign, owner: ast.ClassDef, state: _Strip) -> ast.stmt:
    """What a name annotated in a class body compares as.

    An annotation decides whether a dataclass, a NamedTuple or a Protocol has a field, so the
    rule keeps that decision and drops the type. A bare annotation keeps a label of its own,
    because `x: T` binds and reads nothing and a bare `x` reads x. An assignment in a class
    that can read its annotations keeps a label, because `x: T = v` there is a field and
    `x = v` is not, and a `ClassVar` reads as the plain assignment it declares. ClassVar,
    InitVar and KW_ONLY keep their names, because each changes what a dataclass makes of the
    line.

    Parameters
    ----------
    node : ast.AnnAssign
        The statement, whose target is a plain name.
    owner : ast.ClassDef
        The class.
    state : _Strip
        The module's readings.

    Returns
    -------
    ast.stmt
        The statement to compare in place of it.

    """
    kind = _declared_kind(node.annotation, node.lineno, state)
    if node.value is None:
        return _mark(node, kind or "bare")
    if kind == "ClassVar" or (kind is None and not _reads_fields(owner, state)):
        return _plain_assignment(node, node.value)
    return _mark(node, kind or "field")


def _plain_assignment(node: ast.AnnAssign, value: ast.expr) -> ast.stmt:
    """The assignment `x = v` that `x: T = v` compares as."""
    return ast.copy_location(ast.Assign(targets=[node.target], value=value), node)


def _strip_annotated(
    node: ast.AnnAssign, scope: _Scope, state: _Strip
) -> ast.stmt | None:
    """What an annotated statement compares as, or None when it drops.

    A bare annotation drops only where it is inert. A bare `x: T` at the module level binds
    nothing. A bare `x: T` in a function makes x local, so it drops only when the function
    binds x anyway. A bare annotation of an attribute evaluates its object, so it drops only
    when the object is a parameter of the function, which is always bound, as `self.x: T` is.
    A parenthesized `(x): T`, a subscript and every other bare form keep a label.

    Parameters
    ----------
    node : ast.AnnAssign
        The statement.
    scope : _Scope
        The statement list that holds it.
    state : _Strip
        The module's readings.

    Returns
    -------
    ast.stmt or None
        The statement to compare in place of it.

    """
    target = node.target
    if isinstance(target, ast.Name) and node.simple:
        if isinstance(scope.owner, ast.ClassDef):
            return _class_field(node, scope.owner, state)
        if node.value is not None:
            return _plain_assignment(node, node.value)
        function = isinstance(scope.owner, _FUNCTIONS)
        if not function or target.id in scope.bound:
            return None
        return _mark(node, "local")
    if node.value is not None:
        return _plain_assignment(node, node.value)
    if (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id in scope.parameters
    ):
        return None
    return _mark(node, "target")


def _inert_alias(
    node: ast.Import | ast.ImportFrom, alias: ast.alias, state: _Strip
) -> bool:
    """True for a typing name no run can notice.

    The import must be of `typing` itself, whose import cannot fail, and of a name Python 3.10
    holds. Nothing that runs may read the name and nothing else may bind it, because the name
    otherwise does work, shadows a name or fails. A star never qualifies, and neither does
    typing_extensions, which is a package a run may lack or a module the repository may hold.

    Parameters
    ----------
    node : ast.Import or ast.ImportFrom
        The statement.
    alias : ast.alias
        One of its names.
    state : _Strip
        The module's readings.

    Returns
    -------
    bool
        True when the name leaves the comparison.

    """
    if isinstance(node, ast.Import):
        known = alias.name == "typing"
    else:
        known = (
            node.module == "typing" and not node.level and alias.name in _TYPING_NAMES
        )
    local = alias.asname or alias.name
    return known and local not in state.live and state.binders[local] == 1


def _strip_import(
    node: ast.Import | ast.ImportFrom, scope: _Scope, state: _Strip
) -> ast.stmt | None:
    """An import without its inert typing names and its future annotations, or None if empty.

    A class body keeps its imports, because each is an attribute of the class. The typing
    names that stay are sorted, because their order in the statement does nothing.

    Parameters
    ----------
    node : ast.Import or ast.ImportFrom
        The statement.
    scope : _Scope
        The statement list that holds it.
    state : _Strip
        The module's readings.

    Returns
    -------
    ast.stmt or None
        The statement to compare in place of it.

    """
    if isinstance(scope.owner, ast.ClassDef):
        return node
    future = isinstance(node, ast.ImportFrom) and node.module == "__future__"
    gone = [
        alias
        for alias in node.names
        if (alias.name == "annotations" if future else _inert_alias(node, alias, state))
    ]
    node.names = [alias for alias in node.names if alias not in gone]
    if not node.names:
        return None
    for alias in gone:
        _hide(state, alias)
    if isinstance(node, ast.ImportFrom) and node.module == "typing":
        node.names.sort(key=lambda alias: (alias.name, alias.asname or ""))
    return node


def _bound_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[set, set]:
    """The names a function's own scope binds, and its parameters it never deletes.

    A nested function, class or comprehension has a scope of its own, so a name stored there
    does not count, and a bare annotation does not count as a binding.

    Parameters
    ----------
    node : ast.FunctionDef or ast.AsyncFunctionDef
        The function.

    Returns
    -------
    tuple of set
        The bound names with the parameters among them, and the parameters that stay bound.

    """
    arguments = node.args
    named = {
        item.arg
        for item in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
            *([arguments.vararg] if arguments.vararg else []),
            *([arguments.kwarg] if arguments.kwarg else []),
        )
    }
    stack: list[ast.AST] = list(node.body)
    nodes: list[ast.AST] = []
    while stack:
        item = stack.pop()
        nodes.append(item)
        if not isinstance(item, _NEW_SCOPES):
            stack.extend(ast.iter_child_nodes(item))
    bare = {
        id(item.target)
        for item in nodes
        if isinstance(item, ast.AnnAssign) and item.value is None
    }
    stored = {
        name for item in nodes if id(item) not in bare for name in _names_bound_by(item)
    }
    deleted = {
        item.id
        for item in nodes
        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Del)
    }
    return stored | named, named - deleted


def _strip_nested(node: ast.stmt, scope: _Scope, state: _Strip) -> None:
    """Strip what one kept statement holds: its signature and its statement lists.

    Parameters
    ----------
    node : ast.stmt
        The statement.
    scope : _Scope
        The statement list that holds it.
    state : _Strip
        The module's readings.

    """
    inner = scope
    if isinstance(node, _FUNCTIONS):
        inner = _Scope(node, *_bound_names(node))
        if node.returns:
            _hide(state, node.returns, lead=True)
        node.returns = None
        for argument in ast.walk(node.args):
            if isinstance(argument, ast.arg) and argument.annotation:
                _hide(state, argument.annotation, lead=True)
                argument.annotation = None
    elif isinstance(node, ast.ClassDef):
        inner = _Scope(node, set(), set())
    for field, value in ast.iter_fields(node):
        if not isinstance(value, list) or not value:
            continue
        if isinstance(value[0], ast.stmt):
            setattr(node, field, _strip_body(value, inner, state))
        elif isinstance(value[0], ast.ExceptHandler | ast.match_case):
            for item in value:
                item.body = _strip_body(item.body, inner, state)


def _strip_body(body: list, scope: _Scope, state: _Strip) -> list:
    """A statement list with its hints removed, recursing into nested statements.

    A statement dropped from the head of a list must not promote a string to a docstring, so
    a string left in first place behind a dropped statement keeps a placeholder before it.

    Parameters
    ----------
    body : list of ast.stmt
        The statements.
    scope : _Scope
        The statement list being read.
    state : _Strip
        The module's readings.

    Returns
    -------
    list of ast.stmt
        The statements that remain.

    """
    kept: list = []
    for node in body:
        if isinstance(node, ast.If) and _is_dead(node, scope.owner, state.flags):
            if state.regions is not None:
                last = node.orelse[0] if node.orelse else None
                end = (
                    (last.lineno, last.col_offset)
                    if last
                    else (node.end_lineno, node.end_col_offset)
                )
                state.regions.append((node.lineno, node.col_offset, *end, False))
            kept.extend(_strip_body(node.orelse, scope, state))
            continue
        stripped = node
        if isinstance(node, ast.AnnAssign):
            stripped = _strip_annotated(node, scope, state)
        elif isinstance(node, ast.Import | ast.ImportFrom):
            stripped = _strip_import(node, scope, state)
        if stripped is None:
            _hide(state, node)
            continue
        if isinstance(node, ast.AnnAssign):
            _hide(state, node.annotation, lead=True)
        kept.append(stripped)
        _strip_nested(stripped, scope, state)
    if (
        kept
        and kept[0] is not body[0]
        and isinstance(kept[0], ast.Expr)
        and isinstance(kept[0].value, ast.Constant)
        and isinstance(kept[0].value.value, str)
    ):
        kept.insert(0, ast.copy_location(ast.Pass(), kept[0]))
    return kept


def strip_hints(
    tree: ast.Module,
    *,
    keep: frozenset[str] = frozenset(),
    regions: list | None = None,
) -> ast.Module:
    """Remove from a parsed module what a type hint adds and nothing a program runs.

    The rule removes only what it shows to be inert. Every parameter and return annotation
    goes, `x: T = v` reads as `x = v`, and `from __future__ import annotations` goes. A bare
    annotation goes where it binds and evaluates nothing (_strip_annotated), and a class body
    keeps what an annotation declares (_class_field). A typing name goes when no run can notice
    it (_inert_alias). A module-level `if TYPE_CHECKING:` block goes when the flag is the typing
    constant (_flag_spellings), and its `else` branch stays because it is what runs. A block in
    a function or a class stays, because the compiler reads it (_is_dead). Everything else
    stays, a cast(), an assert or a default included.

    Parameters
    ----------
    tree : ast.Module
        The parsed module, which the call changes.
    keep : frozenset of str
        Names that count as read at run time, for an annotation Python evaluates.
    regions : list, optional
        A list that receives the span of everything stripped, as the start row and column, the
        end row and column, and whether the token before the span goes with it. The columns
        are UTF-8 byte offsets, as `ast` writes them.

    Returns
    -------
    ast.Module
        The same tree.

    """
    binders = binder_names(tree)
    flags = _flag_spellings(tree, binders)
    state = _Strip(
        flags,
        _top_imports(tree, binders),
        binders,
        _live_reads(tree, flags) | keep,
        _passed_names(tree),
        regions,
    )
    tree.body = _strip_body(tree.body, _Scope(tree, set(), set()), state)
    return tree


def fingerprint_bytes(root: str, data: bytes, *, runtime: bool = False) -> str | None:
    """The code fingerprint of a file's bytes, memoized by content.

    A fingerprint is a pure function of the content, the interpreter release and
    FINGERPRINT_RULE, so one file per content under tools/.cache/fingerprints holds it across
    runs. One file per entry means a run that is killed keeps what it finished and two processes
    never contend for one file. An entry of the wrong length is a miss, so a write cut short
    cannot answer. A failed cache read or write only costs the parse.

    Parameters
    ----------
    root : str
        The repository root, which locates the cache.
    data : bytes
        The file's bytes, read under its byte order mark or coding cookie, as Python reads them,
        so that a formatter moving a cookie into the first two lines changes the fingerprint.
    runtime : bool
        True for runtime_fingerprint in place of code_fingerprint.

    Returns
    -------
    str or None
        The fingerprint, or None for a file that does not decode or parse.

    """
    release = "{}.{}".format(*sys.version_info[:2])
    form = f"{FINGERPRINT_RULE}-r{RUNTIME_RULE}" if runtime else FINGERPRINT_RULE
    key = f"{release}-{form}-{hashlib.sha256(data).hexdigest()}"
    if key in _FINGERPRINTS:
        return _FINGERPRINTS[key]
    path = cache_file(root, "fingerprints") / key
    try:
        stored = path.read_text(encoding="ascii").strip()
    except (OSError, ValueError):
        stored = ""
    if stored == "none":
        value = None
    elif len(stored) == 64:
        value = stored
    else:
        decoded = decode_source(data)
        hashed = runtime_fingerprint if runtime else code_fingerprint
        value = None if decoded is None else hashed(decoded)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value or "none", encoding="ascii")
        except OSError:
            pass
    _FINGERPRINTS[key] = value
    return value


def git_blobs(root: str, rev: str, paths: list) -> list:
    """The content of each path at a revision, read in one `git cat-file --batch`.

    Parameters
    ----------
    root : str
        The repository root.
    rev : str
        The revision.
    paths : list of str
        Repository-relative paths with forward slashes.

    Returns
    -------
    list of bytes or None
        One item per path, None for a path the revision does not hold as a blob.

    """
    if not paths:
        return []
    query = "".join(f"{rev}:{p}\n" for p in paths).encode("utf-8")
    r = subprocess.run(
        ["git", "-C", root, "cat-file", "--batch"],
        input=query,
        capture_output=True,
        timeout=60,
    )
    out = r.stdout
    if r.returncode != 0:
        return [None] * len(paths)
    blobs, at = [], 0  # type: ignore[var-annotated]  # element type of blobs is bytes or None
    for _ in paths:
        end = out.index(b"\n", at)
        words = out[at:end].split()
        if len(words) < 3 or not words[-1].isdigit():  # "<query> missing"
            blobs.append(None)
            at = end + 1
            continue
        size = int(words[-1])
        blobs.append(out[end + 1 : end + 1 + size] if words[-2] == b"blob" else None)
        at = end + 1 + size + 1
    return blobs


def same_code_between(
    root: str, old: str, new: str, paths: list, *, runtime: bool = False
) -> set:
    """The Python files whose code is equal at two revisions.

    Parameters
    ----------
    root : str
        The repository root.
    old, new : str
        The two revisions.
    paths : list of str
        Repository-relative paths. A path that is not `.py`, that either revision does not hold,
        or whose fingerprint at either side is None is never in the result.
    runtime : bool
        True to compare the runtime form of the fingerprint.

    Returns
    -------
    set of str
        The paths with equal code fingerprints.

    """
    candidates = [p for p in paths if p.endswith(".py")]
    same = set()
    rows = zip(
        candidates,
        git_blobs(root, old, candidates),
        git_blobs(root, new, candidates),
        strict=True,
    )
    for rel, before, after in rows:
        if before is None or after is None:
            continue
        then = fingerprint_bytes(root, before, runtime=runtime)
        if then is not None and then == fingerprint_bytes(root, after, runtime=runtime):
            same.add(rel)
    return same


def same_code(root: str, rev: str, paths: list, *, runtime: bool = False) -> set:
    """The Python files whose working-tree code equals their code at a revision.

    Parameters
    ----------
    root : str
        The repository root.
    rev : str
        The revision to compare against.
    paths : list of str
        Repository-relative paths. A path that is not `.py`, that the revision does not hold, or
        that is absent from the working tree is never in the result, and neither is a file
        whose fingerprint at either side is None.
    runtime : bool
        True to compare the runtime form of the fingerprint.

    Returns
    -------
    set of str
        The paths with equal code fingerprints.

    """
    candidates = [p for p in paths if p.endswith(".py")]
    same = set()
    for rel, blob in zip(candidates, git_blobs(root, rev, candidates), strict=True):
        if blob is None:
            continue
        try:
            work = (Path(root) / rel).read_bytes()
        except OSError:
            continue
        then = fingerprint_bytes(root, blob, runtime=runtime)
        if then is not None and then == fingerprint_bytes(root, work, runtime=runtime):
            same.add(rel)
    return same
