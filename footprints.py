# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Partition the shared 9-bus network into market footprints (BAs / markets).

A *footprint* is a named subset of buses that one operator runs. The two issue
families use the same structure with different labels:

    congestion-revenue notebooks → two **balancing authorities** ("BA-1", "BA-2"),
        whose cross-footprint lines are **ties**;
    seams notebook                → two **market footprints** ("Market A/B"),
        whose cross-footprint lines are **seams**.

``Footprints`` bundles everything the settlement / allocation / figure helpers
need to know about a partition, derived once from the bus definitions and the
shift-factor topology:

* ``defs``        — ``{name: [bus, …]}`` the bus membership (the visible knob).
* ``line_assign`` — ``{line: name|None}`` which footprint *manages* each line
  (rent assignment / line colour); a line under nobody is unassigned (grey).
* ``monitored``   — ``{name: [line, …]}`` each footprint's activated constraint
  set ℳ^M_act (optional; the seams notebook sets it explicitly).
* ``areas``       — ``defs`` plus a single ``"Non-market"`` area if any bus sits
  outside every footprint (settles at LMP but is never allocated rent).
* ``ties``        — the cross-footprint lines (topological).

Two derivation modes feed ``line_assign``:

* pass ``manage={name: [lines]}``  (the CRA ``BA_LINES`` style — explicit
  management), or
* pass ``monitored={name: [lines]}`` (the seams ``MONITORED_LINES`` style); a
  line monitored by exactly one footprint is coloured for it, by neither/both is
  grey.

The bus membership stays a **visible** notebook knob; this module only derives
the bookkeeping from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Grey for a line assigned to no footprint (or to both).
UNASSIGNED_COLOR = "#AAB7B8"


@dataclass
class Footprints:
    """A two-footprint partition of the shared network (see module docstring)."""

    names: tuple
    defs: dict                      # {name: [bus strings]}
    colors: dict                    # {name: hex}
    line_assign: dict               # {line: name|None}  — management / colour
    monitored: dict                 # {name: [line]}     — activated set ℳ^M_act
    areas: dict                     # defs (+ 'Non-market' if any bus outside)
    ties: list                      # cross-footprint lines
    tie_label: str                  # 'tie' (BA) or 'seam' (market)
    bus_to_fp: dict = field(default_factory=dict)

    # ── bus → footprint ──────────────────────────────────────────────────
    def fp_of(self, bus):
        """Footprint owning ``bus`` (``None`` if it sits outside every footprint)."""
        return self.bus_to_fp.get(str(bus))

    # ── line classification ──────────────────────────────────────────────
    def line_kind(self, pt, l):
        """``('internal', name)`` if both ends share a footprint; ``(tie_label,
        None)`` if ends sit in two different footprints; ``('boundary', None)``
        if one end is outside every footprint (a non-market bus)."""
        i = pt.line_idx[l]
        b0, b1 = pt.line_buses[i]
        a0, a1 = self.fp_of(b0), self.fp_of(b1)
        if a0 is not None and a0 == a1:
            return ("internal", a0)
        if a0 is not None and a1 is not None:
            return (self.tie_label, None)
        return ("boundary", None)

    def monitored_by(self, l):
        """Footprints whose activated set ℳ^M_act contains line ``l``."""
        return [m for m in self.names if l in self.monitored.get(m, [])]

    # ── colours / banding ────────────────────────────────────────────────
    def line_colors(self, pt):
        """``{line: hex}`` — each managed line takes its footprint's colour;
        a line assigned to nobody (or, in the monitored basis, to both) is grey."""
        return {
            l: (self.colors[self.line_assign[l]] if self.line_assign.get(l) else UNASSIGNED_COLOR)
            for l in pt.lines
        }

    def groups(self, pt):
        """``{bus: name}`` for circlize banding (buses outside every footprint
        are left unbanded)."""
        return {b: self.fp_of(b) for b in pt.buses if self.fp_of(b)}


def make(
    pt,
    defs: dict,
    colors: dict,
    *,
    manage: dict | None = None,
    monitored: dict | None = None,
    tie_label: str = "tie",
) -> Footprints:
    """Build a :class:`Footprints` from bus definitions + the network topology.

    Parameters
    ----------
    pt : PTDFData
        Shift factors of the network being partitioned (gives the line set/topology).
    defs : dict
        ``{name: [buses]}`` — the (visible) bus membership of each footprint.
    colors : dict
        ``{name: hex}`` — footprint colour.
    manage : dict, optional
        ``{name: [lines]}`` explicit line management (CRA ``BA_LINES``). Each line
        may appear under at most one footprint; a line under nobody is unassigned.
    monitored : dict, optional
        ``{name: [lines]}`` activated constraint sets ℳ^M_act (seams
        ``MONITORED_LINES``). When ``manage`` is omitted, ``line_assign`` (the
        colour basis) is derived from this: a line monitored by exactly one
        footprint is coloured for it; by neither/both, grey.
    tie_label : str
        Display label for a cross-footprint line: ``"tie"`` or ``"seam"``.
    """
    names = tuple(defs)
    bus_to_fp = {str(b): name for name, buses in defs.items() for b in buses}
    monitored = {m: list(monitored.get(m, [])) for m in names} if monitored else {m: [] for m in names}

    # validate monitored entries
    for m, ls in monitored.items():
        for l in ls:
            assert l in pt.lines, f"unknown line in monitored set for {m}: {l}"

    # line_assign: explicit management, else single-monitor, else None
    if manage is not None:
        owner: dict = {}
        for name, ls in manage.items():
            for l in ls:
                assert l in pt.lines, f"unknown line in manage[{name}]: {l}"
                assert l not in owner, f"{l} listed under both {owner[l]} and {name}"
                owner[l] = name
        line_assign = {l: owner.get(l) for l in pt.lines}
    else:
        def _single(l):
            ms = [m for m in names if l in monitored.get(m, [])]
            return ms[0] if len(ms) == 1 else None
        line_assign = {l: _single(l) for l in pt.lines}

    fp = Footprints(
        names=names, defs={k: [str(b) for b in v] for k, v in defs.items()},
        colors=dict(colors), line_assign=line_assign, monitored=monitored,
        areas={}, ties=[], tie_label=tie_label, bus_to_fp=bus_to_fp,
    )

    # ties (topological) + settlement areas (+ a Non-market area if needed)
    fp.ties = [l for l in pt.lines if fp.line_kind(pt, l)[0] == tie_label]
    areas = {name: [str(b) for b in buses] for name, buses in defs.items()}
    nm = [b for b in pt.buses if b not in bus_to_fp]
    if nm:
        areas["Non-market"] = nm
    fp.areas = areas
    return fp


if __name__ == "__main__":
    import wscc9_model as wm

    pt = wm.shift_factors()

    # CRA-style: balancing authorities with explicit line management
    BA_DEFS = {"BA-1": ["2", "8", "7", "6", "3"], "BA-2": ["1", "9", "4", "5"]}
    BA_LINES = {"BA-1": ["line_2", "line_3", "line_4", "line_5", "line_6"],
                "BA-2": ["line_0", "line_1", "line_7", "line_8"]}
    ba = make(pt, BA_DEFS, {"BA-1": "#993AFF", "BA-2": "#2471A3"},
              manage=BA_LINES, tie_label="tie")
    print("BA ties:", ba.ties)
    print("BA-1 manages:", [l for l in pt.lines if ba.line_assign[l] == "BA-1"])
    print("areas:", {k: v for k, v in ba.areas.items()})

    # seams-style: markets with explicit monitored sets
    MKT_DEFS = {"Market A": ["2", "8", "7", "6", "3"], "Market B": ["1", "9", "4", "5"]}
    MON = {"Market A": ["line_2", "line_3", "line_4", "line_5", "line_6"],
           "Market B": ["line_0", "line_1", "line_7", "line_8"]}
    mk = make(pt, MKT_DEFS, {"Market A": "#993AFF", "Market B": "#2471A3"},
              monitored=MON, tie_label="seam")
    print("\nseam lines:", mk.ties)
    print("monitored by neither:", [l for l in pt.lines if not mk.monitored_by(l)] or "none")
    print("Market A colour for line_2:", mk.line_colors(pt)["line_2"])
