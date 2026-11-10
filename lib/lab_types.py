# SPDX-FileCopyrightText: Market Design Lab contributors
#
# SPDX-License-Identifier: MIT

"""The repository's type aliases, so that one object carries one name in every hint.

A module imports these names only under ``if TYPE_CHECKING:``, with
``from __future__ import annotations`` at its head, so that no notebook or Colab run
gains a runtime import.

An alias names a shape the seams_engine and revenue_allocation functions pass between
them: a label, or a dict keyed by labels, as ``EngineResult.lmp`` and
``MarketEngine.loads`` are. A worker chooses as follows.

1. A dict of those keys and values takes the alias, and the docstring states the unit,
   because one alias holds an LMP in $/MWh and a generation in MW.
2. A ``pd.DataFrame`` or ``pd.Series`` is written as itself, because a table's columns
   are not a key type. An argument that a function accepts in either form is the union,
   ``Loads | pd.Series``.
3. A float array is ``FloatArray``, and the docstring states its axes, which follow
   ``PTDFData.buses`` and ``PTDFData.lines``. An integer or boolean array is written as
   ``npt.NDArray[np.int_]`` or ``npt.NDArray[np.bool_]``.
4. A dict with other keys is written out, and an alias is added here when a second
   module needs the same shape.
"""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

#: A bus label, n, a string on every case: "1" on the 9-bus case, "st_1694_230" on WECC.
Bus: TypeAlias = str
#: A line name, m. A flowgate id is a name of the same kind.
Line: TypeAlias = str
#: A BA area name, a.
Area: TypeAlias = str
#: A resource id, i, the key of a fleet and of ``EngineResult.dispatch``.
Resource: TypeAlias = str
#: An ordered pair of areas, the key of a declared interface.
Interface: TypeAlias = tuple[Area, Area]

#: One float per bus, such as an LMP or ``EngineResult.gen_by_bus``.
BusValues: TypeAlias = dict[Bus, float]
#: One float per line, such as a flow or a signed line dual.
LineValues: TypeAlias = dict[Line, float]
#: One float per area, such as a net position.
AreaValues: TypeAlias = dict[Area, float]
#: Fixed load in MW by bus.
Loads: TypeAlias = BusValues
#: A unit's entry in a fleet, with the keys of ``seams_engine.FLEET_REQUIRED`` and
#: ``FLEET_OPTIONAL``.
Unit: TypeAlias = dict[str, Any]
#: A generator fleet by resource id.
Fleet: TypeAlias = dict[Resource, Unit]
#: The buses of each area, as ``Footprints.areas`` holds them.
AreaBuses: TypeAlias = dict[Area, list[Bus]]
#: Released transfer capacity in MW by interface, None for unlimited.
Interfaces: TypeAlias = dict[Interface, float | None]
#: A float64 array.
FloatArray: TypeAlias = npt.NDArray[np.float64]
