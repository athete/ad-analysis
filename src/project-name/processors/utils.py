"""
Common functions for processors.

Author(s): Ameya Thete
"""

from __future__ import annotations

import awkward as ak
import numpy as np
from coffea.analysis_tools import PackedSelection


def add_selection(
    name: str, sel: ak.Array, selection: PackedSelection, cutflow: dict
) -> None:
    """Adds selection to a PackedSelection object and the cutflow dictionary"""
    selection.add(name, sel)
    cutflow[name] = ak.sum(selection.all(*selection.names))
