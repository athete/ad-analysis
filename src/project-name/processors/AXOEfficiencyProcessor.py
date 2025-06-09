"""
Create and fill histograms to calculate L1 seed efficiencies for a range of
physics object variables.

Author(s): Ameya Thete
"""

from __future__ import annotations

from collections import OrderedDict
from enum import Enum

import awkward as ak
import numpy as np
from coffea import processor
from coffea.analysis_tools import PackedSelection
import hist.dask as hda

from .common import L1s
from .utils import add_selection

jet_selection = {"pt": 30, "eta": 2.3}


class AXOEfficiencyProcessor(processor.ProcessorABC):
    """Accumulates a 1D histogram of different physics objects from input NanoAOD events"""

    def __init__(self):
        super().__init__()

    def process(self, events):
        """Returns a pre- (den) and post- (num) trigger 1D histograms from input NanoAOD events"""
        selection = PackedSelection()

        cutflow = OrderedDict()
        cutflow["begin"] = ak.num(events, axis=0)

        selection_args = (selection, cutflow)

        # objects
        jets = events.ScoutingPFJet

        # passing zero-bias trigger (orthogonal reference)
        zero_bias_triggered = events.L1.ZeroBias

        # jet selection
        jet_selector = (jets.pt > jet_selection["pt"]) * (
            np.abs(jets.eta) < jet_selection["eta"]
        )
        jets = ak.pad_none(jets[jet_selector], 1, axis=1)
        jet_selector = ak.any(jet_selector, axis=1)

        add_selection("jet", jet_selector, *selection_args)

        select = selection.all(*selection.names)

        # initialize histograms
        h = (
            hda.Hist.new.Reg(100, 0, 2000, name="ht", label=r"$H_T$ [GeV]")
            .StrCat([], name="trigger", label="Trigger", growth=True)
            .Double()
        )

        hists = {}

        # select events which pass the zero bias triggers and selection
        den_selections = select * zero_bias_triggered

        hists["den"] = h.copy().fill(
            ht=ak.sum(jets.pt[den_selections], axis=1), trigger="ZeroBias"
        )

        for L1_seed in L1s:
            if L1_seed not in events.L1.fields:
                raise KeyError(f"L1 seed {L1_seed} not found in the L1 menu.\n")
            # passing our triggers
            L1_seed_triggered = events.L1[L1_seed]
            # add our triggers
            num_selections = select * zero_bias_triggered * L1_seed_triggered

            hists["num"] = h.fill(
                ht=ak.sum(jets.pt[num_selections], axis=1), trigger=L1_seed
            )

        hists["cutflow"] = cutflow

        return hists

    def postprocess(self, accumulator):
        return accumulator
