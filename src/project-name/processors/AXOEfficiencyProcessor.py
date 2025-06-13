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
        met = events.ScoutingMET

        # passing zero-bias trigger (orthogonal reference)
        zero_bias_triggered = events.L1.ZeroBias
        L1_axo_nominal = events.L1.AXO_Nominal
        L1_htt360er = events.L1.HTT360er
        L1_etmhf70 = events.L1.ETMHF70
        axo_or_htt = L1_axo_nominal | L1_htt360er
        axo_or_etm = L1_axo_nominal | L1_etmhf70

        # jet selection
        jet_selector = (jets.pt > jet_selection["pt"]) * (
            np.abs(jets.eta) < jet_selection["eta"]
        )
        jets = ak.pad_none(jets[jet_selector], 1, axis=1)
        jet_selector = ak.any(jet_selector, axis=1)

        add_selection("jet", jet_selector, *selection_args)
        select = selection.all(*selection.names)

        # Events that pass both L1_AXO_Nominal and L1_HTT360er
        add_selection("axo_and_htt360", axo_or_htt, *selection_args)

        # Events that pass both L1_AXO_Nominal and L1_HTT360er
        add_selection("axo_and_etmhf70", axo_or_etm, *selection_args)

        hists = {
            "ht": {},
            "met": {},
            "jetpt": {},
            "ht_all": None,
            "jetpt_all": None
        }

        # Define and fill HT and JetPT histograms for all events (no triggers)
        h_ht_all = (
            hda.Hist.new.Reg(100, 0, 2000, name="ht", label=r"$H_T$ [GeV]")
            .Double()
        )
        h_pt_all = (
            hda.Hist.new.Reg(100, 0, 2000, name="pt", label=r"$p_T$ [GeV]")
            .Double()
        )
        h_met_all = (
            hda.Hist.new.Reg(100, 0, 2000, name="met", label=r"$E^{\rm miss}_T$ [GeV]")
            .Double()
        )

        hists["ht_all"] = h_ht_all.fill(
            ht=ak.sum(jets.pt[select], axis=1)
        )

        hists["pt_all"] = h_pt_all.fill(
            pt=ak.flatten(jets.pt[select], axis=1)
        )

        hists["met_all"] = h_met_all.fill(
            met=met.pt[select]
        )

        # Histograms for ZeroBias events (den in efficiency calculations)

        # select events which pass the zero bias triggers and object selection
        den_selections = select * zero_bias_triggered

        # initialize histograms
        h_ht = (
            hda.Hist.new.Reg(100, 0, 2000, name="ht", label=r"$H_T$ [GeV]")
            .StrCat([], name="trigger", label="Trigger", growth=True)
            .Double()
        )

        h_met = (
            hda.Hist.new.Reg(100, 0, 2000, name="met", label=r"$E^{\rm miss}_T$ [GeV]")
            .StrCat([], name="trigger", label="Trigger", growth=True)
            .Double()
        )

        h_pt = (
            hda.Hist.new.Reg(100, 0, 2000, name="pt", label=r"$p_T$ [GeV]")
            .StrCat([], name="trigger", label="Trigger", growth=True)
            .Double()
        )

        hists["ht"]["den"] = h_ht.copy().fill(
            ht=ak.sum(jets.pt[den_selections], axis=1), trigger="ZeroBias"
        )

        hists["met"]["den"] = h_met.copy().fill(
            met=met.pt[den_selections], trigger="ZeroBias"
        )

        hists["jetpt"]["den"] = h_pt.copy().fill(
            pt=ak.flatten(jets.pt[den_selections], axis=1), trigger="ZeroBias"
        )

        # Histograms for L1 seed events (num in efficiency calculations)

        for L1_seed in L1s:
            if L1_seed not in events.L1.fields:
                raise KeyError(f"L1 seed {L1_seed} not found in the L1 menu.\n")
            # passing our triggers
            L1_seed_triggered = events.L1[L1_seed]
            # add our triggers
            num_selections = select * zero_bias_triggered * L1_seed_triggered

            hists["ht"]["num"] = h_ht.fill(
                ht=ak.sum(jets.pt[num_selections], axis=1), 
                trigger=L1_seed
            )

            hists["met"]["num"] = h_met.fill(
                met=met.pt[num_selections], 
                trigger=L1_seed
            )

            hists["jetpt"]["num"] = h_pt.fill(
                pt=ak.flatten(jets.pt[num_selections], axis=1), 
                trigger=L1_seed
            )

        # Fill histograms for (AXO & HTT) as well
        num_selections = select * zero_bias_triggered * axo_or_htt
        hists["ht"]["num"] = h_ht.fill(
            ht=ak.sum(jets.pt[num_selections], axis=1), 
            trigger="AXO_Nominal_HTT360er"
        )
        hists["met"]["num"] = h_met.fill(
            met=met.pt[num_selections], 
            trigger="AXO_Nominal_HTT360er"
        )
        hists["jetpt"]["num"] = h_pt.fill(
            pt=ak.flatten(jets.pt[num_selections], axis=1), 
            trigger="AXO_Nominal_HTT360er"
        )

        # Fill histograms for (AXO & ETM) as well
        num_selections = select * zero_bias_triggered * axo_or_etm
        hists["met"]["num"] = h_met.fill(
            met=met.pt[num_selections], 
            trigger="AXO_Nominal_ETMHF70"
        )

        # Add histograms for top N-leading jets for Quad Jet trigger only
        top_n = 4
        hists["jetpt_topN"] = {}
        h_pt_topn = (
            hda.Hist.new.Reg(100, 0, 1000, name="pt", label=r"Jet $p_T$ [GeV]")
            .StrCat([], name="rank", label="Jet Rank", growth=True)
            .Double()
        )

        # Sort jets by descending pt
        quad_jet_select = events.L1.HTT320er_QuadJet_80_60_er2p1_45_40_er2p3
        selections = select * zero_bias_triggered * quad_jet_select
        sorted_jets = jets[selections]
        sorted_jets = sorted_jets[ak.argsort(sorted_jets.pt, axis=1, ascending=False)]

        for i in range(top_n):
            # mask to keep events with at least (i+1) jets
            has_jet = ak.num(sorted_jets.pt) > i
            jet_pt_i = ak.flatten(sorted_jets.pt[has_jet][:, i:(i + 1)])
            hists["jetpt_topN"][f"jet{i+1}"] = h_pt_topn.copy().fill(
                pt=jet_pt_i,
                rank=f"jet{i+1}"
            )

        hists["cutflow"] = cutflow
        return hists

    def postprocess(self, accumulator):
        return accumulator
