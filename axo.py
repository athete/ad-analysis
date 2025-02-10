import awkward as ak
import dask
import dask_awkward as dak
import hist
import hist.dask as hda
import json
import numpy as np
import vector

vector.register_awkward()


from coffea.nanoevents import NanoEventsFactory, ScoutingNanoAODSchema
import coffea.processor as processor
from coffea.analysis_tools import PackedSelection
from coffea.dataset_tools import apply_to_fileset, max_chunks, preprocess

from utils.hist import create_1d_hist


class AXOHistFactory(processor.ProcessorABC):
    def __init__(self, triggers, hists_to_process):
        object_cuts = {
            "ScoutingPFJet": [("pt", 30.0), ("eta", 3.0)],
            "ScoutingElectron": [("pt", 10.0), ("eta", 2.65)],
            "ScoutingPhoton": [("pt", 10.0), ("eta", 2.65)],
            "ScoutingMuonVtx": [("pt", 3.0), ("eta", 2.4)],
        }

        self.triggers = triggers
        self.hists_to_process = hists_to_process
        self.hists = {}

        self.trigger_axis = hist.axis.StrCategory(
            [], name="trigger", label="Trigger", growth=True
        )
        self.object_axis = hist.axis.StrCategory(
            [], name="object", label="Object", growth=True
        )
        self.pt_axis = hist.axis.Regular(500, 0, 5000, name="pt", label=r"$p_T$ [GeV]")
        self.ht_axis = hist.axis.Regular(100, 0, 2000, name="ht", label=r"$H_T$ [GeV]")
        self.met_axis = hist.axis.Regular(
            250, 0, 2500, name="met", label=r"$p^{\rm miss}_T$ [GeV]"
        )
        self.eta_axis = hist.axis.Regular(150, -5, 5, name="eta", label=r"$\eta$")
        self.phi_axis = hist.axis.Regular(30, -4, 4, name="phi", label=r"$\phi$")

    def process(self, events):
        # Trigger saturation cuts
        selection = PackedSelection()
        selection.add_multiple(
            {
                "saturatedL1Jet": dak.all(events.L1Jet.pt < 1000, axis=1),
                "saturatedL1MET": dak.all(
                    dak.flatten(
                        events.L1EtSum.pt[
                            (events.L1EtSum.etSumType == 2) & (events.L1EtSum.bx == 0)
                        ]
                    )
                    < 1040,
                    axis=1,
                ),
            }
        )
        saturation_cut = {"saturatedL1Jet", "saturatedL1MET"}
        selection.add("saturationCut", selection.all(**saturation_cut))
        events = events[selection.all("saturationCut")]

        # Create histograms
        if "ScoutingHT" in self.hists_to_process:
            self.hists["ScoutingHT"] = create_1d_hist(self.trigger_axis, self.ht_axis)
        if "ScoutingMET" in self.hists_to_process:
            self.hists["ScoutingMET"] = create_1d_hist(self.trigger_axis, self.met_axis)
        if "pt" in self.hists_to_process:
            self.hists["pt"] = create_1d_hist(
                self.trigger_axis, self.pt_axis, self.object_axis
            )
        if "LeadingPt" in self.hists_to_process:
            self.hists["LeadingPt"] = create_1d_hist(
                self.trigger_axis, self.pt_axis, self.object_axis
            )
        if "SubleadingPt" in self.hists_to_process:
            self.hists["SubleadingPt"] = create_1d_hist(
                self.trigger_axis, self.pt_axis, self.object_axis
            )
        if "eta" in self.hists_to_process:
            self.hists["eta"] = create_1d_hist(
                self.trigger_axis, self.eta_axis, self.object_axis
            )
        if "phi" in self.hists_to_process:
            self.hists["phi"] = create_1d_hist(
                self.trigger_axis, self.phi_axis, self.object_axis
            )

        # Apply triggers
        for trigger_path in self.triggers:
            trigger_branch = getattr(events, trigger_path.split("_")[0])
            trig_path = "_".join(trigger_path.split("_")[1:])
            events_triggered = events[getattr(trigger_branch, trig_path)]

            if "ScoutingHT" in self.hists_to_process:
                scouting_ht = dak.sum(events_triggered.ScoutingPFJet.pt, axis=1)
                self.hists["ScoutingHT"].fill(ht=scouting_ht, trigger=trigger_path)
            if "ScoutingMET" in self.hists_to_process:
                scouting_met = events_triggered.ScoutingMET.pt
                self.hists["ScoutingMET"].fill(met=scouting_met, trigger=trigger_path)
