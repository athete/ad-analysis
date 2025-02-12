from typing import List, Dict
import dask_awkward as dak
import hist
import hist.dask as hda

import coffea.processor as processor
from coffea.analysis_tools import PackedSelection


class AXOHistFactory(processor.ProcessorABC):
    def __init__(self, triggers: List[str], hists_to_process: List[str]) -> None:
        self.object_cuts = None if True else {
            "ScoutingPFJet": [("pt", 30.0), ("eta", 3.0)],
            "ScoutingElectron": [("pt", 10.0), ("eta", 2.65)],
            "ScoutingPhoton": [("pt", 10.0), ("eta", 2.65)],
            "ScoutingMuonVtx": [("pt", 3.0), ("eta", 2.4)],
        }

        self.triggers = triggers
        self.hists_to_process = hists_to_process

        self.trigger_axis = hist.axis.StrCategory(
            [], name="trigger", label="Trigger", growth=True
        )
        self.object_axis = hist.axis.StrCategory(
            [], name="object", label="Object", growth=True
        )
        self.pt_axis = hist.axis.Regular(500, 0, 5000, name="pt", label=r"$p_T$ [GeV]")
        self.ht_axis = hist.axis.Regular(70, 0, 2000, name="ht", label=r"$H_T$ [GeV]")
        self.met_axis = hist.axis.Regular(
            250, 0, 2500, name="met", label=r"$p^{\rm miss}_T$ [GeV]"
        )
        self.eta_axis = hist.axis.Regular(150, -5, 5, name="eta", label=r"$\eta$")
        self.phi_axis = hist.axis.Regular(30, -4, 4, name="phi", label=r"$\phi$")

    def process(self, events) -> Dict[str, hda.Hist]:
        # Trigger saturation cuts
        selection = PackedSelection()
        selection.add_multiple(
            {
                "saturatedL1Jet": dak.all(events.L1Jet.pt < 1000, axis=1),
                "saturatedL1MET": dak.all(
                        events.L1EtSum.pt[
                            (events.L1EtSum.etSumType == 2) & (events.L1EtSum.bx == 0)
                        ]
                    < 1040,
                    axis=1,
                ),
            }
        )
        saturation_cut = {"saturatedL1Jet", "saturatedL1MET"}
        selection.add("saturationCut", selection.all(*saturation_cut))
        events = events[selection.all("saturationCut")]

        hists = {}

        # Create histograms
        if "ScoutingHT" in self.hists_to_process:
            hists["ScoutingHT"] = hda.Hist(self.trigger_axis, self.ht_axis)
        if "ScoutingMET" in self.hists_to_process:
            hists["ScoutingMET"] = hda.Hist(self.trigger_axis, self.met_axis)
        if "pt" in self.hists_to_process:
            hists["pt"] = hda.Hist(
                self.trigger_axis, self.pt_axis, self.object_axis
            )
        if "eta" in self.hists_to_process:
            hists["eta"] = hda.Hist(
                self.trigger_axis, self.eta_axis, self.object_axis
            )
        if "phi" in self.hists_to_process:
            hists["phi"] = hda.Hist(
                self.trigger_axis, self.phi_axis, self.object_axis
            )

        # Apply triggers
        for trigger_path in self.triggers:
            trigger_branch = getattr(events, trigger_path.split("_")[0])
            trig_path = "_".join(trigger_path.split("_")[1:])
            events_triggered = events[getattr(trigger_branch, trig_path)]

            if "ScoutingHT" in self.hists_to_process:
                jets = events_triggered.ScoutingJet
                scouting_ht = dak.sum(jets[(jets.pt > 30) & (jets.eta < 3.0)].pt, axis=1)
                hists["ScoutingHT"].fill(ht=scouting_ht, trigger=trigger_path.split("_")[-1])
            if "ScoutingMET" in self.hists_to_process:
                scouting_met = events_triggered.ScoutingMET.pt
                hists["ScoutingMET"].fill(met=scouting_met, trigger=trigger_path.split("_")[-1])

            if self.object_cuts is None:
                continue
            
            for obj, obj_cutlist in self.object_cuts.items():
                obj_branch = getattr(events_triggered, obj)
                for var, cut in obj_cutlist:
                    mask = (getattr(obj_branch, var) > cut)
                    obj_branch = obj_branch[mask] 
                
                if "pt" in self.hists_to_process:
                    hists["pt"].fill(pt=dak.flatten(obj_branch.pt), object=obj, trigger=trigger_path.split("_")[-1])
                if "eta" in self.hists_to_process:
                    hists["eta"].fill(eta=dak.flatten(obj_branch.eta), object=obj, trigger=trigger_path.split("_")[-1])
                if "phi" in self.hists_to_process:
                    hists["phi"].fill(pt=dak.flatten(obj_branch.phi), object=obj, trigger=trigger_path.split("_")[-1])

        return hists
    
    def postprocess(self, accumulator):
        return accumulator