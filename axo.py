from typing import List, Dict
import dask_awkward as dak
import hist
import hist.dask as hda

import coffea.processor as processor


def make_1d_hist(hist_dict:Dict[str, hda.hist.Hist], hist_name: str, trigger_axis: hist.axis, observable_axis: hist.axis, object_axis: hist.axis=None) -> Dict[str, hda.hist.Hist]:
    if object_axis == None:
        h = hda.hist.Hist(trigger_axis, observable_axis, storage='weight')
    else:
        h = hda.hist.Hist(trigger_axis, object_axis, observable_axis, storage='weight')
    
    hist_dict[f"{hist_name}"] = h
    return hist_dict

def fill_1d_hist(hist_dict:Dict[str, hda.hist.Hist], hist_name: str, trigger, observable, observable_name: str, object_name: str=None) -> Dict[str, hda.hist.Hist]:
    kwargs = {
        observable_name: observable,
        "trigger": trigger,
    }
    if object_name is not None:
        kwargs["object"] = object_name
    
    hist_dict[f"{hist_name}"].fill(**kwargs)
    return hist_dict


class AXOHistFactory(processor.ProcessorABC):
    def __init__(self, triggers: List[str], hists_to_process: Dict[str, List]) -> None:
        
        self.object_dict =  {
                "ScoutingPFJet": {
                    'cuts': [("pt", 30.0)], 
                    'label': 'j'
                },
                "ScoutingElectron": {
                    'cuts': [("pt", 10.0)], 
                    'label': 'e'
                },
                "ScoutingMuonVtx": {
                    'cuts': [("pt", 3.0)], 
                    'label': '\mu'
                },
                "ScoutingPhoton": {
                    'cuts': [("pt", 10.0)], 
                    'label': '\gamma'
                },
                "L1Jet": {
                    'cuts': [("pt", 0.1)], 
                    'label': 'L1j'
                },
                "L1EG": {
                    'cuts': [("pt", 0.1)], 
                    'label': 'L1e'
                },
                "L1Mu": {
                    'cuts': [("pt", 0.1)], 
                    'label': 'L1\mu'
                }
        }

        self.triggers = triggers
        self.hists_to_process = hists_to_process

        self.trigger_axis = hist.axis.StrCategory(
            [], name="trigger", label="Trigger", growth=True
        )
        self.object_axis = hist.axis.StrCategory(
            [], name="object", label="Object", growth=True
        )
        self.pt_axis = hist.axis.Regular(
            500, 0, 5000, name="pt", label=r"$p_T$ [GeV]"
        )
        self.ht_axis = hist.axis.Regular(
            70, 0, 2000, name="ht", label=r"$H_T$ [GeV]"
        )
        self.met_axis = hist.axis.Regular(
            250, 0, 2500, name="met", label=r"$p^{\rm miss}_T$ [GeV]"
        )
        self.eta_axis = hist.axis.Regular(
            150, -5, 5, name="eta", label=r"$\eta$"
        )
        self.phi_axis = hist.axis.Regular(
            30, -4, 4, name="phi", label=r"$\phi$"
        )
        self.mult_axis = hist.axis.Regular(
            200, 0, 201, name="mult", label=r'$N_{\rm obj}$'
        )

    def process(self, events) -> Dict[str, hda.hist.Hist]:
        dataset = events.metadata['dataset']
        hist_dict = {}
        # Trigger saturation cuts for quality
        events = events[dak.all(events.L1Jet.pt<1000, axis=1)]
        events = events[dak.flatten(events.L1EtSum.pt[(events.L1EtSum.etSumType==2) & (events.L1EtSum.bx==0)])<1040]
        
        
        if ("l1ht" in self.hists_to_process["1d_scalar"]):
            hist_dict = make_1d_hist(
                hist_dict,
                "l1ht",
                self.trigger_axis,
                self.ht_axis
            )
        if ("l1met" in self.hists_to_process["1d_scalar"]):
            hist_dict = make_1d_hist(
                hist_dict,
                "l1met",
                self.trigger_axis,
                self.met_axis
            )
        if ("total_l1mult" in self.hists_to_process["1d_scalar"]):
            hist_dict = make_1d_hist(
                hist_dict, 
                "total_l1mult", 
                self.trigger_axis, 
                self.mult_axis 
            )
        if ("total_l1pt" in self.hists_to_process["1d_scalar"]):
            hist_dict = make_1d_hist()(
                hist_dict, 
                "total_l1pt", 
                self.trigger_axis, 
                self.pt_axis
            )
        if ("scoutinght" in self.hists_to_process["1d_scalar"]):
            hist_dict = make_1d_hist(
                hist_dict,
                "scoutinght",
                self.trigger_axis,
                self.ht_axis
            )
        if ("scoutingmet" in self.hists_to_process["1d_scalar"]):
            hist_dict = make_1d_hist(
                hist_dict,
                "scoutingmet",
                self.trigger_axis,
                self.met_axis
            )
        if ("total_scoutingmult" in self.hists_to_process["1d_scalar"]):
            hist_dict = make_1d_hist(
                hist_dict, 
                "total_scoutingmult", 
                self.trigger_axis, 
                self.mult_axis
            )
        if ("total_scoutingpt" in self.hists_to_process["1d_scalar"]):
            hist_dict = make_1d_hist(
                hist_dict, 
                "total_scoutingpt", 
                self.trigger_axis, 
                self.pt_axis
            )
        if ("n" in self.hists_to_process["1d_object"]):
            hist_dict = make_1d_hist(
                hist_dict, 
                "n_obj",
                self.trigger_axis,
                self.mult_axis,
                object_axis=self.object_axis
            )
        if ("pt" in self.hists_to_process["1d_object"]):
            hist_dict = make_1d_hist(
                hist_dict, 
                "pt_obj",
                self.trigger_axis,
                self.pt_axis,
                object_axis=self.object_axis
            )
        if ("pt0" in self.hists_to_process["1d_object"]):
            hist_dict = make_1d_hist(
                hist_dict, 
                "pt0_obj",
                self.trigger_axis,
                self.pt_axis,
                object_axis=self.object_axis
            )
        if ("pt1" in self.hists_to_process["1d_object"]):
            hist_dict = make_1d_hist(
                hist_dict, 
                "pt1_obj",
                self.trigger_axis,
                self.pt_axis,
                object_axis=self.object_axis
            )
        if ("eta" in self.hists_to_process["1d_object"]):
            hist_dict = make_1d_hist(
                hist_dict, 
                "eta_obj",
                self.trigger_axis,
                self.eta_axis,
                object_axis=self.object_axis
            )
        if ("phi" in self.hists_to_process["1d_object"]):
            hist_dict = make_1d_hist(
                hist_dict, 
                "phi_obj",
                self.trigger_axis,
                self.phi_axis,
                object_axis=self.object_axis
            )

        for trigger_path in self.triggers:
            events_trig = None

            trigger_branch = getattr(events, trigger_path.split("_")[0])
            path = "_".join(trigger_path.split("_")[1:])
            events_trig = events[getattr(trigger_branch, path)]

            # fill L1 branches
            if (("l1ht" in self.hists_to_process["1d_scalar"]) or ("l1met" in self.hists_to_process["1d_scalar"])):
                l1_etsums = events_trig.L1EtSum
                if ("l1ht" in self.hists_to_process["1d_scalar"]):
                    l1_ht = l1_etsums[(events_trig.L1EtSum.etSumType==1) & (events_trig.L1EtSum.bx==0)]
                    hist_dict = fill_1d_hist(
                        hist_dict,
                        "l1ht",
                        trigger_path.split("_")[-1],
                        dak.flatten(l1_ht.pt),
                        "ht"
                    )
                if ("l1met" in self.hists_to_process["1d_scalar"]):
                    l1_met = l1_etsums[(events_trig.L1EtSum.etSumType==2) & (events_trig.L1EtSum.bx==0)]
                    hist_dict = fill_1d_hist(
                        hist_dict,
                        "l1met",
                        trigger_path.split("_")[-1],
                        dak.flatten(l1_met.pt),
                        "met"
                    )
            if ("total_l1mult" in self.hists_to_process["1d_scalar"]):
                l1_total_mult = (dak.num(events_trig.L1Jet.bx[events.trig.L1Jet.bx == 0])
                                    + dak.num(events_trig.L1Mu.bx[events.trig.L1Mu.bx == 0])
                                    + dak.num(events_trig.L1EG.bx[events.trig.L1EG.bx == 0]))
                hist_dict = fill_1d_hist(
                    hist_dict,
                    "total_l1mult",
                    trigger_path.split("_")[-1],
                    l1_total_mult,
                    "mult"
                )
            if ("total_l1pt" in self.hists_to_process["1d_scalar"]):
                l1_total_pt = (dak.num(events_trig.L1Jet.pt[events.trig.L1Jet.bx == 0], axis=1)
                                + dak.num(events_trig.L1Mu.pt[events.trig.L1Mu.bx == 0], axis=1)
                                + dak.num(events_trig.L1EG.pt[events.trig.L1EG.bx == 0], axis=1))
                hist_dict = fill_1d_hist(
                    hist_dict,
                    "total_l1pt",
                    trigger_path.split("_")[-1],
                    l1_total_pt,
                    "pt"
                )

            # fill scouting branches
            if ("scoutinght" in self.hists_to_process["1d_scalar"]):
                scouting_ht = dak.sum(events_trig.ScoutingPFJet.pt, axis=1)
                hist_dict = fill_1d_hist(
                    hist_dict,
                    "scoutinght",
                    trigger_path.split("_")[-1],
                    scouting_ht,
                    "ht"
                )
            if ("scoutingmet" in self.hists_to_process["1d_scalar"]):
                scouting_met = events_trig.ScoutingMET.pt
                hist_dict = fill_1d_hist(
                    hist_dict,
                    "scoutingmet",
                    trigger_path.split("_")[-1],
                    scouting_met,
                    "met"
                )
            if ("total_scoutingmult" in self.hists_to_process["1d_scalar"]):
                scouting_total_mult = dak.num(events_trig.ScoutingPFJet) + dak.num(events_trig.ScoutingElectron) + dak.num(events_trig.ScoutingMuonVtx) + dak.num(events_trig.ScoutingPhoton)
                hist_dict = fill_1d_hist(
                    hist_dict,
                    "total_scoutingmult",
                    trigger_path.split("_")[-1],
                    scouting_total_mult,
                    "mult"
                )
            if ("total_scoutingpt" in self.hists_to_process["1d_scalar"]):
                scouting_total_pt = dak.sum(events_trig.ScoutingPFJet.pt, axis=1) 
                + dak.sum(events_trig.ScoutingElectron.pt, axis=1) 
                + dak.num(events_trig.ScoutingMuonVtx.pt, axis=1) 
                + dak.num(events_trig.ScoutingPhoton.pt, axis=1)
                hist_dict = fill_1d_hist(
                    hist_dict,
                    "total_scoutingpt",
                    trigger_path.split("_")[-1],
                    scouting_total_pt,
                    "pt"
                )
            for obj, obj_dict in self.object_dict.items():
                cuts = obj_dict['cuts']
                label = obj_dict['label']
                isL1Obj = 'L1' in obj
                isScoutingObj = 'Scouting' in obj
                obj_branch = getattr(events_trig, obj)

                if isL1Obj:
                    obj_branch = obj_branch[obj_branch.bx == 0]

                for var, cut in cuts:
                    if var == 'pt':
                        mask = (getattr(obj_branch, var) > cut)
                    if var == 'eta':
                        mask = mask & (abs(getattr(obj_branch, var)) < cut)
                obj_branch = obj_branch[mask]

                if ("n" in self.hists_to_process["1d_object"]):
                    hist_dict = fill_1d_hist(
                        hist_dict,
                        "n_obj",
                        trigger_path.split("_")[-1],
                        dak.num(obj_branch),
                        "mult",
                        object_name=obj
                    )
                if ("pt" in self.hists_to_process["1d_object"]):
                    hist_dict = fill_1d_hist(
                        hist_dict,
                        "pt_obj",
                        trigger_path.split("_")[-1],
                        dak.flatten(obj_branch.pt),
                        "pt",
                        object_name=obj
                    )
                if ("pt0" in self.hists_to_process["1d_object"]):
                    hist_dict = fill_1d_hist(
                        hist_dict,
                        "pt0_obj",
                        trigger_path.split("_")[-1],
                        dak.flatten(obj_branch.pt[:, 0:1]),
                        "pt",
                        object_name=obj
                    )
                if ("pt1" in self.hists_to_process["1d_object"]):
                    hist_dict = fill_1d_hist(
                        hist_dict,
                        "pt1_obj",
                        trigger_path.split("_")[-1],
                        dak.flatten(obj_branch.pt[:, 1:2]),
                        "pt",
                        object_name=obj
                    )
                if ("eta" in self.hists_to_process["1d_object"]):
                    hist_dict = fill_1d_hist(
                        hist_dict,
                        "eta_obj",
                        trigger_path.split("_")[-1],
                        dak.flatten(obj_branch.eta),
                        "eta",
                        object_name=obj
                    )
                if ("phi" in self.hists_to_process["1d_object"]):
                    hist_dict = fill_1d_hist(
                        hist_dict,
                        "phi_obj",
                        trigger_path.split("_")[-1],
                        dak.flatten(obj_branch.phi),
                        "phi",
                        object_name=obj
                    )


        return hist_dict


        # hists = {
        # "AXONominalPure": 0,
        # "AXOTightPure": 0,
        # "JetHTPure": 0,
        # "Nominal+HT": 0,
        # "Tight+HT": 0,
        # "None": 0,
        # "All": 0,
        # "AXOs": 0,
        # "AXONominal": 0,
        # "AXOTight": 0,
        # "JetHT": 0,
        # }

        # axonom = events.DST.PFScouting_AXONominal
        # axotight = events.DST.PFScouting_AXOTight
        # jetht = events.DST.PFScouting_JetHT


        # hists["None"] += dak.num(events, axis=0)
        
        # hists["All"] += dak.num(events[axonom & (axotight) & (jetht)], axis=0)
        # hists["AXONominalPure"] += dak.num(events[axonom & (~axotight) & (~jetht)], axis=0)
        # hists["AXOTightPure"] += dak.num(events[(~axonom) & (axotight) & (~jetht)], axis=0)
        # hists["JetHTPure"] += dak.num(events[(~axonom) & (~axotight) & (jetht)], axis=0)
        # hists["Nominal+HT"] += dak.num(events[axonom & (~axotight) & (jetht)], axis=0)
        # hists["Tight+HT"] += dak.num(events[(~axonom) & (axotight) & (jetht)], axis=0)
        # hists["AXOs"] += dak.num(events[(axonom) & (axotight) & (~jetht)], axis=0)

        # # Create histograms
        # if "ScoutingHT" in self.hists_to_process:
        #     hists["ScoutingHT"] = hda.Hist(self.trigger_axis, self.ht_axis)
        # if "L1HT" in self.hists_to_process:
        #     hists["L1HT"] = hda.Hist(self.trigger_axis, self.ht_axis)
        # if "ScoutingMET" in self.hists_to_process:
        #     hists["ScoutingMET"] = hda.Hist(self.trigger_axis, self.met_axis)
        # if "L1MET" in self.hists_to_process:
        #     hists["L1MET"] = hda.Hist(self.trigger_axis, self.met_axis)
        # if "mult" in self.hists_to_process:
        #     hists["mult"] = hda.Hist(self.trigger_axis, self.mult_axis, self.object_axis)
        #     # hists["njets_HT_gt_500"] = hda.Hist(self.trigger_axis, self.mult_axis)
        # if "npv" in self.hists_to_process:
        #     hists["npv"] = hda.Hist(self.trigger_axis, self.mult_axis)
        # if "pt" in self.hists_to_process:
        #     hists["pt"] = hda.Hist(self.trigger_axis, self.pt_axis, self.object_axis)
        # if "eta" in self.hists_to_process:
        #     hists["eta"] = hda.Hist(self.trigger_axis, self.eta_axis, self.object_axis)
        # if "phi" in self.hists_to_process:
        #     hists["phi"] = hda.Hist(self.trigger_axis, self.phi_axis, self.object_axis)


        # # Apply triggers
        # for trigger_path in self.triggers:
        #     if trigger_path == 'None':
        #         events_triggered = events
        #         hists["None"] += dak.num(events_triggered, axis=0)
        #     else:
        #         trigger_branch = getattr(events, trigger_path.split("_")[0])
        #         trig_path = "_".join(trigger_path.split("_")[1:])
        #         events_triggered = events[getattr(trigger_branch, trig_path)]
        #         hists[trigger_path.split("_")[-1]] += dak.num(events_triggered, axis=0)


        #     if (("L1HT" in self.hists_to_process) or ("L1MET" in self.hists_to_process)):
        #         l1_etsums = events_triggered.L1EtSum
        #         if ("L1HT" in self.hists_to_process):
        #             l1_ht = l1_etsums[(events_triggered.L1EtSum.etSumType==1) & (events_triggered.L1EtSum.bx==0)]
        #             hists["L1HT"].fill(
        #                 ht=dak.flatten(l1_ht.pt), trigger=trigger_path.split("_")[-1]
        #             )
        #         if ("l1met" in self.hists_to_process):
        #             l1_met = l1_etsums[(events_triggered.L1EtSum.etSumType==2) & (events_triggered.L1EtSum.bx==0)]
        #             hists["L1MET"].fill(
        #                 met=dak.flatten(l1_met.pt), trigger=trigger_path.split("_")[-1]
        #             )

        #     if "ScoutingHT" in self.hists_to_process:
        #         jets = events_triggered.ScoutingPFJet
        #         scouting_ht = dak.sum(
        #             jets[(jets.pt > 30) & (abs(jets.eta) < 3.0)].pt, axis=1
        #         )
        #         hists["ScoutingHT"].fill(
        #             ht=scouting_ht, trigger=trigger_path.split("_")[-1]
        #         )

        #     if "ScoutingMET" in self.hists_to_process:
        #         scouting_met = events_triggered.ScoutingMET.pt
        #         hists["ScoutingMET"].fill(
        #             met=scouting_met, trigger=trigger_path.split("_")[-1]
        #         )
        #     if "npv" in self.hists_to_process:
        #         hists["npv"].fill(
        #             mult=dak.num(events_triggered.ScoutingPrimaryVertex, axis=-1), trigger=trigger_path.split("_")[-1]
        #         )

        #     if self.object_cuts is None:
        #         continue

        #     for obj, obj_cutlist in self.object_cuts.items():
        #         obj_branch = getattr(events_triggered, obj)
        #         mask = True
        #         for var, cut in obj_cutlist:
        #             if var == 'pt':
        #                 mask = mask & (getattr(obj_branch, var) > cut)
        #             if var == 'eta':
        #                 mask = mask & (abs(getattr(obj_branch, var)) < cut)
        #         obj_branch = obj_branch[mask]

        #         if "pt" in self.hists_to_process:
        #             hists["pt"].fill(
        #                 pt=dak.flatten(obj_branch.pt),
        #                 object=obj,
        #                 trigger=trigger_path.split("_")[-1]
        #             )
        #         if "eta" in self.hists_to_process:
        #             hists["eta"].fill(
        #                 eta=dak.flatten(obj_branch.eta),
        #                 object=obj,
        #                 trigger=trigger_path.split("_")[-1]
        #             )
        #         if "phi" in self.hists_to_process:
        #             hists["phi"].fill(
        #                 phi=dak.flatten(obj_branch.phi),
        #                 object=obj,
        #                 trigger=trigger_path.split("_")[-1]
        #             )
        #         if "mult" in self.hists_to_process:
        #             hists["mult"].fill(
        #                 mult=dak.num(obj_branch),
        #                 object=obj,
        #                 trigger=trigger_path.split("_")[-1]
        #             )

        # return hists

    def postprocess(self, accumulator):
        return accumulator
