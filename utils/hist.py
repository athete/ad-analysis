import hist.dask as hda

def create_1d_hist(storage="weight", label="nEvents", **hist_axes) -> hda.Hist:
    return hda.hist.Hist(storage=storage, label=label, **hist_axes)