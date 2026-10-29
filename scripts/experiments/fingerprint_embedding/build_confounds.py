"""Spec v4 E6: per-subject confound proxies (Glasser) for the fingerprint check. Head motion (framewise displacement)
is not in our data (neither the xcp-d outputs nor the HCP unrestricted table), so motion is proxied from the rest
timeseries.

  dvars       mean over runs of the RMS across parcels of the first temporal difference (raw BOLD units)
  gs_sd       mean over runs of the SD over time of the global (parcel-mean) signal
  mean_fc     mean rest FC over all edges (motion inflates it)
  icv         FreeSurfer intracranial volume (HCP unrestricted)
  parcel_vol  total Glasser parcel volume (Conn2Conn_data/parcel_node_features)

    python scripts/experiments/fingerprint_embedding/build_confounds.py   -> results/cache/experiments/fingerprint_embedding/
"""
import numpy as np
import pandas as pd

from fe_common import CONFOUNDS, UNRESTRICTED
from data.timeseries import TimeseriesCache

NODE = "/scratch/asr655/neuroinformatics/Conn2Conn_data/parcel_node_features/parc-Glasser_hemi-both_vol-volume_mm3_cent-centroid_mm"
FC_DIR = "/scratch/asr655/neuroinformatics/Conn2Conn_data/fc/parc-Glasser_hemi-both"


def main():
    ts = TimeseriesCache("rest", "Glasser")
    rows = []
    for k, sid in enumerate(ts.subject_ids):
        x = ts.subject(sid)
        dv, gs = [], []
        for a, b in ts.run_bounds(sid):
            r = x[a:b].astype(np.float64)
            dv.append(np.sqrt((np.diff(r, axis=0) ** 2).mean(1)).mean())
            gs.append(r.mean(1).std())
        rows.append(dict(subject=int(sid), dvars=float(np.mean(dv)), gs_sd=float(np.mean(gs))))
        if k % 200 == 0:
            print(f"{k} / {len(ts.subject_ids)}", flush=True)
    df = pd.DataFrame(rows)
    ids = np.load(f"{FC_DIR}/subject_ids.npy")
    X = np.load(f"{FC_DIR}/upper_triangles.npy", mmap_mode="r")
    df = df.merge(pd.DataFrame(dict(subject=ids.astype(int), mean_fc=[float(np.mean(X[i])) for i in range(len(ids))])), on="subject", how="left")
    u = pd.read_csv(UNRESTRICTED, usecols=["Subject", "FS_IntraCranial_Vol"]).rename(columns={"Subject": "subject", "FS_IntraCranial_Vol": "icv"})
    df = df.merge(u, on="subject", how="left")
    nid = np.load(f"{NODE}/subject_ids.npy").astype(int)
    vol = np.load(f"{NODE}/parcel_volume.npy")
    df = df.merge(pd.DataFrame(dict(subject=nid, parcel_vol=vol.reshape(len(nid), -1).sum(1))), on="subject", how="left")
    CONFOUNDS.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CONFOUNDS, index=False, float_format="%.6g")
    print(df.describe().T.to_string())
    print(df.drop(columns="subject").corr().round(2).to_string())


if __name__ == "__main__":
    main()
