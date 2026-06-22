"""Validate F8 localization math against the notebook's saved seed-0 PC loadings.

depth1.1 saved `sc_pc_loadings.npy` (seed-0 SC PCA components, 10 x 64620) and the localization
outputs. The localization probes (energy concentration, interhemispheric fraction, network-pair
enrichment, top-edge region mapping) depend ONLY on the loadings + the Glasser atlas — not on the
connectome data — so we reproduce them off-cluster and assert they match the notebook.
(rich-club needs node strength = SC data, so that probe is validated on Torch, not here.)

    python reproduction/family_mechanism/tests/test_f8_localization_matches_notebook.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import _f8_common as f8

_FE = _HERE.parents[3] / ("notebooks-FC_to_SC-experimental/model_overviews/results/"
                          "local_results/further_exploration")
LOADINGS = _FE / "depth1_spectral_mechanism" / "sc_pc_loadings.npy"
LOC_CSV = _FE / "depth1.1_pc_stability_and_confounds" / "pc3_localization_per_seed.csv"
TOPEDGE_CSV = _FE / "depth1.1_pc_stability_and_confounds" / "pc3_top50_edges_seed0_labeled.csv"


def test_pc3_localization_matches_notebook_seed0():
    load = np.load(LOADINGS)
    pc3 = load[2]                                   # seed-0 PC3 (0-indexed)
    n_edges = pc3.size
    N = f8.n_regions_from_edges(n_edges)
    ti, tj = np.triu_indices(N, k=1)
    atlas, netcol = f8.load_atlas("Glasser")
    assert len(atlas) == N == 360
    all_pair_counts, interhemi_base = f8.base_rates(atlas, netcol, ti, tj, n_edges)

    ref = pd.read_csv(LOC_CSV)
    ref0 = ref[ref.seed == 0].set_index("K")

    # energy concentration (orientation-invariant)
    eg = f8.energy_top1pct(pc3)
    assert abs(eg - float(ref0.iloc[0]["energy_top1pct_frac"])) < 1e-5, eg
    # interhemi base
    assert abs(interhemi_base - float(ref0.iloc[0]["interhemi_frac_base"])) < 1e-6

    dummy_hub = np.zeros(N, dtype=bool)
    loc, _ = f8.pc3_localization(pc3, atlas, netcol, ti, tj, all_pair_counts,
                                 interhemi_base, dummy_hub, 0.0, k_list=(100, 200, 500))
    for r in loc:
        ref_ih = float(ref0.loc[r["K"]]["interhemi_frac_top"])
        assert abs(r["interhemi_frac_top"] - ref_ih) < 1e-6, (r["K"], r["interhemi_frac_top"], ref_ih)

    # top-edge region mapping: |loading| ordering is orientation-invariant -> region idx pairs match
    ref_top = pd.read_csv(TOPEDGE_CSV).head(50)
    order = np.argsort(-np.abs(pc3))[:50]
    got_pairs = set(zip(ti[order].tolist(), tj[order].tolist()))
    ref_pairs = set(zip(ref_top.region_i_idx.tolist(), ref_top.region_j_idx.tolist()))
    overlap = len(got_pairs & ref_pairs)
    assert overlap >= 48, f"top-50 edge region mapping overlap only {overlap}/50"
    return dict(energy=eg, interhemi_base=interhemi_base, topedge_overlap=overlap)


if __name__ == "__main__":
    info = test_pc3_localization_matches_notebook_seed0()
    print("PASS  F8 localization reproduces the notebook (seed 0, Glasser)")
    print(f"  energy_top1pct      = {info['energy']:.6f}  (notebook 0.630678)")
    print(f"  interhemi base      = {info['interhemi_base']:.6f}  (notebook 0.501393)")
    print(f"  top-50 edge overlap = {info['topedge_overlap']}/50  (region->triu mapping correct)")
