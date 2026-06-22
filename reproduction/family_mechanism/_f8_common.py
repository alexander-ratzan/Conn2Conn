"""F8 (PC mechanism) helpers — faithful port of depth1_spectral_mechanism.ipynb and
depth1.1_pc_stability_and_confounds.ipynb.

Per-seed compute (PCA on SC/FC, per-PC FC->R², per-PC family AUC, sex+bv confound, PC1
residualization, node strength for rich-club) lives in compute_seed() and uses the data layer
LAZILY (via _fm_common._data()), so the torch-free localization/alignment/enrichment helpers
below run + test anywhere off-cluster (validated against the notebook's saved sc_pc_loadings.npy).

Notebook is canonical. Parcellation-aware where the notebook was Glasser-hardcoded:
  - network column: Glasser=community_yeo, 4S456Parcels=network_label
  - N regions inferred from edge count (no 360/64620 hardcode)
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import _fm_common as fm                                   # noqa: E402 (lazy data layer + auc_vs_unrelated)

CONN2CONN_ROOT = fm.REPRO_ROOT.parent
K_PCA_TOP = 10            # top SC PCs tracked
K_PCA_FC = 256           # FC PCA dim for FC->PC regression
NETWORK_COL = {"Glasser": "community_yeo", "4S456Parcels": "network_label"}


# ============================================================================
# torch-free primitives (verbatim numpy; match _setup / notebook)
# ============================================================================
def extract_pair_sims(sim_matrix, pairs_by_rel):
    out = {}
    for rel, pair_list in pairs_by_rel.items():
        if not pair_list:
            out[rel] = np.array([], dtype=np.float32)
            continue
        idx = np.asarray(pair_list)
        out[rel] = sim_matrix[idx[:, 0], idx[:, 1]].astype(np.float32)
    return out


def per_pc_aucs(score_vec, pairs_by_rel):
    """1-D PC score -> sim = -|z_i - z_j| -> AUC(relation vs unrelated). Mirrors depth1 cell 5."""
    diff = np.abs(score_vec[:, None] - score_vec[None, :])
    sims_by_rel = extract_pair_sims(-diff, pairs_by_rel)
    return fm.auc_vs_unrelated(sims_by_rel)


def cosine_sim_matrix(A, B):
    """Row-wise cosine similarity matrix (depth1.1 cell 4)."""
    A = A / np.linalg.norm(A, axis=1, keepdims=True).clip(1e-12)
    B = B / np.linalg.norm(B, axis=1, keepdims=True).clip(1e-12)
    return A @ B.T


def energy_top1pct(loading_vec, n_edges=None):
    """Fraction of squared-loading energy in the top 1% of edges (depth1.1 cell 10)."""
    sq = np.asarray(loading_vec, dtype=np.float64) ** 2
    n_edges = sq.size if n_edges is None else n_edges
    sq_sorted = np.sort(sq)[::-1]
    n_top1 = max(1, n_edges // 100)
    return float(sq_sorted[:n_top1].sum() / sq_sorted.sum())


# ============================================================================
# atlas + triu mapping (parcellation-aware)
# ============================================================================
def n_regions_from_edges(n_edges):
    N = int((1 + np.sqrt(1 + 8 * n_edges)) / 2)
    assert N * (N - 1) // 2 == n_edges, f"edge count {n_edges} not triangular"
    return N


def load_atlas(parc):
    """Return (atlas_indexed_by_idx0, network_col_name). idx0 = id-1 (0-indexed region)."""
    f = CONN2CONN_ROOT / "data" / "atlas_info" / f"{parc}_dseg_reformatted.csv"
    atlas = pd.read_csv(f)
    atlas["idx0"] = atlas["id"].astype(int) - 1
    return atlas.set_index("idx0").sort_index(), NETWORK_COL[parc]


def base_rates(atlas_idx, net_col, triu_i, triu_j, n_edges):
    """Network-pair counts over ALL edges + interhemispheric base fraction (depth1.1 cell 10)."""
    hemi_i = atlas_idx.loc[triu_i, "hemisphere"].values
    hemi_j = atlas_idx.loc[triu_j, "hemisphere"].values
    net_i = atlas_idx.loc[triu_i, net_col].astype(str).values
    net_j = atlas_idx.loc[triu_j, net_col].astype(str).values
    net_pair_all = np.array([" || ".join(sorted([a, b])) for a, b in zip(net_i, net_j)])
    all_pair_counts = pd.Series(net_pair_all).value_counts()
    interhemi_base = float((hemi_i != hemi_j).mean())
    return all_pair_counts, interhemi_base


def pc3_localization(pc_vec, atlas_idx, net_col, triu_i, triu_j, all_pair_counts,
                     interhemi_base, is_hub, richclub_base, k_list=(100, 200, 500)):
    """Per-K localization probes + network-pair enrichment rows for one PC (depth1.1 cell 10)."""
    n_edges = pc_vec.size
    eg = energy_top1pct(pc_vec, n_edges)
    loc_rows, enr_rows = [], []
    for K in k_list:
        order = np.argsort(-np.abs(pc_vec))
        top = order[:K]
        ti, tj = triu_i[top], triu_j[top]
        interhemi_frac = float((atlas_idx.loc[ti, "hemisphere"].values
                                != atlas_idx.loc[tj, "hemisphere"].values).mean())
        richclub_frac = float((is_hub[ti] & is_hub[tj]).mean())
        top_pair = np.array([" || ".join(sorted([a, b])) for a, b in zip(
            atlas_idx.loc[ti, net_col].astype(str).values,
            atlas_idx.loc[tj, net_col].astype(str).values)])
        for pair, n_obs in pd.Series(top_pair).value_counts().items():
            n_exp = all_pair_counts.get(pair, 0) * (K / n_edges)
            enr = (n_obs / n_exp) if n_exp > 0 else np.nan
            enr_rows.append({"K": K, "net_pair": pair, "n_obs": int(n_obs),
                             "n_exp": float(n_exp), "enrichment": float(enr)})
        loc_rows.append({"K": K, "interhemi_frac_top": interhemi_frac,
                         "interhemi_frac_base": interhemi_base,
                         "richclub_frac_top": richclub_frac, "richclub_frac_base": richclub_base,
                         "energy_top1pct_frac": eg})
    return loc_rows, enr_rows


# ============================================================================
# per-seed heavy compute (data layer; runs on Torch) — depth1 cells 3/5/7 + depth1.1 cells 3/6
# ============================================================================
def compute_seed(parc, seed):
    """Return a dict of per-seed F8 artifacts for (parc, seed). Uses the data layer."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import BayesianRidge, LinearRegression
    from sklearn.metrics import r2_score
    D = fm._data()
    D["set_parcellation"](parc)
    sp = D["load_seed_split"](seed=seed, source="FC", target="SC")
    SC_tr, SC_te = sp["SC_train"], sp["SC_test"]
    FC_tr, FC_te = sp["FC_train"], sp["FC_test"]
    base, test_idx, train_idx = sp["base"], sp["test_idx"], sp["train_idx"]

    pca_sc = PCA(n_components=K_PCA_TOP, random_state=0).fit(SC_tr)
    pca_fc = PCA(n_components=K_PCA_FC, random_state=0).fit(FC_tr)
    sc_tr = pca_sc.transform(SC_tr); sc_te = pca_sc.transform(SC_te)
    Z_FC_tr = pca_fc.transform(FC_tr); Z_FC_te = pca_fc.transform(FC_te)

    fc_r2 = np.zeros(K_PCA_TOP)
    for k in range(K_PCA_TOP):
        m = BayesianRidge(max_iter=300).fit(Z_FC_tr, sc_tr[:, k])
        fc_r2[k] = r2_score(sc_te[:, k], m.predict(Z_FC_te))

    rng = np.random.default_rng(42 + seed)
    pairs = D["pair_indices_by_relation"](base.metadata_df, test_idx, rng, fm.PAIR_AGE_TOL)
    auc_mz = np.full(K_PCA_TOP, np.nan); auc_dz = np.full(K_PCA_TOP, np.nan); auc_sb = np.full(K_PCA_TOP, np.nan)
    for k in range(K_PCA_TOP):
        a = per_pc_aucs(sc_te[:, k], pairs)
        auc_mz[k], auc_dz[k], auc_sb[k] = a.get("MZ", np.nan), a.get("DZ", np.nan), a.get("sibling", np.nan)

    # confound: OLS [sex || bv] -> PC_k  (depth1.1 cell 6)
    sex_tr, sex_te = base.sex_oh[train_idx], base.sex_oh[test_idx]
    Xc_tr = np.concatenate([sex_tr, sp["bv_train"]], axis=1)
    Xc_te = np.concatenate([sex_te, sp["bv_test"]], axis=1)
    conf_r2 = np.zeros(K_PCA_TOP)
    for k in range(K_PCA_TOP):
        ols = LinearRegression().fit(Xc_tr, sc_tr[:, k])
        conf_r2[k] = r2_score(sc_te[:, k], ols.predict(Xc_te))
    # PC1 residualization retest
    ols1 = LinearRegression().fit(Xc_tr, sc_tr[:, 0])
    pc1_res_tr = sc_tr[:, 0] - ols1.predict(Xc_tr)
    pc1_res_te = sc_te[:, 0] - ols1.predict(Xc_te)
    r2_raw = r2_score(sc_te[:, 0], BayesianRidge(max_iter=300).fit(Z_FC_tr, sc_tr[:, 0]).predict(Z_FC_te))
    r2_resid = r2_score(pc1_res_te, BayesianRidge(max_iter=300).fit(Z_FC_tr, pc1_res_tr).predict(Z_FC_te))

    # node strength + rich-club hubs from SC train mean edge (depth1.1 cell 10)
    n_edges = SC_tr.shape[1]
    N = n_regions_from_edges(n_edges)
    ti, tj = np.triu_indices(N, k=1)
    sc_mean = SC_tr.mean(axis=0)
    sym = np.zeros((N, N), dtype=np.float32); sym[ti, tj] = sc_mean; sym[tj, ti] = sc_mean
    node_strength = sym.mean(axis=1)
    is_hub = node_strength >= np.quantile(node_strength, 0.80)
    richclub_base = float((is_hub[ti] & is_hub[tj]).mean())

    return dict(
        loadings=pca_sc.components_.astype(np.float32), expl_var=pca_sc.explained_variance_ratio_,
        fc_r2=fc_r2, auc_mz=auc_mz, auc_dz=auc_dz, auc_sb=auc_sb, conf_r2=conf_r2,
        pc1_r2_raw=np.array([r2_raw]), pc1_r2_resid=np.array([r2_resid]),
        node_strength=node_strength.astype(np.float32), is_hub=is_hub,
        richclub_base=np.array([richclub_base]),
        pair_count_keys=np.array(list(pairs.keys())),
        pair_counts=np.array([len(v) for v in pairs.values()]),
        seed=np.array([seed]),
    )
