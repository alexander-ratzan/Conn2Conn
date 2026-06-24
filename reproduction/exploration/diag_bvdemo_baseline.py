#!/usr/bin/env python3
"""Diagnose why bv+demo does NOT trivially predict its own sex/age columns.

Hypothesis (user): "PCA compresses the information to nothing." Refined: bv+demo is ~22-dim, so
PCA(min(256,22))=PCA(22) is a LOSSLESS rotation — it doesn't truncate. The loss comes from
PCA(variance-ordered) + BayesianRidge's variance-dependent L2 shrinkage: the demographic columns
(age_z, sex_oh), being relatively independent of the correlated brain-volume block, land in
LOW-VARIANCE principal directions, whose ridge coefficients are shrunk toward zero. Net effect:
the model can't read off the in-feature label. Whitening the PCA scores (StandardScaler) before
the ridge removes the variance dependence and recovers the label.

Two modes:
  (default) SYNTHETIC — faithful analog (correlated brain-vol block + independent age/sex/race),
            runs anywhere (no torch). Proves the mechanism + the fix.
  --real    REAL — load seed-0 bv+demo via the grid data layer (Torch only) and run the same ladder.

    python reproduction/exploration/diag_bvdemo_baseline.py            # synthetic, local
    python reproduction/exploration/diag_bvdemo_baseline.py --real     # on Torch (needs data)
"""
import argparse
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.metrics import balanced_accuracy_score
from scipy.stats import pearsonr


def ladder(Xtr, Xte, y_tr, y_te, kind):
    """Run the method ladder; return dict of score (pearson for age, bal_acc for sex)."""
    def score(pred):
        if kind == "sex":
            return balanced_accuracy_score((y_te >= 0.5).astype(int), (pred >= 0.5).astype(int))
        return pearsonr(pred, y_te)[0]
    out = {}
    # (a) raw OLS, no PCA, no ridge — the label is a feature, so this is the recoverability ceiling
    out["raw_OLS"] = score(LinearRegression().fit(Xtr, y_tr).predict(Xte))
    # (b) CURRENT path: PCA(width) -> BayesianRidge, no whitening
    k = min(256, Xtr.shape[1])
    p = PCA(n_components=k, random_state=0).fit(Xtr)
    out["PCA->BR (current)"] = score(BayesianRidge(max_iter=500).fit(p.transform(Xtr), y_tr).predict(p.transform(Xte)))
    # (c) FIX: PCA -> whiten (StandardScaler on scores) -> BayesianRidge
    Ztr, Zte = p.transform(Xtr), p.transform(Xte)
    sc = StandardScaler().fit(Ztr)
    out["PCA->whiten->BR (fix)"] = score(BayesianRidge(max_iter=500).fit(sc.transform(Ztr), y_tr).predict(sc.transform(Zte)))
    # (d) PCA(whiten=True) built-in — equivalent fix
    pw = PCA(n_components=k, whiten=True, random_state=0).fit(Xtr)
    out["PCA(whiten=True)->BR"] = score(BayesianRidge(max_iter=500).fit(pw.transform(Xtr), y_tr).predict(pw.transform(Xte)))
    # (e) k-sweep on the CURRENT path: does small k (real compression) hurt more?
    for kk in [2, 5, 10, k]:
        pk = PCA(n_components=min(kk, Xtr.shape[1]), random_state=0).fit(Xtr)
        out[f"PCA(k={kk})->BR"] = score(
            BayesianRidge(max_iter=500).fit(pk.transform(Xtr), y_tr).predict(pk.transform(Xte)))
    return out


def make_synthetic(seed=0):
    rng = np.random.default_rng(seed)
    n_tr, n_te = 683, 195
    n = n_tr + n_te
    # brain-volume block: 16 columns from 3 correlated latent factors -> few PCs dominate variance
    L = rng.normal(size=(n, 3))
    W = rng.normal(size=(3, 16))
    bv = L @ W + 0.3 * rng.normal(size=(n, 16))
    bv = (bv - bv[:n_tr].mean(0)) / bv[:n_tr].std(0)          # train z-score (like fs_volumes_z)
    # age: independent, z-scored (like age_z); target = raw age (linear fn of age_z)
    age_raw = rng.normal(60, 8, size=n)
    age_z = ((age_raw - age_raw[:n_tr].mean()) / age_raw[:n_tr].std())
    # sex: balanced binary -> one-hot 2 cols (like pd.get_dummies)
    sex = (rng.random(n) < 0.5).astype(float)
    sex_oh = np.stack([1 - sex, sex], axis=1)
    # race: 3-cat one-hot
    race = rng.integers(0, 3, size=n)
    race_oh = np.eye(3)[race]
    X = np.concatenate([bv, age_z[:, None], sex_oh, race_oh], axis=1)   # 16+1+2+3 = 22 cols
    return (X[:n_tr], X[n_tr:], age_raw[:n_tr], age_raw[n_tr:], sex[:n_tr], sex[n_tr:])


def pc_diagnosis(Xtr, label_vec_tr):
    """Where does the label direction live in the variance-ordered PCA spectrum?"""
    k = min(256, Xtr.shape[1])
    p = PCA(n_components=k, random_state=0).fit(Xtr)
    Z = p.transform(Xtr)
    evr = p.explained_variance_ratio_
    corr = np.array([abs(pearsonr(Z[:, j], label_vec_tr)[0]) for j in range(Z.shape[1])])
    j = int(np.argmax(corr))
    return j, evr[j], corr[j], evr


def run(Xtr, Xte, age_tr, age_te, sex_tr, sex_te, tag):
    print(f"\n{'='*68}\n{tag}: bv+demo shape train={Xtr.shape} test={Xte.shape}\n{'='*68}")
    print("  column std (train):", np.round(Xtr.std(0), 2))
    # where does the age direction live?
    j, evr_j, corr_j, evr = pc_diagnosis(Xtr, age_tr.astype(float))
    print(f"  age aligns best with PC#{j+1} (|corr|={corr_j:.2f}), that PC explains "
          f"{100*evr_j:.1f}% var (top PC explains {100*evr[0]:.1f}%) "
          f"-> age is a {'LOW' if evr_j < evr[0]/2 else 'mid'}-variance direction")
    print("\n  AGE (pearson):")
    for m, v in ladder(Xtr, Xte, age_tr, age_te, "age").items():
        print(f"    {m:24s} {v:.3f}")
    print("\n  SEX (balanced acc):")
    for m, v in ladder(Xtr, Xte, sex_tr, sex_te, "sex").items():
        print(f"    {m:24s} {v:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="load seed-0 bv+demo via grid data layer (Torch)")
    ap.add_argument("--parc", default="Glasser")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.real:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import _grid_common as gc
        gc.set_parcellation(args.parc)
        sp = gc.load_split_checked(args.seed, args.parc) if hasattr(gc, "load_split_checked") else gc.load_seed_split(seed=args.seed)
        Xtr = np.asarray(sp["bvdemo_train"], float); Xte = np.asarray(sp["bvdemo_test"], float)
        age_tr, age_te = gc.load_scalar_target(sp, "age")
        sex_tr, sex_te = gc.load_scalar_target(sp, "sex")
        ok = ~np.isnan(age_tr)
        run(Xtr[ok], Xte, age_tr[ok], age_te, sex_tr[ok], sex_te, f"REAL {args.parc} seed{args.seed}")
    else:
        Xtr, Xte, age_tr, age_te, sex_tr, sex_te = make_synthetic()
        run(Xtr, Xte, age_tr, age_te, sex_tr, sex_te, "SYNTHETIC A: feature == clean linear(target)")
        print("\n  -> At full rank (k=22) PCA->BR recovers ~1.0. Compression only bites under "
              "truncation (k=2 kills it). bv+demo is 22-dim with k=22, so NO truncation here.")

        # SCENARIO B: benign mismatch — the age TARGET is a binned/noisy version of the age_z FEATURE
        # (HCP public age is 5-yr bins; if the feature is continuous, recovery caps at their corr).
        rng = np.random.default_rng(1)
        # rebuild with a binned age target while the feature stays continuous age_z
        Xtr2, Xte2, age_tr2, age_te2, sex_tr2, sex_te2 = make_synthetic(seed=2)
        def binned(a):  # 5-year bins -> bin midpoint (coarsens the target)
            return (np.floor(a / 5.0) * 5 + 2.5)
        bt_tr, bt_te = binned(age_tr2), binned(age_te2)
        print(f"\n{'='*68}\nSYNTHETIC B: target = 5-yr-binned age, feature = continuous age_z "
              f"(corr cont~binned = {pearsonr(age_te2, bt_te)[0]:.2f})\n{'='*68}")
        for m, v in ladder(Xtr2, Xte2, bt_tr, bt_te, "age").items():
            print(f"    {m:24s} {v:.3f}")
        print("  -> Even with NO truncation, PCA->BR caps near the feature/target correlation. "
              "If real age_z (feature) and age (target) are encoded differently, 0.84 is BENIGN, not a bug.")
        print("\nNEXT: run `--real` on Torch — it prints the real bvdemo width + raw_OLS vs PCA->BR. "
              "If raw_OLS also ~0.84 => encoding mismatch (benign). If raw_OLS ~1.0 but PCA->BR ~0.84 "
              "=> a real path bug to fix.")


if __name__ == "__main__":
    main()
