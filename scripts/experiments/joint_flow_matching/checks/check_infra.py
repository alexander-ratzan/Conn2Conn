"""Spec v4 E2 gate G0 (I1 acceptance), CPU. Prints one line per check and ALL_OK / FAILED.

  1. cohort, subject order and train / val / test lists identical to v3:E0's HCP_Base (fc_source_condition=rest);
  2. code space fit on train only: stored means / scales equal the train-row values; each basis is orthonormal and
     spans the top eigenvectors of its train-only operator (relative residual of X^T X W = W Lambda);
  3. encode / decode: decode(encode(x)) is the orthogonal projection (Pythagoras on val rows);
  4. integrators: exact on a constant field; RK4 matches expm(A) z0 for a linear field at 64 steps;
  5. L3: the Gaussian OT map satisfies A S_x A = S_y;
  6. L4: with two views and lambda -> 0 the MAXVAR eigenvalues minus 1 are the canonical correlations (QR / SVD of
     the two views); G is orthonormal and D_m = G^T Z_m.

    python scripts/experiments/joint_flow_matching/checks/check_infra.py --seed 0
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.linalg
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CFG, SeedData, load_views  # noqa: E402
from linear_arms import gaussian_ot, maxvar  # noqa: E402
from models.architectures.crossmodal_flow_matching import integrate  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"{'OK  ' if ok else 'FAIL'} {name} {detail}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from data.dataset_utils import FC_TASK_CONDITIONS
    from data.hcp_dataset import HCP_Base

    sd = SeedData(args.seed, "main")
    data, splits, ids = load_views(args.seed)

    # 1. cohort and splits vs v3:E0
    bench = HCP_Base(parcellation=CFG["parcellation"], hemi="both", source="FC", target="SC",
                     data_load_mode="precomputed", shuffle_seed=args.seed, fc_conditions=list(FC_TASK_CONDITIONS),
                     expose_fc_sessions=True, fc_source_condition="rest")
    bids = np.asarray([int(s) for s in bench.metadata_df["subject"]])
    check("cohort size", len(ids) == CFG["cohort"] == len(bids), f"{len(ids)} vs {len(bids)}")
    check("subject order == v3:E0", np.array_equal(ids, bids) and np.array_equal(sd.subject_ids, bids))
    for k in ("train", "val", "test"):
        check(f"split {k} == v3:E0", np.array_equal(sd.splits[k], np.asarray(bench.trainvaltest_partition_indices[k])),
              f"n={len(sd.splits[k])}")
    check("SC edges == v3:E0", np.array_equal(data["SC"], bench.sc_upper_triangles))
    check("rest edges == v3:E0", np.array_equal(data["rest"], bench.fc_upper_triangles))

    # 2. train-only fit
    tr = sd.splits["train"]
    for v in sd.cs.views:
        Xc = data[v][tr].astype(np.float64) - data[v][tr].astype(np.float64).mean(0)
        check(f"mean/scale train-only [{v}]",
              np.allclose(sd.cs.mean[v], data[v][tr].mean(0), atol=1e-5)
              and np.isclose(sd.cs.scale[v], np.linalg.norm(Xc), rtol=1e-5))
    for name, W in sd.cs.bases.items():
        W = W.astype(np.float64)
        members = [v for v in CFG["views"] if sd.cs.basis_of[v] == name]
        X = np.vstack([(data[v][tr].astype(np.float64) - sd.cs.mean[v]) / sd.cs.scale[v] for v in members])
        XW = X @ W
        lam = np.square(XW).sum(0)
        resid = np.linalg.norm(X.T @ XW - W * lam) / np.linalg.norm(W * lam)
        ortho = np.abs(W.T @ W - np.eye(W.shape[1])).max()
        check(f"basis [{name}] orthonormal + eigenvectors of train operator", ortho < 1e-4 and resid < 1e-3,
              f"ortho {ortho:.1e}, eig resid {resid:.1e}, members {members}")

    # 3. encode / decode = orthogonal projection
    for v in ("SC", "rest", "wm", "rest_S1"):
        X = data[v][sd.splits["val"]].astype(np.float64)
        Z = sd.cs.encode(v, X).astype(np.float64)
        Xh = sd.cs.decode(v, Z).astype(np.float64)
        dev = (X - sd.cs.mean[v]) / sd.cs.scale[v]
        lhs = np.square(dev).sum(1)
        rhs = np.square(Z).sum(1) + np.square(dev - (Xh - sd.cs.mean[v]) / sd.cs.scale[v]).sum(1)
        check(f"decode(encode) is the projection [{v}]", np.allclose(lhs, rhs, rtol=1e-3),
              f"max rel {np.max(np.abs(lhs - rhs) / lhs):.1e}")
        check(f"cached codes == encode [{v}]", np.allclose(sd.code(v, "val", sd.cs.k_max), Z, atol=1e-5))

    # 4. integrators
    z0 = torch.randn(5, 7, dtype=torch.float64)
    c = torch.randn(7, dtype=torch.float64)
    for m in ("euler", "rk4"):
        out = integrate(lambda z, t: c.expand_as(z), z0, n_steps=3, method=m)
        check(f"{m} exact on constant field", torch.allclose(out, z0 + c, atol=1e-12))
    A = torch.randn(7, 7, dtype=torch.float64) * 0.5
    out = integrate(lambda z, t: z @ A.T, z0, n_steps=64, method="rk4")
    ref = z0 @ torch.as_tensor(scipy.linalg.expm(A.numpy())).T
    check("rk4 linear field vs expm", (out - ref).abs().max().item() < 1e-6, f"{(out - ref).abs().max().item():.1e}")

    # 5. L3 Gaussian OT
    rng = np.random.default_rng(0)
    Xs, Ys = rng.normal(size=(500, 16)) @ rng.normal(size=(16, 16)), rng.normal(size=(500, 16)) @ rng.normal(size=(16, 16))
    from sklearn.covariance import LedoitWolf
    _, Amap, _ = gaussian_ot(Xs, Ys)
    Sx, Sy = LedoitWolf().fit(Xs).covariance_, LedoitWolf().fit(Ys).covariance_
    err = np.linalg.norm(Amap @ Sx @ Amap - Sy) / np.linalg.norm(Sy)
    check("L3 OT map: A Sx A = Sy", err < 1e-6, f"rel {err:.1e}")

    # 6. L4 MAXVAR
    Z1 = sd.code("rest", "train", 16).astype(np.float64)
    Z2 = sd.code("SC", "train", 16).astype(np.float64)
    M = Z1 @ np.linalg.inv(Z1.T @ Z1) @ Z1.T + Z2 @ np.linalg.inv(Z2.T @ Z2) @ Z2.T
    w = np.sort(np.linalg.eigvalsh(M))[::-1][:8] - 1
    rho = np.linalg.svd(np.linalg.qr(Z1)[0].T @ np.linalg.qr(Z2)[0], compute_uv=False)[:8]
    check("L4 two-view MAXVAR == CCA", np.allclose(w, rho, atol=1e-6), f"max diff {np.max(np.abs(w - rho)):.1e}")
    Zall = {v: sd.code(v, "train", 32).astype(np.float64) for v in CFG["views"]}
    Aall, Dall, G = maxvar(Zall, d=16, lam_rel=1.0)
    check("L4 G orthonormal", np.allclose(G.T @ G, np.eye(16), atol=1e-8))
    check("L4 decoders D_m = G^T Z_m", np.allclose(Dall["rest"], G.T @ Zall["rest"], atol=1e-8))

    print("ALL_OK" if all(RESULTS) else f"FAILED ({len(RESULTS) - sum(RESULTS)} of {len(RESULTS)})")


if __name__ == "__main__":
    main()
