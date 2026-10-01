"""Raw and weighted Krakencoder loss terms in its own training space at a checkpoint (batches of 64): basis of the
var anchor in config.yml.

    python scripts/experiments/composite_loss/krakencoder/checks/term_magnitudes.py results/krakencoder/<tag>/seed0 64
"""
import glob, sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "main.py").exists())))
from models.architectures.krakencoder import _vendor_entry
kc = _vendor_entry.activate()
from krakencoder.data import load_transformers_from_file, generate_adapt_transformer
from krakencoder.model import Krakencoder
from krakencoder.loss import correye, distance_neighbor_loss, var_match_loss
from krakencoder.utils import numpyvar, torchfloat, torchint
from data.hcp_dataset import HCP_Base
run = sys.argv[1]; bs = int(sys.argv[2]) if len(sys.argv) > 2 else 64
ck = sorted(glob.glob(f"{run}/kraken_chkpt_*_ep002000.pt"))[-1]
iox = glob.glob(f"{run}/kraken_ioxfm_*.npy")
b = HCP_Base(parcellation="Glasser", hemi="both", shuffle_seed=0, source="SC", target="FC", data_load_mode="precomputed")
tr = np.asarray(b.trainvaltest_partition_indices["train"])
data = {"SC": np.asarray(b.sc_upper_triangles, np.float32), "FC": np.asarray(b.fc_upper_triangles, np.float32)}
fl = {"SC": "SCifod2act_Glasser_volnorm", "FC": "FCcorr_Glasser_hpf"}
T, info = load_transformers_from_file(iox, input_names=list(fl.values()), quiet=True)
Z = {m: torchfloat(T[fl[m]].transform(data[m][tr])) for m in data}           # training-space targets/inputs
net, ex = Krakencoder.load_checkpoint(ck, eval_mode=True); net.eval(); names = list(ex["input_name_list"])
W = {"mse": 1000, "correye": 1, "neidist": 1, "var": 1}
for src, tgt in (("SC", "FC"), ("FC", "SC"), ("SC", "SC"), ("FC", "FC")):
    with torch.no_grad():
        _, y = net(Z[src], torchint(names.index(fl[src])), torchint(names.index(fl[tgt])))
    t = Z[tgt]; acc = {k: [] for k in W}
    for s in range(0, len(tr) - bs + 1, bs):
        p, q = y[s:s+bs], t[s:s+bs]
        acc["mse"].append(float(torch.mean((q - p) ** 2)))
        acc["correye"].append(float(correye(q, p)))
        acc["neidist"].append(float(distance_neighbor_loss(q, p)))
        acc["var"].append(float(var_match_loss(p, q)))
    raw = {k: np.mean(v) for k, v in acc.items()}
    print(f"{src}->{tgt} raw: " + "  ".join(f"{k} {raw[k]:.4g}" for k in W) +
          " | weighted at paper weights: " + "  ".join(f"{k} {W[k]*raw[k]:.4g}" for k in W) +
          f" | ratio to mse.w1000: " + "  ".join(f"{k} {raw[k]/(1000*raw['mse']):.3g}" for k in ("correye", "neidist", "var")), flush=True)
