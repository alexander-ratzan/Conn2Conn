import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from scipy.io import loadmat


# ---------------------------------------------------------------------------
# Krakencoder precomputed dummy model
# ---------------------------------------------------------------------------

# Input-type key fragments used by krakencoder inference outputs
_KRAKEN_INPUT_KEY = {
    "SC": "SCifod2act_{parc}_volnorm",
    "FC": "FCcorr_{parc}_hpf",
}
_KRAKEN_OUTPUT_KEY = "FCcorr_{parc}_hpf"


class KrakencoderPrecomputed(nn.Module):
    """
    Dummy Conn2Conn model that serves precomputed Krakencoder predictions.

    At construction time the model loads the per-seed inference `.mat` file
    produced by `krakencoder_runner.ipynb` and stores the full [957 x N_edges]
    prediction array together with the corresponding FC ground-truth array from
    `base`.  `predict_split()` then slices both by the partition indices stored
    in `base`, returning ready-to-evaluate (preds, targets) pairs without ever
    running a forward pass through a neural network.

    The model is wired as a closed-form (learned=false) model so it follows the
    same wandb prod-run path as CrossModalPCA / CrossModal_PLS_SVD — with one
    small guard in `Sim._evaluate_model` that calls `predict_split()` instead of
    `predict_from_loader()`.

    No tuning is possible: the YAML `search_space` is intentionally empty, which
    causes `Sim.run_tune()` to raise `ValueError` immediately.

    Args:
        base: `HCP_Base` instance.  `base.shuffle_seed`, `base.parcellation`,
              and `base.source` determine which inference file is loaded.
        kraken_predictions_dir: Directory that contains the inference `.mat`
              files.  Defaults to `<repo_root>/krakencoder/example_data/`.
    """

    is_precomputed = True

    def __init__(self, base, kraken_predictions_dir=None, **kwargs):
        super().__init__()

        seed = base.shuffle_seed
        parc = base.parcellation
        # base.source may be composite (e.g. "SC+SC_r2t"); use the first modality
        conn_type = (
            base.source_modalities[0]
            if hasattr(base, "source_modalities")
            else base.source
        )

        if conn_type not in _KRAKEN_INPUT_KEY:
            raise ValueError(
                f"KrakencoderPrecomputed: unsupported source modality '{conn_type}'. "
                f"Expected one of {list(_KRAKEN_INPUT_KEY)}."
            )

        if kraken_predictions_dir is None:
            repo_root = Path(__file__).resolve().parent.parent.parent
            kraken_predictions_dir = repo_root / "krakencoder" / "example_data"

        mat_path = (
            Path(kraken_predictions_dir)
            / f"mydata_kraken_seed{seed}_source_{parc}.{conn_type}.mat"
        )
        if not mat_path.exists():
            raise FileNotFoundError(
                f"KrakencoderPrecomputed: inference file not found:\n  {mat_path}\n"
                "Run `krakencoder_runner.ipynb` (inference loop) for this seed first."
            )

        mat = loadmat(str(mat_path), simplify_cells=True)
        input_key  = _KRAKEN_INPUT_KEY[conn_type].format(parc=parc)
        output_key = _KRAKEN_OUTPUT_KEY.format(parc=parc)

        try:
            preds_all = np.array(
                mat["predicted_alltypes"][input_key][output_key], dtype=np.float32
            )
        except (KeyError, TypeError) as exc:
            available = list(mat.get("predicted_alltypes", {}).keys())
            raise KeyError(
                f"KrakencoderPrecomputed: could not find "
                f"predicted_alltypes['{input_key}']['{output_key}'] "
                f"in {mat_path.name}.  Available input keys: {available}"
            ) from exc

        self._preds_all     = preds_all                                    # [N_subj, N_edges]
        self._targets_all   = base.fc_upper_triangles.astype(np.float32)   # [N_subj, N_edges]
        self._split_indices = base.trainvaltest_partition_indices

        n_subj = self._preds_all.shape[0]
        n_base = self._targets_all.shape[0]
        if n_subj != n_base:
            raise ValueError(
                f"KrakencoderPrecomputed: prediction array has {n_subj} subjects but "
                f"base has {n_base}.  Ensure the inference file matches participants.tsv."
            )

        print(
            f"KrakencoderPrecomputed: loaded  seed={seed}  parc={parc}  "
            f"source={conn_type}  shape={preds_all.shape}"
        )

    def predict_split(self, split: str):
        """
        Return (preds, targets) numpy float32 arrays for the requested split.

        Args:
            split: one of 'train', 'val', 'test'.

        Returns:
            preds:   ndarray [N_split, N_edges]
            targets: ndarray [N_split, N_edges]
        """
        idx = self._split_indices[split]
        return self._preds_all[idx], self._targets_all[idx]

    def forward(self, x):
        raise RuntimeError(
            "KrakencoderPrecomputed.forward() should never be called directly. "
            "Use predict_split() or run through Sim._evaluate_model."
        )
