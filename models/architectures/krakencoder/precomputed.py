import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from scipy.io import loadmat


# ---------------------------------------------------------------------------
# Krakencoder precomputed dummy model
# ---------------------------------------------------------------------------

# Connectome-flavor keys used in Krakencoder inference outputs (`predicted_alltypes[input][output]`).
_KRAKEN_FLAVOR_KEY = {
    "SC": "SCifod2act_{parc}_volnorm",
    "FC": "FCcorr_{parc}_hpf",
}
# Legacy name kept for callers that imported it.
_KRAKEN_INPUT_KEY = _KRAKEN_FLAVOR_KEY

_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "main.py").exists())
# Cached March 2026 predictions (gitignored local Krakencoder copy).
LEGACY_PREDICTIONS_DIR = _REPO_ROOT / "krakencoder_experimental" / "example_data"
LEGACY_FILE_PATTERN = "mydata_kraken_seed{seed}_source_{parc}.{conn}.mat"
# Retrained runs (models/architectures/krakencoder/retrain.py): <root>/<tag>/seed{seed}/<pattern>.
RETRAINED_PREDICTIONS_ROOT = _REPO_ROOT / "results" / "krakencoder"
RETRAINED_FILE_PATTERN = "predictions_source_{parc}.{conn}.mat"


def krakencoder_prediction_path(seed, parc, conn, tag=None, predictions_root=None, kraken_predictions_dir=None):
    """Path of the inference file for one seed / parcellation / source modality.

    `tag` selects a retrained run under `predictions_root` (default `results/krakencoder`); otherwise the legacy
    cache directory `kraken_predictions_dir` (default `krakencoder_experimental/example_data`) is used.
    """
    if tag:
        root = Path(predictions_root) if predictions_root else RETRAINED_PREDICTIONS_ROOT
        return root / str(tag) / f"seed{seed}" / RETRAINED_FILE_PATTERN.format(parc=parc, conn=conn)
    directory = Path(kraken_predictions_dir) if kraken_predictions_dir else LEGACY_PREDICTIONS_DIR
    return directory / LEGACY_FILE_PATTERN.format(seed=seed, parc=parc, conn=conn)


class KrakencoderPrecomputed(nn.Module):
    """
    Conn2Conn model that serves Krakencoder predictions computed outside the training loop.

    At construction time the model loads the per-seed inference `.mat` file for the run's source modality, takes the
    prediction for the run's **target** modality (so both `SC -> FC` and `FC -> SC` work: every inference file holds
    all input -> output types), and stores it with the matching ground truth from `base`. `predict_split()` slices
    both by the partition indices in `base`, so no forward pass is run.

    Two prediction sources:
      - registry name `Krakencoder_precomputed` (no `tag`): the cached March 2026 predictions in
        `krakencoder_experimental/example_data/` (`mydata_kraken_seed{seed}_source_{parc}.{SC|FC}.mat`);
      - registry name `Krakencoder` (`tag` set): a retrained run written by `retrain.py` (this package)
        to `results/krakencoder/<tag>/seed{seed}/predictions_source_{parc}.{SC|FC}.mat`.

    The model is wired as closed-form (`learned: false`), so it follows the CrossModalPCA prod-run path; the YAML
    `search_space` is empty, so `Sim.run_tune()` raises immediately (variants are separate retrain tags).

    Args:
        base: `HCP_Base` instance; `shuffle_seed`, `parcellation`, `source` and `target` pick the file and key.
        kraken_predictions_dir: legacy cache directory (only used without `tag`).
        tag: retrained-run tag under `predictions_root`.
        predictions_root: root of retrained runs (default `<repo>/results/krakencoder`).
    """

    is_precomputed = True

    def __init__(self, base, kraken_predictions_dir=None, tag=None, predictions_root=None, **kwargs):
        super().__init__()

        seed = base.shuffle_seed
        parc = base.parcellation
        # base.source may be composite (e.g. "SC+SC_r2t"); use the first modality
        conn_type = (
            base.source_modalities[0]
            if hasattr(base, "source_modalities")
            else base.source
        )
        target = base.target
        for role, modality in (("source", conn_type), ("target", target)):
            if modality not in _KRAKEN_FLAVOR_KEY:
                raise ValueError(
                    f"KrakencoderPrecomputed: unsupported {role} modality '{modality}'. "
                    f"Expected one of {list(_KRAKEN_FLAVOR_KEY)}."
                )

        mat_path = krakencoder_prediction_path(seed, parc, conn_type, tag=tag, predictions_root=predictions_root,
                                               kraken_predictions_dir=kraken_predictions_dir)
        if not mat_path.exists():
            hint = (f"Run `python -m models.architectures.krakencoder.retrain --seed {seed}` with the config whose tag is {tag} first." if tag
                    else "Cached predictions come from the local Krakencoder copy (krakencoder_experimental/).")
            raise FileNotFoundError(f"KrakencoderPrecomputed: inference file not found:\n  {mat_path}\n{hint}")

        mat = loadmat(str(mat_path), simplify_cells=True)
        input_key  = _KRAKEN_FLAVOR_KEY[conn_type].format(parc=parc)
        output_key = _KRAKEN_FLAVOR_KEY[target].format(parc=parc)

        try:
            preds_all = np.array(
                mat["predicted_alltypes"][input_key][output_key], dtype=np.float32
            )
        except (KeyError, TypeError) as exc:
            available = {k: list(v) for k, v in mat.get("predicted_alltypes", {}).items()}
            raise KeyError(
                f"KrakencoderPrecomputed: could not find "
                f"predicted_alltypes['{input_key}']['{output_key}'] "
                f"in {mat_path.name}.  Available input -> output keys: {available}"
            ) from exc

        targets_all = base.fc_upper_triangles if target == "FC" else base.sc_upper_triangles
        self._preds_all     = preds_all                                    # [N_subj, N_edges]
        self._targets_all   = np.asarray(targets_all, dtype=np.float32)    # [N_subj, N_edges]
        self._split_indices = base.trainvaltest_partition_indices
        self.prediction_file = str(mat_path)

        n_subj = self._preds_all.shape[0]
        n_base = self._targets_all.shape[0]
        if n_subj != n_base:
            raise ValueError(
                f"KrakencoderPrecomputed: prediction array has {n_subj} subjects but "
                f"base has {n_base}.  Ensure the inference file matches participants.tsv."
            )

        print(
            f"KrakencoderPrecomputed: loaded  seed={seed}  parc={parc}  "
            f"{conn_type}->{target}  tag={tag}  file={mat_path}  shape={preds_all.shape}"
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
