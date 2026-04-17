"""General model runtime utilities.

These helpers are shared by training, evaluation, and top-level experiment code.
Architecture-specific construction helpers live in `models.architectures.utils`.
"""

import torch

from models.architectures.utils import get_model_input


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def get_batch_cov(batch):
    """Return covariate features from a dataloader batch when present."""
    return batch.get("cov")


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def predict_from_loader(model, data_loader, device=None):
    """Generate predictions and targets from a model over a dataloader."""
    model.eval()
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            try:
                device = next(model.buffers()).device
            except StopIteration:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in data_loader:
            x = get_model_input(batch)
            y = batch["y"].to(device)

            kwargs = {}
            if getattr(model, "uses_cov", False):
                kwargs["cov"] = get_batch_cov(batch)
                if getattr(model, "use_target_scores_in_projector", False) and "y" in batch:
                    kwargs["y"] = batch["y"]
            if getattr(model, "uses_node_features", False) and "node_features" in batch:
                kwargs["node_features"] = batch["node_features"]

            out = model(x, **kwargs) if kwargs else model(x)
            preds = out[0] if isinstance(out, tuple) else out
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())

    return torch.cat(all_preds, dim=0), torch.cat(all_targets, dim=0)
