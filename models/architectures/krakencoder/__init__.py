"""Krakencoder baseline: vendored upstream code (vendor/), retrain wrapper (retrain.py), prediction loader (precomputed.py).

    python -m models.architectures.krakencoder.retrain --config models/configs/deep/Krakencoder.yml --seed 0
"""

from .precomputed import KrakencoderPrecomputed, krakencoder_prediction_path

__all__ = ["KrakencoderPrecomputed", "krakencoder_prediction_path"]
