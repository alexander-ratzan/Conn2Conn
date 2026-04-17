import numpy as np
import torch
import torch.nn as nn

from models.architectures.utils import _build_mlp, compute_reg_loss, get_modality_data


class CrossModalVAE(nn.Module):
    """
    Non-linear cross-modal VAE: MLP encoder -> Gaussian latent (mu, logvar) -> MLP decoder.
    Trained with reconstruction (MSE) + beta * KLD to standard Gaussian prior.
    Optional PCA projection on input (default off).

    Args:
        base: Dataset base object (e.g., HCP_Base) providing source/target means and loadings.
        latent_dim: Latent dimension (e.g. 64 or 128).
        hidden_dims: List of hidden layer sizes for encoder and decoder (e.g. [512, 256]).
        use_pca_encoder: If True, project input to PCA space before encoder. Default False.
        n_pca_components_encoder: Number of PCA components for encoder when use_pca_encoder=True.
        use_pca_decoder: If True, have decoder output PCA scores and reconstruct via target PCA. Default False.
        n_pca_components_decoder: Number of PCA components for decoder when use_pca_decoder=True.
        dropout: Dropout probability. Default 0.1.
        l1_l2_tuple: Tuple of (l1_reg, l2_reg) weights. Default (0.0, 0.0).
        device: torch device.
        **kwargs: Ignored (lr, epochs, loss_fn, beta passed to Lightning module).

    Note:
        For backward compatibility, the deprecated arguments `use_pca` and `n_pca_components`
        are still accepted and mapped to encoder PCA when the new encoder arguments are not set.
    """
    def __init__(self, base, latent_dim=64, hidden_dims=(512, 256),
                 use_pca=False, n_pca_components=256,
                 use_pca_encoder=None, n_pca_components_encoder=None,
                 use_pca_decoder=False, n_pca_components_decoder=256,
                 dropout=0.1, l1_l2_tuple=(0.0, 0.0), device=None, **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Backwards compatibility: if new encoder args not provided, fall back to old ones
        if use_pca_encoder is None:
            use_pca_encoder = use_pca
        if n_pca_components_encoder is None:
            n_pca_components_encoder = n_pca_components
        
        self.use_pca_encoder = bool(use_pca_encoder)
        self.n_pca_components_encoder = int(n_pca_components_encoder)
        self.use_pca_decoder = bool(use_pca_decoder)
        self.n_pca_components_decoder = int(n_pca_components_decoder)
        self.dropout_p = dropout
        
        self.l1_reg, self.l2_reg = l1_l2_tuple
        self.l1_l2_tuple = l1_l2_tuple

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Get modality data using helper function
        data = get_modality_data(base, device=device, include_scores=False)
        source_mean = data['source_mean']
        source_loadings = data['source_loadings']
        target_mean = data['target_mean']
        target_loadings = data['target_loadings']

        d_source = source_loadings.shape[0]
        d_target = target_mean.shape[0]

        # Buffers: means and optional PCA loadings
        self.register_buffer('source_mean_', torch.tensor(source_mean, dtype=torch.float32, device=device))
        self.register_buffer('target_mean_', torch.tensor(target_mean, dtype=torch.float32, device=device))
        # Encoder PCA
        if self.use_pca_encoder:
            enc_loadings_k = source_loadings[:, :self.n_pca_components_encoder]
            self.register_buffer('source_loadings_enc', torch.tensor(enc_loadings_k, dtype=torch.float32, device=device))
            encoder_in_dim = self.n_pca_components_encoder
            print(f"CrossModalVAE: use_pca_encoder=True, encoder input dim={encoder_in_dim}")
        else:
            self.register_buffer('source_loadings_enc', None)
            encoder_in_dim = d_source
            print(f"CrossModalVAE: use_pca_encoder=False, encoder input dim={encoder_in_dim}")

        # Decoder PCA
        if self.use_pca_decoder:
            dec_loadings_k = target_loadings[:, :self.n_pca_components_decoder]
            self.register_buffer('target_loadings_dec', torch.tensor(dec_loadings_k, dtype=torch.float32, device=device))
            decoder_out_dim = self.n_pca_components_decoder
            print(f"CrossModalVAE: use_pca_decoder=True, decoder output dim={decoder_out_dim}")
        else:
            self.register_buffer('target_loadings_dec', None)
            decoder_out_dim = d_target
            print(f"CrossModalVAE: use_pca_decoder=False, decoder output dim={decoder_out_dim}")

        # Encoder: input -> hidden_dims -> latent_dim*2 (mu and logvar)
        hidden_dims = list(hidden_dims)
        self.encoder = _build_mlp(encoder_in_dim, hidden_dims, latent_dim * 2, dropout_p=dropout)

        # Decoder: latent_dim -> reversed(hidden_dims) -> decoder_out_dim
        decoder_hidden = list(reversed(hidden_dims))
        self.decoder = _build_mlp(latent_dim, decoder_hidden, decoder_out_dim, dropout_p=dropout)

        print(f"CrossModalVAE: latent_dim={latent_dim}, hidden_dims={hidden_dims}, d_target={d_target}")

    def forward(self, x):
        # Use parameter device so model works after being moved by Lightning (self.device can be stale)
        device = self.source_mean_.device
        x = x.to(device).to(torch.float32)
        x = x - self.source_mean_

        if self.use_pca_encoder and self.source_loadings_enc is not None:
            x = torch.matmul(x, self.source_loadings_enc)

        h = self.encoder(x)
        mu = h[:, :self.latent_dim]
        logvar = h[:, self.latent_dim:]
        logvar = torch.clamp(logvar, -20.0, 2.0)

        if self.training:
            eps = torch.randn_like(mu, device=mu.device, dtype=mu.dtype)
            std = torch.exp(0.5 * logvar)
            z = mu + eps * std
        else:
            z = mu # deterministic as compared to VAE standard stochastic sampling from prior

        y_latent = self.decoder(z)
        if self.use_pca_decoder and self.target_loadings_dec is not None:
            y_pred = torch.matmul(y_latent, self.target_loadings_dec.t()) + self.target_mean_
        else:
            y_pred = y_latent + self.target_mean_
        
        return (y_pred, mu, logvar)

    def get_reg_loss(self):
        """Compute L1/L2 regularization on learnable parameters using global helper."""
        return compute_reg_loss(self.parameters(), self.l1_l2_tuple)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Krakencoder precomputed dummy model
# ---------------------------------------------------------------------------

# Input-type key fragments used by krakencoder inference outputs
_KRAKEN_INPUT_KEY = {
    "SC": "SCifod2act_{parc}_volnorm",
    "FC": "FCcorr_{parc}_hpf",
}
_KRAKEN_OUTPUT_KEY = "FCcorr_{parc}_hpf"
