"""Joint multidirectional translation in a shared code space (spec v4 E2 / I1).

- `CodeSpace`: per-view train mean and block scale, a pooled PCA basis for the FC views and one basis per other view
  (SC), encode / decode between edges and k-dimensional codes. Nothing here sees val / test rows.
- `TokenBackbone`: residual MLP over a code, with optional source / target view tokens and flow-time embedding. Used as
  the per-pair MLP (no tokens), the joint regression (`predict_regression`, residual form z + f) and the paired-bridge
  velocity field (`predict_flow`).
- `integrate`: fixed-step Euler / RK4 for dz/dtau = v(z, tau) on [0, 1].
"""
import math

import numpy as np
import torch
import torch.nn as nn


def _top_right_singular_vectors(X, k):
    """Top-k right singular vectors of X (rows << cols) through the row Gram matrix, float64. Returns (E, k)."""
    evals, U = np.linalg.eigh(X @ X.T)
    order = np.argsort(evals)[::-1][:k]
    s = np.sqrt(np.clip(evals[order], 1e-300, None))
    return (X.T @ U[:, order]) / s


class CodeSpace:
    """Edge vectors <-> codes. z = W_m^T (x - mu_m) / s_m and x_hat = mu_m + s_m W_m z.

    Args:
        train: {view: (n_train, E) array}, train subjects only.
        pooled_views: views that share one basis (the FC views); every other view gets its own basis.
        k_max: number of basis vectors kept (codes are sliced to any k <= k_max).
        scaling: 'frob' (block divided by its total norm: equal total variance per view), 'sv1' (first singular
            value) or 'none'.
        per_view: if True, every view gets its own basis (the per-condition ablation).
    """

    def __init__(self, train, pooled_views, k_max, scaling="frob", per_view=False):
        self.views = list(train)
        self.pooled_views = [] if per_view else [v for v in pooled_views if v in train]
        self.k_max = int(k_max)
        self.scaling = scaling
        self.mean, self.scale, self.basis_of = {}, {}, {}
        centered = {}
        for v, X in train.items():
            X = np.asarray(X, dtype=np.float64)
            self.mean[v] = X.mean(0)
            Xc = X - self.mean[v]
            if scaling == "frob":
                self.scale[v] = float(np.linalg.norm(Xc))
            elif scaling == "sv1":
                self.scale[v] = float(np.sqrt(np.linalg.eigvalsh(Xc @ Xc.T)[-1]))
            elif scaling == "none":
                self.scale[v] = 1.0
            else:
                raise ValueError(f"unknown scaling {scaling!r}")
            centered[v] = Xc / self.scale[v]
        self.bases = {}
        if self.pooled_views:
            self.bases["pooled"] = _top_right_singular_vectors(
                np.vstack([centered[v] for v in self.pooled_views]), self.k_max).astype(np.float32)
            for v in self.pooled_views:
                self.basis_of[v] = "pooled"
        for v in self.views:
            if v not in self.basis_of:
                self.bases[v] = _top_right_singular_vectors(centered[v], self.k_max).astype(np.float32)
                self.basis_of[v] = v
        for v in self.views:
            self.mean[v] = self.mean[v].astype(np.float32)

    def add_view(self, view, X_train, like):
        """Attach a view that did not shape the bases: its own train mean and scale, `like`'s basis."""
        Xc = np.asarray(X_train, dtype=np.float64)
        mu = Xc.mean(0)
        Xc = Xc - mu
        self.scale[view] = {"frob": lambda: float(np.linalg.norm(Xc)),
                            "sv1": lambda: float(np.sqrt(np.linalg.eigvalsh(Xc @ Xc.T)[-1])),
                            "none": lambda: 1.0}[self.scaling]()
        self.mean[view] = mu.astype(np.float32)
        self.basis_of[view] = self.basis_of[like]
        self.views.append(view)

    def basis(self, view):
        return self.bases[self.basis_of[view]]

    def encode(self, view, X, k=None):
        k = self.k_max if k is None else k
        return ((np.asarray(X, dtype=np.float32) - self.mean[view]) / self.scale[view]) @ self.basis(view)[:, :k]

    def decode(self, view, Z):
        Z = np.asarray(Z, dtype=np.float32)
        return self.mean[view] + self.scale[view] * (Z @ self.basis(view)[:, :Z.shape[1]].T)

    def decode_torch(self, view, Z, device):
        """Decode on `device`; basis and mean tensors are cached per (view, k, device)."""
        key = (view, Z.shape[1], str(device))
        cache = self.__dict__.setdefault("_torch_cache", {})
        if key not in cache:
            cache[key] = (torch.as_tensor(self.mean[view], device=device),
                          torch.as_tensor(self.basis(view)[:, :Z.shape[1]].T.copy(), device=device))
        mu, Wt = cache[key]
        return mu + self.scale[view] * (Z @ Wt)

    def state(self):
        """Arrays to save (np.savez) and reload with `from_state`."""
        out = {"views": np.array(self.views), "pooled_views": np.array(self.pooled_views),
               "k_max": self.k_max, "scaling": self.scaling}
        for v in self.views:
            out[f"mean__{v}"] = self.mean[v]
            out[f"scale__{v}"] = self.scale[v]
            out[f"basis_of__{v}"] = self.basis_of[v]
        for name, W in self.bases.items():
            out[f"basis__{name}"] = W
        return out

    @classmethod
    def from_state(cls, st):
        cs = cls.__new__(cls)
        cs.views = [str(v) for v in st["views"]]
        cs.pooled_views = [str(v) for v in st["pooled_views"]]
        cs.k_max = int(st["k_max"])
        cs.scaling = str(st["scaling"])
        cs.mean = {v: np.asarray(st[f"mean__{v}"], dtype=np.float32) for v in cs.views}
        cs.scale = {v: float(st[f"scale__{v}"]) for v in cs.views}
        cs.basis_of = {v: str(st[f"basis_of__{v}"]) for v in cs.views}
        cs.bases = {name: np.asarray(st[f"basis__{name}"], dtype=np.float32) for name in set(cs.basis_of.values())}
        return cs


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        half = dim // 2
        self.register_buffer("freqs", torch.exp(-math.log(10000.0) * torch.arange(half) / half))

    def forward(self, tau):
        a = tau[:, None] * 1000.0 * self.freqs[None, :]
        return torch.cat([torch.sin(a), torch.cos(a)], dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(width, width))

    def forward(self, h):
        return h + self.net(h)


class TokenBackbone(nn.Module):
    """f(z, [e_src(m), e_tgt(m')], [gamma(tau)], [c]) -> R^k: input projection, `depth` residual blocks, linear output.

    The output layer starts at zero, so at initialisation the regression predicts the source code and the velocity
    field is zero. Optional: `cond_dim` > 0 adds a conditioning vector c (e.g. SC codes) to the input, alongside the
    state z (spec v4 E2.1: z is the ridge prediction or the flow state, c the subject's SC codes).
    """

    def __init__(self, k, n_views=0, token_dim=32, width=512, depth=3, dropout=0.1, use_tokens=True,
                 use_time=False, time_dim=64, cond_dim=0):
        super().__init__()
        self.use_tokens = bool(use_tokens)
        self.use_time = bool(use_time)
        self.cond_dim = int(cond_dim)
        in_dim = k + self.cond_dim
        if self.use_tokens:
            self.src_token = nn.Embedding(n_views, token_dim)
            self.tgt_token = nn.Embedding(n_views, token_dim)
            in_dim += 2 * token_dim
        if self.use_time:
            self.time_embed = SinusoidalEmbedding(time_dim)
            in_dim += time_dim
        self.inp = nn.Linear(in_dim, width)
        self.blocks = nn.Sequential(*[ResidualBlock(width, dropout) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, k))
        nn.init.zeros_(self.out[1].weight)
        nn.init.zeros_(self.out[1].bias)

    def forward(self, z, src=None, tgt=None, tau=None, cond=None):
        parts = [z] if cond is None else [z, cond]
        if self.use_tokens:
            parts += [self.src_token(src), self.tgt_token(tgt)]
        if self.use_time:
            parts.append(self.time_embed(tau))
        return self.out(self.blocks(self.inp(torch.cat(parts, dim=1))))

    def predict_regression(self, z, src=None, tgt=None, cond=None):
        """Residual regression (A2 / A3; E2.1 corrector): z_hat = z + f(z, tokens, cond)."""
        return z + self(z, src, tgt, cond=cond)

    def predict_flow(self, z, src=None, tgt=None, n_steps=16, method="rk4", cond=None):
        """Integrate the velocity field (A4; E2.1 bridge and flow) from z at tau = 0 to tau = 1."""
        return integrate(lambda x, t: self(x, src, tgt, t, cond=cond), z, n_steps=n_steps, method=method)


def integrate(velocity, z0, n_steps=16, method="rk4"):
    """Fixed-step solution of dz/dtau = velocity(z, tau) from tau = 0 to 1. `velocity(z, tau)` takes tau of shape (n,)."""
    z = z0
    h = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((z.shape[0],), i * h, dtype=z.dtype, device=z.device)
        if method == "euler":
            z = z + h * velocity(z, t)
        elif method == "rk4":
            k1 = velocity(z, t)
            k2 = velocity(z + 0.5 * h * k1, t + 0.5 * h)
            k3 = velocity(z + 0.5 * h * k2, t + 0.5 * h)
            k4 = velocity(z + h * k3, t + h)
            z = z + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise ValueError(f"unknown method {method!r}")
    return z
