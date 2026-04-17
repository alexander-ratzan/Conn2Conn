# Latent vs. Raw-Space Loss for Fixed PCA Decoding

For the current Conn2Conn PCA-decoder setup, optimizing mean squared error in raw FC space is equivalent to optimizing squared Euclidean loss in the target PCA latent space, provided that the target PCA decoder is fixed.

## Setup

Let

$$
B \in \mathbb{R}^{d \times k}
$$

denote the fixed target PCA loading matrix, with orthonormal columns:

$$
B^\top B = I_k .
$$

Let

$$
\mu \in \mathbb{R}^d
$$

be the fixed target mean, and let

$$
z \in \mathbb{R}^k
$$

be the predicted target latent code.

Let the true target PCA score be

$$
c \in \mathbb{R}^k,
$$

and let the true FC target be

$$
y \in \mathbb{R}^d.
$$

The decoder is

$$
\hat{y} = \mu + B z .
$$

The true target can be decomposed into a component inside the retained PCA subspace and a residual outside that subspace:

$$
y = \mu + B c + r,
$$

where

$$
r \in \mathbb{R}^d
$$

is orthogonal to the span of $B$, so that

$$
B^\top r = 0.
$$

## Raw-space loss

Consider the squared reconstruction loss:

$$
\|\hat{y} - y\|_2^2.
$$

Substituting the expressions above,

$$
\|\hat{y} - y\|_2^2
= \|(\mu + B z) - (\mu + B c + r)\|_2^2
= \|B(z-c) - r\|_2^2 .
$$

Expanding the norm,

$$
\|B(z-c) - r\|_2^2
= \|B(z-c)\|_2^2 + \|r\|_2^2 - 2\, r^\top B(z-c).
$$

Since $r$ is orthogonal to the PCA subspace,

$$
r^\top B = 0,
$$

so the cross-term vanishes:

$$
\|B(z-c) - r\|_2^2
= \|B(z-c)\|_2^2 + \|r\|_2^2.
$$

Now use orthonormality of the PCA loading matrix:

$$
\|B(z-c)\|_2^2
= (z-c)^\top B^\top B (z-c)
= (z-c)^\top (z-c)
= \|z-c\|_2^2.
$$

Therefore,

$$
\|\hat{y} - y\|_2^2
= \|z-c\|_2^2 + \|r\|_2^2.
$$

The term

$$
\|r\|_2^2
$$

does not depend on the model parameters. Hence minimizing raw-space reconstruction error is equivalent to minimizing latent-space squared error:

$$
\arg\min_z \|\hat{y} - y\|_2^2
=
\arg\min_z \|z-c\|_2^2.
$$

## Gradient view

The same conclusion appears from the gradient with respect to $z$:

$$
\nabla_z \|Bz - (y-\mu)\|_2^2
= 2 B^\top \bigl(Bz - (y-\mu)\bigr).
$$

Using

$$
y - \mu = Bc + r,
$$

we get

$$
\nabla_z \|Bz - (y-\mu)\|_2^2
= 2 B^\top \bigl(Bz - Bc - r\bigr)
= 2(B^\top B z - B^\top B c - B^\top r).
$$

Since $B^\top B = I$ and $B^\top r = 0$,

$$
\nabla_z \|Bz - (y-\mu)\|_2^2
= 2(z-c),
$$

which is exactly the gradient of

$$
\|z-c\|_2^2.
$$

## Practical implication for Conn2Conn

For models such as:

- `CrossModal_PCA_PLS`
- `CrossModal_PCA_PLS_learnable`
- `CrossModal_PCA_PLS_FSResidual`

if the target PCA decoder is fixed, then training with

$$
\mathcal{L}_{\text{raw}} = \|\hat{y} - y\|_2^2
$$

or with

$$
\mathcal{L}_{\text{latent}} = \|z-c\|_2^2
$$

defines the same optimization problem up to the additive constant $\|r\|_2^2$.

So if the modeling goal is to recover the target FC PCA score, then a latent-space loss is fully justified.

## Why latent-space loss may still be preferable

Even though the objectives are equivalent in this setting, latent-space loss can still be more convenient in practice because it is:

- cheaper to compute
- directly aligned with the desired target quantity $c$
- easier to interpret when inspecting latent branches such as `z_base`, `z_fs_resid`, and `z_target`

## Important caveats

This equivalence holds only when:

1. The target decoder is fixed.
2. The target mean $\mu$ is fixed.
3. The true target latent $c$ is defined using the same PCA basis $B$ and mean $\mu$.
4. The PCA basis has orthonormal columns.

The equivalence breaks if:

- the decoder becomes learnable
- the decoder is not orthonormal
- the latent target is defined by a different transform

## Note on `mse` vs. `demeaned_mse`

If the same fixed target mean $\mu$ is subtracted from both prediction and target, then:

$$
\|(\hat{y}-\mu) - (y-\mu)\|_2^2
= \|\hat{y} - y\|_2^2.
$$

Therefore, in the current fixed-mean setup,

$$
\mathcal{L}_{\text{mse}} = \mathcal{L}_{\text{demeaned\_mse}}
$$

numerically. So the current `demeaned_mse` implementation is not a different optimization objective from `mse`; it is simply the same loss written in centered coordinates.
