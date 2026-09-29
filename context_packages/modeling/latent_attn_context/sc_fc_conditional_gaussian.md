# Conditional Gaussian SC→FC Baseline (Expanded)

## Purpose
A **linear probabilistic baseline** for predicting FC from SC using a joint Gaussian model and its conditional mean.

This spec extends the earlier draft by making explicit:
- the mathematical equivalence to multivariate linear regression / ridge regression
- practical regularization strategies for both **PCA space** and **raw edge space**
- implementation details, tensor shapes, and diagnostics

---

## 1. Data representations

For each subject, define:

- SC edge vector:
  `x_sc ∈ R^E`
- FC edge vector:
  `x_fc ∈ R^E`

where `E = r(r-1)/2` if using vectorized upper-triangle edges for an `r × r` connectome.

Two possible fitting domains:

### A. PCA latent domain (recommended first)
- `z_sc ∈ R^k`
- `z_fc ∈ R^k`

obtained from:
- `z_sc = x_sc B_sc`
- `z_fc = x_fc B_fc`

where:
- `B_sc ∈ R^(E × k)`
- `B_fc ∈ R^(E × k)`

### B. Raw edge domain
- use `x_sc` and `x_fc` directly, without PCA compression

### Optional covariates
Additional tabular covariates can be concatenated directly into the conditioning
variable without changing the closed-form nature of the model.

Examples:
- continuous standardized features such as age or FreeSurfer volumes
- one-hot demographic variables such as sex or race/ethnicity

Then the conditioning variable becomes:

- PCA space:
  `u = [ z_sc ; c ]`
- raw edge space:
  `u = [ x_sc ; c ]`

where `c ∈ R^d` is the covariate vector.

---

## 2. Joint Gaussian model

### PCA-space form
Let

`z = [ u ; z_fc ] ∈ R^(k+d+k)` when covariates are used,
or `z = [ z_sc ; z_fc ] ∈ R^(2k)` otherwise

Assume

`z ~ N( [μ_sc ; μ_fc], [[Σ_11, Σ_12], [Σ_21, Σ_22]] )`

with:
- `μ_u ∈ R^(k+d)` or `μ_sc ∈ R^k`
- `μ_fc ∈ R^k`
- `Σ_11 ∈ R^((k+d) × (k+d))` or `R^(k × k)`
- `Σ_22 ∈ R^(k × k)`
- `Σ_21 ∈ R^(k × (k+d))` or `R^(k × k)`

### Raw-edge form
Let

`x = [ u ; x_fc ] ∈ R^(E+d+E)` when covariates are used,
or `x = [ x_sc ; x_fc ] ∈ R^(2E)` otherwise

Assume

`x ~ N( [μ_sc ; μ_fc], [[Σ_11, Σ_12], [Σ_21, Σ_22]] )`

with:
- `μ_u ∈ R^(E+d)` or `μ_sc ∈ R^E`
- `μ_fc ∈ R^E`
- `Σ_11 ∈ R^((E+d) × (E+d))` or `R^(E × E)`
- `Σ_22 ∈ R^(E × E)`
- `Σ_21 ∈ R^(E × (E+d))` or `R^(E × E)`

---

## 3. Conditional predictor

The conditional mean gives the prediction.

### PCA-space predictor
`z_fc_hat = μ_fc + Σ_21 Σ_11^{-1} (u - μ_u)`

### Raw-edge predictor
`x_fc_hat = μ_fc + Σ_21 Σ_11^{-1} (u - μ_u)`

This is the core model.

### Practical raw-edge note
In raw edge space, explicitly forming and inverting
`Σ_11 ∈ R^(E × E)` is usually not practical because `E` is very large.

With ridge / Tikhonov regularization, the same predictor is most naturally
implemented through the **dual (sample-space) form of multivariate ridge
regression**:

- primal form:
  `B = (X_sc^T X_sc + λ I)^(-1) X_sc^T X_fc`

- dual/sample-space form:
  `B = X_sc^T (X_sc X_sc^T + λ I)^(-1) X_fc`

and prediction becomes:

`x_fc_hat = μ_fc + (u - μ_u) U^T (U U^T + λ I)^(-1) (X_fc - μ_fc)`

So in raw edge space, it is often best to interpret the implementation as:

> the raw-edge conditional Gaussian mean realized through the dual/sample-space
> form of regularized multivariate ridge regression.

---

## 4. Conditional uncertainty

A useful distinction from ordinary regression is that the Gaussian model also gives uncertainty.

### PCA-space conditional covariance
`Σ_fc|sc = Σ_22 - Σ_21 Σ_11^{-1} Σ_12`

### Raw-edge conditional covariance
Same formula, replacing PCA-space matrices with raw-edge matrices.

This can be reported as:
- full covariance
- per-component / per-edge marginal variances
- summary uncertainty diagnostics

---

## 5. Equivalence to linear regression

Define

`W = Σ_21 Σ_11^{-1}`

`b = μ_fc - W μ_u`

Then the conditional mean can be written as:

### PCA space
`z_fc_hat = W u + b`

### Raw edge space
`x_fc_hat = W u + b`

So this model is equivalent to a multivariate linear map, but derived from a **joint distributional assumption**.

The main added value of the Gaussian framing is:
- natural uncertainty estimate
- explicit covariance modeling
- clearer regularization choices at the covariance level

---

## 6. Training / fitting procedure

Given training data:
- `U ∈ R^(N × (k+d))` or `Z_sc ∈ R^(N × k)`
- `Z_fc ∈ R^(N × k)`

or raw:
- `U ∈ R^(N × (E+d))` or `X_sc ∈ R^(N × E)`
- `X_fc ∈ R^(N × E)`

Estimate:

### Means
- `μ_u` or `μ_sc`
- `μ_fc`

### Joint covariance
Concatenate along features:

- PCA: `Z = [U, Z_fc] ∈ R^(N × (k+d+k))`
- raw: `X = [U, X_fc] ∈ R^(N × (E+d+E))`

Then compute empirical covariance and extract:
- `Σ_11`
- `Σ_21`

Prediction requires only these blocks plus the means.

---

## 7. Regularization strategies

This is the most important practical issue.

### 7.1 Ridge / Tikhonov regularization
Replace the inverse by

`Σ_11^{-1} → (Σ_11 + λ I)^{-1}`

This is the simplest and strongest first regularization strategy.

#### Why useful
- stabilizes inversion
- damps noisy directions
- works well in both PCA and raw spaces

#### Hyperparameter
- `λ > 0`
- tune on validation data

### 7.1b Dual ridge view for raw edges
For the raw-edge model, ridge is not just a numerical convenience; it is
effectively the computational form of the estimator.

When `E` is very large, the regularized conditional mean is best implemented
through the sample-space Gram matrix:

`K = X_sc X_sc^T`

and predictor:

`x_fc_hat = μ_fc + k(x_sc)^T (K + λ I)^(-1) (X_fc - μ_fc)`

where:
- `k(x_sc) = X_sc (x_sc - μ_sc)` under the linear kernel
- `K + λ I` is only `N × N`

This makes the raw-edge estimator:
- computationally feasible
- explicitly regularized
- equivalent to linear multivariate ridge regression in dual form
- naturally extendable to direct covariate conditioning by replacing `X_sc`
  with the concatenated conditioning matrix `U = [X_sc ; C]`

---

### 7.2 Shrinkage covariance + ridge
Estimate covariance with shrinkage:

`Σ_hat = (1 - α) S + α D`

where:
- `S` = empirical covariance
- `D` = diagonal target
- `α` = shrinkage intensity

Then use:
`(Σ_11_hat + λ I)^{-1}`

#### Good estimators
- Ledoit–Wolf
- OAS

#### Why useful
- especially strong in `p >> n`
- more stable than raw empirical covariance

---

### 7.3 Graphical lasso / sparse precision (optional)
Estimate a sparse inverse covariance / precision matrix.

This is more structurally motivated in connectomics, but:
- more brittle
- more hyperparameter-sensitive
- less natural as a first baseline than ridge

Recommendation:
- treat as a later structured baseline, not the first one

---

## 8. PCA-space vs raw-edge-space considerations

### PCA-space fitting
#### Advantages
- low-dimensional
- `Σ_11` is small and often near-diagonal
- numerically stable
- easy to validate
- strong baseline for comparison to attention models

#### Notes
If PCA is fit on the training set:
- PCA scores are mean-centered
- score covariance is diagonal on the fit set
- cross-modal coupling is carried primarily by `Σ_21`

This is not a problem; it just means the predictive structure lives in the cross-covariance block.

---

### Raw-edge-space fitting
#### Advantages
- avoids information loss from PCA
- more direct edge-level model
- can test whether PCA compression is the main limiting factor

#### Problems
- `E` is large (`p >> n`)
- empirical covariance is singular or unstable
- regularization is essential
- explicit feature-space covariance inversion is usually not feasible

#### Most promising regularizers here
1. ridge / Tikhonov
2. shrinkage covariance + ridge

These are the two most practical first choices.

#### Implementation note
In practice, the first raw-edge implementation should usually use the
**dual ridge** form rather than explicitly estimating the full raw-edge
covariance blocks. That keeps the model faithful to the same linear
conditional-mean idea while remaining tractable.

---

## 9. Optional normalization

If working in PCA space, it may help to normalize PCA scores before fitting.

### Option A: z-score each component
Normalize both:
- `z_sc`
- `z_fc`

using training statistics.

### Option B: whiten PCA scores
Scale by component standard deviation / square root eigenvalue.

#### Why
- balances component scales
- prevents early high-variance PCs from dominating covariance estimation
- can simplify the predictor

If training on normalized latent scores:
- prediction is made in normalized latent space
- then unnormalize before PCA decode

---

## 10. Decode step (PCA-space model only)

If fitting in PCA space and wanting edge-space output:

1. predict `z_fc_hat`
2. unnormalize if needed
3. decode:

`x_fc_hat = z_fc_hat B_fc^T`

This yields predicted FC in edge space.

---

## 11. Diagnostics

### 11.1 Inspect covariance structure
Useful plots:
- full joint covariance
- full joint correlation
- cross-covariance block `Σ_21`
- cross-correlation block

These help determine whether:
- within-modality structure is as expected
- cross-modal signal is strong, weak, diffuse, or low-rank

### 11.2 Condition number
Check conditioning of `Σ_11`.

High condition number implies stronger regularization needed.

### 11.3 Performance metrics
Evaluate:
- latent-space MSE
- edge-space MSE
- Pearson `r`
- subnetwork metrics if relevant

### 11.4 Ablations
Recommended ablations:
- PCA-space vs raw-edge-space
- ridge vs shrinkage+ridget
- varying `k`
- normalized vs unnormalized PCA scores

---

## 12. Practical implementation outline

### PCA-space version
Inputs:
- `Z_sc: (N, k)`
- `Z_fc: (N, k)`

Steps:
1. compute means
2. concatenate `[Z_sc, Z_fc]`
3. estimate covariance
4. extract `Σ_11`, `Σ_21`
5. compute regularized inverse
6. predict via conditional mean

### Raw-edge version
Inputs:
- `X_sc: (N, E)`
- `X_fc: (N, E)`

Same conceptual steps, but in practice:
- covariance dimension is much larger
- use ridge and/or shrinkage
- prefer the dual/sample-space ridge form for implementation

---

## 13. Minimal NumPy pseudocode

```python
import numpy as np
from sklearn.covariance import LedoitWolf

def fit_conditional_gaussian(Z_sc, Z_fc, lam=1e-3, use_shrinkage=False):
    mu_sc = Z_sc.mean(axis=0)
    mu_fc = Z_fc.mean(axis=0)

    Z = np.concatenate([Z_sc, Z_fc], axis=1)

    if use_shrinkage:
        Sigma = LedoitWolf().fit(Z).covariance_
    else:
        Sigma = np.cov(Z, rowvar=False)

    k = Z_sc.shape[1]
    Sigma_11 = Sigma[:k, :k]
    Sigma_21 = Sigma[k:, :k]

    W = Sigma_21 @ np.linalg.inv(Sigma_11 + lam * np.eye(k))
    b = mu_fc - W @ mu_sc

    return {
        "mu_sc": mu_sc,
        "mu_fc": mu_fc,
        "Sigma": Sigma,
        "Sigma_11": Sigma_11,
        "Sigma_21": Sigma_21,
        "W": W,
        "b": b,
    }

def predict_conditional_gaussian(Z_sc_test, model):
    return Z_sc_test @ model["W"].T + model["b"]
```

### Raw-edge dual ridge pseudocode

```python
import numpy as np

def fit_raw_edge_dual_ridge(X_sc, X_fc, lam=1e-3):
    mu_sc = X_sc.mean(axis=0)
    mu_fc = X_fc.mean(axis=0)

    Xc = X_sc - mu_sc
    Yc = X_fc - mu_fc

    K = Xc @ Xc.T
    A = np.linalg.solve(K + lam * np.eye(K.shape[0]), Yc)

    return {
        "mu_sc": mu_sc,
        "mu_fc": mu_fc,
        "Xc": Xc,
        "A": A,
    }

def predict_raw_edge_dual_ridge(x_sc_test, model):
    xc = x_sc_test - model["mu_sc"]
    kx = xc @ model["Xc"].T
    return kx @ model["A"] + model["mu_fc"]
```

---

## 14. Key model interpretation

This model says:

> FC can be predicted as the conditional expectation of a jointly Gaussian SC–FC representation.

In practice, that becomes a regularized linear map:
- simple
- interpretable
- uncertainty-aware
- good as a baseline

It is especially useful to determine whether more complex models are gaining from:
- genuine nonlinear structure
- or simply from better regularization / latent encoding

---

## 15. Recommended baseline sequence

### First
PCA-space conditional Gaussian with ridge

### Second
PCA-space conditional Gaussian with shrinkage + ridge

### Third
Raw-edge conditional Gaussian with ridge

### Fourth
Raw-edge conditional Gaussian with shrinkage + ridge

Graphical lasso can be deferred until later.

---

## One-line summary
A closed-form linear SC→FC baseline:
`target = cross-covariance × regularized inverse source covariance × centered input + mean`,
with optional uncertainty from the conditional covariance.

For raw edge space, the same idea is most practically realized as the
dual/sample-space form of multivariate ridge regression.
