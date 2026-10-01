# Composite-loss protocol

**Status:** running (spec v2 E1) · **Owner:** agent:modeling · **Grid:** [`grid.yml`](grid.yml) (v1)

## Question

With MSE fixed at weight 1, how do `varmatch`, `correye` and `neidist` shape training dynamics and trade test
`demeaned_pearson` against `avg_rank`? The protocol is run per model on identical combination ids, so the loss
landscape can be compared across configurations within a model and across models.

## Layout

```
composite_loss/
├── composite_loss.md        # this file: protocol + cross-model comparison
├── grid.yml                 # protocol grid (versioned; never edited after release)
├── checks/                  # loss-code checks for the protocol (E1.1)
└── <model>/                 # one folder per model instance
    ├── <model>.md           # instance write-up
    ├── config.yml           # model, Stage 1 config, consensus, reference scales, stop conditions
    └── stage1/              # MSE-only tune config + packed launcher
```

## Protocol v1

- **Stage 1:** MSE-only tune of the model's own search space (24 trials × seeds 0–4, packed), with `varmatch`,
  `correye`, `neidist` logged as monitor-only terms.
- **Consensus + reference scales:** one consensus config (categorical by majority, `lr` / `l2_reg` by geometric
  median; accepted within 1 SE of the per-seed bests); fixed scales `c_t = s_t / s_mse` measured on it.
  If the check fails, `protocol.py rebaseline` (`launch_rebaseline.sh`) retrains each seed's own best config on its
  seed (epochs capped at what ASHA let the trial train) and re-checks against those retrained values, which removes the
  best-of-24 selection bias. If it still fails, the consensus is accepted as a recorded fallback
  (`state.yml` `consensus_check.basis: fallback_accept`, echoed in `tables/consensus_note.txt`) so Stage 2 runs.
- **Stage 2:** the 16 grid combinations (8-cell on/off factorial at w = 0.5 + 8 dose-response points) × 5 seeds,
  Stage 1 hyperparameters fixed, `loss_normalize: none`, all four terms logged every epoch.
- **Fixed across models:** `batch_size` 64 (D4), seeds, selection on `val_demeaned_r`, output schema
  (`seed_records.csv`, `epoch_history.csv`), W&B tags (`<model>`, `loss_grid:v1`, `combo:<id>`).
- **Scope:** the grid maps the landscape with each model's Stage 1 hyperparameters held fixed; magnitude tuning for a
  final model is spec v2 E3.

## Loss terms (math)

Transcribed from `models/train/loss.py` (`CompositeLoss`, `compute_*_loss`); evaluation metrics from
`models/eval/metrics.py`. Every loss term is computed on one training **batch**, not the full dataset.

**Notation.** A batch has $B$ subjects (here $B = 64$) and each connectome is vectorized to $E$ edges.
$Y \in \mathbb{R}^{B \times E}$ are the targets (FC) and $\hat Y$ the predictions; $y_i, \hat y_i \in \mathbb{R}^E$
are the rows for subject $i$. $\bar y \in \mathbb{R}^E$ is the training-set mean target. For a row $x$,
$\tilde x = x - \tfrac{1}{E}\sum_e x_e$ is the row centered over its edges, and $\varepsilon$ is a small
numerical constant ($10^{-10}$).

### Total loss (protocol v1, `loss_normalize: none`)

$$
\mathcal{L}  =  \mathcal{L}_{\text{mse}}
 +  \sum_{t \in \lbrace \text{varmatch}, \text{correye}, \text{neidist} \rbrace} w_t \frac{\mathcal{L}_t}{c_t},
\qquad
c_t = \frac{\overline{\lvert \mathcal{L}_t \rvert}}{\overline{\mathcal{L}_{\text{mse}}}}
$$

Here $w_t$ is the grid weight (0, 0.1, 0.5 or 1) and $c_t$ is the fixed reference scale (the per-term `scale`
kwarg). Each mean $\overline{\cdot}$ is over all training batches of the trained consensus MSE-only model,
then over seeds 0–4. The scaling means that at the consensus model each active term contributes about
$w_t \mathcal{L}_{\text{mse}}$, so $w_t$ reads as "fraction of the MSE magnitude". The scale is fixed: it never
adapts during training. That differs from `loss_normalize: ema`, which divides by a running mean of
$\lvert \mathcal{L}_t \rvert$.

Measured for `linear_backbone` (in `state.yml`):

| Term | $\overline{\lvert\mathcal{L}_t\rvert}$ | $c_t$ |
|---|---|---|
| mse | 0.0122 | 1 |
| varmatch | 0.966 | 79.3 |
| correye | 52.9 | 4342 |
| neidist | 1.25 | 103 |

### MSE (`mse`)

$$
\mathcal{L}_{\text{mse}} = \frac{1}{BE}\sum_{i=1}^{B}\sum_{e=1}^{E}\left(\hat y_{ie} - y_{ie}\right)^2
$$

### Variance matching (`varmatch`, Krakencoder relative form)

Each edge is centered across the subjects in the batch, then one pooled variance is taken over all
subjects × edges:

$$
\sigma^2(Y) = \frac{1}{BE}\sum_{i,e}\Big(y_{ie} - \tfrac{1}{B}\textstyle\sum_{k} y_{ke}\Big)^2,
\qquad
\mathcal{L}_{\text{varmatch}} = \left(\frac{\sigma^2(Y) - \sigma^2(\hat Y)}{\sigma^2(Y) + \varepsilon}\right)^2
$$

This penalizes the shrinkage toward the mean that MSE produces: the predictions' between-subject spread should
match the targets'. It constrains one scalar per batch, not each edge's variance, and says nothing about whether
the spread points in the right subject-specific direction.

### Correlation-identity (`correye`, Krakencoder)

$C \in \mathbb{R}^{B \times B}$ holds the edge-wise Pearson correlation between every target and every prediction
in the batch (rows = targets):

$$
C_{ij} = \frac{\langle \tilde y_i, \tilde{\hat y}_j \rangle}
{\sqrt{\lVert \tilde y_i \rVert^2 + \varepsilon} \sqrt{\lVert \tilde{\hat y}_j \rVert^2 + \varepsilon}},
\qquad
\mathcal{L}_{\text{correye}} = \lVert C - I_B \rVert_F
= \sqrt{\sum_i (C_{ii} - 1)^2 + \sum_{i \ne j} C_{ij}^2}
$$

The term asks for own-subject correlations of 1 and cross-subject correlations of 0. Note that it is the
Frobenius norm, not its square, and that the correlations are on the **raw** (not de-meaned) connectomes.

*Reading of the E1 result (interpretation, not a separate test):* raw FC rows are dominated by the shared group
connectome, so every entry of $C$ sits near the raw Pearson $r \approx 0.83$. Plugging in diagonal 0.84 and
off-diagonal 0.83 gives $\sqrt{64 \cdot 0.16^2 + 4032 \cdot 0.83^2} \approx 52.7$, which matches the measured
mean of 52.9. So the off-diagonal sum (4032 terms) carries almost all of the loss. Lowering it means pushing
predictions away from the group mean, which MSE resists. That is consistent with `correye` having no measurable
effect at any weight. The `grad_cosine` figure tests this directly: its gradient should be near-parallel or
anti-parallel to MSE's on `W_mid`.

### Nearest-neighbor distance (`neidist`, Krakencoder; `margin: null`)

$D \in \mathbb{R}^{B \times B}$ holds the Euclidean distances between targets and predictions:

$$
D_{ij} = \lVert y_i - \hat y_j \rVert_2,
\qquad
d_{\text{self}} = \frac{1}{B}\sum_i D_{ii},
$$

$$
d_{\text{other}} = \frac{1}{B}\sum_{i}\frac{1}{2}\Big(\min_{k \ne i} D_{ki} + \min_{k \ne i} D_{ik}\Big),
\qquad
\mathcal{L}_{\text{neidist}} = d_{\text{self}} - d_{\text{other}}
$$

In the code, $k \ne i$ is enforced by adding $\max D$ to the diagonal before taking the minimum. The first min is
the nearest *other target* to prediction $i$; the second is the nearest *other prediction* to target $i$. The
loss is **signed**: it is negative once every prediction is closer to its own target than to its closest
competitor. That is why the reference scale uses $\lvert\mathcal{L}_t\rvert$. The measured mean is +1.25, so at
the consensus model own-subject distances are still larger than nearest-competitor distances on average. Only
the hardest competitor per subject receives gradient, so this is a margin-style identifiability objective, much
closer to `avg_rank` than any other term. That matches the E1 finding that `neidist` is the only term that
raises `avg_rank`. With `margin` $= m$ set, the code instead uses
$d_{\text{self}} + \max(0, m - d_{\text{other}})$ (unused in v1).

### Other terms in `loss.py` (not in grid v1)

`demeaned_mse` ($\mathcal{L}_{\text{dm-mse}}$), `pairwise_corr` ($\mathcal{L}_{\text{pcorr}}$) and `kld`:

$$
\mathcal{L}_{\text{dm-mse}} = \frac{1}{BE}\sum_{i,e}\big((\hat y_{ie} - \bar y_e) - (y_{ie} - \bar y_e)\big)^2
= \mathcal{L}_{\text{mse}}
$$

(identical in value and gradient, since $\bar y$ cancels; kept for logging parity),

$$
\mathcal{L}_{\text{pcorr}} = \left\lvert \frac{1}{B(B-1)}\sum_{i \ne j} \operatorname{corr}(\hat y_i, \hat y_j) - \rho^\ast \right\rvert,
\quad \rho^\ast = 0.4 \text{ (Sarwar et al.)},
\qquad
\mathcal{L}_{\text{kld}} = -\frac{1}{2B}\sum_{i}\sum_{k}\left(1 + \log\sigma_{ik}^2 - \mu_{ik}^2 - \sigma_{ik}^2\right)
$$

### Evaluation metrics (full test split, not batches)

`demeaned_pearson`:

$$
r^{\text{dm}} = \frac{1}{N}\sum_i
\frac{\langle \hat y_i - \bar y, y_i - \bar y \rangle}{\lVert \hat y_i - \bar y \rVert \lVert y_i - \bar y \rVert + \varepsilon}
$$

`avg_rank`:

$$
\text{avg rank} = 1 - \frac{1}{N}\sum_i \frac{\operatorname{rank}_i}{N},
\qquad
\operatorname{rank}_i = \left\lvert \lbrace j \ne i : C_{ij} > C_{ii} \rbrace \right\rvert
$$

Here $C$ is the raw-FC Pearson matrix as in `correye`, but over all $N$ test subjects. $\operatorname{rank}_i$ is
the 0-based position of the true prediction when target $i$'s row is sorted in descending order, so 1 is a
perfect match and about 0.5 is chance. `demeaned_pearson` rewards reproducing each subject's *deviation* from the
group mean edge by edge. `avg_rank` only rewards being *closer* to the right subject than to the others. That is
the trade-off E1 maps: `neidist` buys rank by sacrificing edge-level deviation fidelity.

## Instances

| Instance | Role | Status |
|---|---|---|
| [`linear_backbone`](linear_backbone/linear_backbone.md) | primary | grid v1 complete (80 runs); report in `linear_backbone/figures/` |
| [`pca_pls_learnable`](pca_pls_learnable/pca_pls_learnable.md) | replicability | Stage 1 seeds 0–2 done; 3–4 paused |

## Cross-model comparison

Pending: concatenates the instances' `seed_records.csv` / `epoch_history.csv` on `combo_id`.
