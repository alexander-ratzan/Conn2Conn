# Metrics glossary — Phase 0 closed-form evaluation

What each of the 6 metrics in the cross-step panel means, when to trust each,
and the formula gotcha I caught at end-of-day 2026-05-26.

Companion to [`session_log_2026-05-26.md`](session_log_2026-05-26.md).

---

## The 6 metrics

All metrics are computed on the test split (n=195 subjects, Glasser parcellation,
64,620 upper-triangular edges per connectome) by `compute_basic_regression_metrics`
in [`../models/eval/metrics.py`](../models/eval/metrics.py).

### 1. `mse` — Mean Squared Error

```
mse = mean((y_pred - y_true) ** 2)         # averaged over subjects and edges
```

Total reconstruction error. Lower is better. Mixes group-pattern error and
individual-pattern error. **Almost useless for our pivot question** because the
"easy" part (group mean) dominates and individual differences contribute only a
tiny fraction. Reported for completeness — don't read into small differences.

### 2. `r2` — Coefficient of Determination

```
r2 = 1 - sum((y_true - y_pred)**2) / sum((y_true - y_true.mean())**2)
```

Variance explained vs the target's own mean. **Almost always negative in our
table** because we're predicting onto a high-dim target (64,620 edges) with
small N=195 and limited model capacity. Negative r² doesn't mean "worse than
nothing"; it means "worse than predicting the test-set's own mean." For
residual targets, r² is meaningless (residual mean is 0 by construction, so the
denominator measures total residual variance, not informative spread).

**Trust this metric for:** confirming a model isn't catastrophically broken
(very-negative r² like -2.0). **Don't trust for:** anything subtle.

### 3. `pearson` — Raw per-subject Pearson correlation

```
pearson_per_subject_i = Pearson(y_pred[i, :], y_true[i, :])    # across edges
pearson = mean over subjects
```

This is the "edge-pattern correlation" — for each subject, how well does the
predicted edge vector correlate with the true edge vector? Then averaged across
subjects.

**Problem**: this is *dominated by the group mean*. Every connectome has the
same large-scale structure (strong intra-hemispheric edges, weak inter-hemispheric,
etc.), so even a model that predicts the train mean for everyone gets very high
raw pearson (~0.8). The number looks impressive but is mostly "are you in the
right neighborhood of connectome space" — which is easy. Subject-specific signal
is hidden under the group pattern.

This is exactly what trapped the SC→FC literature at "raw pearson 0.82–0.84
across all architectures." Everyone hit the wall because they were measuring
group-pattern reconstruction, not individual signal.

**Trust this metric for:** sanity check that your model isn't outputting noise
(if raw pearson is 0.05, something's wrong). **Don't trust for:** comparing
models or measuring individual signal.

### 4. `demeaned_pearson` — Subject-specific signal (THE HEADLINE)

The fix to the problem above: subtract the **train mean** from both prediction
and truth *before* correlating. What's left is the deviation from the group
pattern — i.e., individual subject signal.

**The project's convention** (in `compute_demeaned_pearson_r`):

```
yp_dm = y_pred - target_train_mean         # subtract train mean from prediction
yt_dm = y_true - target_train_mean         # subtract train mean from truth
demeaned_per_subject_i = (yp_dm[i] · yt_dm[i]) / (||yp_dm[i]|| · ||yt_dm[i]||)
                         # cosine similarity of demeaned vectors
demeaned_pearson = mean over subjects
```

This is **cosine similarity of (y − train_mean) vectors**, NOT Pearson on
demeaned data. The distinction matters (see the gotcha section below).

**Why it's the headline:** if the model just predicts the train mean for every
subject, demeaned_pearson is undefined / zero. If the model captures any
subject-specific deviation from the population mean, demeaned_pearson is positive.
This isolates the "is this subject *this* subject" signal from the easier
"is this a connectome" signal.

**Trust this metric for:** comparing models or directions on individual signal.
This is the right thing to read for the pivot question.

**Caveat:** it's still a vector-similarity metric. Two predictions can have the
same demeaned_pearson but rank very differently in identifiability (see top1_acc
below). Demeaned_pearson asks "are the right deviations there?"; identifiability
asks "can we ID the right subject?"

### 5. `top1_acc` — Top-1 identification accuracy

For each test subject, build a similarity matrix between predicted edges and
all true edges. Top-1 accuracy = fraction of subjects whose predicted edges
have the highest similarity to their *own* true edges (vs everyone else's).

```
corr_matrix[i, j] = Pearson(true_edges[i], predicted_edges[j])
top1_acc = mean over i of (argmax_j corr_matrix[i, j] == i)
```

In our table the gallery is the 195 test subjects (so chance = 1/195 ≈ 0.0051).

**Trust this metric for:** "can the model fingerprint individuals?" This is the
strict identifiability test. Above ~5× chance is statistically meaningful.

**Don't trust for:** small differences. Top-1 is **discrete** — every correct
match is +0.005. At n=195, a 1-subject change (say 14 hits vs 15 hits) gives a
0.5% difference that's pure sampling noise. The fact that our PLS top1=0.082 vs
BR top1=0.056 = ratio 1.5× *could be entirely* a difference of 5 correct matches
between two random samples.

This is why the bootstrap is the main next-session task — to put CIs on the top1
numbers and see if the differences are real.

### 6. `avg_rank` — Average rank percentile (SOFT identifiability)

Like top-1, but instead of "did the right subject rank #1?" we ask "what
percentile rank did the right subject get?"

```
For each query subject i:
  rank_i = position of i in argsort(corr_matrix[i, :], descending=True)
  rank_percentile_i = 1 - rank_i / N
avg_rank = mean over subjects of rank_percentile_i
```

If the right subject ranks #1 → rank_percentile = 1.0. If they rank #N (last)
→ rank_percentile = 1/N ≈ 0. Chance is 0.5.

**Why it's the most stable metric in our results:** top-1 is binary (hit or
miss), so a single subject flipping from rank-2 to rank-1 changes top1_acc by
0.005 but barely moves avg_rank (changes one number from 0.989 → 0.994).
avg_rank averages over the full ranking, smoothing out the brittle "did the
right subject squeak past the second-place subject?" noise.

**Trust this metric for:** stable cross-model and cross-direction comparisons of
identifiability. In our results avg_rank ratios sit at 1.20–1.25× across every
condition tested — the most reliable indicator that the FC→SC > SC→FC asymmetry
is real.

**Don't trust for:** intuitive "fraction of correct IDs" interpretation. Above
0.85 is "very identifiable"; below 0.6 is "barely above chance."

---

## How to read a row of the cross-step table

For each predictor, the 6 metrics tell a layered story:

| metric | reads as |
|---|---|
| `mse` ↓ | "edges are in the right magnitude range" — almost always similar across models |
| `r2` | "did we beat predicting the test mean?" — often negative, ignore small differences |
| `pearson` ↑ | "the group connectome pattern is correct" — basically a sanity check |
| `demeaned_pearson` ↑ | "individual deviations from the group pattern are correctly predicted" — **headline** |
| `top1_acc` ↑ | "the model correctly ID'd the subject" — strict but noisy |
| `avg_rank` ↑ | "the right subject ranks high in the gallery" — most stable id metric |

Headline triad for the pivot question: **demeaned_pearson + top1_acc + avg_rank**.
The other three (mse, r2, pearson) are background sanity, not informative.

---

## The formula gotcha I caught (helper bug, 2026-05-26)

I built a wrapper `_full_panel_eval` in cell 20 of the closed-form notebook to
compute all 6 metrics in one call. My initial implementation had a subtle bug
in the `demeaned_pearson` calculation.

### The bug

I computed the demeaned corr matrix as:
```
cc_dm = compute_corr_matrix(y_true - train_mean, y_pred - train_mean)
```

`compute_corr_matrix` is row-Pearson, which **re-demeans each row to its own
mean** before computing the correlation. So my "demeaned_pearson" was actually
"Pearson of (y − train_mean) vectors, with each row further demeaned to its own
mean." Two layers of demeaning.

### The project's convention

`compute_demeaned_pearson_r` does only one layer:
```
yp_dm = y_pred - train_mean
yt_dm = y_true - train_mean
r_i = (yp_dm[i] · yt_dm[i]) / (||yp_dm[i]|| · ||yt_dm[i]||)   # NO further row-mean subtraction
```

This is **cosine similarity** of demeaned vectors. The row-mean of (y − train_mean)
is NOT generally zero — a subject whose connectome is uniformly above the train
mean has a positive row mean after demeaning. The project's convention preserves
this DC offset as informative; my buggy version threw it away.

### How big the discrepancy is

It depends on how non-zero the post-demeaning row means are:

| prediction | row means of (y − train_mean) | buggy demeaned | true demeaned | diff |
|---|---|---|---|---|
| residual predictions (3.5/3.6 main) | ~0 by construction | matches exactly | matches | 0% |
| raw `brain-vol → SC` | large per-subject offsets | 0.1577 | 0.1670 | -5.9% |
| raw `FC → SC` sanity | moderate offsets | 0.1269 | 0.1322 | -4.2% |
| raw `SC → FC` sanity | small offsets | 0.0848 | 0.0849 | -0.1% |

**Fix applied** in cell 20 at end of session — re-run the affected cells to refresh.

**None of the other 5 metrics were affected** — pearson, top1_acc, avg_rank,
mse, r2 all use the raw corr matrix, which the bug didn't touch.

### Lesson for the future

When reusing `compute_corr_matrix` for "Pearson on transformed data," remember
it always row-re-centers. For the project's "demeaned cosine" convention, you
must compute the cosine directly without going through `compute_corr_matrix`.

---

## When metrics disagree (the actual interesting question)

In our results, metrics disagree on the anatomy-vs-FC comparison:

| predictor → SC | demeaned | top1_acc | avg_rank |
|---|---|---|---|
| brain-vol → SC | **0.167** | 0.103 | **0.879** |
| FC raw → SC | 0.132 | **0.154** | 0.853 |

Brain-vol wins demeaned and avg_rank; FC wins top1.

**Why?** Brain-vol predicts edge *values* well in aggregate (good demeaned, good
avg_rank because predictions are in the right neighborhood). But the predictions
have a "biological feel" — they cluster around the group mean modulated by brain
size — and don't *uniquely* fingerprint subjects. FC predictions are more
unique-per-subject (good top1) even though they're less accurate on edge values
(slightly lower demeaned).

This is a real methodological insight: **demeaned_pearson and top1_acc are
measuring different things**. demeaned-r asks "does the prediction's deviation
from the mean match the truth's deviation"; top1 asks "is the prediction
distinctively this subject?" A predictor can do well on the first and badly on
the second if it produces *generic* "biological-looking" predictions.

For our pivot question, **both views matter**: edge-value accuracy (demeaned)
and individual fingerprinting (top1/avg_rank). The recalibrated headline keeps
both.
