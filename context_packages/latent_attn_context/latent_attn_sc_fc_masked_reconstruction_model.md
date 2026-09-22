# SC/FC Masked Reconstruction Single-Head Attention Model

## Purpose
A **masked reconstruction variant** that can be used for pretraining.

This model uses:
- SC component tokens as persistent context
- FC component tokens as optional observed context and fixed output/query slots
- single-head self-attention over a fixed `2k` token sequence
- loss only on FC components that are masked from the input

At inference, **SC is the only true context**, but the model still receives **masked FC query tokens**.

This is still a lightweight single-attention model, not a transformer stack.

---

## Core design

### Fixed sequence structure
Always use:

`X = [ SC tokens ; FC tokens ]`

Sequence length:
- `2k` tokens during both pretraining and inference

Interpretation:
- first `k` tokens = SC context tokens
- second `k` tokens = FC query/output tokens

### Why fixed `2k` tokens
This avoids train/test mismatch:
- pretraining and inference use the same token layout
- FC tokens always define where predictions are written

---

## Modeling assumptions

### PCA role
PCA provides the **subject-specific scalar signal**:
- `c_sc[i]` and, when visible, `c_fc[i]`

### PLS role
PLS barcodes provide **component identity / FC-aware similarity structure**:
- `e[i]` is fixed across subjects
- it helps define what component `i` means

### Masking role
Masking removes the subject-specific FC value but preserves:
- component identity
- output position
- the ability to query context

---

## Symbols and tensor shapes

- `k`: number of PCA components
- `l`: PLS barcode dimension
- `d`: query/key dimension
- `d_v`: value dimension

### Inputs
- `c_sc`: `(B, k)`
- `c_fc`: `(B, k)`
- `e`: `(k, l)`

### Mask metadata
- `m_fc`: `(B, k)` with:
  - `0` = observed FC value available
  - `1` = masked FC value hidden and must be predicted

### Shared learned mask scalar
- `theta_mask`: scalar parameter, shared across all masked FC positions

---

## Token definitions

### SC tokens
For component `i`:

`token_sc[i] = [ c_sc[i] ; e[i] ; 0 ]`

Shape per token:
- `(l + 2,)`

The last entry can be interpreted as an SC/FC-type-or-mask bit.
For a first pass, keep SC tokens simple and set final slot to `0`.

### FC tokens
For component `i`:

If observed:
`token_fc[i] = [ c_fc[i] ; e[i] ; 0 ]`

If masked:
`token_fc[i] = [ theta_mask ; e[i] ; 1 ]`

So FC tokens preserve component identity even when masked.

### Token matrices
- `tokens_sc`: `(B, k, l + 2)`
- `tokens_fc`: `(B, k, l + 2)`

Concatenate:
- `tokens = torch.cat([tokens_sc, tokens_fc], dim=1)` → `(B, 2k, l + 2)`

---

## Why separate Q/K/V projections by modality
Recommended:

- `W_Q_sc`, `W_K_sc`, `W_V_sc`
- `W_Q_fc`, `W_K_fc`, `W_V_fc`

Reason:
masked FC tokens should still be useful queries, but ideally weak value contributors.

This is easiest if SC and FC token streams have separate linear projections before concatenation.

---

## Forward pass

### 1. Build SC tokens
Using `c_sc` and `e`

### 2. Build FC tokens
Using:
- observed `c_fc`
- masked placeholder `theta_mask`
- mask flag

### 3. Project each stream separately

#### SC projections
- `W_Q_sc`: `((l + 2), d)`
- `W_K_sc`: `((l + 2), d)`
- `W_V_sc`: `((l + 2), d_v)`

Compute:
- `Q_sc`: `(B, k, d)`
- `K_sc`: `(B, k, d)`
- `V_sc`: `(B, k, d_v)`

#### FC projections
- `W_Q_fc`: `((l + 2), d)`
- `W_K_fc`: `((l + 2), d)`
- `W_V_fc`: `((l + 2), d_v)`

Compute:
- `Q_fc`: `(B, k, d)`
- `K_fc`: `(B, k, d)`
- `V_fc`: `(B, k, d_v)`

### 4. Concatenate streams
- `Q = cat([Q_sc, Q_fc], dim=1)` → `(B, 2k, d)`
- `K = cat([K_sc, K_fc], dim=1)` → `(B, 2k, d)`
- `V = cat([V_sc, V_fc], dim=1)` → `(B, 2k, d_v)`

### 5. Single-head self-attention
`A = softmax( Q K^T / sqrt(d) )`

Shape:
- `A`: `(B, 2k, 2k)`

Contextualized features:
`Z = A V`

Shape:
- `Z`: `(B, 2k, d_v)`

### 6. Readout
Use shared output head:
- `W_o`: `(d_v, 1)`

Compute:
`c_hat_all = Z @ W_o + b_o`

Shape:
- raw: `(B, 2k, 1)`
- squeeze → `(B, 2k)`

Split:
- `c_hat_sc = c_hat_all[:, :k]`
- `c_hat_fc = c_hat_all[:, k:]`

Main target is `c_hat_fc`.

---

## Loss

### Masked FC loss
Only supervise FC positions that were hidden:

`L_mask = mse( c_hat_fc[m_fc == 1], c_fc[m_fc == 1] )`

This is the core objective.

### Important rule
Whatever FC component is masked in the target must also be masked in the FC input token.

---

## Multitask training strategy

### Easy implementation
For each batch, sample an FC mask ratio `rho` in `[0, 1]`.

Then:
- `rho = 0.0` → all FC tokens observed
- `0 < rho < 1` → partial masked reconstruction
- `rho = 1.0` → all FC tokens masked (SC-only context)

Loss always comes from the masked FC positions.

### Practical schedule
Example mixture:
- 20% of batches: `rho = 1.0`
- 60% of batches: `rho ~ Uniform(0.3, 0.7)`
- 20% of batches: `rho ~ Uniform(0.0, 0.2)`

This keeps inference-time behavior in-distribution.

---

## Inference

### Inputs
- SC tokens contain real SC PCA scores
- FC tokens are all masked query tokens

So:

`token_fc[i] = [ theta_mask ; e[i] ; 1 ]`

for all `i`.

### Sequence shape
Still:
- `(B, 2k, l + 2)`

### Forward
Same exact code path as pretraining.

### Output
Take:
- `c_hat_fc = c_hat_all[:, k:]`

This is the full predicted FC latent vector.

### Key interpretation
At inference:
- SC tokens provide context
- masked FC tokens ask: “what should FC component `i` be for this subject?”

---

## Minimal PyTorch module outline

```python
class SCFCMaskedReconAttention(nn.Module):
    def __init__(self, k, l, d, d_v):
        super().__init__()
        self.theta_mask = nn.Parameter(torch.zeros(1))

        self.W_Q_sc = nn.Linear(l + 2, d, bias=False)
        self.W_K_sc = nn.Linear(l + 2, d, bias=False)
        self.W_V_sc = nn.Linear(l + 2, d_v, bias=False)

        self.W_Q_fc = nn.Linear(l + 2, d, bias=False)
        self.W_K_fc = nn.Linear(l + 2, d, bias=False)
        self.W_V_fc = nn.Linear(l + 2, d_v, bias=False)

        self.W_o = nn.Linear(d_v, 1, bias=True)

    def forward(self, c_sc, c_fc, e, m_fc):
        B, k = c_sc.shape
        e_batch = e.unsqueeze(0).expand(B, -1, -1)  # (B, k, l)

        zeros_flag = torch.zeros(B, k, 1, device=c_sc.device, dtype=c_sc.dtype)
        mask_flag = m_fc.unsqueeze(-1).to(c_sc.dtype)  # (B, k, 1)

        tokens_sc = torch.cat([c_sc.unsqueeze(-1), e_batch, zeros_flag], dim=-1)  # (B, k, l+2)

        c_fc_in = torch.where(
            m_fc.bool(),
            self.theta_mask.expand_as(c_fc),
            c_fc
        )
        tokens_fc = torch.cat([c_fc_in.unsqueeze(-1), e_batch, mask_flag], dim=-1)  # (B, k, l+2)

        Q_sc = self.W_Q_sc(tokens_sc)
        K_sc = self.W_K_sc(tokens_sc)
        V_sc = self.W_V_sc(tokens_sc)

        Q_fc = self.W_Q_fc(tokens_fc)
        K_fc = self.W_K_fc(tokens_fc)
        V_fc = self.W_V_fc(tokens_fc)

        Q = torch.cat([Q_sc, Q_fc], dim=1)  # (B, 2k, d)
        K = torch.cat([K_sc, K_fc], dim=1)  # (B, 2k, d)
        V = torch.cat([V_sc, V_fc], dim=1)  # (B, 2k, d_v)

        A = torch.softmax((Q @ K.transpose(-2, -1)) / math.sqrt(Q.shape[-1]), dim=-1)
        Z = A @ V  # (B, 2k, d_v)

        c_hat_all = self.W_o(Z).squeeze(-1)  # (B, 2k)
        c_hat_fc = c_hat_all[:, k:]

        return c_hat_fc, A, Z, c_hat_all
```

---

## Key ablations / future hyperparameters

### Token design
- use `0` vs learned `theta_mask`
- include explicit mask bit vs omit it
- use one type bit for SC/FC vs separate bits

### Projection design
- modality-specific Q/K/V vs shared Q/K/V
- shared readout vs separate FC-only readout

### Attention size
- `d`
- `d_v`

### Training schedule
- FC mask ratio distribution
- probability of all-FC-masked batches
- curriculum vs fixed sampling

### Regularization
- SC scalar dropout
- attention dropout
- diagonal penalty on attention
- entropy regularization

### Optional future extension
- auxiliary masked SC reconstruction
- residual SC→FC linear bypass
- learned refinement of PLS barcodes

---

## Recommended first implementation
Implement first with:
- one attention layer
- modality-specific Q/K/V
- shared learned mask scalar
- explicit FC mask bit
- masked FC loss only
- variable batchwise mask ratio, including some fully masked-FC batches

Return:
- `c_hat_fc`
- `A`
- `Z`
- `c_hat_all`

for debugging and attention inspection.
