# SC→FC Single-Head Attention Translation Model

## Purpose
A simple **SC-only translation model** that predicts subject-level FC latent coefficients from SC latent coefficients.

This model uses:
- **PCA** for the subject-specific latent signal
- **PLS-derived loadings** as a component identity / alignment barcode
- **single-head self-attention** as a lightweight translation module
- **linear readout** to FC latent coefficients

This is **not** a transformer stack. It is a single attention operation plus a linear decoder.

---

## Modeling assumptions

### PCA role
PCA provides the **subject-specific scalar signal**:
- `c_sc[i]` tells us how much SC PCA component `i` is expressed in this subject.

### PLS role
PLS provides the **component identity embedding**:
- `e[i]` is a fixed loading-based barcode for PCA component `i`
- it softly encodes how this SC component relates to FC-aligned latent structure
- it is **not** the primary signal source

### Token interpretation
For component `i`, the token is:

`token_sc[i] = [ c_sc[i] ; e[i] ]`

So each token means:

> how much this SC component is present + what that component means in FC-aligned latent space

---

## Symbols and tensor shapes

### Subject-level latent variables
- `k`: number of PCA components retained
- `l`: PLS barcode dimension
- `d`: attention query/key dimension
- `d_v`: attention value dimension

### Inputs
- `c_sc`: shape `(B, k)`
- `c_fc_target`: shape `(B, k)`
- `e`: shape `(k, l)`  
  Fixed across subjects. Derived from PLS loadings / barcode construction.

### Tokenized input
- `tokens_sc`: shape `(B, k, l + 1)`

Construct:
- first channel = subject-specific scalar `c_sc`
- remaining `l` channels = fixed `e`

---

## Forward pass

### 1. Build SC tokens
For batch item `b` and component `i`:

`tokens_sc[b, i] = concat( c_sc[b, i], e[i] )`

Shape:
- `tokens_sc`: `(B, k, l + 1)`

### 2. Linear projections
Use separate learned matrices:

- `W_Q_sc`: shape `((l + 1), d)`
- `W_K_sc`: shape `((l + 1), d)`
- `W_V_sc`: shape `((l + 1), d_v)`

Compute:

- `Q = tokens_sc @ W_Q_sc` → `(B, k, d)`
- `K = tokens_sc @ W_K_sc` → `(B, k, d)`
- `V = tokens_sc @ W_V_sc` → `(B, k, d_v)`

### 3. Single-head attention
Attention weights:

`A = softmax( Q K^T / sqrt(d) )`

Shape:
- `A`: `(B, k, k)`

Contextualized features:

`Z = A V`

Shape:
- `Z`: `(B, k, d_v)`

Interpretation:
- each row `Z[:, i, :]` is a contextualized feature vector for FC component `i`
- query identity is anchored by token `i`, especially by its PLS barcode `e[i]`

### 4. Linear readout
Use shared linear decoder:

- `W_z`: shape `(d_v, 1)`
- optional `b_z`: shape `(1,)`

Compute:

`c_fc_hat = Z @ W_z + b_z`

Shape:
- raw output: `(B, k, 1)`
- squeeze last dim → `(B, k)`

### 5. Optional PCA decode
If reconstructing full FC edge vector:

- `B_fc`: shape `(k, E)` if using row-basis convention
- then:

`x_fc_hat = c_fc_hat @ B_fc`

Shape:
- `x_fc_hat`: `(B, E)`

---

## Loss

### Latent-space loss
Default:

`L = mse(c_fc_hat, c_fc_target)`

Possible alternatives:
- Pearson correlation loss
- combined Pearson + MSE
- weighted latent component loss

### Recommended initial training target
Train first in **latent FC space**:
- more stable
- lower dimensional
- easier to debug

---

## Minimal PyTorch module outline

```python
class SCFCTranslationAttention(nn.Module):
    def __init__(self, k, l, d, d_v):
        super().__init__()
        self.W_Q_sc = nn.Linear(l + 1, d, bias=False)
        self.W_K_sc = nn.Linear(l + 1, d, bias=False)
        self.W_V_sc = nn.Linear(l + 1, d_v, bias=False)
        self.W_z = nn.Linear(d_v, 1, bias=True)

    def forward(self, c_sc, e):
        B, k = c_sc.shape
        e_batch = e.unsqueeze(0).expand(B, -1, -1)            # (B, k, l)
        tokens = torch.cat([c_sc.unsqueeze(-1), e_batch], -1) # (B, k, l+1)

        Q = self.W_Q_sc(tokens)  # (B, k, d)
        K = self.W_K_sc(tokens)  # (B, k, d)
        V = self.W_V_sc(tokens)  # (B, k, d_v)

        A = torch.softmax((Q @ K.transpose(-2, -1)) / math.sqrt(Q.shape[-1]), dim=-1)
        Z = A @ V  # (B, k, d_v)

        c_fc_hat = self.W_z(Z).squeeze(-1)  # (B, k)
        return c_fc_hat, A, Z
```

---

## Key design decisions / ablations

### 1. Token construction
Base:
- `[c_sc[i] ; e[i]]`

Ablations:
- scalar only
- barcode only
- scalar + normalized barcode
- scalar + projected barcode

### 2. PLS barcode definition
Possible choices for `e[i]`:
- row of SC-side PLS loading matrix
- row of FC-side PLS loading matrix
- concatenation of both
- reduced projection of loadings
- learned embedding initialized from PLS

### 3. Attention dimensions
Tune:
- `d`
- `d_v`

Likely small ranges:
- `d ∈ {4, 8, 16, 32}`
- `d_v ∈ {4, 8, 16, 32}`

### 4. Readout
Current:
- shared linear readout `W_z`

Possible later variants:
- small MLP head
- residual linear SC→FC bypass
- component-specific bias terms

### 5. Normalization
Possible choices:
- z-score PCA scores across train set
- normalize PLS barcodes
- layernorm on token matrix before attention
- no normalization for first pass

### 6. Attention regularization
Possible later:
- dropout on attention weights
- entropy regularization
- diagonal penalty to reduce identity collapse

### 7. SC dropout
Useful later if attention collapses:
- randomly drop / corrupt some SC scalar entries during training
- encourages inter-component dependence

---

## Expected behavior
This model should learn:

- which SC components matter for each FC component
- how SC components should be mixed to form FC component estimates
- soft component-to-component relations guided by PLS barcodes

It should be viewed as:

> SC PCA signal + PLS component barcode → single-head latent translation → FC coefficients

---

## Recommended first implementation
Start with:
- latent-space supervision only
- one single attention block
- no masking
- no residual bypass
- no normalization beyond what is already in your PCA pipeline

Return:
- `c_fc_hat`
- `A`
- `Z`

so attention maps and latent features can be inspected during debugging.
