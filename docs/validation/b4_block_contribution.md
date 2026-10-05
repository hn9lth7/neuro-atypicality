# B4 — Block Contribution Analysis

**Status:** CLOSED (descriptive)  
**Cohort:** Canonical / discovery (`nai_v10.csv`), $n = 41$ ($\mathrm{TD} = 39$, $\mathrm{ASD} = 2$)  
**Model:** Frozen NAI v1.0 — no refit, $\lambda = 0.10$, equal weights unchanged  

**Inputs:** `results/normative/nai_v10.csv`  
**Outputs:**  
- `results/validation/b4_block_contribution.csv`  
- `results/validation/b4_block_summary.json`  
- `results/validation/b4_concentrated_subjects.csv`  

---

## 1. Purpose

B4 describes how the four frozen block distances

$$D_{\mathrm{SE}},\quad D_{\mathrm{C}},\quad D_{\mathrm{G}},\quad D_{\mathrm{D}}$$

relate to each other and to

$$\mathrm{NAI} = rac{1}{4} igl(D_{\mathrm{SE}} + D_{\mathrm{C}} + D_{\mathrm{G}} + D_{\mathrm{D}} igr)$$

on the canonical cohort.  
It does **not** re-estimate the normative model, change weights, or constitute external validation.

Normalized contribution:

$$C_B = rac{D_B}{D_{\mathrm{SE}} + D_{\mathrm{C}} + D_{\mathrm{G}} + D_{\mathrm{D}}},\qquad \sum_B C_B = 1$$

Formula integrity check: $\max|\mathrm{NAI} - \mathrm{mean}(D_B)| = 4.44 	imes 10^{-16}$.

---

## 2. Observed Results

### 2.1 Mean Distances and Contributions

Across the canonical cohort, mean block distances were:

$$ ar{D}_{\mathrm{SE}} = 1.453,\quad  ar{D}_{\mathrm{C}} = 1.904,\quad  ar{D}_{\mathrm{G}} = 2.083,\quad  ar{D}_{\mathrm{D}} = 3.393$$

Mean NAI:

$$\overline{\mathrm{NAI}} = 2.208$$

| Block | Mean $C_B$ |
| :--- | :---: |
| **SE** | 0.162 |
| **C** | 0.212 |
| **G** | 0.232 |
| **D** | 0.394 |
| **Sum** | **1.000** |

TD mean $C_{\mathrm{D}}  pprox 0.392$; ASD ($n = 2$) mean $C_{\mathrm{D}}  pprox 0.430$ — **descriptive only**.

### 2.2 Spearman Correlation of Each $D_B$ with NAI

| Block | Spearman $\rho$ |
| :--- | :---: |
| $D_{\mathrm{SE}}$ | 0.721 |
| $D_{\mathrm{C}}$ | 0.913 |
| $D_{\mathrm{G}}$ | 0.930 |
| $D_{\mathrm{D}}$ | 0.811 |

These associations are partly induced by the definition of NAI as the average of the four distances.

### 2.3 Spearman Correlation Among Block Distances

| Pair | Spearman $\rho$ |
| :--- | :---: |
| $D_{\mathrm{SE}}, D_{\mathrm{C}}$ | 0.556 |
| $D_{\mathrm{SE}}, D_{\mathrm{G}}$ | 0.530 |
| $D_{\mathrm{SE}}, D_{\mathrm{D}}$ | 0.464 |
| $D_{\mathrm{C}}, D_{\mathrm{G}}$ | **0.917** |
| $D_{\mathrm{C}}, D_{\mathrm{D}}$ | 0.622 |
| $D_{\mathrm{G}}, D_{\mathrm{D}}$ | 0.697 |

The strongest rank association is between connectivity and graph distances
($\rho = 0.917$). These are descriptive within-cohort associations and do
not establish causal or biological relationships.

### 2.4 Dominant Block and Concentration

| Metric | Value |
| :--- | :---: |
| **Dominant block = D** | **41 / 41** ($\mathrm{TD}\ 39/39$, $\mathrm{ASD}\ 2/2$) |
| **$\max C_B \ge 0.40$** | **19 / 41** ($ pprox 46\%$) |

---

## 3. Descriptive Interpretation

1. **Composition of Equal-Weight NAI:**  
   The largest average share of the sum of distances is dynamic ($D$), followed by graph ($G$), connectivity ($C$), and spectral/entropy ($SE$). This describes the **current** frozen composite; it is not a reason to change weights.

2. **Universal D-Dominance:**  
   $D_{\mathrm{D}}$ is the largest contribution for all 41 subjects. Its mean distance ($3.393$) is also higher than $SE$ ($1.453$), $C$ ($1.904$), and $G$ ($2.083$). This pattern is compatible with differences in block dimensionality, covariance structure, and regularization. It does **not** establish that dynamic features are intrinsically more informative or biologically more important.

3. **C–G Coupling:**  
   High rank correlation between $D_{\mathrm{C}}$ and $D_{\mathrm{G}}$ is consistent with graph metrics being derived from the same connectivity matrices; the blocks are related, not orthogonal latent factors.

4. **Moderate Concentration:**  
   About 46% of subjects have $\max C_B \ge 0.40$; the rest are more multi-block in contribution space.

5. **ASD Cohort:**  
   With $n_{\mathrm{ASD}} = 2$, group means are recorded only as description, not as inference.

---

## 4. Mathematical Caveats

- $C_B$ inherits relative scales of Mahalanobis distances.  
- $\mathrm{Corr}(D_B, \mathrm{NAI})$ is partly structural.  
- Equal weights remain frozen.  
- Discovery/canonical cohort only; not EV-A.

---

## 5. What Cannot Be Concluded

- Dynamic features as ASD biomarker or primary mechanism.  
- ASD-vs-TD effects.  
- External generalizability.  
- Weight retuning based on mean $C_B$.

---

## 6. Status Summary

| Item | Status |
| :--- | :---: |
| **Frozen distances only** | Yes |
| **Model / $\lambda$ / weights unchanged** | Yes |
| **Block means, contributions, inter-block $
ho$ documented** | Yes |
| **External validation** | No |
| **B4 Status** | **CLOSED** |

---

## 7. Next Step — B5 Heterogeneity

Study subject-level profiles $\mathbf{C}_i = (C_{\mathrm{SE}}, C_{\mathrm{C}}, C_{\mathrm{G}}, C_{\mathrm{D}})$: dispersion of $C_B$, distribution of max contribution, multi-block vs. concentrated subjects, and whether high-NAI subjects share one contribution pattern — still without changing NAI v1.0.