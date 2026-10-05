# B5 ‚Äî Contribution-Profile Heterogeneity

**Status:** CLOSED (descriptive)  
**Input:** `results/validation/b4_block_contribution.csv` only  
**Model:** Frozen NAI v1.0 unchanged  

---

## 1. Purpose

Describe between-subject variation of 

$$\mathbf{C}_i = (C_{\mathrm{SE}}, C_{\mathrm{C}}, C_{\mathrm{G}}, C_{\mathrm{D}})$$

and whether high-NAI subjects share one contribution pattern.

---

## 2. Results ($n = 41$)

- **Mean contributions:** $C_{\mathrm{SE}} = 0.162$, $C_{\mathrm{C}} = 0.212$, $C_{\mathrm{G}} = 0.232$, $C_{\mathrm{D}} = 0.394$  
- **Maximum $C_B$ ($\max C_B$):** Mean $0.394$ (range $0.292	ext{--}0.528$)  
- **Profile entropy:** $H = -\sum C_B \log C_B$, mean $1.311$ (max possible $\ln 4  pprox 1.386$)  
- **Distribution Bins:** 
  - $\max C_B < 0.30 
ightarrow 2$
  - $[0.30, 0.40) 
ightarrow 20$
  - $[0.40, 0.50) 
ightarrow 17$
  - $\ge 0.50 
ightarrow 2$  
- **High-NAI Subset:** Top quartile ($\mathrm{NAI} \ge 2.612$), $n = 11$:  
  Mean $\mathbf{C}  pprox (0.183, 0.227, 0.239, \mathbf{0.350})$

---

## 3. Interpretation

1. Contribution profiles are mostly multi-block. Extreme single-block concentration is rare.  
2. High-NAI subjects on average show **lower** $C_{\mathrm{D}}$ than the full cohort ($0.350$ vs. $0.394$), consistent with composite elevation across blocks rather than pure dynamic dominance.

---

## 4. Non-Claims

- No ASD-vs-TD inference;
- No weight change;
- Not external validation.

---

## 5. Artifacts

- `b5_subject_profiles.csv`
- `b5_high_nai_profiles.csv`
- `b5_heterogeneity_summary.json`