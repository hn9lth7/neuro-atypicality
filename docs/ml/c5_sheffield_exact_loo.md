# C5 — Exact LOO vs In-Sample (Sheffield)

## 1. Design

- **In-sample:** Normative fit on all Sheffield CTRL ($n = 18$), score all 46 subjects.
- **Exact LOO:** CTRL $i$ fitted on $\mathrm{CTRL} \setminus \{i\}$; ASD $i$ fitted on all CTRL.

---

## 2. Results

| Mode | AUC | Mean NAI ASD | Mean NAI CTRL |
| :--- | :---: | :---: | :---: |
| **In-sample** | 0.849 | 2.96 | 2.04 |
| **Exact LOO** | 0.492 | 2.96 | 3.21 |

---

## 3. Interpretation

In-sample AUC (0.849) matches the previously reported C4 figure and is optimistic: CTRL subjects are scored against a reference that includes themselves, which artificially shrinks their Mahalanobis distances.

Under subject-independent leave-one-out (LOO) within Sheffield, mean CTRL NAI rises (2.04 $\rightarrow$ 3.21) and AUC collapses to chance (0.492). This does not refute the 54-D feature pipeline; it demonstrates that discrimination on this cohort was not stable when the normative CTRL reference was estimated without the scored control subject.

---

## 4. Implication for Diagnostic Claims

Sheffield-internal NAI scoring with $n_{\mathrm{CTRL}} = 18$ is **not sufficient evidence** for an out-of-sample diagnostic model. A proper external test requires a normative model fitted on an independent discovery TD cohort (e.g., `ds006780`), then applied to Sheffield without refitting on Sheffield CTRL.

---

## 5. Summary Table

| Mode | AUC | Mean NAI ASD | Mean NAI CTRL | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **In-sample** | 0.849 | 2.96 | 2.04 | Matches prior C4 figure (optimistic) |
| **Exact LOO** | 0.492 | 2.96 | 3.21 | Collapses to chance; see `c4_sheffield_final.md` |