# Metabolic Disease Screening — Honest experiments (real cohorts, held-out AUC)

BDI Hackathon 2026 | PhenoInsure Tech | Calvin (KMITL AI Engineering) | env `superaiss6`

Three cleanly-separated, poster-ready experiments. Each follows
**`ข้อมูลจากไหน → ใช้วิธีอะไร → ผลยังไง`** (Data source → Method → Results), and every
ROC figure writes its **data provenance** and **evaluation method** directly on the plot
so a reviewer never has to ask "measured how?".

| Notebook | Disease | Data (real) | Eval | Result |
|---|---|---|---|---|
| [`exp_obesity_nmr_mtbls242.ipynb`](exp_obesity_nmr_mtbls242.ipynb) | Obesity (NMR) | MTBLS242 serum ¹H-NMR, n=177 | stratified 5-fold CV | **ROC-AUC 0.973 ± 0.020** |
| [`exp_t2dm_clinical_nhanes.ipynb`](exp_t2dm_clinical_nhanes.ipynb) | T2DM (clinical) | NHANES 2013–14 Cycle H, n=6,113 | stratified 5-fold CV | **ROC-AUC 0.833 ± 0.014** |
| [`exp_t2dm_nmr_mtbls1.ipynb`](exp_t2dm_nmr_mtbls1.ipynb) | T2DM (NMR) | MTBLS1 urine ¹H-NMR, n=132 (42 subj) | **subject-grouped** 5-fold CV | **ROC-AUC 0.976 ± 0.022** |

Figures live in [`poster_figures/`](poster_figures/) (300 DPI, print-ready) and are also
embedded inline in each notebook. `run_approach6.py` is the script form of Experiment A;
its CSV outputs are in [`results/`](results/).

**Data:** the notebooks read the source cohorts (MTBLS242, MTBLS1, NHANES) by absolute path
from the parent BDI project; the raw data is **not** included in this repo. This repo holds
the analysis notebooks, figures, and documentation.

**📊 For the poster:** [`poster_figures/`](poster_figures/) holds print-ready 300 DPI
figures + [`POSTER.md`](poster_figures/POSTER.md) documenting every experiment and figure
in detail (data source, method, results, captions, honest caveats, judge Q&A prep).

## Run

```bash
conda run -n superaiss6 jupyter nbconvert --to notebook --execute --inplace \
  exp_obesity_nmr_mtbls242.ipynb exp_t2dm_clinical_nhanes.ipynb
# or open them in Jupyter with the superaiss6 kernel
```

## Design principles (why these are defensible)

1. **Real cohorts only.** Obesity uses measured MTBLS242 NMR; T2DM uses the NHANES
   national survey. No simulated spectra. A "leak-free" pipeline on *synthetic* data would
   only be **pipeline validation**, not evidence a biomarker is real — so we don't show that.
2. **Held-out AUC, stated on the figure.** Every ROC is stratified 5-fold cross-validation
   (mean ± std), annotated as held-out — never training performance. ROC/AUC is the headline,
   not permutation/feature importance (which shows *which* feature matters, not *how well* it predicts).
3. **Leakage-free labels.** T2DM excludes HbA1c **and** glucose (they define the label); an
   `assert` in the notebook enforces it. Honest AUC ≈ 0.83, not a circular ~1.0.

## Experiment A — Obesity (MTBLS242 NMR)

- **Data:** bariatric cohort; **obese = `preop` (n=106)** vs **healthy = `12 months after
  surgery` (n=71)**; 21 metabolites, `log1p`. Mid time points (3/6/9 mo) excluded.
- **Method:** XGBoost (`max_depth=3`), stratified 5-fold CV; SHAP for interpretation.
- **Honest caveats:** preop vs 12mo are cohort *extremes* (high AUC expected). Top SHAP
  drivers `dimethyl sulfone / isopropanol / methanol` are exogenous — **isopropanol is a
  surgical skin-prep contaminant**, so part of the separation is sample context, not
  metabolism. Real obesity markers (`L-valine`/BCAA, `lipoproteins`, `L-tyrosine`, lactate)
  rank just below. Patient IDs aren't recoverable → same-patient fold leakage can't be fully excluded.

## Experiment B — Type-2 Diabetes (NHANES clinical)

- **Data:** NHANES 2013–14, 7 files merged on `SEQN`; adults ≥18; label = self-reported
  diagnosis **or** HbA1c ≥ 6.5%; prevalence 14.3%.
- **Method:** 13 routine features (age, sex, race, BP, cholesterol/HDL, TC/HDL, creatinine,
  albumin, BUN, triglycerides, uric acid) — **HbA1c + glucose excluded**; XGBoost, 5-fold CV.
- **Honest caveat:** AUC ≈ 0.83 is the genuine screening number from non-diagnostic signals.

## Experiment C — Type-2 Diabetes (MTBLS1 urine NMR)

- **Data:** MTBLS1, *"urinary changes in type 2 diabetes"*; **48 diabetes vs 84 control** samples
  from 42 people; 220 NMR variables. Same person has NMR **and** label (a real merge). Glucose
  region excluded from the NMR — no shortcut.
- **Method:** XGBoost (`max_depth=3`); **StratifiedGroupKFold** on subjects reconstructed from
  consecutive sample-number runs (controls → 12 clean blocks of 7) so no person spans train/test.
- **Result:** subject-grouped **AUC 0.976 ± 0.022** (naive 0.986 → leakage negligible). SHAP
  recovers BCAAs (2-oxoisovalerate, isoleucine), hippurate region, N-methylnicotinamide, allantoin.
- **Honest caveats:** small (42 people); **urine** not serum; diabetic subject boundaries approximate.
