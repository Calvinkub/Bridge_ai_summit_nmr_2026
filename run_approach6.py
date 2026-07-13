"""
Approach 6 - Obesity Screener (MTBLS242, NMR serum -> obese vs healthy)
=======================================================================
BDI Hackathon 2026 | PhenoInsure Tech

MTBLS242 is a bariatric (weight-loss) surgery cohort. We use the time-point
factor as a natural obesity signal and frame a binary classification:

    obese   (class 1) = preop                     (before operation, n=106)
    healthy (class 0) = 12 months after surgery   (weight-normalized, n=71)

The intermediate 3/6/9-month samples (patients mid-weight-loss, neither clearly
obese nor healthy) are excluded to sharpen the obese-vs-healthy contrast.

Model      : XGBoost, stratified 5-fold cross-validation
Headline   : roc_auc_cv.png  -> per-fold ROC + mean ROC (+/-1 std band) + AUC
Interpret  : shap_summary.png -> which NMR metabolites drive the obese call

Honesty note: preop vs 12mo are the two extremes of the cohort, so expect a
very high AUC. The CV +/-std band shows the real variance. This measures how
separable the extremes are, not proof of general (population) screening power.

Run:  conda run -n superaiss6 python run_approach6.py
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "Highly_Imagination", "real_data"))
NPZ_PATH = os.path.join(DATA_DIR, "MTBLS242_parsed.npz")
SAMPLES_PATH = os.path.join(DATA_DIR, "MTBLS242_samples.txt")

OBESE_TIMEPOINTS = {"preop"}                       # class 1
HEALTHY_TIMEPOINTS = {"12 months after surgery"}   # class 0
# To use "healthy = all post-op" instead, set:
# HEALTHY_TIMEPOINTS = {"3 months after surgery", "6 months after surgery",
#                       "9 months after surgery", "12 months after surgery"}

N_SPLITS = 5
SEED = 42

XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=3,          # shallow: only 177 samples, guard against overfit
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    eval_metric="logloss",
    random_state=SEED,
    n_jobs=-1,
)


# --------------------------------------------------------------------------- #
# 1. Load data + attach time points
# --------------------------------------------------------------------------- #
def load_dataset():
    d = np.load(NPZ_PATH, allow_pickle=True)
    metabolites = [str(m) for m in d["metabolites"]]
    conc = d["concentrations"].astype(float)          # (465, 21)
    sample_ids = [str(s) for s in d["sample_ids"]]

    # sample_name -> time point
    tp_map = {}
    with open(SAMPLES_PATH) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            tp_map[row["Sample Name"]] = row["Factor Value[time point]"]
    timepoints = np.array([tp_map.get(s, "MISSING") for s in sample_ids])

    # keep only obese/healthy anchor time points
    is_obese = np.isin(timepoints, list(OBESE_TIMEPOINTS))
    is_healthy = np.isin(timepoints, list(HEALTHY_TIMEPOINTS))
    keep = is_obese | is_healthy

    X = conc[keep]
    y = is_obese[keep].astype(int)                    # 1=obese(preop), 0=healthy
    X_log = np.log1p(X)                               # NMR abundances -> log scale

    df = pd.DataFrame(X_log, columns=metabolites)
    print(f"[data] samples kept: {keep.sum()}  (obese={y.sum()}, healthy={(y==0).sum()})")
    print(f"[data] features: {len(metabolites)} NMR metabolites")
    return df, y, metabolites


# --------------------------------------------------------------------------- #
# 2. Stratified 5-fold CV -> collect ROC per fold
# --------------------------------------------------------------------------- #
def cross_validate(df, y):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    mean_fpr = np.linspace(0, 1, 200)
    tprs, aucs, fold_rows = [], [], []

    for fold, (tr, te) in enumerate(skf.split(df, y), start=1):
        model = XGBClassifier(**XGB_PARAMS)
        model.fit(df.iloc[tr], y[tr])
        prob = model.predict_proba(df.iloc[te])[:, 1]

        fpr, tpr, _ = roc_curve(y[te], prob)
        fold_auc = auc(fpr, tpr)
        aucs.append(fold_auc)

        interp = np.interp(mean_fpr, fpr, tpr)
        interp[0] = 0.0
        tprs.append(interp)

        fold_rows.append({"fold": fold, "n_test": len(te), "auc": fold_auc})
        print(f"[cv] fold {fold}: n_test={len(te):3d}  AUC={fold_auc:.4f}")

    return mean_fpr, np.array(tprs), np.array(aucs), pd.DataFrame(fold_rows)


# --------------------------------------------------------------------------- #
# 3. Plot the AUC / ROC curve  (the headline deliverable)
# --------------------------------------------------------------------------- #
def plot_roc(mean_fpr, tprs, aucs, n_obese, n_healthy, out_path):
    mean_tpr = tprs.mean(axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = aucs.std()
    std_tpr = tprs.std(axis=0)
    upper = np.minimum(mean_tpr + std_tpr, 1)
    lower = np.maximum(mean_tpr - std_tpr, 0)

    fig, ax = plt.subplots(figsize=(7.6, 7))

    for i, (tpr, a) in enumerate(zip(tprs, aucs), start=1):
        ax.plot(mean_fpr, tpr, lw=1, alpha=0.35,
                label=f"Fold {i}  (AUC = {a:.3f})")

    ax.plot(mean_fpr, mean_tpr, color="#b2182b", lw=3,
            label=f"Mean ROC  (AUC = {mean_auc:.3f} $\\pm$ {std_auc:.3f})")
    ax.fill_between(mean_fpr, lower, upper, color="#b2182b", alpha=0.18,
                    label=r"$\pm$ 1 std. dev.")
    ax.plot([0, 1], [0, 1], linestyle="--", lw=1.2, color="grey",
            label="Chance (AUC = 0.500)")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(
        "Approach 6 — Obesity Screener (MTBLS242 NMR)\n"
        f"XGBoost, stratified 5-fold CV  |  obese/preop n={n_obese}  vs  "
        f"healthy/12mo n={n_healthy}",
        fontsize=12.5,
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {out_path}   mean AUC = {mean_auc:.4f} +/- {std_auc:.4f}")
    return mean_auc, std_auc


# --------------------------------------------------------------------------- #
# 4. SHAP -> which metabolites drive the obese prediction
# --------------------------------------------------------------------------- #
def plot_shap(df, y, out_path):
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(df, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)

    plt.figure()
    shap.summary_plot(shap_values, df, show=False, plot_size=(8, 7))
    plt.title("Approach 6 — SHAP: NMR metabolites driving 'obese' call",
              fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[fig] wrote {out_path}")

    # rank metabolites by mean |SHAP|
    imp = np.abs(shap_values).mean(axis=0)
    order = np.argsort(imp)[::-1]
    ranking = pd.DataFrame(
        {"metabolite": df.columns[order], "mean_abs_shap": imp[order]}
    )
    print("[shap] top drivers:")
    print(ranking.head(8).to_string(index=False))
    return ranking


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    print("=" * 70)
    print("Approach 6 - Obesity Screener (MTBLS242 NMR -> obese vs healthy)")
    print("=" * 70)

    df, y, _ = load_dataset()
    n_obese, n_healthy = int(y.sum()), int((y == 0).sum())

    mean_fpr, tprs, aucs, fold_df = cross_validate(df, y)

    roc_path = os.path.join(HERE, "roc_auc_cv.png")
    mean_auc, std_auc = plot_roc(mean_fpr, tprs, aucs, n_obese, n_healthy, roc_path)

    shap_path = os.path.join(HERE, "shap_summary.png")
    ranking = plot_shap(df, y, shap_path)

    # persist results
    fold_df.to_csv(os.path.join(HERE, "approach6_results.csv"), index=False)
    ranking.to_csv(os.path.join(HERE, "approach6_shap_ranking.csv"), index=False)

    print("-" * 70)
    print(f"SUMMARY  mean ROC-AUC = {mean_auc:.4f} +/- {std_auc:.4f}  "
          f"(5-fold CV, n={n_obese + n_healthy})")
    print("Figures: roc_auc_cv.png (AUC curve), shap_summary.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
