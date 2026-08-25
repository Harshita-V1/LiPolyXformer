import os
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

BASE = "results_pavia"
runs = sorted(os.listdir(BASE))

from sklearn.metrics import roc_curve, auc
import numpy as np

def safe_auc_from_roc(y_true, scores):
    labels = np.asarray(y_true).astype(int)
    # need both classes present
    if labels.min() == labels.max():
        return np.nan, None, None
    scores = np.asarray(scores)
    # need finite and some variance
    mask_finite = np.isfinite(scores)
    if not np.any(mask_finite):
        return np.nan, None, None
    if np.nanstd(scores[mask_finite]) == 0.0:
        return np.nan, None, None
    try:
        fpr, tpr, _ = roc_curve(labels, scores)
        return float(auc(fpr, tpr)), fpr, tpr
    except Exception:
        return np.nan, None, None

def compute_auc_metrics(gt, scr, tau=None):
    # flatten and mask invalid samples
    gt = np.asarray(gt).reshape(-1)
    scr = np.asarray(scr).reshape(-1)

    mask = np.isfinite(gt) & np.isfinite(scr)
    gt = gt[mask].astype(int)
    scr = scr[mask]

    # --------------------------------------------------
    # AUC(D,F): standard ROC on reconstruction error
    # --------------------------------------------------
    auc_df, _, _ = safe_auc_from_roc(gt, scr)

    # --------------------------------------------------
    # TAEF-style AUC(D,τ) and AUC(F,τ)
    # τ is a THRESHOLD on reconstruction error
    # --------------------------------------------------

    # Normalize reconstruction error
    score = scr.astype(np.float64)
    score = (score - score.min()) / (score.max() - score.min() + 1e-12)

    taus = np.unique(score)
    taus.sort()

    P = np.sum(gt == 1)  # anomaly pixels
    N = np.sum(gt == 0)  # background pixels

    PD = []
    PF = []

    for tau_thr in taus:
        detected = (score >= tau_thr)

        TP = np.sum((detected == 1) & (gt == 1))
        FP = np.sum((detected == 1) & (gt == 0))

        PD.append(TP / (P + 1e-12))
        PF.append(FP / (N + 1e-12))

    PD = np.array(PD)
    PF = np.array(PF)

    auc_dtau = np.trapz(PD, taus)
    auc_ftau = np.trapz(PF, taus)

    # --------------------------------------------------
    # Composite metrics (TAEF equations 14a–14e)
    # --------------------------------------------------
    auc_bs = auc_df - auc_ftau
    auc_td = auc_df + auc_dtau
    auc_snpr = auc_dtau / (auc_ftau + 1e-12)
    auc_td_bs = auc_dtau - auc_ftau
    auc_od = auc_df + auc_dtau - auc_ftau

    return {
        "AUC_DF": auc_df,
        "AUC_Dtau": auc_dtau,
        "AUC_Ftau": auc_ftau,
        "AUC_BS": auc_bs,
        "AUC_TD": auc_td,
        "AUC_SNPR": auc_snpr,
        "AUC_TD_BS": auc_td_bs,
        "AUC_OD": auc_od
    }



summary = {}

for run in runs:
    run_dir = os.path.join(BASE, run)
    if not os.path.isdir(run_dir):
        continue

    required = ["gt_flat.npy", "score_flat.npy", "var_map.npy"]
    if not all(os.path.exists(os.path.join(run_dir, f)) for f in required):
        print(f"Skipping: {run}  (missing required outputs)")
        continue

    print(f"Processing: {run}")

    gt = np.load(os.path.join(run_dir, "gt_flat.npy")).reshape(-1)
    scr = np.load(os.path.join(run_dir, "score_flat.npy")).reshape(-1)
    tau = np.load(os.path.join(run_dir, "var_map.npy")).reshape(-1)

    aucs = compute_auc_metrics(gt, scr, tau)
    summary[run] = aucs

    with open(os.path.join(run_dir, "aucs.json"), "w") as f:
        json.dump(aucs, f, indent=4)

    # Plot per-run ROCs only if computable
    # ROC(D,F)
    try:
        if not np.isnan(aucs["AUC_DF"]):
            fpr, tpr, _ = roc_curve(gt, scr)
            plt.figure(figsize=(5,4))
            plt.plot(fpr, tpr, lw=2)
            plt.plot([0,1],[0,1],'k--', lw=0.6)
            plt.xlabel("P_F"); plt.ylabel("P_D"); plt.title(f"{run}: ROC(D,F) AUC={aucs['AUC_DF']:.4f}")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, "roc_df.png"), dpi=200)
            plt.close()
    except Exception:
        pass

    # ROC(D,τ)
    try:
        if not np.isnan(aucs["AUC_Dtau"]):
            fpr2, tpr2, _ = roc_curve(gt, -tau)
            plt.figure(figsize=(5,4))
            plt.plot(fpr2, tpr2, lw=2)
            plt.plot([0,1],[0,1],'k--', lw=0.6)
            plt.xlabel("P_F"); plt.ylabel("P_D"); plt.title(f"{run}: ROC(D,τ) AUC={aucs['AUC_Dtau']:.4f}")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, "roc_dtau.png"), dpi=200)
            plt.close()
    except Exception:
        pass

    # ROC(F,τ)
    try:
        if not np.isnan(aucs["AUC_Ftau"]):
            bg = (gt == 0).astype(int)
            fpr3, tpr3, _ = roc_curve(bg, tau)
            plt.figure(figsize=(5,4))
            plt.plot(fpr3, tpr3, lw=2)
            plt.plot([0,1],[0,1],'k--', lw=0.6)
            plt.xlabel("P_F"); plt.ylabel("P_τ"); plt.title(f"{run}: ROC(F,τ) AUC={aucs['AUC_Ftau']:.4f}")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, "roc_ftau.png"), dpi=200)
            plt.close()
    except Exception:
        pass

# Save CSV summary
import csv
csv_path = os.path.join(BASE, "summary_auc_new4ja.csv")
with open(csv_path, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Run", "AUC_DF", "AUC_Dtau", "AUC_Ftau",
                     "AUC_BS", "AUC_TD", "AUC_SNPR", "AUC_TD_BS", "AUC_OD"])

    for run, aucs in summary.items():
        writer.writerow([
            run,
            aucs["AUC_DF"], aucs["AUC_Dtau"], aucs["AUC_Ftau"],
            aucs["AUC_BS"], aucs["AUC_TD"], aucs["AUC_SNPR"],
            aucs["AUC_TD_BS"], aucs["AUC_OD"]
        ])

print("DONE! Summary at:", csv_path)
