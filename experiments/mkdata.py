"""Consolidate every number the report quotes into one file, so prose and
figures are generated from an identical source."""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score,
                             roc_curve)

sys.stdout.reconfigure(encoding="utf-8")
S = r"C:\Users\jains\AppData\Local\Temp\claude\c--Desktop-Hate-Speech-Detection-\28db352e-a58e-413a-a1b3-1c38eabd9d6a\scratchpad"
D = os.path.join(S, "data")

base = json.load(open(os.path.join(D, "results.json"), encoding="utf-8"))
tuned = json.load(open(os.path.join(D, "tuned.json"), encoding="utf-8"))
stats = json.load(open(os.path.join(D, "stats.json"), encoding="utf-8"))
y_test = np.load(os.path.join(D, "y_test.npy"))
proba = np.load(os.path.join(D, "tuned_proba.npy"))
pred = proba.argmax(1)

NAMES = ["Hate speech", "Offensive language", "Neither"]

acc = accuracy_score(y_test, pred)
p, r, f, _ = precision_recall_fscore_support(y_test, pred, average="weighted",
                                            zero_division=0)
pc = precision_recall_fscore_support(y_test, pred, average=None, labels=[0, 1, 2],
                                     zero_division=0)
cm = confusion_matrix(y_test, pred, labels=[0, 1, 2])
Y = np.eye(3)[y_test]
auc_macro = roc_auc_score(Y, proba, average="macro", multi_class="ovr")

roc = {}
for c in range(3):
    fpr, tpr, _ = roc_curve(Y[:, c], proba[:, c])
    idx = np.linspace(0, len(fpr) - 1, 250).astype(int)
    roc[NAMES[c]] = {"fpr": fpr[idx].tolist(), "tpr": tpr[idx].tolist(),
                     "auc": float(roc_auc_score(Y[:, c], proba[:, c]))}

# threshold sweep on the tuned model, using the report's decision rule
sweep = []
harmful = proba[:, 0] + proba[:, 1]
neutral = proba[:, 2]
true_harmful = np.isin(y_test, [0, 1])
for th in np.arange(0.05, 0.96, 0.05):
    flagged = harmful > th
    notflag = neutral > (harmful + th)
    incon = ~(flagged | notflag)
    tp = int((flagged & true_harmful).sum()); fp = int((flagged & ~true_harmful).sum())
    sweep.append({
        "threshold": round(float(th), 2),
        "flagged_pct": float(100 * flagged.mean()),
        "inconclusive_pct": float(100 * incon.mean()),
        "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "recall": float(tp / true_harmful.sum()),
    })

out = {
    "corpus": {
        "total": 24783,
        "class_counts": stats["class_counts"],
        "agreement_by_class": stats["agreement_by_class"],
        "unanimous_by_class": stats["unanimous_by_class"],
        "agreement_overall": stats["agreement_overall"],
        "clean_len": stats["clean_len"],
        "raw_len_mean": stats["raw_len_mean"],
        "vocab_size": stats["vocab_size"],
        "total_tokens": stats["total_tokens"],
        "distinctive": stats["distinctive"],
        "len_hist_by_class": stats["len_hist_by_class"],
        "n_train": base["meta"]["n_train"],
        "n_test": base["meta"]["n_test"],
        "test_class_counts": base["meta"]["test_class_counts"],
        "glove_type_coverage": base["meta"]["glove_type_coverage"],
        "glove_token_coverage": base["meta"]["glove_token_coverage"],
    },
    "models": {
        "SVM": {k: base["SVM"][k] for k in
                ("accuracy", "precision_w", "recall_w", "f1_w", "auc_macro",
                 "per_class", "confusion_matrix")},
        "LSTM": {k: base["LSTM"][k] for k in
                 ("accuracy", "precision_w", "recall_w", "f1_w", "auc_macro",
                  "per_class", "confusion_matrix")},
        "BiLSTM_baseline": {k: base["Bidirectional LSTM"][k] for k in
                            ("accuracy", "precision_w", "recall_w", "f1_w",
                             "auc_macro", "per_class", "confusion_matrix")},
        "BiLSTM_tuned": {
            "accuracy": float(acc), "precision_w": float(p), "recall_w": float(r),
            "f1_w": float(f), "macro_f1": float(f1_score(y_test, pred, average="macro")),
            "auc_macro": float(auc_macro),
            "per_class": {"precision": [float(x) for x in pc[0]],
                          "recall": [float(x) for x in pc[1]],
                          "f1": [float(x) for x in pc[2]],
                          "support": [int(x) for x in pc[3]]},
            "confusion_matrix": cm.tolist(),
            "roc": roc,
        },
    },
    "history": {
        "LSTM": base["LSTM"]["history"],
        "BiLSTM_baseline": base["Bidirectional LSTM"]["history"],
    },
    "best_config": tuned["best_config"],
    "sweep_configs": tuned["sweep"],
    "threshold_sweep": sweep,
}


def binary_view(cm):
    """Collapse the 3x3 matrix to harmful ({hate, offensive}) vs neutral, which is
    the decision the deployed system actually reports."""
    cm = np.array(cm)
    tp = cm[:2, :2].sum()          # harmful predicted harmful
    fn = cm[:2, 2].sum()           # harmful predicted neutral
    fp = cm[2, :2].sum()           # neutral predicted harmful
    tn = cm[2, 2]
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return {"precision": float(prec), "recall": float(rec),
            "f1": float(2 * prec * rec / (prec + rec)),
            "accuracy": float((tp + tn) / cm.sum()),
            "harmful_missed": int(fn), "neutral_over_flagged": int(fp)}


b_cm = out["models"]["BiLSTM_baseline"]["confusion_matrix"]
t_cm = out["models"]["BiLSTM_tuned"]["confusion_matrix"]
out["comparison"] = {
    "before": {
        "config": {"class_weights": "balanced", "lr": 1e-4},
        "weighted": {k: out["models"]["BiLSTM_baseline"][k] for k in
                     ("accuracy", "precision_w", "recall_w", "f1_w", "auc_macro")},
        "per_class": out["models"]["BiLSTM_baseline"]["per_class"],
        "confusion_matrix": b_cm,
        "binary": binary_view(b_cm),
    },
    "after": {
        "config": {"class_weights": tuned["best_config"]["class_weights"],
                   "lr": tuned["best_config"]["lr"]},
        "weighted": {k: out["models"]["BiLSTM_tuned"][k] for k in
                     ("accuracy", "precision_w", "recall_w", "f1_w", "auc_macro")},
        "per_class": out["models"]["BiLSTM_tuned"]["per_class"],
        "confusion_matrix": t_cm,
        "binary": binary_view(t_cm),
    },
}
print("\nBINARY (harmful vs neutral) VIEW")
for k in ("before", "after"):
    v = out["comparison"][k]["binary"]
    print(f"  {k:7s} P {v['precision']:.4f}  R {v['recall']:.4f}  "
          f"F1 {v['f1']:.4f}  harmful missed {v['harmful_missed']}  "
          f"neutral over-flagged {v['neutral_over_flagged']}")
json.dump(out, open(os.path.join(D, "report_data.json"), "w", encoding="utf-8"),
          indent=2)

print("TUNED BiLSTM")
print(f"  acc {acc:.4f}  P {p:.4f}  R {r:.4f}  F1 {f:.4f}  "
      f"macroF1 {out['models']['BiLSTM_tuned']['macro_f1']:.4f}  AUC {auc_macro:.4f}")
print(classification_report(y_test, pred, target_names=NAMES, digits=4,
                            zero_division=0))
print(cm)
print("\nbest config:", tuned["best_config"])
print("\nMODEL COMPARISON (weighted)")
for k, v in out["models"].items():
    print(f"  {k:18s} acc {v['accuracy']:.4f}  P {v['precision_w']:.4f} "
          f" R {v['recall_w']:.4f}  F1 {v['f1_w']:.4f}  AUC {v['auc_macro']:.4f}")
print("\nwrote report_data.json")
