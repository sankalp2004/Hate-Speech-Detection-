"""Generate every report figure directly from data/report_data.json."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

S = r"C:\Users\jains\AppData\Local\Temp\claude\c--Desktop-Hate-Speech-Detection-\28db352e-a58e-413a-a1b3-1c38eabd9d6a\scratchpad"
F = os.path.join(S, "figs")
os.makedirs(F, exist_ok=True)
D = json.load(open(os.path.join(S, "data", "report_data.json"), encoding="utf-8"))

DARK, FG = "#1E1E1E", "#D4D4D4"
NAMES = ["Hate speech", "Offensive language", "Neither"]
SHORT = ["Hate\nSpeech", "Offensive\nLanguage", "Neither"]
COLORS = ["#C0392B", "#E67E22", "#27AE60"]


def ascii_fig(lines, path, fontsize=13, pad=0.30):
    ncols = max(len(l) for l in lines)
    w = ncols * fontsize * 0.62 / 72 + 2 * pad
    h = len(lines) * fontsize * 1.32 / 72 + 2 * pad
    fig = plt.figure(figsize=(w, h), dpi=200)
    fig.patch.set_facecolor(DARK)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor(DARK); ax.axis("off")
    ax.text(0.5, 0.5, "\n".join(lines), family="DejaVu Sans Mono", color=FG,
            fontsize=fontsize, ha="center", va="center", linespacing=1.32)
    fig.savefig(path, facecolor=DARK); plt.close(fig)


def finish(fig, ax, name, legend=False):
    if isinstance(ax, plt.Axes):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle=":", alpha=0.5); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(os.path.join(F, name)); plt.close(fig)


# ---------------------------------------------------- 3.1 architecture
ascii_fig([
    "+-----------------------+",
    "|         User          |",
    "+-----------+-----------+",
    "            |",
    "   Streamlit / Tkinter UI",
    "            |",
    "   Sensitivity Threshold",
    "            |",
    "     Preprocessing Layer",
    "            |",
    "+-----------+-----------+",
    "|                       |",
    "  Keras Tokenizer   GloVe Twitter",
    "  (10,000 words)     100d vectors",
    "|                       |",
    "+-----------+-----------+",
    "            |",
    "     Embedding Matrix",
    "            |",
    "   BiLSTM 128 (sequences)",
    "            |",
    "       Dropout 0.5",
    "            |",
    "       BiLSTM 64",
    "            |",
    "       Dropout 0.5",
    "            |",
    "     Softmax (3 classes)",
    "            |",
    "     Threshold Mapping",
    "            |",
    "  Label + Word-Level Evidence",
], os.path.join(F, "arch.png"))

# ---------------------------------------------------- 3.2 workflow
ascii_fig([
    "Raw Tweet Text", "     |", "     v",
    "Remove Non-Letters", "     |", "     v",
    "Lowercase", "     |", "     v",
    "Stopword Removal", "     |", "     v",
    "Tokenization", "     |", "     v",
    "Index + Pad Sequence", "     |", "     v",
    "GloVe Embedding", "     |", "     v",
    "Stacked BiLSTM", "     |", "     v",
    "Class Probabilities", "     |", "     v",
    "Threshold Decision", "     |", "     v",
    "Label + Explanation",
], os.path.join(F, "workflow.png"), fontsize=12)

ascii_fig([
    "User Input Text", "      |", "      v",
    "Preprocessing", "      |", "      v",
    "Tokenizer + Padding", "      |", "      v",
    "GloVe Embedding Layer", "      |", "      v",
    "BiLSTM Classifier", "      |", "      v",
    "Class Probabilities", "      |", "      v",
    "Sensitivity Threshold", "      |", "      v",
    "Final Classification",
], os.path.join(F, "workflow2.png"), fontsize=12)

# ---------------------------------------------------- 4.1 class distribution
cc = D["corpus"]["class_counts"]
counts = [cc["0"], cc["1"], cc["2"]]
fig, ax = plt.subplots(figsize=(7.0, 3.4), dpi=200)
bars = ax.bar([f"{n}\n(Class {i})" for i, n in enumerate(NAMES)], counts,
              color=COLORS, width=0.55)
for b, c in zip(bars, counts):
    ax.text(b.get_x() + b.get_width() / 2, c + 350,
            f"{c:,}\n({100*c/sum(counts):.1f}%)", ha="center", fontsize=9,
            fontweight="bold")
ax.set_ylabel("Number of tweets"); ax.set_ylim(0, 22500)
finish(fig, ax, "classdist.png")

# ---------------------------------------------------- 4.2 tweet length
hist = D["corpus"]["len_hist_by_class"]
fig, ax = plt.subplots(figsize=(7.0, 3.4), dpi=200)
x = np.arange(1, 31)
for i, c in enumerate(["0", "1", "2"]):
    h = np.array(hist[c][1:31], dtype=float)
    ax.plot(x, 100 * h / h.sum(), marker="o", ms=3, color=COLORS[i],
            label=NAMES[i], linewidth=1.5)
cl = D["corpus"]["clean_len"]
ax.axvline(cl["median"], color="grey", linestyle="--", linewidth=1)
ax.text(cl["median"] + 0.4, ax.get_ylim()[1] * 0.88,
        f"median = {cl['median']:.0f}", fontsize=8, color="grey")
ax.set_xlabel("Tokens per tweet after cleaning")
ax.set_ylabel("% of class")
ax.legend(fontsize=8)
finish(fig, ax, "lendist.png")

# ---------------------------------------------------- 4.3 annotator agreement
agr = D["corpus"]["agreement_by_class"]
una = D["corpus"]["unanimous_by_class"]
x = np.arange(3); w = 0.36
fig, ax = plt.subplots(figsize=(7.0, 3.4), dpi=200)
b1 = ax.bar(x - w / 2, [100 * agr[str(i)] for i in range(3)], w,
            label="Mean annotator agreement", color="#2E86C1")
b2 = ax.bar(x + w / 2, [100 * una[str(i)] for i in range(3)], w,
            label="Unanimously labelled", color="#8E44AD")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2,
                f"{b.get_height():.1f}%", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(NAMES, fontsize=9)
ax.set_ylabel("Percent"); ax.set_ylim(0, 108)
ax.legend(fontsize=8)
finish(fig, ax, "agreement.png")

# ---------------------------------------------------- 4.4 distinctive words
# Slurs and explicit profanity are masked for presentation; the frequency
# ratios plotted are computed on the unmasked tokens.
MASK = {"niggers", "nigger", "niggas", "nigga", "faggots", "faggot", "fag",
        "fucking", "fucked", "bitches", "bitch", "pussy", "pussies", "hoes",
        "hoe", "thot", "booty", "trash"}


def mask(w):
    return (w[0] + "*" * (len(w) - 2) + w[-1]) if w in MASK and len(w) > 2 else w


fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.2), dpi=200)
for i, axx in enumerate(axes):
    items = D["corpus"]["distinctive"][str(i)][:8][::-1]
    words = [mask(w) for w, n, r in items]
    ratios = [r for w, n, r in items]
    axx.barh(range(len(words)), ratios, color=COLORS[i])
    axx.set_yticks(range(len(words)))
    axx.set_yticklabels(words, fontsize=7)
    axx.set_title(NAMES[i], fontsize=9, fontweight="bold")
    axx.set_xlabel("Frequency ratio vs rest", fontsize=7)
    axx.tick_params(axis="x", labelsize=7)
    axx.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(F, "words.png")); plt.close(fig)

# ---------------------------------------------------- 4.5 model comparison
order = ["SVM", "LSTM", "BiLSTM_baseline", "BiLSTM_tuned"]
labels = ["SVM", "LSTM", "BiLSTM\n(weighted loss)", "BiLSTM\n(tuned)"]
prec = [D["models"][m]["precision_w"] for m in order]
rec = [D["models"][m]["recall_w"] for m in order]
f1 = [D["models"][m]["f1_w"] for m in order]
x = np.arange(4); w = 0.26
fig, ax = plt.subplots(figsize=(7.2, 3.5), dpi=200)
for off, vals, lab, col in [(-w, prec, "Precision", "#2E86C1"),
                            (0, rec, "Recall", "#E67E22"),
                            (w, f1, "F1-score", "#27AE60")]:
    bb = ax.bar(x + off, vals, w, label=lab, color=col)
    for b in bb:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.004,
                f"{b.get_height():.4f}", ha="center", fontsize=6.5)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylim(0.78, 0.96); ax.set_ylabel("Weighted score")
ax.legend(fontsize=8, ncol=3, loc="upper left")
finish(fig, ax, "modelcmp.png")

# ---------------------------------------------------- 4.6 ROC
fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=200)
for i, n in enumerate(NAMES):
    rc = D["models"]["BiLSTM_tuned"]["roc"][n]
    ax.plot(rc["fpr"], rc["tpr"], color=COLORS[i], linewidth=1.6,
            label=f"{n} (AUC = {rc['auc']:.4f})")
ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.legend(fontsize=8, loc="lower right")
ax.grid(linestyle=":", alpha=0.4)
fig.tight_layout(); fig.savefig(os.path.join(F, "roc.png")); plt.close(fig)

# ---------------------------------------------------- 4.7 confusion matrix
cm = np.array(D["models"]["BiLSTM_tuned"]["confusion_matrix"])
fig, ax = plt.subplots(figsize=(5.4, 4.2), dpi=200)
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(3)); ax.set_xticklabels(SHORT, fontsize=8)
ax.set_yticks(range(3)); ax.set_yticklabels(SHORT, fontsize=8)
ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
th = cm.max() / 2
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=10,
                color="white" if cm[i, j] > th else "black")
fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout(); fig.savefig(os.path.join(F, "confmat.png")); plt.close(fig)

# ---------------------------------------------------- 4.8 per-class metrics
pcm = D["models"]["BiLSTM_tuned"]["per_class"]
x = np.arange(3); w = 0.26
fig, ax = plt.subplots(figsize=(7.0, 3.4), dpi=200)
for off, key, lab, col in [(-w, "precision", "Precision", "#2E86C1"),
                           (0, "recall", "Recall", "#E67E22"),
                           (w, "f1", "F1-score", "#27AE60")]:
    bb = ax.bar(x + off, pcm[key], w, label=lab, color=col)
    for b in bb:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                f"{b.get_height():.3f}", ha="center", fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels([f"{n}\n(n = {s:,})" for n, s in zip(NAMES, pcm["support"])],
                   fontsize=8)
ax.set_ylim(0, 1.12); ax.set_ylabel("Score")
ax.legend(fontsize=8, ncol=3, loc="upper left")
finish(fig, ax, "perclass.png")

# ---------------------------------------------------- 4.9 training curves
h = D["history"]["BiLSTM_baseline"]
ep = np.arange(1, len(h["train_loss"]) + 1)
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), dpi=200)
axes[0].plot(ep, h["train_loss"], "o-", ms=3, label="Training", color="#2E86C1")
axes[0].plot(ep, h["val_loss"], "s-", ms=3, label="Validation", color="#C0392B")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend(fontsize=8)
axes[1].plot(ep, h["train_acc"], "o-", ms=3, label="Training", color="#2E86C1")
axes[1].plot(ep, h["val_acc"], "s-", ms=3, label="Validation", color="#C0392B")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].legend(fontsize=8)
for a in axes:
    a.grid(linestyle=":", alpha=0.5); a.set_axisbelow(True)
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(F, "curves.png")); plt.close(fig)

# ---------------------------------------------------- 4.10 config sweep
sw = D["sweep_configs"]
lab = [f"{'frozen' if c['freeze'] else 'fine-tuned'}\n{c['weights']}" for c in sw]
val = [c["val_f1"] for c in sw]
o = np.argsort(val)
fig, ax = plt.subplots(figsize=(7.0, 3.2), dpi=200)
cols = ["#27AE60" if v == max(val) else "#7F8C8D" for v in np.array(val)[o]]
bb = ax.barh(range(len(sw)), np.array(val)[o], color=cols)
ax.set_yticks(range(len(sw))); ax.set_yticklabels(np.array(lab)[o], fontsize=7.5)
for b in bb:
    ax.text(b.get_width() + 0.002, b.get_y() + b.get_height() / 2,
            f"{b.get_width():.4f}", va="center", fontsize=7.5)
ax.set_xlim(0.85, 0.935); ax.set_xlabel("Validation weighted F1-score")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", linestyle=":", alpha=0.5); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(os.path.join(F, "sweep.png")); plt.close(fig)

# ---------------------------------------------------- 4.11 threshold sweep
ts = D["threshold_sweep"]
th = [t["threshold"] for t in ts]
fig, ax = plt.subplots(figsize=(7.0, 3.4), dpi=200)
ax.plot(th, [t["flagged_pct"] for t in ts], "o-", ms=3.5, color="#C0392B",
        label="Flagged as harmful (%)")
ax.plot(th, [t["inconclusive_pct"] for t in ts], "s-", ms=3.5, color="#7F8C8D",
        label="Inconclusive (%)")
ax.set_xlabel("User-defined sensitivity threshold")
ax.set_ylabel("% of test samples")
ax.legend(fontsize=8)
ax.grid(linestyle=":", alpha=0.5); ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(F, "threshold.png")); plt.close(fig)

# ---------------------------------------------------- 4.12 precision/recall vs threshold
fig, ax = plt.subplots(figsize=(7.0, 3.4), dpi=200)
ax.plot(th, [t["precision"] for t in ts], "o-", ms=3.5, color="#2E86C1",
        label="Precision (harmful)")
ax.plot(th, [t["recall"] for t in ts], "s-", ms=3.5, color="#E67E22",
        label="Recall (harmful)")
ax.set_xlabel("User-defined sensitivity threshold"); ax.set_ylabel("Score")
ax.legend(fontsize=8)
ax.grid(linestyle=":", alpha=0.5); ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(F, "prthreshold.png")); plt.close(fig)

# ------------------------------------- before / after: per-class metrics
CMP = D["comparison"]
fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.0), dpi=200, sharey=True)
keys = ["precision", "recall", "f1"]
klab = ["Precision", "Recall", "F1"]
for i, axx in enumerate(axes):
    before = [CMP["before"]["per_class"][k][i] for k in keys]
    after = [CMP["after"]["per_class"][k][i] for k in keys]
    xx = np.arange(3); ww = 0.36
    b1 = axx.bar(xx - ww / 2, before, ww, label="Before (balanced weights)",
                 color="#95A5A6")
    b2 = axx.bar(xx + ww / 2, after, ww, label="After (tuned)", color="#27AE60")
    for bars in (b1, b2):
        for b in bars:
            axx.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                     f"{b.get_height():.2f}", ha="center", fontsize=6.5)
    axx.set_xticks(xx); axx.set_xticklabels(klab, fontsize=8)
    axx.set_title(NAMES[i], fontsize=9, fontweight="bold")
    axx.set_ylim(0, 1.18)
    axx.spines[["top", "right"]].set_visible(False)
    axx.grid(axis="y", linestyle=":", alpha=0.5); axx.set_axisbelow(True)
axes[0].set_ylabel("Score")
axes[1].legend(fontsize=7, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, -0.16), frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(F, "beforeafter.png")); plt.close(fig)

# ------------------------------------- before / after: confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), dpi=200)
for axx, key, title in zip(axes, ("before", "after"),
                           ("Before: balanced class weights",
                            "After: no class weighting")):
    m = np.array(CMP[key]["confusion_matrix"])
    im = axx.imshow(m, cmap="Blues", vmin=0, vmax=3800)
    axx.set_xticks(range(3)); axx.set_xticklabels(SHORT, fontsize=7)
    axx.set_yticks(range(3)); axx.set_yticklabels(SHORT, fontsize=7)
    axx.set_title(title, fontsize=9, fontweight="bold")
    axx.set_xlabel("Predicted", fontsize=8)
    for i in range(3):
        for j in range(3):
            axx.text(j, i, f"{m[i, j]}", ha="center", va="center", fontsize=8.5,
                     color="white" if m[i, j] > 1900 else "black")
axes[0].set_ylabel("True", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(F, "cmpair.png")); plt.close(fig)

print("figures written:")
for f_ in sorted(os.listdir(F)):
    print("  ", f_)
