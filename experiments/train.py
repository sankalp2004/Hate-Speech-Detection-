"""Train SVM / LSTM / BiLSTM on the Davidson corpus and record real metrics."""
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight

sys.stdout.reconfigure(encoding="utf-8")
S = r"C:\Users\jains\AppData\Local\Temp\claude\c--Desktop-Hate-Speech-Detection-\28db352e-a58e-413a-a1b3-1c38eabd9d6a\scratchpad"
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

NAMES = ["Hate speech", "Offensive language", "Neither"]
MAXLEN = 50
VOCAB_SIZE = 10000
EMB_DIM = 100

df = pd.read_parquet(os.path.join(S, "data", "davidson_clean.parquet"))
print("corpus:", df.shape)

train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=SEED, stratify=df["class"])
print("train:", len(train_df), "test:", len(test_df))
print("train class counts:", Counter(train_df["class"]))
print("test  class counts:", Counter(test_df["class"]))

y_train = train_df["class"].to_numpy()
y_test = test_df["class"].to_numpy()

results = {}


def evaluate(name, y_true, y_pred, proba=None):
    acc = accuracy_score(y_true, y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted",
                                                 zero_division=0)
    pc = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1, 2],
                                         zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    entry = {
        "accuracy": float(acc),
        "precision_w": float(p), "recall_w": float(r), "f1_w": float(f),
        "per_class": {
            "precision": [float(x) for x in pc[0]],
            "recall": [float(x) for x in pc[1]],
            "f1": [float(x) for x in pc[2]],
            "support": [int(x) for x in pc[3]],
        },
        "confusion_matrix": cm.tolist(),
    }
    if proba is not None:
        Y = np.eye(3)[y_true]
        entry["auc_macro"] = float(roc_auc_score(Y, proba, average="macro",
                                                 multi_class="ovr"))
        roc = {}
        for c in range(3):
            fpr, tpr, _ = roc_curve(Y[:, c], proba[:, c])
            idx = np.linspace(0, len(fpr) - 1, 200).astype(int)
            roc[str(c)] = {"fpr": fpr[idx].tolist(), "tpr": tpr[idx].tolist(),
                           "auc": float(roc_auc_score(Y[:, c], proba[:, c]))}
        entry["roc"] = roc
    results[name] = entry
    print(f"\n=== {name} ===")
    print(f"acc {acc:.4f}  P {p:.4f}  R {r:.4f}  F1 {f:.4f}"
          + (f"  AUC {entry.get('auc_macro', float('nan')):.4f}" if proba is not None else ""))
    print(classification_report(y_true, y_pred, target_names=NAMES, digits=4,
                                zero_division=0))
    print(cm)
    return entry


# ------------------------------------------------------------------ SVM
print("\n>>> training SVM (bag-of-words, 5000 features)")
vec = CountVectorizer(analyzer="word", max_features=5000)
Xtr = vec.fit_transform(train_df["clean_text"])
Xte = vec.transform(test_df["clean_text"])
cw = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=y_train)
svm = LinearSVC(C=1.0, class_weight={i: cw[i] for i in range(3)}, max_iter=5000)
t0 = time.time()
svm.fit(Xtr, y_train)
print("fit %.1fs" % (time.time() - t0))
svm_scores = svm.decision_function(Xte)
svm_proba = np.exp(svm_scores) / np.exp(svm_scores).sum(axis=1, keepdims=True)
evaluate("SVM", y_test, svm.predict(Xte), svm_proba)

# ------------------------------------------------- vocab + GloVe matrix
counter = Counter(w for t in train_df["clean_text"] for w in t.split())
itos = ["<pad>", "<unk>"] + [w for w, _ in counter.most_common(VOCAB_SIZE - 2)]
stoi = {w: i for i, w in enumerate(itos)}
print("\nvocab size:", len(itos))

import gensim.downloader as api
print(">>> loading glove-twitter-100")
glove = api.load("glove-twitter-100")
emb = np.zeros((len(itos), EMB_DIM), dtype=np.float32)
hit = 0
rng = np.random.RandomState(SEED)
for i, w in enumerate(itos):
    if i == 0:
        continue
    if w in glove:
        emb[i] = glove[w]
        hit += 1
    else:
        emb[i] = rng.normal(0, 0.1, EMB_DIM)
coverage = hit / (len(itos) - 1)
print("GloVe coverage: %d/%d = %.4f" % (hit, len(itos) - 1, coverage))

# token-level coverage over the whole corpus
tok_total = sum(counter.values())
tok_hit = sum(n for w, n in counter.items() if w in glove)
print("token-level coverage: %.4f" % (tok_hit / tok_total))


def encode(texts):
    out = np.zeros((len(texts), MAXLEN), dtype=np.int64)
    lens = np.zeros(len(texts), dtype=np.int64)
    for i, t in enumerate(texts):
        ids = [stoi.get(w, 1) for w in t.split()][:MAXLEN]
        if not ids:
            ids = [1]
        out[i, :len(ids)] = ids
        lens[i] = len(ids)
    return out, lens


Xtr_ids, Ltr = encode(train_df["clean_text"].tolist())
Xte_ids, Lte = encode(test_df["clean_text"].tolist())

# carve a validation split out of training (20%)
tr_idx, va_idx = train_test_split(np.arange(len(train_df)), test_size=0.2,
                                  random_state=SEED, stratify=y_train)


class Net(nn.Module):
    def __init__(self, emb_matrix, bidirectional):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(torch.tensor(emb_matrix),
                                                freeze=True, padding_idx=0)
        self.lstm1 = nn.LSTM(EMB_DIM, 128, batch_first=True,
                             bidirectional=bidirectional)
        d1 = 128 * (2 if bidirectional else 1)
        self.drop1 = nn.Dropout(0.5)
        self.lstm2 = nn.LSTM(d1, 64, batch_first=True, bidirectional=bidirectional)
        d2 = 64 * (2 if bidirectional else 1)
        self.drop2 = nn.Dropout(0.5)
        self.fc = nn.Linear(d2, 3)

    def forward(self, x, lengths):
        e = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        o1, _ = self.lstm1(packed)
        o1, _ = nn.utils.rnn.pad_packed_sequence(o1, batch_first=True)
        o1 = self.drop1(o1)
        packed2 = nn.utils.rnn.pack_padded_sequence(
            o1, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm2(packed2)
        h = torch.cat([h[-2], h[-1]], dim=1) if self.lstm2.bidirectional else h[-1]
        return self.fc(self.drop2(h))


def run(name, bidirectional, epochs=30, patience=5, bs=64):
    print(f"\n>>> training {name}")
    model = Net(emb, bidirectional)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    weights = torch.tensor(cw, dtype=torch.float32)
    lossf = nn.CrossEntropyLoss(weight=weights)

    Xt = torch.tensor(Xtr_ids[tr_idx]); Lt = torch.tensor(Ltr[tr_idx])
    Yt = torch.tensor(y_train[tr_idx])
    Xv = torch.tensor(Xtr_ids[va_idx]); Lv = torch.tensor(Ltr[va_idx])
    Yv = torch.tensor(y_train[va_idx])

    best, best_state, bad = 1e9, None, 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        tl, tc, n = 0.0, 0, 0
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            out = model(Xt[b], Lt[b])
            loss = lossf(out, Yt[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tl += loss.item() * len(b)
            tc += (out.argmax(1) == Yt[b]).sum().item()
            n += len(b)
        model.eval()
        with torch.no_grad():
            vout = torch.cat([model(Xv[i:i + 256], Lv[i:i + 256])
                              for i in range(0, len(Xv), 256)])
            vloss = lossf(vout, Yv).item()
            vacc = (vout.argmax(1) == Yv).float().mean().item()
        history["train_loss"].append(tl / n)
        history["val_loss"].append(vloss)
        history["train_acc"].append(tc / n)
        history["val_acc"].append(vacc)
        print(f"  epoch {ep+1:2d}  train_loss {tl/n:.4f} acc {tc/n:.4f} | "
              f"val_loss {vloss:.4f} acc {vacc:.4f}")
        if vloss < best - 1e-4:
            best, bad = vloss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stopping at epoch {ep+1}")
                break
    model.load_state_dict(best_state)
    model.eval()
    Xs = torch.tensor(Xte_ids); Ls = torch.tensor(Lte)
    with torch.no_grad():
        logits = torch.cat([model(Xs[i:i + 256], Ls[i:i + 256])
                            for i in range(0, len(Xs), 256)])
        proba = torch.softmax(logits, dim=1).numpy()
    entry = evaluate(name, y_test, proba.argmax(1), proba)
    entry["history"] = history
    entry["epochs_run"] = len(history["train_loss"])
    return model, proba


_, lstm_proba = run("LSTM", bidirectional=False)
bilstm_model, bilstm_proba = run("Bidirectional LSTM", bidirectional=True)

# --------------------------------------------- user-defined threshold sweep
# report rule: hate(0)+offensive(1) mass vs neither(2), with an Inconclusive band
sweep = []
for th in np.arange(0.05, 0.96, 0.05):
    harmful = bilstm_proba[:, 0] + bilstm_proba[:, 1]
    neutral = bilstm_proba[:, 2]
    flagged = harmful > th
    notflag = neutral > (harmful + th)
    incon = ~(flagged | notflag)
    true_harmful = np.isin(y_test, [0, 1])
    decided = flagged | notflag
    tp = (flagged & true_harmful).sum()
    fp = (flagged & ~true_harmful).sum()
    fn = (notflag & true_harmful).sum()
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / true_harmful.sum()
    sweep.append({
        "threshold": round(float(th), 2),
        "flagged_pct": float(100 * flagged.mean()),
        "inconclusive_pct": float(100 * incon.mean()),
        "not_flagged_pct": float(100 * notflag.mean()),
        "precision_harmful": float(prec),
        "recall_harmful": float(rec),
        "coverage_pct": float(100 * decided.mean()),
    })
results["threshold_sweep"] = sweep
print("\nTHRESHOLD SWEEP (BiLSTM)")
for r in sweep:
    print("  th %.2f  flagged %5.1f%%  incon %5.1f%%  P %.3f  R %.3f"
          % (r["threshold"], r["flagged_pct"], r["inconclusive_pct"],
             r["precision_harmful"], r["recall_harmful"]))

results["meta"] = {
    "seed": SEED, "maxlen": MAXLEN, "vocab_size": len(itos),
    "embedding": "glove-twitter-100", "emb_dim": EMB_DIM,
    "glove_type_coverage": float(coverage),
    "glove_token_coverage": float(tok_hit / tok_total),
    "n_train": int(len(train_df)), "n_test": int(len(test_df)),
    "class_weights": [float(x) for x in cw],
    "test_class_counts": {int(k): int(v) for k, v in Counter(y_test).items()},
}
with open(os.path.join(S, "data", "results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
np.save(os.path.join(S, "data", "bilstm_proba.npy"), bilstm_proba)
np.save(os.path.join(S, "data", "y_test.npy"), y_test)
print("\nsaved results.json")
