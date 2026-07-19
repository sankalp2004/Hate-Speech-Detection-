"""Legitimate tuning sweep: unfreeze embeddings, soften class weights.
Selects the best configuration on the validation set, then reports test metrics."""
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

sys.stdout.reconfigure(encoding="utf-8")
S = r"C:\Users\jains\AppData\Local\Temp\claude\c--Desktop-Hate-Speech-Detection-\28db352e-a58e-413a-a1b3-1c38eabd9d6a\scratchpad"
SEED, MAXLEN, VOCAB_SIZE, EMB_DIM = 42, 50, 10000, 100
NAMES = ["Hate speech", "Offensive language", "Neither"]
np.random.seed(SEED); torch.manual_seed(SEED)

df = pd.read_parquet(os.path.join(S, "data", "davidson_clean.parquet"))
train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED,
                                     stratify=df["class"])
y_train = train_df["class"].to_numpy()
y_test = test_df["class"].to_numpy()

counter = Counter(w for t in train_df["clean_text"] for w in t.split())
itos = ["<pad>", "<unk>"] + [w for w, _ in counter.most_common(VOCAB_SIZE - 2)]
stoi = {w: i for i, w in enumerate(itos)}

import gensim.downloader as api
glove = api.load("glove-twitter-100")
emb = np.zeros((len(itos), EMB_DIM), dtype=np.float32)
rng = np.random.RandomState(SEED)
for i, w in enumerate(itos):
    if i == 0:
        continue
    emb[i] = glove[w] if w in glove else rng.normal(0, 0.1, EMB_DIM)


def encode(texts):
    out = np.zeros((len(texts), MAXLEN), dtype=np.int64)
    lens = np.zeros(len(texts), dtype=np.int64)
    for i, t in enumerate(texts):
        ids = [stoi.get(w, 1) for w in t.split()][:MAXLEN] or [1]
        out[i, :len(ids)] = ids
        lens[i] = len(ids)
    return out, lens


Xtr_ids, Ltr = encode(train_df["clean_text"].tolist())
Xte_ids, Lte = encode(test_df["clean_text"].tolist())
tr_idx, va_idx = train_test_split(np.arange(len(train_df)), test_size=0.2,
                                  random_state=SEED, stratify=y_train)


class Net(nn.Module):
    def __init__(self, freeze, bidir=True):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(torch.tensor(emb), freeze=freeze,
                                                padding_idx=0)
        self.l1 = nn.LSTM(EMB_DIM, 128, batch_first=True, bidirectional=bidir)
        self.d1 = nn.Dropout(0.5)
        self.l2 = nn.LSTM(128 * (2 if bidir else 1), 64, batch_first=True,
                          bidirectional=bidir)
        self.d2 = nn.Dropout(0.5)
        self.fc = nn.Linear(64 * (2 if bidir else 1), 3)
        self.bidir = bidir

    def forward(self, x, lengths):
        e = self.emb(x)
        p = nn.utils.rnn.pack_padded_sequence(e, lengths.cpu(), batch_first=True,
                                              enforce_sorted=False)
        o, _ = self.l1(p)
        o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True)
        o = self.d1(o)
        p2 = nn.utils.rnn.pack_padded_sequence(o, lengths.cpu(), batch_first=True,
                                               enforce_sorted=False)
        _, (h, _) = self.l2(p2)
        h = torch.cat([h[-2], h[-1]], 1) if self.bidir else h[-1]
        return self.fc(self.d2(h))


bal = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=y_train)
WEIGHTS = {
    "none": np.ones(3, dtype=np.float32),
    "sqrt": np.sqrt(bal).astype(np.float32),
    "balanced": bal.astype(np.float32),
}


def train_cfg(freeze, wname, lr, epochs=25, patience=5, bs=64, bidir=True):
    torch.manual_seed(SEED)
    model = Net(freeze, bidir)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(WEIGHTS[wname]))
    Xt = torch.tensor(Xtr_ids[tr_idx]); Lt = torch.tensor(Ltr[tr_idx])
    Yt = torch.tensor(y_train[tr_idx])
    Xv = torch.tensor(Xtr_ids[va_idx]); Lv = torch.tensor(Ltr[va_idx])
    Yv = torch.tensor(y_train[va_idx])
    best_f1, best_state, bad = -1, None, 0
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
            "val_f1": []}
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
            tl += loss.item() * len(b); n += len(b)
            tc += (out.argmax(1) == Yt[b]).sum().item()
        model.eval()
        with torch.no_grad():
            vo = torch.cat([model(Xv[i:i + 256], Lv[i:i + 256])
                            for i in range(0, len(Xv), 256)])
            vl = lossf(vo, Yv).item()
            vp = vo.argmax(1).numpy()
            vf1 = f1_score(Yv.numpy(), vp, average="weighted")
            vacc = accuracy_score(Yv.numpy(), vp)
        hist["train_loss"].append(tl / n); hist["val_loss"].append(vl)
        hist["train_acc"].append(tc / n); hist["val_acc"].append(vacc)
        hist["val_f1"].append(vf1)
        if vf1 > best_f1 + 1e-4:
            best_f1, bad = vf1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_f1, hist, ep + 1


print("=== validation sweep (weighted F1) ===")
cands = []
for freeze in (True, False):
    for wname in ("none", "sqrt", "balanced"):
        lr = 1e-3 if freeze else 5e-4
        m, vf1, hist, eps = train_cfg(freeze, wname, lr)
        tag = f"freeze={freeze} weights={wname} lr={lr}"
        print(f"  {tag:42s} val_F1 {vf1:.4f}  ({eps} epochs)")
        cands.append((vf1, freeze, wname, lr, m, hist, eps))

cands.sort(key=lambda x: -x[0])
vf1, freeze, wname, lr, model, hist, eps = cands[0]
print(f"\nBEST CONFIG: freeze={freeze} weights={wname} lr={lr} val_F1={vf1:.4f}")

model.eval()
Xs = torch.tensor(Xte_ids); Ls = torch.tensor(Lte)
with torch.no_grad():
    logits = torch.cat([model(Xs[i:i + 256], Ls[i:i + 256])
                        for i in range(0, len(Xs), 256)])
    proba = torch.softmax(logits, 1).numpy()
pred = proba.argmax(1)

acc = accuracy_score(y_test, pred)
p, r, f, _ = precision_recall_fscore_support(y_test, pred, average="weighted",
                                             zero_division=0)
mf1 = f1_score(y_test, pred, average="macro")
Y = np.eye(3)[y_test]
auc = roc_auc_score(Y, proba, average="macro", multi_class="ovr")
print(f"\nTEST  acc {acc:.4f}  P {p:.4f}  R {r:.4f}  F1 {f:.4f} "
      f" macroF1 {mf1:.4f}  AUC {auc:.4f}")
print(classification_report(y_test, pred, target_names=NAMES, digits=4,
                            zero_division=0))
print(confusion_matrix(y_test, pred, labels=[0, 1, 2]))

json.dump({
    "best_config": {"freeze_embeddings": bool(freeze), "class_weights": wname,
                    "lr": lr, "epochs_run": eps, "val_f1": float(vf1)},
    "sweep": [{"freeze": bool(c[1]), "weights": c[2], "lr": c[3],
               "val_f1": float(c[0]), "epochs": c[6]} for c in cands],
    "test": {"accuracy": float(acc), "precision_w": float(p), "recall_w": float(r),
             "f1_w": float(f), "macro_f1": float(mf1), "auc_macro": float(auc)},
}, open(os.path.join(S, "data", "tuned.json"), "w"), indent=2)
np.save(os.path.join(S, "data", "tuned_proba.npy"), proba)
print("\nsaved tuned.json")
