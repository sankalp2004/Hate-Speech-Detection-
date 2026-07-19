"""Load the Davidson corpus and compute real descriptive statistics."""
import json
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
S = r"C:\Users\jains\AppData\Local\Temp\claude\c--Desktop-Hate-Speech-Detection-\28db352e-a58e-413a-a1b3-1c38eabd9d6a\scratchpad"
os.makedirs(os.path.join(S, "data"), exist_ok=True)

from datasets import load_dataset

ds = load_dataset("tdavidson/hate_speech_offensive")
df = ds["train"].to_pandas()
df.to_parquet(os.path.join(S, "data", "davidson.parquet"))

print("SHAPE:", df.shape)
print("COLUMNS:", list(df.columns))
print()
print("CLASS COUNTS:")
print(df["class"].value_counts().sort_index())
print()
print("ANNOTATOR COUNT distribution:")
print(df["count"].value_counts().sort_index())
print()

names = {0: "Hate speech", 1: "Offensive language", 2: "Neither"}

# ---- annotator agreement: share of annotators voting the majority label ----
cols = ["hate_speech_count", "offensive_language_count", "neither_count"]
votes = df[cols].to_numpy()
total = votes.sum(axis=1)
maj = votes.max(axis=1)
agreement = maj / total
df["agreement"] = agreement
print("AGREEMENT overall mean: %.4f" % agreement.mean())
print("AGREEMENT by class:")
print(df.groupby("class")["agreement"].agg(["mean", "median", "count"]))
print()
print("Share of tweets with unanimous annotation: %.4f" % (agreement == 1.0).mean())
print("Unanimous by class:")
print(df.groupby("class")["agreement"].apply(lambda s: (s == 1.0).mean()))
print()

# ---- tweet length (raw whitespace tokens, and after cleaning) ----
import nltk
for pkg in ["stopwords", "punkt", "punkt_tab"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass
from nltk.corpus import stopwords

STOPS = set(stopwords.words("english"))


def clean(t):
    letters = re.sub("[^a-zA-Z]", " ", t)
    return " ".join(w for w in letters.lower().split() if w not in STOPS)


df["clean_text"] = df["tweet"].map(clean)
df["raw_len"] = df["tweet"].str.split().map(len)
df["clean_len"] = df["clean_text"].str.split().map(len)

print("RAW length: mean %.2f  median %.0f  p95 %.0f  max %d"
      % (df.raw_len.mean(), df.raw_len.median(),
         df.raw_len.quantile(.95), df.raw_len.max()))
print("CLEAN length: mean %.2f  median %.0f  p95 %.0f  max %d"
      % (df.clean_len.mean(), df.clean_len.median(),
         df.clean_len.quantile(.95), df.clean_len.max()))
print("Share of cleaned tweets with <=50 tokens: %.4f" % (df.clean_len <= 50).mean())
print("Share of cleaned tweets with <=100 tokens: %.4f" % (df.clean_len <= 100).mean())
print()
print("CLEAN length by class:")
print(df.groupby("class")["clean_len"].agg(["mean", "median"]))
print()

# ---- vocabulary ----
all_tokens = [w for t in df.clean_text for w in t.split()]
vocab = Counter(all_tokens)
print("TOTAL tokens (cleaned): %d" % len(all_tokens))
print("VOCAB size (cleaned): %d" % len(vocab))
cum = np.cumsum([c for _, c in vocab.most_common()]) / len(all_tokens)
for k in (1000, 2000, 5000, 10000):
    if k <= len(cum):
        print("  top %5d words cover %.2f%% of tokens" % (k, 100 * cum[k - 1]))
print()

# ---- most distinctive words per class (log-odds vs rest) ----
per_class = {c: Counter(w for t in df[df["class"] == c].clean_text for w in t.split())
             for c in (0, 1, 2)}
tot_class = {c: sum(per_class[c].values()) for c in per_class}
print("TOP DISTINCTIVE WORDS (min freq 40, ratio vs rest of corpus):")
distinct = {}
for c in (0, 1, 2):
    scores = []
    for w, n in per_class[c].items():
        if n < 40:
            continue
        other = sum(per_class[o][w] for o in per_class if o != c)
        tot_other = sum(tot_class[o] for o in tot_class if o != c)
        p_in = n / tot_class[c]
        p_out = (other + 1) / (tot_other + 1)
        scores.append((p_in / p_out, w, n))
    scores.sort(reverse=True)
    distinct[c] = [(w, int(n), round(r, 2)) for r, w, n in scores[:12]]
    print(f"  {names[c]}: {[w for w, _, _ in distinct[c]]}")
print()

out = {
    "class_counts": {int(k): int(v) for k, v in df["class"].value_counts().items()},
    "agreement_by_class": {int(k): float(v) for k, v in
                           df.groupby("class")["agreement"].mean().items()},
    "unanimous_by_class": {int(k): float(v) for k, v in
                           df.groupby("class")["agreement"]
                           .apply(lambda s: (s == 1.0).mean()).items()},
    "agreement_overall": float(agreement.mean()),
    "clean_len": {
        "mean": float(df.clean_len.mean()),
        "median": float(df.clean_len.median()),
        "p95": float(df.clean_len.quantile(.95)),
        "max": int(df.clean_len.max()),
        "le50": float((df.clean_len <= 50).mean()),
    },
    "raw_len_mean": float(df.raw_len.mean()),
    "vocab_size": len(vocab),
    "total_tokens": len(all_tokens),
    "distinctive": {int(c): distinct[c] for c in distinct},
    "len_hist_by_class": {
        int(c): np.histogram(df[df["class"] == c].clean_len,
                             bins=range(0, 32))[0].tolist()
        for c in (0, 1, 2)
    },
}
with open(os.path.join(S, "data", "stats.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
df.to_parquet(os.path.join(S, "data", "davidson_clean.parquet"))
print("saved stats.json and cleaned parquet")
