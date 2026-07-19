import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
S = r"C:\Users\jains\AppData\Local\Temp\claude\c--Desktop-Hate-Speech-Detection-\28db352e-a58e-413a-a1b3-1c38eabd9d6a\scratchpad"
D = json.load(open(os.path.join(S, "data", "report_data.json"), encoding="utf-8"))
NAMES = ["Hate speech", "Offensive language", "Neither"]

b = D["models"]["BiLSTM_baseline"]
t = D["models"]["BiLSTM_tuned"]

print("                       BEFORE (balanced)   AFTER (tuned)     delta")
for k in ("accuracy", "precision_w", "recall_w", "f1_w", "auc_macro"):
    print(f"  {k:14s}  {b[k]:18.4f} {t[k]:15.4f} {t[k]-b[k]:+10.4f}")

print("\nPER-CLASS")
for i, n in enumerate(NAMES):
    print(f"\n  {n} (support {t['per_class']['support'][i]})")
    for k in ("precision", "recall", "f1"):
        bv, tv = b["per_class"][k][i], t["per_class"][k][i]
        print(f"    {k:10s} {bv:.4f} -> {tv:.4f}   ({tv-bv:+.4f})")

print("\nCONFUSION BEFORE")
print(np.array(b["confusion_matrix"]))
print("CONFUSION AFTER")
print(np.array(t["confusion_matrix"]))

cb, ct = np.array(b["confusion_matrix"]), np.array(t["confusion_matrix"])
print("\noffensive wrongly called hate:  before %d  after %d" % (cb[1][0], ct[1][0]))
print("neutral wrongly called harmful: before %d  after %d"
      % (cb[2][0] + cb[2][1], ct[2][0] + ct[2][1]))
print("hate caught (as hate):          before %d  after %d" % (cb[0][0], ct[0][0]))
print("hate missed entirely (neutral): before %d  after %d" % (cb[0][2], ct[0][2]))
