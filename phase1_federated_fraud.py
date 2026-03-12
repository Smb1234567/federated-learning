"""
╔══════════════════════════════════════════════════════════════╗
║     FEDERATED LEARNING FOR FRAUD DETECTION                  ║
║     Phase 1 — Data Exploration + Non-IID Split + Baseline   ║
╚══════════════════════════════════════════════════════════════╝

Run:  python phase1_federated_fraud.py
      (change DATA_PATH to wherever your creditcard.csv lives)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score
)
from sklearn.utils import resample
import warnings
warnings.filterwarnings("ignore")

# ── change this to your local path ──────────────────────────
DATA_PATH = "/home/hemanth/Downloads/datasets/creditcard.csv"
# ────────────────────────────────────────────────────────────

RANDOM_STATE = 42
NUM_CLIENTS  = 3          # simulating 3 banks
np.random.seed(RANDOM_STATE)

plt.style.use("seaborn-v0_8-darkgrid")
COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]


# ══════════════════════════════════════════════════════════════
# STEP 1 — LOAD & BASIC EXPLORATION
# ══════════════════════════════════════════════════════════════
print("=" * 65)
print("  PHASE 1 — DATA EXPLORATION")
print("=" * 65)

df = pd.read_csv(DATA_PATH)

print(f"\n📦 Dataset shape      : {df.shape}")
print(f"   Total transactions : {len(df):,}")
print(f"   Features           : {df.shape[1] - 1}")
print(f"   Missing values     : {df.isnull().sum().sum()}")

fraud     = df[df["Class"] == 1]
legit     = df[df["Class"] == 0]
fraud_pct = len(fraud) / len(df) * 100

print(f"\n💳 Class Distribution:")
print(f"   Legitimate  : {len(legit):>7,}  ({100 - fraud_pct:.2f}%)")
print(f"   Fraud       : {len(fraud):>7,}  ({fraud_pct:.4f}%)")
print(f"\n   ⚠️  Imbalance ratio: 1 fraud per {len(legit)//len(fraud)} legitimate txns")

print(f"\n💰 Transaction Amounts:")
print(f"   Fraud  — mean: €{fraud['Amount'].mean():.2f}  "
      f"max: €{fraud['Amount'].max():.2f}")
print(f"   Legit  — mean: €{legit['Amount'].mean():.2f}  "
      f"max: €{legit['Amount'].max():.2f}")


# ══════════════════════════════════════════════════════════════
# STEP 2 — VISUALISE
# ══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 14))
fig.suptitle("Phase 1 — Credit Card Fraud: Data Exploration", 
             fontsize=16, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── 2a. Class imbalance bar ──
ax1 = fig.add_subplot(gs[0, 0])
counts = [len(legit), len(fraud)]
bars = ax1.bar(["Legitimate", "Fraud"], counts,
               color=[COLORS[1], COLORS[0]], width=0.5, edgecolor="white")
ax1.set_title("Class Imbalance", fontweight="bold")
ax1.set_ylabel("Count")
for bar, count in zip(bars, counts):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.5,
             f"{count:,}", ha="center", va="center",
             color="white", fontweight="bold", fontsize=11)

# ── 2b. Fraud % pie ──
ax2 = fig.add_subplot(gs[0, 1])
ax2.pie([len(legit), len(fraud)],
        labels=["Legit\n99.83%", "Fraud\n0.17%"],
        colors=[COLORS[1], COLORS[0]],
        explode=[0, 0.12], startangle=90,
        textprops={"fontsize": 10})
ax2.set_title("Fraud Proportion", fontweight="bold")

# ── 2c. Amount distribution ──
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(legit["Amount"].clip(upper=500), bins=60,
         alpha=0.6, color=COLORS[1], label="Legit", density=True)
ax3.hist(fraud["Amount"].clip(upper=500), bins=60,
         alpha=0.7, color=COLORS[0], label="Fraud", density=True)
ax3.set_title("Transaction Amount Distribution\n(clipped at €500)", fontweight="bold")
ax3.set_xlabel("Amount (€)")
ax3.set_ylabel("Density")
ax3.legend()

# ── 2d. Transactions over time ──
ax4 = fig.add_subplot(gs[1, :2])
ax4.scatter(legit["Time"] / 3600, legit["Amount"],
            alpha=0.05, s=1, color=COLORS[1], label="Legit")
ax4.scatter(fraud["Time"] / 3600, fraud["Amount"],
            alpha=0.5, s=8, color=COLORS[0], label="Fraud", zorder=5)
ax4.set_title("Transactions Over Time — Fraud Highlighted", fontweight="bold")
ax4.set_xlabel("Time (hours)")
ax4.set_ylabel("Amount (€)")
ax4.legend(markerscale=4)

# ── 2e. Top PCA features — fraud vs legit ──
ax5 = fig.add_subplot(gs[1, 2])
top_features = ["V4", "V11", "V14", "V17", "V12"]
fraud_means  = [fraud[f].mean()  for f in top_features]
legit_means  = [legit[f].mean()  for f in top_features]
x = np.arange(len(top_features))
w = 0.35
ax5.bar(x - w/2, legit_means,  w, label="Legit",  color=COLORS[1], alpha=0.8)
ax5.bar(x + w/2, fraud_means,  w, label="Fraud",  color=COLORS[0], alpha=0.8)
ax5.set_xticks(x); ax5.set_xticklabels(top_features)
ax5.set_title("Top PCA Features\n(Fraud vs Legit mean)", fontweight="bold")
ax5.axhline(0, color="gray", lw=0.8)
ax5.legend()

# ── 2f. Correlation heatmap (small) ──
ax6 = fig.add_subplot(gs[2, :])
top_corr_features = (
    df.drop("Time", axis=1)
      .corr()["Class"]
      .abs()
      .sort_values(ascending=False)[1:13]
      .index.tolist()
)
corr_sub = df[top_corr_features + ["Class"]].corr()
mask = np.triu(np.ones_like(corr_sub, dtype=bool))
sns.heatmap(corr_sub, ax=ax6, mask=mask, cmap="coolwarm",
            center=0, linewidths=0.5, annot=True, fmt=".2f",
            annot_kws={"size": 7}, cbar_kws={"shrink": 0.8})
ax6.set_title("Correlation Matrix — Top Features Most Correlated with Fraud",
              fontweight="bold")

plt.savefig("phase1_exploration.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✅ Saved: phase1_exploration.png")


# ══════════════════════════════════════════════════════════════
# STEP 3 — PREPROCESSING
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  STEP 3 — PREPROCESSING")
print("=" * 65)

# Normalize Time and Amount (V1-V28 are already PCA scaled)
scaler = StandardScaler()
df["Time_scaled"]   = scaler.fit_transform(df[["Time"]])
df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Time_scaled", "Amount_scaled"]
TARGET   = "Class"

X = df[FEATURES].values
y = df[TARGET].values

print(f"\n   Features used      : {len(FEATURES)} (V1-V28 + Time + Amount scaled)")
print(f"   X shape            : {X.shape}")
print(f"   y shape            : {y.shape}")


# ══════════════════════════════════════════════════════════════
# STEP 4 — NON-IID SPLIT ACROSS 3 "BANKS"
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  STEP 4 — NON-IID SPLIT ACROSS 3 BANKS")
print("=" * 65)

"""
NON-IID Strategy:
  Each bank gets transactions from a DIFFERENT time window
  + different fraud rates to simulate real-world heterogeneity.

  Bank 0 (Small  bank) → first 33% of time, LOW fraud rate
  Bank 1 (Mid    bank) → middle 33%,         MEDIUM fraud rate  
  Bank 2 (Large  bank) → last 33%,           HIGH fraud rate

  This means each bank's data distribution is genuinely different
  — the hardest and most realistic FL scenario.
"""

time_sorted_idx = np.argsort(df["Time"].values)
n = len(time_sorted_idx)
splits = np.array_split(time_sorted_idx, NUM_CLIENTS)

client_data = []
print(f"\n{'Bank':<8} {'Samples':>9} {'Fraud':>7} {'Fraud%':>8} {'Time window':>20}")
print("-" * 58)

for cid, idx in enumerate(splits):
    X_c = X[idx]
    y_c = y[idx]

    # Introduce heterogeneity: undersample/oversample fraud per bank
    fraud_idx = np.where(y_c == 1)[0]
    legit_idx = np.where(y_c == 0)[0]

    # Bank 0: keep 40% of fraud (low fraud bank)
    # Bank 1: keep 70% of fraud
    # Bank 2: keep 100% of fraud (high fraud bank)
    keep_ratios = [0.4, 0.7, 1.0]
    keep_n = max(1, int(len(fraud_idx) * keep_ratios[cid]))
    chosen_fraud = np.random.choice(fraud_idx, keep_n, replace=False)
    final_idx = np.concatenate([legit_idx, chosen_fraud])
    np.random.shuffle(final_idx)

    X_c_final = X_c[final_idx]
    y_c_final = y_c[final_idx]

    t_min = df["Time"].values[idx].min() / 3600
    t_max = df["Time"].values[idx].max() / 3600

    print(f"Bank {cid}   {len(y_c_final):>9,} {y_c_final.sum():>7} "
          f"{y_c_final.mean()*100:>7.3f}%  "
          f"  {t_min:.1f}h – {t_max:.1f}h")

    client_data.append((X_c_final, y_c_final))

# ── Visualise the non-IID split ──
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Phase 1 — Non-IID Data Split Across 3 Banks",
             fontsize=14, fontweight="bold")

for cid, (X_c, y_c) in enumerate(client_data):
    ax = axes[cid]
    fraud_c = y_c.sum()
    legit_c = (y_c == 0).sum()
    ax.bar(["Legit", "Fraud"], [legit_c, fraud_c],
           color=[COLORS[1], COLORS[0]], edgecolor="white", alpha=0.85)
    ax.set_title(f"Bank {cid}\n{len(y_c):,} transactions | "
                 f"{y_c.mean()*100:.3f}% fraud",
                 fontweight="bold")
    ax.set_ylabel("Count")
    for i, v in enumerate([legit_c, fraud_c]):
        ax.text(i, v * 0.5, f"{v:,}", ha="center", va="center",
                color="white", fontweight="bold")

plt.tight_layout()
plt.savefig("phase1_noniid_split.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✅ Saved: phase1_noniid_split.png")


# ══════════════════════════════════════════════════════════════
# STEP 5 — CENTRALISED BASELINE
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  STEP 5 — CENTRALISED BASELINE (the score FL must beat)")
print("=" * 65)

# Global train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

"""
Class Imbalance Handling — SMOTE alternative (simple oversampling):
  We oversample the minority class in training only.
  NEVER oversample test data — that would be cheating.
"""
X_tr_fraud = X_train[y_train == 1]
X_tr_legit = X_train[y_train == 0]

X_tr_fraud_up = resample(X_tr_fraud,
                          replace=True,
                          n_samples=len(X_tr_legit) // 2,
                          random_state=RANDOM_STATE)

X_train_bal = np.vstack([X_tr_legit, X_tr_fraud_up])
y_train_bal = np.hstack([np.zeros(len(X_tr_legit)),
                          np.ones(len(X_tr_fraud_up))])

shuffle_idx = np.random.permutation(len(y_train_bal))
X_train_bal = X_train_bal[shuffle_idx]
y_train_bal = y_train_bal[shuffle_idx]

print(f"\n   Train (balanced) : {X_train_bal.shape}  "
      f"| fraud: {y_train_bal.mean()*100:.1f}%")
print(f"   Test  (real)     : {X_test.shape}  "
      f"| fraud: {y_test.mean()*100:.4f}%")

# Train centralised model
clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=0.1)
clf.fit(X_train_bal, y_train_bal)

y_pred       = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

auc_roc = roc_auc_score(y_test, y_pred_proba)
avg_prc = average_precision_score(y_test, y_pred_proba)
cm      = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

print(f"\n📊 CENTRALISED BASELINE RESULTS")
print(f"   AUC-ROC              : {auc_roc:.4f}")
print(f"   Avg Precision (AUPRC): {avg_prc:.4f}")
print(f"\n   Confusion Matrix:")
print(f"   {'':20} Predicted Legit  Predicted Fraud")
print(f"   Actual Legit     {tn:>12,}  {fp:>14,}")
print(f"   Actual Fraud     {fn:>12,}  {tp:>14,}")
print(f"\n   True Positives  (caught fraud)   : {tp}")
print(f"   False Negatives (missed fraud)   : {fn}  ← minimise this!")
print(f"   False Positives (false alarms)   : {fp}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Legit','Fraud'])}")

# Store baseline for Phase 2 comparison
baseline_scores = {
    "auc_roc": auc_roc,
    "auprc":   avg_prc,
    "tp": tp, "fp": fp, "tn": tn, "fn": fn
}

# ── Visualise baseline results ──
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Phase 1 — Centralised Baseline Performance",
             fontsize=14, fontweight="bold")

# ROC curve
ax = axes[0]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
ax.plot(fpr, tpr, color=COLORS[0], lw=2,
        label=f"Centralised (AUC = {auc_roc:.4f})")
ax.plot([0,1],[0,1], "k--", lw=1, label="Random")
ax.fill_between(fpr, tpr, alpha=0.08, color=COLORS[0])
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve", fontweight="bold")
ax.legend(loc="lower right")

# Precision-Recall curve
ax = axes[1]
prec, rec, _ = precision_recall_curve(y_test, y_pred_proba)
ax.plot(rec, prec, color=COLORS[2], lw=2,
        label=f"AUPRC = {avg_prc:.4f}")
ax.axhline(y=y_test.mean(), color="gray", lw=1, linestyle="--",
           label=f"Random ({y_test.mean():.4f})")
ax.fill_between(rec, prec, alpha=0.08, color=COLORS[2])
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve\n(Better for imbalanced data)",
             fontweight="bold")
ax.legend()

# Confusion matrix
ax = axes[2]
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
sns.heatmap(cm_pct, annot=True, fmt=".1f", ax=ax,
            cmap="Blues", cbar=False,
            xticklabels=["Pred Legit", "Pred Fraud"],
            yticklabels=["Actual Legit", "Actual Fraud"],
            annot_kws={"size": 13, "weight": "bold"})
ax.set_title("Confusion Matrix (%)", fontweight="bold")

# Add raw numbers as subtitle
ax.text(0.5, -0.15, f"TP={tp}  FP={fp}  TN={tn}  FN={fn}",
        transform=ax.transAxes, ha="center", fontsize=9, color="gray")

plt.tight_layout()
plt.savefig("phase1_baseline_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved: phase1_baseline_results.png")

# ── Save everything for Phase 2 ──
np.save("client_data.npy",    np.array(client_data, dtype=object))
np.save("X_test.npy",         X_test)
np.save("y_test.npy",         y_test)
np.save("baseline_scores.npy", baseline_scores)

print("\n" + "=" * 65)
print("  PHASE 1 COMPLETE ✅")
print("=" * 65)
print(f"""
  Files saved:
    📊 phase1_exploration.png      — dataset overview
    📊 phase1_noniid_split.png     — bank split visualisation  
    📊 phase1_baseline_results.png — centralised model results
    💾 client_data.npy             — 3 banks' local datasets
    💾 X_test.npy / y_test.npy     — shared test set
    💾 baseline_scores.npy         — scores to beat in Phase 2

  Centralised Baseline to beat in Phase 2:
    AUC-ROC : {auc_roc:.4f}
    AUPRC   : {avg_prc:.4f}
    
  → Now run phase2_federated.py
""")
