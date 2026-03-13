"""
╔══════════════════════════════════════════════════════════════╗
║     FEDERATED LEARNING FOR FRAUD DETECTION                  ║
║     Phase 2 — FedAvg Training + Comparison vs Baseline      ║
╚══════════════════════════════════════════════════════════════╝

Run AFTER phase1_federated_fraud.py
Loads: client_data.npy, X_test.npy, y_test.npy, baseline_scores.npy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.utils import resample
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

plt.style.use("seaborn-v0_8-darkgrid")
COLORS      = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
CLIENT_COLS = ["#3498db", "#2ecc71", "#f39c12"]

# ── FL Hyperparameters ─────────────────────────────────────────
NUM_ROUNDS    = 10      # communication rounds
LOCAL_EPOCHS  = 5       # local SGD epochs per round
LEARNING_RATE = 0.01
BATCH_SIZE    = 256
# ──────────────────────────────────────────────────────────────

SEPARATOR = "=" * 65


# ══════════════════════════════════════════════════════════════
# LOAD PHASE 1 ARTIFACTS
# ══════════════════════════════════════════════════════════════
print(SEPARATOR)
print("  PHASE 2 — FEDERATED LEARNING (FedAvg)")
print(SEPARATOR)

client_data     = np.load("npy_output/client_data.npy",    allow_pickle=True)
X_test          = np.load("npy_output/X_test.npy",         allow_pickle=True)
y_test          = np.load("npy_output/y_test.npy",         allow_pickle=True)
baseline_scores = np.load("npy_output/baseline_scores.npy", allow_pickle=True).item()

NUM_CLIENTS = len(client_data)
NUM_FEATURES = client_data[0][0].shape[1]

print(f"\n✅ Loaded Phase 1 artifacts")
print(f"   Clients       : {NUM_CLIENTS}")
print(f"   Features      : {NUM_FEATURES}")
print(f"   Test samples  : {len(y_test):,}")
print(f"\n   🎯 Baseline to beat:")
print(f"      AUC-ROC : {baseline_scores['auc_roc']:.4f}")
print(f"      AUPRC   : {baseline_scores['auprc']:.4f}")


# ══════════════════════════════════════════════════════════════
# MODEL — Logistic Regression (weights = w, b)
# ══════════════════════════════════════════════════════════════

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

def forward(X, w, b):
    return sigmoid(X @ w + b)

def predict(X, w, b, threshold=0.5):
    return (forward(X, w, b) >= threshold).astype(int)

def compute_loss(X, y, w, b):
    """Binary cross-entropy loss."""
    p   = forward(X, w, b)
    eps = 1e-9
    return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

def evaluate(X, y, w, b):
    proba = forward(X, w, b)
    preds = (proba >= 0.5).astype(int)
    auc   = roc_auc_score(y, proba)  if len(np.unique(y)) > 1 else 0.0
    auprc = average_precision_score(y, proba) if len(np.unique(y)) > 1 else 0.0
    return auc, auprc, preds, proba


# ══════════════════════════════════════════════════════════════
# LOCAL TRAINING (runs on each client)
# ══════════════════════════════════════════════════════════════

def balance_data(X, y):
    """Oversample fraud to 1:2 ratio for local training."""
    X_fraud = X[y == 1];  X_legit = X[y == 0]
    if len(X_fraud) == 0:
        return X, y
    target_n = min(len(X_legit), len(X_fraud) * 2)
    X_fraud_up = resample(X_fraud, replace=True,
                          n_samples=target_n, random_state=RANDOM_STATE)
    X_bal = np.vstack([X_legit, X_fraud_up])
    y_bal = np.hstack([np.zeros(len(X_legit)), np.ones(target_n)])
    idx   = np.random.permutation(len(y_bal))
    return X_bal[idx], y_bal[idx]

def local_train(X, y, w_init, b_init,
                epochs=LOCAL_EPOCHS, lr=LEARNING_RATE,
                batch_size=BATCH_SIZE):
    """
    Mini-batch SGD on local client data.
    Returns updated (w, b) and training loss history.
    """
    X_bal, y_bal = balance_data(X, y)
    w, b   = w_init.copy(), float(b_init)
    n      = len(y_bal)
    losses = []

    for epoch in range(epochs):
        # shuffle each epoch
        perm  = np.random.permutation(n)
        X_shuf, y_shuf = X_bal[perm], y_bal[perm]
        epoch_loss = 0.0
        batches    = 0

        for start in range(0, n, batch_size):
            Xb = X_shuf[start : start + batch_size]
            yb = y_shuf[start : start + batch_size]
            nb = len(yb)

            p   = forward(Xb, w, b)
            err = p - yb                   # gradient of BCE
            gw  = (Xb.T @ err) / nb
            gb  = err.mean()

            w -= lr * gw
            b -= lr * gb

            epoch_loss += compute_loss(Xb, yb, w, b)
            batches    += 1

        losses.append(epoch_loss / batches)

    return w, b, losses


# ══════════════════════════════════════════════════════════════
# FEDAVG AGGREGATION (runs on server)
# ══════════════════════════════════════════════════════════════

def fed_avg(client_weights, client_biases, client_sizes):
    """
    FedAvg: weighted average of client models.
    Weight = proportion of data each client contributed.
    
    global_w = Σ (n_k / N) * w_k
    """
    total = sum(client_sizes)
    w_agg = sum(w * s for w, s in zip(client_weights, client_sizes)) / total
    b_agg = sum(b * s for b, s in zip(client_biases,  client_sizes)) / total
    return w_agg, b_agg


# ══════════════════════════════════════════════════════════════
# FEDERATED TRAINING LOOP
# ══════════════════════════════════════════════════════════════
print(f"\n{SEPARATOR}")
print(f"  TRAINING — {NUM_ROUNDS} rounds × {LOCAL_EPOCHS} local epochs")
print(SEPARATOR)

# Initialise global model
global_w = np.zeros(NUM_FEATURES)
global_b = 0.0

# History trackers
history = {
    "round":          [],
    "global_auc":     [],
    "global_auprc":   [],
    "client_auc":     [[] for _ in range(NUM_CLIENTS)],
    "client_loss":    [[] for _ in range(NUM_CLIENTS)],
    "client_samples": [len(cd[1]) for cd in client_data],
}

for rnd in range(1, NUM_ROUNDS + 1):
    print(f"\n{'─'*65}")
    print(f"  📡 Round {rnd:>2}/{NUM_ROUNDS}")
    print(f"{'─'*65}")

    client_ws, client_bs, sizes = [], [], []

    for cid in range(NUM_CLIENTS):
        X_c, y_c = client_data[cid]

        # ── Client receives global model, trains locally ──
        local_w, local_b, losses = local_train(
            X_c, y_c, global_w, global_b
        )

        local_auc, local_auprc, _, _ = evaluate(X_c, y_c, local_w, local_b)

        print(f"  Bank {cid}  |  n={len(y_c):>6,}  "
              f"fraud={y_c.sum():>3}  "
              f"local AUC={local_auc:.4f}  "
              f"loss={losses[-1]:.4f}")

        history["client_auc"][cid].append(local_auc)
        history["client_loss"][cid].append(losses[-1])

        client_ws.append(local_w)
        client_bs.append(local_b)
        sizes.append(len(y_c))

    # ── Server aggregates ──
    global_w, global_b = fed_avg(client_ws, client_bs, sizes)

    # ── Evaluate global model on held-out test set ──
    g_auc, g_auprc, _, _ = evaluate(X_test, y_test, global_w, global_b)

    history["round"].append(rnd)
    history["global_auc"].append(g_auc)
    history["global_auprc"].append(g_auprc)

    gap_auc   = baseline_scores["auc_roc"] - g_auc
    gap_auprc = baseline_scores["auprc"]   - g_auprc

    print(f"\n  ✅ Global  |  AUC-ROC={g_auc:.4f} (gap={gap_auc:+.4f})  "
          f"|  AUPRC={g_auprc:.4f} (gap={gap_auprc:+.4f})")


# ══════════════════════════════════════════════════════════════
# FINAL EVALUATION
# ══════════════════════════════════════════════════════════════
print(f"\n{SEPARATOR}")
print("  FINAL RESULTS")
print(SEPARATOR)

final_auc, final_auprc, final_preds, final_proba = evaluate(
    X_test, y_test, global_w, global_b
)

cm     = confusion_matrix(y_test, final_preds)
tn, fp, fn, tp = cm.ravel()

print(f"\n  Federated Learning (after {NUM_ROUNDS} rounds):")
print(f"    AUC-ROC : {final_auc:.4f}  "
      f"(centralised: {baseline_scores['auc_roc']:.4f}  "
      f"gap: {baseline_scores['auc_roc'] - final_auc:+.4f})")
print(f"    AUPRC   : {final_auprc:.4f}  "
      f"(centralised: {baseline_scores['auprc']:.4f}  "
      f"gap: {baseline_scores['auprc'] - final_auprc:+.4f})")
print(f"\n  Confusion Matrix:")
print(f"    Caught fraud (TP) : {tp}  |  Missed fraud (FN) : {fn}")
print(f"    False alarms (FP) : {fp}  |  True legit   (TN) : {tn}")
print(f"\n{classification_report(y_test, final_preds, target_names=['Legit','Fraud'])}")

privacy_overhead = (baseline_scores["auc_roc"] - final_auc) * 100
print(f"  🔒 Privacy cost: {privacy_overhead:.2f}% AUC-ROC reduction")
print(f"     Raw data never left any bank — only weights were shared ✅")


# ══════════════════════════════════════════════════════════════
# VISUALISATIONS
# ══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 14))
fig.suptitle("Phase 2 — Federated Learning Results", 
             fontsize=16, fontweight="bold")
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

rounds = history["round"]
b_auc  = baseline_scores["auc_roc"]
b_prc  = baseline_scores["auprc"]

# ── Plot 1: Global AUC-ROC across rounds ──
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(rounds, history["global_auc"],
         "o-", color=COLORS[0], lw=2.5, ms=7, label="Federated (global)")
ax1.axhline(b_auc, color="gray", lw=2, ls="--",
            label=f"Centralised baseline ({b_auc:.4f})")
ax1.fill_between(rounds, history["global_auc"], b_auc,
                 alpha=0.08, color=COLORS[0])

for cid in range(NUM_CLIENTS):
    ax1.plot(rounds, history["client_auc"][cid],
             "s--", color=CLIENT_COLS[cid], lw=1.2,
             alpha=0.6, ms=4, label=f"Bank {cid} (local)")

ax1.set_xlabel("Communication Round")
ax1.set_ylabel("AUC-ROC")
ax1.set_title("Global Model AUC-ROC vs Centralised Baseline", fontweight="bold")
ax1.legend(loc="lower right", fontsize=9)
ax1.set_ylim(0.85, 1.01)
ax1.set_xticks(rounds)

# ── Plot 2: AUPRC across rounds ──
ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(rounds, history["global_auprc"],
         "o-", color=COLORS[2], lw=2.5, ms=7, label="Federated")
ax2.axhline(b_prc, color="gray", lw=2, ls="--",
            label=f"Centralised ({b_prc:.4f})")
ax2.set_xlabel("Communication Round")
ax2.set_ylabel("AUPRC")
ax2.set_title("AUPRC Convergence\n(Key metric for imbalanced data)",
              fontweight="bold")
ax2.legend(fontsize=9)
ax2.set_xticks(rounds)

# ── Plot 3: Client loss curves ──
ax3 = fig.add_subplot(gs[1, 0])
for cid in range(NUM_CLIENTS):
    ax3.plot(rounds, history["client_loss"][cid],
             "o-", color=CLIENT_COLS[cid], lw=2, ms=5,
             label=f"Bank {cid}")
ax3.set_xlabel("Round"); ax3.set_ylabel("Training Loss")
ax3.set_title("Local Training Loss per Bank", fontweight="bold")
ax3.legend()
ax3.set_xticks(rounds)

# ── Plot 4: ROC comparison ──
ax4 = fig.add_subplot(gs[1, 1])
fpr_f, tpr_f, _ = roc_curve(y_test, final_proba)
ax4.plot(fpr_f, tpr_f, color=COLORS[0], lw=2,
         label=f"Federated  (AUC={final_auc:.4f})")
ax4.plot([0,1],[0,1], "k--", lw=1, label="Random")
ax4.fill_between(fpr_f, tpr_f, alpha=0.08, color=COLORS[0])
ax4.set_xlabel("False Positive Rate")
ax4.set_ylabel("True Positive Rate")
ax4.set_title("Final ROC Curve", fontweight="bold")
ax4.legend(loc="lower right")

# ── Plot 5: Precision-Recall ──
ax5 = fig.add_subplot(gs[1, 2])
prec_f, rec_f, _ = precision_recall_curve(y_test, final_proba)
ax5.plot(rec_f, prec_f, color=COLORS[2], lw=2,
         label=f"Federated  (AUPRC={final_auprc:.4f})")
ax5.axhline(y_test.mean(), color="gray", lw=1, ls="--",
            label="Random baseline")
ax5.fill_between(rec_f, prec_f, alpha=0.08, color=COLORS[2])
ax5.set_xlabel("Recall"); ax5.set_ylabel("Precision")
ax5.set_title("Final Precision-Recall Curve", fontweight="bold")
ax5.legend()

plt.savefig("png_output/phase2_federated_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✅ Saved: png_output/phase2_federated_results.png")

# ── Save for Phase 3 ──
np.save("npy_output/global_model.npy",  {"w": global_w, "b": global_b})
np.save("npy_output/fl_history.npy",    history)

print(f"\n{SEPARATOR}")
print("  PHASE 2 COMPLETE ✅")
print(SEPARATOR)
print(f"""
  Saved:
    📊 phase2_federated_results.png  — full results dashboard
    💾 global_model.npy              — trained FL model weights
    💾 fl_history.npy                — round-by-round history

  Summary:
    Federated AUC-ROC : {final_auc:.4f}  (centralised: {b_auc:.4f})
    Federated AUPRC   : {final_auprc:.4f}  (centralised: {b_prc:.4f})
    Privacy cost      : {privacy_overhead:.2f}% AUC reduction
    Data shared       : ZERO raw transactions — only model weights ✅

  → Next: phase3_fedprox_vs_fedavg.py
    We'll tackle the non-IID problem head-on with FedProx
    and show the privacy-accuracy trade-off with Differential Privacy
""")
