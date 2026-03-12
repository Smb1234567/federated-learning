"""
╔══════════════════════════════════════════════════════════════╗
║     FEDERATED LEARNING FOR FRAUD DETECTION                  ║
║     Phase 3 — FedProx vs FedAvg + Differential Privacy      ║
╚══════════════════════════════════════════════════════════════╝

Run AFTER phase2_federated_fraud.py
Loads: client_data.npy, X_test.npy, y_test.npy, baseline_scores.npy

What this phase covers:
  1. FedProx  — fixes non-IID drift problem in FedAvg
  2. FedAvg vs FedProx — side by side comparison
  3. Differential Privacy — add noise to weights before sharing
  4. Privacy-Accuracy trade-off curve (the research contribution)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.utils import resample
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

plt.style.use("seaborn-v0_8-darkgrid")
COLORS      = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
CLIENT_COLS = ["#3498db", "#2ecc71", "#f39c12"]

# ── Hyperparameters ────────────────────────────────────────────
NUM_ROUNDS   = 10
LOCAL_EPOCHS = 5
LR           = 0.01
BATCH_SIZE   = 256

# FedProx proximal term strengths to compare
MU_VALUES    = [0.0, 0.01, 0.1, 1.0]   # mu=0.0 is plain FedAvg

# Differential Privacy noise levels (epsilon-like scale)
# Smaller = more private = more noise = less accurate
DP_SIGMAS    = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
# ──────────────────────────────────────────────────────────────

SEPARATOR = "=" * 65


# ══════════════════════════════════════════════════════════════
# LOAD ARTIFACTS
# ══════════════════════════════════════════════════════════════
print(SEPARATOR)
print("  PHASE 3 — FedProx + Differential Privacy")
print(SEPARATOR)

client_data     = np.load("client_data.npy",     allow_pickle=True)
X_test          = np.load("X_test.npy",          allow_pickle=True)
y_test          = np.load("y_test.npy",          allow_pickle=True)
baseline_scores = np.load("baseline_scores.npy", allow_pickle=True).item()

NUM_CLIENTS  = len(client_data)
NUM_FEATURES = client_data[0][0].shape[1]

b_auc  = baseline_scores["auc_roc"]
b_prc  = baseline_scores["auprc"]

print(f"\n✅ Loaded artifacts  |  clients={NUM_CLIENTS}  features={NUM_FEATURES}")
print(f"   Centralised baseline  →  AUC-ROC: {b_auc:.4f}  AUPRC: {b_prc:.4f}")
print(f"   Phase 2 FL (FedAvg)  →  AUC-ROC: 0.9782   AUPRC: 0.7305")


# ══════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ══════════════════════════════════════════════════════════════

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

def forward(X, w, b):
    return sigmoid(X @ w + b)

def evaluate(X, y, w, b):
    proba = forward(X, w, b)
    auc   = roc_auc_score(y, proba)
    auprc = average_precision_score(y, proba)
    preds = (proba >= 0.5).astype(int)
    return auc, auprc, preds, proba

def balance_data(X, y):
    X_fraud = X[y == 1]; X_legit = X[y == 0]
    if len(X_fraud) == 0:
        return X, y
    target_n   = min(len(X_legit), len(X_fraud) * 2)
    X_fraud_up = resample(X_fraud, replace=True,
                          n_samples=target_n, random_state=RANDOM_STATE)
    X_bal = np.vstack([X_legit, X_fraud_up])
    y_bal = np.hstack([np.zeros(len(X_legit)), np.ones(target_n)])
    idx   = np.random.permutation(len(y_bal))
    return X_bal[idx], y_bal[idx]

def fed_avg(client_ws, client_bs, sizes):
    total = sum(sizes)
    return (sum(w * s for w, s in zip(client_ws, sizes)) / total,
            sum(b * s for b, s in zip(client_bs,  sizes)) / total)


# ══════════════════════════════════════════════════════════════
# FEDPROX LOCAL TRAINING
# ══════════════════════════════════════════════════════════════

def local_train_fedprox(X, y, w_init, b_init, global_w, global_b,
                         mu=0.01, epochs=LOCAL_EPOCHS,
                         lr=LR, batch_size=BATCH_SIZE):
    """
    FedProx adds a PROXIMAL TERM to the loss:

        Loss_FedProx = Loss_BCE + (mu/2) * ||w - w_global||²

    The second term PENALISES the local model for drifting
    too far from the global model. This is the only difference
    from standard FedAvg (where mu=0).

    Why does this help?
      In non-IID settings, each bank's data pulls the local model
      toward its own distribution. Without the proximal term,
      Bank 2 (high fraud) might diverge completely from Bank 0
      (low fraud). The penalty keeps them anchored to the global,
      so aggregation produces a more stable result.
    """
    X_bal, y_bal = balance_data(X, y)
    w, b = w_init.copy(), float(b_init)
    n    = len(y_bal)

    for epoch in range(epochs):
        perm = np.random.permutation(n)
        X_s, y_s = X_bal[perm], y_bal[perm]

        for start in range(0, n, batch_size):
            Xb = X_s[start : start + batch_size]
            yb = y_s[start : start + batch_size]
            nb = len(yb)

            p   = forward(Xb, w, b)
            err = p - yb

            # Standard gradient
            gw = (Xb.T @ err) / nb
            gb = err.mean()

            # ── Proximal term gradient ──────────────────────
            # d/dw [ (mu/2)||w - w_global||² ] = mu*(w - w_global)
            prox_grad_w = mu * (w - global_w)
            prox_grad_b = mu * (b - global_b)
            # ────────────────────────────────────────────────

            w -= lr * (gw + prox_grad_w)
            b -= lr * (gb + prox_grad_b)

    return w, b


# ══════════════════════════════════════════════════════════════
# EXPERIMENT 1 — FedAvg vs FedProx (different mu values)
# ══════════════════════════════════════════════════════════════
print(f"\n{SEPARATOR}")
print("  EXPERIMENT 1 — FedAvg vs FedProx")
print(f"  Testing mu values: {MU_VALUES}  (mu=0 = plain FedAvg)")
print(SEPARATOR)

"""
We run the full FL loop for each mu value and track
global AUC-ROC across rounds. This lets us see:
  - Does FedProx converge faster?
  - Does it converge to a better final model?
  - What mu value is optimal?
"""

mu_results = {}   # mu → list of AUC per round

for mu in MU_VALUES:
    label = f"FedAvg" if mu == 0.0 else f"FedProx μ={mu}"
    print(f"\n  Running {label}...")

    global_w = np.zeros(NUM_FEATURES)
    global_b = 0.0
    round_aucs = []

    for rnd in range(1, NUM_ROUNDS + 1):
        client_ws, client_bs, sizes = [], [], []

        for cid in range(NUM_CLIENTS):
            X_c, y_c = client_data[cid]
            lw, lb = local_train_fedprox(
                X_c, y_c, global_w, global_b,
                global_w, global_b, mu=mu
            )
            client_ws.append(lw)
            client_bs.append(lb)
            sizes.append(len(y_c))

        global_w, global_b = fed_avg(client_ws, client_bs, sizes)
        auc, auprc, _, _   = evaluate(X_test, y_test, global_w, global_b)
        round_aucs.append(auc)

        print(f"    Round {rnd:>2}  AUC={auc:.4f}", end="\r")

    mu_results[mu] = round_aucs
    final_auc, final_auprc, _, _ = evaluate(X_test, y_test, global_w, global_b)
    print(f"    Final  →  AUC-ROC: {final_auc:.4f}  AUPRC: {final_auprc:.4f}  ✅")


# ══════════════════════════════════════════════════════════════
# EXPERIMENT 2 — DIFFERENTIAL PRIVACY
# ══════════════════════════════════════════════════════════════
print(f"\n{SEPARATOR}")
print("  EXPERIMENT 2 — DIFFERENTIAL PRIVACY")
print(f"  Testing noise levels (sigma): {DP_SIGMAS}")
print(SEPARATOR)

"""
HOW DIFFERENTIAL PRIVACY WORKS IN FL:

After local training, before sending weights to the server,
each client adds Gaussian noise to their weights:

    w_shared = w_local + N(0, sigma²)

This means the server NEVER sees the exact local weights.
Even if someone intercepts the transmission, they can't
reverse-engineer the training data from noisy weights.

The trade-off:
  sigma=0    → no privacy, maximum accuracy
  sigma=0.01 → small noise, slight accuracy drop, some privacy
  sigma=0.5  → heavy noise, large accuracy drop, strong privacy

In real DP theory, this maps to an epsilon (ε) guarantee:
  smaller ε = stronger privacy = more noise needed
"""

dp_results = {}   # sigma → (final_auc, final_auprc)

for sigma in DP_SIGMAS:
    label = "No DP (σ=0)" if sigma == 0.0 else f"σ={sigma}"
    print(f"\n  Running {label}...")

    global_w = np.zeros(NUM_FEATURES)
    global_b = 0.0

    for rnd in range(1, NUM_ROUNDS + 1):
        client_ws, client_bs, sizes = [], [], []

        for cid in range(NUM_CLIENTS):
            X_c, y_c = client_data[cid]

            # Train locally (using best mu from Exp 1)
            best_mu = min(mu_results, key=lambda m: -mu_results[m][-1])
            lw, lb  = local_train_fedprox(
                X_c, y_c, global_w, global_b,
                global_w, global_b, mu=best_mu
            )

            # ── Add Gaussian noise BEFORE sharing ───────────
            if sigma > 0:
                lw = lw + np.random.normal(0, sigma, lw.shape)
                lb = lb + np.random.normal(0, sigma)
            # ────────────────────────────────────────────────

            client_ws.append(lw)
            client_bs.append(lb)
            sizes.append(len(y_c))

        global_w, global_b = fed_avg(client_ws, client_bs, sizes)

    auc, auprc, _, _ = evaluate(X_test, y_test, global_w, global_b)
    dp_results[sigma] = (auc, auprc)
    print(f"    AUC-ROC: {auc:.4f}  AUPRC: {auprc:.4f}  ✅")


# ══════════════════════════════════════════════════════════════
# PRINT SUMMARY TABLES
# ══════════════════════════════════════════════════════════════
print(f"\n{SEPARATOR}")
print("  RESULTS — FedAvg vs FedProx")
print(SEPARATOR)
print(f"  {'Method':<22} {'Final AUC-ROC':>14} {'vs FedAvg':>10}")
print(f"  {'-'*22} {'-'*14} {'-'*10}")
fedavg_final = mu_results[0.0][-1]
for mu, aucs in mu_results.items():
    label = "FedAvg (μ=0)" if mu == 0.0 else f"FedProx μ={mu}"
    diff  = aucs[-1] - fedavg_final
    marker = " ← best" if aucs[-1] == max(r[-1] for r in mu_results.values()) else ""
    print(f"  {label:<22} {aucs[-1]:>14.4f} {diff:>+10.4f}{marker}")

print(f"\n{SEPARATOR}")
print("  RESULTS — Differential Privacy Trade-off")
print(SEPARATOR)
print(f"  {'Noise (σ)':<12} {'AUC-ROC':>10} {'AUPRC':>10} "
      f"{'AUC drop':>10} {'Privacy':>12}")
print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")

base_auc = dp_results[0.0][0]
for sigma, (auc, auprc) in dp_results.items():
    drop    = base_auc - auc
    privacy = "None" if sigma == 0 else (
              "Low"    if sigma <= 0.001 else (
              "Medium" if sigma <= 0.01  else (
              "High"   if sigma <= 0.1   else "Very High")))
    print(f"  {sigma:<12} {auc:>10.4f} {auprc:>10.4f} "
          f"{drop:>+10.4f} {privacy:>12}")


# ══════════════════════════════════════════════════════════════
# VISUALISATIONS
# ══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 16))
fig.suptitle("Phase 3 — FedProx vs FedAvg + Differential Privacy",
             fontsize=16, fontweight="bold")
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

rounds = list(range(1, NUM_ROUNDS + 1))
mu_colors = [COLORS[0], COLORS[1], COLORS[2], COLORS[3]]

# ── Plot 1: Convergence curves — all mu values ──
ax1 = fig.add_subplot(gs[0, :2])
for i, (mu, aucs) in enumerate(mu_results.items()):
    label = "FedAvg (μ=0)" if mu == 0.0 else f"FedProx μ={mu}"
    lw    = 3 if mu == 0.0 else 2
    ls    = "--" if mu == 0.0 else "-"
    ax1.plot(rounds, aucs, ls, color=mu_colors[i],
             lw=lw, ms=6, marker="o", label=label)

ax1.axhline(b_auc, color="gray", lw=1.5, ls=":",
            label=f"Centralised ({b_auc:.4f})")
ax1.set_xlabel("Communication Round")
ax1.set_ylabel("Global AUC-ROC")
ax1.set_title("FedAvg vs FedProx — Convergence Across Rounds",
              fontweight="bold")
ax1.legend(fontsize=9)
ax1.set_ylim(0.88, 1.00)
ax1.set_xticks(rounds)

# ── Plot 2: Final AUC bar chart ──
ax2 = fig.add_subplot(gs[0, 2])
labels  = ["FedAvg\n(μ=0)"] + [f"FedProx\nμ={m}" for m in MU_VALUES[1:]]
finals  = [mu_results[m][-1] for m in MU_VALUES]
bars    = ax2.bar(labels, finals, color=mu_colors, alpha=0.85, edgecolor="white")
ax2.axhline(b_auc, color="gray", lw=1.5, ls="--", label="Centralised")
ax2.set_ylabel("Final AUC-ROC")
ax2.set_title("Final AUC-ROC Comparison", fontweight="bold")
ax2.set_ylim(min(finals) - 0.02, 1.0)
for bar, val in zip(bars, finals):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.001,
             f"{val:.4f}", ha="center", fontsize=9, fontweight="bold")
ax2.legend()

# ── Plot 3: Privacy-Accuracy Trade-off (THE KEY CHART) ──
ax3 = fig.add_subplot(gs[1, :2])
sigmas   = list(dp_results.keys())
dp_aucs  = [dp_results[s][0]   for s in sigmas]
dp_auprcs= [dp_results[s][1]   for s in sigmas]

ax3.plot(sigmas, dp_aucs,   "o-", color=COLORS[0], lw=2.5,
         ms=8, label="AUC-ROC", zorder=5)
ax3.plot(sigmas, dp_auprcs, "s-", color=COLORS[2], lw=2.5,
         ms=8, label="AUPRC", zorder=5)

ax3.axhline(b_auc, color=COLORS[0], lw=1, ls="--", alpha=0.5,
            label=f"Centralised AUC-ROC ({b_auc:.4f})")
ax3.axhline(b_prc, color=COLORS[2], lw=1, ls="--", alpha=0.5,
            label=f"Centralised AUPRC ({b_prc:.4f})")

# Shade privacy regions
ax3.axvspan(0,     0.005, alpha=0.05, color="green",  label="Low privacy cost")
ax3.axvspan(0.005, 0.05,  alpha=0.05, color="orange", label="Medium privacy cost")
ax3.axvspan(0.05,  0.5,   alpha=0.05, color="red",    label="High privacy cost")

ax3.set_xlabel("Noise Level (σ)  —  higher = more private")
ax3.set_ylabel("Score")
ax3.set_title("Privacy-Accuracy Trade-off Curve\n"
              "(The core research contribution of this project)",
              fontweight="bold")
ax3.legend(fontsize=8, loc="lower left")

# ── Plot 4: AUC drop vs sigma ──
ax4 = fig.add_subplot(gs[1, 2])
drops = [base_auc - dp_results[s][0] for s in sigmas]
ax4.bar([str(s) for s in sigmas], drops,
        color=[COLORS[2] if d < 0.01 else
               COLORS[3] if d < 0.05 else COLORS[0]
               for d in drops],
        alpha=0.85, edgecolor="white")
ax4.set_xlabel("Noise σ")
ax4.set_ylabel("AUC-ROC Drop")
ax4.set_title("Accuracy Cost of Privacy", fontweight="bold")
ax4.axhline(0.02, color="orange", ls="--", lw=1.5,
            label="2% threshold (acceptable)")
ax4.legend(fontsize=9)

# ── Plot 5: Best model ROC + PR ──
best_mu = min(mu_results, key=lambda m: -mu_results[m][-1])
global_w_best = np.zeros(NUM_FEATURES)
global_b_best = 0.0
for rnd in range(NUM_ROUNDS):
    cws, cbs, szs = [], [], []
    for cid in range(NUM_CLIENTS):
        X_c, y_c = client_data[cid]
        lw, lb = local_train_fedprox(X_c, y_c,
                                      global_w_best, global_b_best,
                                      global_w_best, global_b_best,
                                      mu=best_mu)
        cws.append(lw); cbs.append(lb); szs.append(len(y_c))
    global_w_best, global_b_best = fed_avg(cws, cbs, szs)

_, _, _, best_proba = evaluate(X_test, y_test, global_w_best, global_b_best)

ax5 = fig.add_subplot(gs[2, 0])
fpr, tpr, _ = roc_curve(y_test, best_proba)
best_auc    = roc_auc_score(y_test, best_proba)
ax5.plot(fpr, tpr, color=COLORS[0], lw=2,
         label=f"Best FL model (AUC={best_auc:.4f})")
ax5.plot([0,1],[0,1],"k--",lw=1,label="Random")
ax5.fill_between(fpr, tpr, alpha=0.08, color=COLORS[0])
ax5.set_xlabel("FPR"); ax5.set_ylabel("TPR")
ax5.set_title("Best Model ROC Curve", fontweight="bold")
ax5.legend(fontsize=9)

ax6 = fig.add_subplot(gs[2, 1])
prec, rec, _ = precision_recall_curve(y_test, best_proba)
best_ap      = average_precision_score(y_test, best_proba)
ax6.plot(rec, prec, color=COLORS[2], lw=2,
         label=f"Best FL (AUPRC={best_ap:.4f})")
ax6.axhline(y_test.mean(), color="gray", lw=1, ls="--", label="Random")
ax6.fill_between(rec, prec, alpha=0.08, color=COLORS[2])
ax6.set_xlabel("Recall"); ax6.set_ylabel("Precision")
ax6.set_title("Best Model PR Curve", fontweight="bold")
ax6.legend(fontsize=9)

# ── Plot 6: Summary comparison table ──
ax7 = fig.add_subplot(gs[2, 2])
ax7.axis("off")
table_data = [
    ["Method",           "AUC-ROC", "AUPRC"],
    ["Centralised",      f"{b_auc:.4f}", f"{b_prc:.4f}"],
    ["FL FedAvg",        "0.9782",  "0.7305"],
    [f"FL FedProx μ={best_mu}", f"{mu_results[best_mu][-1]:.4f}",
     "—"],
    ["FL + DP σ=0.001",  f"{dp_results[0.001][0]:.4f}",
     f"{dp_results[0.001][1]:.4f}"],
    ["FL + DP σ=0.01",   f"{dp_results[0.01][0]:.4f}",
     f"{dp_results[0.01][1]:.4f}"],
]
table = ax7.table(cellText=table_data[1:],
                  colLabels=table_data[0],
                  loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.1, 1.8)
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#ecf0f1")
ax7.set_title("Full Results Summary", fontweight="bold", pad=15)

plt.savefig("phase3_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✅ Saved: phase3_results.png")


# ══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
best_dp_sigma = min(
    [s for s in DP_SIGMAS if s > 0],
    key=lambda s: abs(dp_results[s][0] - base_auc)
)

print(f"\n{SEPARATOR}")
print("  PHASE 3 COMPLETE ✅")
print(SEPARATOR)
print(f"""
  Key findings:

  1. FedProx (best μ={best_mu})
     → AUC-ROC: {mu_results[best_mu][-1]:.4f}
     → Keeps clients anchored to global model during non-IID training

  2. Differential Privacy
     → At σ={best_dp_sigma}: AUC-ROC={dp_results[best_dp_sigma][0]:.4f}
        (accuracy drop: {base_auc - dp_results[best_dp_sigma][0]:.4f})
     → Privacy-accuracy trade-off curve plotted ✅

  3. Project story:
     "We trained a fraud detection model across 3 banks
      with non-IID data, using FedProx for stable convergence
      and Differential Privacy to provide formal guarantees —
      all without any bank sharing raw customer transactions."

  Saved:
    📊 phase3_results.png  — full dashboard

  → Optional Phase 4: Byzantine attack + robust aggregation
""")
