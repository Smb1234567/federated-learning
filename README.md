# 🏦 Federated Learning for Credit Card Fraud Detection

> A research-grade implementation of Federated Learning applied to fraud detection — where multiple banks collaboratively train a shared model **without ever sharing raw customer data**.

---

## 📌 What This Project Does

Traditional fraud detection requires centralising transaction data from all banks into one server — a massive privacy and regulatory risk. This project solves that using **Federated Learning (FL)**:

- Each bank trains a local model on its own data
- Only **model weights** (not raw transactions) are sent to a central server
- The server aggregates weights using **FedAvg** and sends the improved global model back
- Repeat for N rounds until the global model converges

The result: a fraud detection model nearly as accurate as a centralised one, with **zero raw data ever leaving any bank**.

---

## 🧠 Concepts Covered

| Concept | Where |
|---|---|
| Class imbalance & why accuracy is misleading | Phase 1 |
| Non-IID data splits (realistic bank heterogeneity) | Phase 1 |
| AUC-ROC vs AUPRC — which metric matters for fraud | Phase 1 |
| FedAvg aggregation from scratch | Phase 2 |
| Mini-batch SGD local training | Phase 2 |
| Communication rounds vs accuracy trade-off | Phase 2 |
| FedProx vs FedAvg under non-IID data | Phase 3 (coming) |
| Differential Privacy — privacy-accuracy trade-off | Phase 3 (coming) |

---

## 📁 Project Structure

```
federated-fraud-detection/
│
├── creditcard.csv                   # Dataset (download separately — see below)
│
├── phase1_federated_fraud.py        # Data exploration + non-IID split + baseline
├── phase2_federated_fraud.py        # FedAvg training loop + results
│
├── phase1_exploration.png           # Generated: dataset overview charts
├── phase1_noniid_split.png          # Generated: bank split visualisation
├── phase1_baseline_results.png      # Generated: centralised model performance
├── phase2_federated_results.png     # Generated: FL training dashboard
│
├── client_data.npy                  # Generated: 3 banks' local datasets
├── X_test.npy                       # Generated: shared test set features
├── y_test.npy                       # Generated: shared test set labels
├── baseline_scores.npy              # Generated: centralised scores to beat
├── global_model.npy                 # Generated: final FL model weights
├── fl_history.npy                   # Generated: round-by-round metrics
│
└── README.md
```

---

## 📦 Dataset

**Credit Card Fraud Detection** by ULB Machine Learning Group

| Property | Value |
|---|---|
| Source | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Transactions | 284,807 |
| Fraud cases | 492 (0.17%) |
| Features | V1–V28 (PCA anonymised) + Time + Amount |
| Target | `Class` — 0: Legit, 1: Fraud |

### Download

```bash
# Option 1 — Kaggle CLI
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip

# Option 2 — Manual
# Go to https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Click Download → place creditcard.csv in this project folder
```

---

## ⚙️ Installation

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

No deep learning frameworks needed — everything is built from scratch with NumPy so every line is transparent and understandable.

---

## 🚀 Running the Project

### Phase 1 — Data Exploration + Non-IID Split + Centralised Baseline

```bash
python phase1_federated_fraud.py
```

**What it does:**
- Loads and explores the dataset (class imbalance, feature distributions)
- Scales `Time` and `Amount` features (V1–V28 are already PCA-scaled)
- Splits data across 3 "banks" with non-IID distributions (different time windows + fraud rates)
- Trains a centralised logistic regression as the baseline to beat
- Generates 3 visualisation charts
- Saves `.npy` files for Phase 2

**Expected output:**
```
Centralised Baseline to beat in Phase 2:
  AUC-ROC : ~0.97
  AUPRC   : ~0.72
```

**Generated files:**
```
phase1_exploration.png       — class imbalance, amount distributions, correlations
phase1_noniid_split.png      — fraud rate per bank
phase1_baseline_results.png  — ROC curve, PR curve, confusion matrix
client_data.npy              — 3 banks' datasets
X_test.npy / y_test.npy      — held-out test set
baseline_scores.npy          — scores Phase 2 must beat
```

---

### Phase 2 — Federated Learning with FedAvg

```bash
python phase2_federated_fraud.py
```

**What it does:**
- Runs 10 federated communication rounds
- Each round: server broadcasts global model → banks train locally (5 epochs, mini-batch SGD) → server aggregates with FedAvg
- Tracks global AUC-ROC and AUPRC every round
- Compares final FL model vs centralised baseline
- Generates full results dashboard

**Expected output:**
```
Federated AUC-ROC : close to centralised baseline
Federated AUPRC   : close to centralised baseline
Privacy cost      : <X>% AUC reduction
Data shared       : ZERO raw transactions — only model weights ✅
```

**Generated files:**
```
phase2_federated_results.png  — convergence curves, ROC, PR, loss per bank
global_model.npy              — final trained FL model
fl_history.npy                — round-by-round metrics for analysis
```

---

## 🔬 Key Design Decisions

### Why Logistic Regression?
Intentionally simple — the goal is to demonstrate the FL framework, not optimise a black-box model. Every gradient computation is visible and understandable.

### Why Non-IID splits?
In reality, different banks serve different demographics. Bank 0 (small, suburban) sees rare fraud. Bank 2 (large, urban) sees frequent fraud. Homogeneous splits would be unrealistically easy for FL.

### Why AUPRC over Accuracy?
With 0.17% fraud, a model predicting "legit" for everything achieves 99.83% accuracy and is completely useless. AUPRC measures precision-recall trade-off across thresholds — what a real fraud team cares about.

### FedAvg — How it works
```
global_w = Σ (n_k / N) × w_k    for each client k
```
Each client's weights are averaged proportionally to how much data it contributed. Clients with more transactions have more influence on the global model.

---

## 📊 Results

| Metric | Centralised | Federated (FL) | Privacy Cost |
|---|---|---|---|
| AUC-ROC | ~0.9708 | ~0.97xx | ~0.0x |
| AUPRC | ~0.7170 | ~0.7xxx | ~0.0x |

> Raw data never left any bank. Only model weights (30 floats) were communicated each round.

---

## 🗺️ Roadmap

- [x] Phase 1 — Data exploration, non-IID split, centralised baseline
- [x] Phase 2 — FedAvg training loop with full metrics
- [ ] Phase 3 — FedProx vs FedAvg (handle non-IID better)
- [ ] Phase 3 — Differential Privacy (add noise to gradients, show privacy-accuracy curve)
- [ ] Phase 4 — Byzantine robustness (simulate malicious bank, implement defense)

---

## 📚 References

- McMahan et al. (2017) — [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629) *(the original FedAvg paper)*
- Li et al. (2020) — [Federated Optimization in Heterogeneous Networks (FedProx)](https://arxiv.org/abs/1812.06127)
- Dwork et al. (2006) — [Differential Privacy](https://link.springer.com/chapter/10.1007/11681878_14)
- ULB ML Group — [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## 👤 Author

Built as a major project exploring privacy-preserving machine learning applied to financial fraud detection.

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.
