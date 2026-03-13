# Federated Learning for Credit Card Fraud Detection

This repository explores privacy-preserving fraud detection with federated learning. The project starts from the ULB credit card fraud dataset, builds a centralized baseline, trains a federated logistic-regression model with FedAvg, and then pushes the system through progressively harsher stress tests, including synthetic large-scale red-team scenarios.

## What This Project Covers

- Phase 1: data exploration, preprocessing, non-IID bank split, centralized baseline
- Phase 2: federated learning with FedAvg
- Phase 3: FedProx comparison and differential privacy experiments
- Phase 4: stress testing under extreme non-IID, label noise, feature shift, dropout, and Byzantine corruption
- Phase 5: large-scale synthetic stress experiments
- Phase 6: deployment-oriented red-team stress testing with robust aggregation comparisons

The implementation is intentionally simple and transparent. The model is logistic regression built with NumPy, so the focus stays on federated-learning behavior, privacy tradeoffs, and robustness rather than black-box model complexity.

## Repository Layout

```text
federated-learning/
├── creditcard.csv
├── README.md
├── findings.md
├── generate_synthetic_dataset.py
├── phase1_federated_fraud.py
├── phase2_federated_fraud.py
├── phase3_fedprox_dp.py
├── phase4_stress_test.py
├── phase5_hardcore_stress.py
├── phase6_red_team_stress.py
├── npy_output/
├── png_output/
├── stress_output/
└── synthetic_data/
```

## Dataset

Primary dataset:
- ULB Credit Card Fraud Detection dataset from Kaggle
- `284,807` transactions
- `492` fraud cases
- Features: `V1` to `V28`, `Time`, `Amount`
- Target: `Class`

Expected local file:
- `creditcard.csv`

Synthetic dataset support:
- `generate_synthetic_dataset.py` creates multi-bank synthetic fraud datasets
- output is written under `synthetic_data/`
- these files can be used directly by phase 6

## Installation

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Recommended Run Order

### 1. Phase 1: baseline and bank split

```bash
python3 phase1_federated_fraud.py
```

Outputs:
- `npy_output/client_data.npy`
- `npy_output/X_test.npy`
- `npy_output/y_test.npy`
- `npy_output/baseline_scores.npy`
- `png_output/phase1_exploration.png`
- `png_output/phase1_noniid_split.png`
- `png_output/phase1_baseline_results.png`

### 2. Phase 2: FedAvg training

```bash
python3 phase2_federated_fraud.py
```

Outputs:
- `npy_output/global_model.npy`
- `npy_output/fl_history.npy`
- `png_output/phase2_federated_results.png`

### 3. Phase 3: FedProx and differential privacy

```bash
python3 phase3_fedprox_dp.py
```

Output:
- `png_output/phase3_results.png`

### 4. Phase 4: classical stress test

```bash
python3 phase4_stress_test.py
```

Output:
- `png_output/phase4_stress_test.png`

### 5. Generate large synthetic bank data

```bash
python3 generate_synthetic_dataset.py --scale 100 --save-combined
```

Outputs:
- `synthetic_data/bank_0.csv` ... `bank_4.csv`
- `synthetic_data/combined_all_banks.csv`
- `synthetic_data/test_set.csv`
- `synthetic_data/stressed_label_noise/`
- `synthetic_data/stressed_feature_shift/`
- `synthetic_data/stressed_extreme_noniid/`

Note:
- this is CPU and disk I/O heavy
- GPU does not materially help with the current CSV-based generation flow

### 6. Phase 6: red-team stress testing on synthetic data

```bash
python3 phase6_red_team_stress.py \
  --synthetic-dir synthetic_data \
  --scales 1 10 100 \
  --rounds 8 \
  --epochs 2 \
  --aggregators fedavg coord_median trimmed_mean \
  --output-prefix phase6_synth_full
```

Outputs:
- `stress_output/phase6_synth_full_results.json`
- `stress_output/phase6_synth_full_dashboard.png`
- `stress_output/phase6_synth_full_summary.md`

When `--synthetic-dir synthetic_data` is used, `--scales 1 10 100` means approximately `1%`, `10%`, and `100%` of the prepared synthetic corpus.

## What Phase 6 Tests

Phase 6 is the most deployment-oriented script in the repository. It compares aggregation rules under adversarial and failure-heavy conditions.

Scenarios include:
- benign
- prior shift
- feature shift
- label noise
- missingness and outliers
- client dropout
- concept drift
- sign-flip attacks
- model-replacement attacks
- sybil attacks
- combined full red-team scenario

Aggregation rules:
- `fedavg`
- `coord_median`
- `trimmed_mean`

Metrics tracked:
- AUC-ROC
- AUPRC
- recall
- precision
- false positive rate
- training loss
- pass/warn/fail deployment verdict

## Current Outputs

Generated artifacts already used by this repo are stored in:
- `npy_output/`
- `png_output/`
- `stress_output/`

Research-style narrative summary:
- `findings.md`

## Practical Notes

- The repository ignores generated CSVs, NPY files, and stress outputs via `.gitignore`.
- Large synthetic dataset generation is slow mainly because of CSV writing.
- If you want faster synthetic pipelines later, the next step is binary output formats such as Parquet or NumPy arrays rather than larger CSV workflows.
- Phase 6 currently supports both:
  - on-demand synthetic generation
  - direct loading from `synthetic_data/`

## References

- McMahan et al. (2017), FedAvg
- Li et al. (2020), FedProx
- Dwork et al. (2006), Differential Privacy
- ULB Machine Learning Group credit-card fraud dataset

## License

MIT
