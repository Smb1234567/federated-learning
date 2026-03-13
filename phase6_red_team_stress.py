"""
Deployment-oriented synthetic red-team stress test for federated fraud detection.

This script replaces the earlier "demo-style" large-scale stress tests with a
more systematic harness that:
  1. generates synthetic non-IID bank datasets on demand,
  2. evaluates multiple attack/failure scenarios,
  3. compares aggregation rules under attack,
  4. reports deployment readiness with explicit pass/fail thresholds.

Examples:
  python3 phase6_red_team_stress.py
  python3 phase6_red_team_stress.py --scales 1 10 100 --features 64
  python3 phase6_red_team_stress.py --banks 7 --rounds 12 --epochs 2
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score


SEED = 42
RNG = np.random.default_rng(SEED)
OUTPUT_DIR = Path("stress_output")


@dataclass(frozen=True)
class Scenario:
    name: str
    prior_shift: float = 0.0
    feature_shift: float = 0.0
    label_noise: float = 0.0
    missing_rate: float = 0.0
    outlier_rate: float = 0.0
    dropout_rate: float = 0.0
    concept_drift: float = 0.0
    sign_flip: bool = False
    model_replacement: float = 0.0
    sybil_copies: int = 0


SCENARIOS = [
    Scenario("benign"),
    Scenario("prior_shift", prior_shift=0.9),
    Scenario("feature_shift", feature_shift=1.1),
    Scenario("label_noise", label_noise=0.20),
    Scenario("missingness_outliers", missing_rate=0.08, outlier_rate=0.02),
    Scenario("dropout", dropout_rate=0.35),
    Scenario("concept_drift", concept_drift=0.8),
    Scenario("sign_flip_attack", sign_flip=True),
    Scenario("model_replacement_attack", model_replacement=6.0),
    Scenario("sybil_attack", sybil_copies=2),
    Scenario(
        "full_red_team",
        prior_shift=1.0,
        feature_shift=1.2,
        label_noise=0.15,
        missing_rate=0.08,
        outlier_rate=0.02,
        dropout_rate=0.30,
        concept_drift=1.0,
        sign_flip=True,
        model_replacement=4.0,
        sybil_copies=2,
    ),
]

AGGREGATORS = ("fedavg", "coord_median", "trimmed_mean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic red-team FL stress test")
    parser.add_argument("--banks", type=int, default=5, help="Number of simulated banks")
    parser.add_argument("--features", type=int, default=48, help="Number of synthetic features")
    parser.add_argument(
        "--scales",
        type=int,
        nargs="+",
        default=[1, 10, 100],
        help="Row-count multipliers applied to --base-samples",
    )
    parser.add_argument(
        "--base-samples",
        type=int,
        default=4000,
        help="Base samples per bank before scale multiplier",
    )
    parser.add_argument("--fraud-rate", type=float, default=0.003, help="Base fraud rate")
    parser.add_argument("--rounds", type=int, default=8, help="Federated communication rounds")
    parser.add_argument("--epochs", type=int, default=2, help="Local epochs")
    parser.add_argument("--lr", type=float, default=0.03, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=512, help="Mini-batch size")
    parser.add_argument(
        "--aggregators",
        nargs="+",
        choices=AGGREGATORS,
        default=list(AGGREGATORS),
        help="Aggregation rules to compare",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=[s.name for s in SCENARIOS],
        help="Scenario names to run",
    )
    parser.add_argument(
        "--malicious-bank",
        type=int,
        default=4,
        help="Bank index used for adversarial update attacks",
    )
    parser.add_argument(
        "--output-prefix",
        default="phase6_red_team",
        help="Prefix for generated output files in stress_output/",
    )
    parser.add_argument(
        "--synthetic-dir",
        default="",
        help="Optional directory containing bank_*.csv and test_set.csv generated offline",
    )
    return parser.parse_args()


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -80.0, 80.0)))


def predict_proba(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return sigmoid(X @ w + b)


def compute_metrics(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> dict[str, float]:
    proba = predict_proba(X, w, b)
    preds = (proba >= 0.5).astype(np.int8)
    metrics = {
        "auc": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else 0.5,
        "auprc": float(average_precision_score(y, proba)) if len(np.unique(y)) > 1 else 0.0,
        "recall": float(recall_score(y, preds, zero_division=0)),
        "precision": float(precision_score(y, preds, zero_division=0)),
    }
    negatives = (y == 0).sum()
    false_positives = int(((preds == 1) & (y == 0)).sum())
    metrics["fpr"] = float(false_positives / negatives) if negatives else 0.0
    return metrics


def compute_loss(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
    p = predict_proba(X, w, b)
    eps = 1e-8
    return float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))


def balance_binary_data(X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    fraud_idx = np.flatnonzero(y == 1)
    legit_idx = np.flatnonzero(y == 0)
    if len(fraud_idx) == 0 or len(legit_idx) == 0:
        return X, y
    target_fraud = min(len(legit_idx) // 2, max(len(fraud_idx), 1) * 3)
    resampled_fraud = rng.choice(fraud_idx, size=target_fraud, replace=True)
    idx = np.concatenate([legit_idx, resampled_fraud])
    rng.shuffle(idx)
    return X[idx], y[idx]


def local_train(
    X: np.ndarray,
    y: np.ndarray,
    w_init: np.ndarray,
    b_init: float,
    epochs: int,
    lr: float,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, float]:
    X_bal, y_bal = balance_binary_data(X, y, rng)
    w = w_init.copy()
    b = float(b_init)
    n = len(y_bal)
    if n == 0:
        return w, b, 0.0

    for _ in range(epochs):
        perm = rng.permutation(n)
        X_shuf = X_bal[perm]
        y_shuf = y_bal[perm]
        for start in range(0, n, batch_size):
            Xb = X_shuf[start : start + batch_size]
            yb = y_shuf[start : start + batch_size]
            if len(yb) == 0:
                continue
            p = predict_proba(Xb, w, b)
            err = p - yb
            w -= lr * ((Xb.T @ err) / len(yb))
            b -= lr * float(err.mean())
    return w, b, compute_loss(X_bal, y_bal, w, b)


def aggregate_updates(
    global_w: np.ndarray,
    global_b: float,
    deltas_w: list[np.ndarray],
    deltas_b: list[float],
    sizes: list[int],
    method: str,
) -> tuple[np.ndarray, float]:
    if not deltas_w:
        return global_w, global_b

    delta_matrix = np.stack(deltas_w, axis=0)
    delta_bias = np.asarray(deltas_b, dtype=np.float64)

    if method == "fedavg":
        weights = np.asarray(sizes, dtype=np.float64)
        weights = weights / weights.sum()
        agg_w = np.sum(delta_matrix * weights[:, None], axis=0)
        agg_b = float(np.sum(delta_bias * weights))
    elif method == "coord_median":
        agg_w = np.median(delta_matrix, axis=0)
        agg_b = float(np.median(delta_bias))
    elif method == "trimmed_mean":
        trim = max(1, len(deltas_w) // 5) if len(deltas_w) >= 5 else 0
        sorted_w = np.sort(delta_matrix, axis=0)
        sorted_b = np.sort(delta_bias)
        if trim > 0 and 2 * trim < len(deltas_w):
            sorted_w = sorted_w[trim:-trim]
            sorted_b = sorted_b[trim:-trim]
        agg_w = np.mean(sorted_w, axis=0)
        agg_b = float(np.mean(sorted_b))
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

    return global_w + agg_w, global_b + agg_b


def make_bank_profile(
    bank_id: int,
    n_banks: int,
    base_samples: int,
    scale: int,
    base_fraud_rate: float,
    scenario: Scenario,
) -> tuple[int, float]:
    samples = base_samples * scale
    bank_weight = 0.75 + (bank_id / max(1, n_banks - 1))
    samples = int(samples * bank_weight)

    heterogeneity = (bank_id - (n_banks - 1) / 2) / max(1, n_banks - 1)
    fraud_rate = base_fraud_rate * (1.0 + scenario.prior_shift * 2.5 * heterogeneity)
    fraud_rate = float(np.clip(fraud_rate, 0.0005, 0.08))
    return samples, fraud_rate


def generate_bank_data(
    bank_id: int,
    n_banks: int,
    n_samples: int,
    fraud_rate: float,
    n_features: int,
    round_idx: int,
    scenario: Scenario,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_fraud = max(1, int(n_samples * fraud_rate))
    n_legit = max(1, n_samples - n_fraud)

    bank_pos = (bank_id - (n_banks - 1) / 2) / max(1, n_banks - 1)
    feature_axis = np.linspace(-1.0, 1.0, n_features, dtype=np.float32)
    bank_shift = bank_pos * 0.5 * feature_axis
    drift = scenario.concept_drift * (round_idx / 10.0)

    legit = rng.normal(0.0, 1.0, size=(n_legit, n_features)).astype(np.float32)
    legit += bank_shift
    legit[:, : min(6, n_features)] += bank_pos * 0.4

    fraud = rng.normal(0.0, 1.2, size=(n_fraud, n_features)).astype(np.float32)
    signal_end = min(12, n_features)
    fraud[:, :signal_end] += 2.0 + drift
    if n_features > 12:
        fraud[:, 12 : min(24, n_features)] -= 1.2 - bank_pos * 0.3

    if scenario.feature_shift > 0:
        shift_slice = slice(0, min(10, n_features))
        legit[:, shift_slice] += scenario.feature_shift * bank_pos
        fraud[:, shift_slice] += scenario.feature_shift * (bank_pos + 0.5)

    X = np.vstack([legit, fraud]).astype(np.float32)
    y = np.concatenate([
        np.zeros(n_legit, dtype=np.int8),
        np.ones(n_fraud, dtype=np.int8),
    ])

    idx = rng.permutation(len(y))
    X = X[idx]
    y = y[idx]

    if scenario.missing_rate > 0:
        mask = rng.random(X.shape) < scenario.missing_rate
        X[mask] = 0.0

    if scenario.outlier_rate > 0:
        row_mask = rng.random(len(y)) < scenario.outlier_rate
        X[row_mask] += rng.normal(0, 8.0, size=(row_mask.sum(), n_features)).astype(np.float32)

    if scenario.label_noise > 0:
        n_flip = int(len(y) * scenario.label_noise)
        flip_idx = rng.choice(len(y), size=n_flip, replace=False)
        y[flip_idx] = 1 - y[flip_idx]

    return X, y


def load_csv_bank(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    feature_cols = [col for col in df.columns if col != "Class"]
    X = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    y = df["Class"].to_numpy(dtype=np.int8, copy=True)
    return X, y


def subset_bank_data(
    X: np.ndarray,
    y: np.ndarray,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if fraction >= 0.999:
        return X.copy(), y.copy()
    rng = np.random.default_rng(seed)
    n = max(1, int(len(y) * fraction))
    idx = rng.choice(len(y), size=n, replace=False)
    return X[idx].copy(), y[idx].copy()


def apply_scenario_to_loaded_data(
    X: np.ndarray,
    y: np.ndarray,
    scenario: Scenario,
    bank_id: int,
    n_banks: int,
    round_idx: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X_mod = X.copy()
    y_mod = y.copy()

    if scenario.prior_shift > 0:
        bank_pos = (bank_id - (n_banks - 1) / 2) / max(1, n_banks - 1)
        legit_idx = np.flatnonzero(y_mod == 0)
        fraud_idx = np.flatnonzero(y_mod == 1)
        if len(legit_idx) > 0 and len(fraud_idx) > 0:
            fraud_keep = np.clip(0.55 + scenario.prior_shift * bank_pos, 0.20, 1.0)
            legit_keep = np.clip(1.0 - 0.25 * scenario.prior_shift * max(bank_pos, 0.0), 0.65, 1.0)
            fraud_sel = rng.choice(fraud_idx, size=max(1, int(len(fraud_idx) * fraud_keep)), replace=False)
            legit_sel = rng.choice(legit_idx, size=max(1, int(len(legit_idx) * legit_keep)), replace=False)
            idx = np.concatenate([legit_sel, fraud_sel])
            rng.shuffle(idx)
            X_mod = X_mod[idx]
            y_mod = y_mod[idx]

    if scenario.feature_shift > 0:
        shift_span = min(10, X_mod.shape[1])
        shift = (bank_id - (n_banks - 1) / 2) * 0.25 * scenario.feature_shift
        X_mod[:, :shift_span] += np.float32(shift)

    if scenario.concept_drift > 0:
        drift_cols = slice(0, min(12, X_mod.shape[1]))
        drift = np.float32(scenario.concept_drift * round_idx * 0.08)
        X_mod[:, drift_cols] += drift

    if scenario.missing_rate > 0:
        mask = rng.random(X_mod.shape) < scenario.missing_rate
        X_mod[mask] = 0.0

    if scenario.outlier_rate > 0:
        row_mask = rng.random(len(y_mod)) < scenario.outlier_rate
        X_mod[row_mask] += rng.normal(0, 8.0, size=(row_mask.sum(), X_mod.shape[1])).astype(np.float32)

    if scenario.label_noise > 0:
        n_flip = int(len(y_mod) * scenario.label_noise)
        if n_flip > 0:
            flip_idx = rng.choice(len(y_mod), size=n_flip, replace=False)
            y_mod[flip_idx] = 1 - y_mod[flip_idx]

    return X_mod, y_mod


def apply_attack_to_update(
    bank_id: int,
    malicious_bank: int,
    scenario: Scenario,
    delta_w: np.ndarray,
    delta_b: float,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[float]]:
    updates_w = [delta_w]
    updates_b = [delta_b]

    if bank_id != malicious_bank:
        return updates_w, updates_b

    attacked_w = delta_w.copy()
    attacked_b = float(delta_b)

    if scenario.sign_flip:
        attacked_w = -4.0 * attacked_w
        attacked_b = -4.0 * attacked_b

    if scenario.model_replacement > 0:
        attacked_w = attacked_w * scenario.model_replacement + rng.normal(0, 0.5, size=attacked_w.shape)
        attacked_b = attacked_b * scenario.model_replacement + float(rng.normal(0, 0.5))

    updates_w = [attacked_w]
    updates_b = [attacked_b]

    for _ in range(scenario.sybil_copies):
        noise = rng.normal(0, 0.05, size=attacked_w.shape)
        updates_w.append(attacked_w + noise)
        updates_b.append(attacked_b + float(rng.normal(0, 0.05)))

    return updates_w, updates_b


def make_test_set(
    n_samples: int,
    n_features: int,
    fraud_rate: float,
    round_idx: int,
    n_banks: int,
    scenario: Scenario,
) -> tuple[np.ndarray, np.ndarray]:
    return generate_bank_data(
        bank_id=n_banks // 2,
        n_banks=n_banks,
        n_samples=n_samples,
        fraud_rate=fraud_rate,
        n_features=n_features,
        round_idx=round_idx,
        scenario=Scenario(
            "eval",
            prior_shift=min(scenario.prior_shift, 0.3),
            feature_shift=min(scenario.feature_shift, 0.4),
            missing_rate=min(scenario.missing_rate, 0.02),
            outlier_rate=min(scenario.outlier_rate, 0.005),
            concept_drift=scenario.concept_drift,
        ),
        seed=SEED + 999 + round_idx,
    )


def load_synthetic_dataset(args: argparse.Namespace) -> tuple[list[tuple[np.ndarray, np.ndarray]], tuple[np.ndarray, np.ndarray], int]:
    source_dir = Path(args.synthetic_dir)
    bank_paths = [source_dir / f"bank_{bank_id}.csv" for bank_id in range(args.banks)]
    missing = [str(path) for path in bank_paths if not path.exists()]
    test_path = source_dir / "test_set.csv"
    if not test_path.exists():
        missing.append(str(test_path))
    if missing:
        raise FileNotFoundError("Missing synthetic dataset files: " + ", ".join(missing))

    banks = [load_csv_bank(path) for path in bank_paths]
    X_test, y_test = load_csv_bank(test_path)
    features = banks[0][0].shape[1]
    return banks, (X_test, y_test), features


def scenario_lookup(names: list[str]) -> list[Scenario]:
    by_name = {scenario.name: scenario for scenario in SCENARIOS}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(f"Unknown scenarios: {', '.join(missing)}")
    return [by_name[name] for name in names]


def pass_fail(metrics: dict[str, float], baseline: dict[str, float]) -> str:
    if metrics["auc"] >= baseline["auc"] - 0.03 and metrics["recall"] >= baseline["recall"] - 0.10:
        return "PASS"
    if metrics["auc"] >= baseline["auc"] - 0.07 and metrics["recall"] >= baseline["recall"] - 0.20:
        return "WARN"
    return "FAIL"


def run_experiment(args: argparse.Namespace) -> dict[str, dict]:
    selected_scenarios = scenario_lookup(args.scenarios)
    results: dict[str, dict] = {}
    loaded_banks = None
    loaded_test = None
    max_scale = max(args.scales)
    feature_count = args.features

    if args.synthetic_dir:
        loaded_banks, loaded_test, feature_count = load_synthetic_dataset(args)

    for scale in args.scales:
        scale_key = f"scale_{scale}"
        results[scale_key] = {}
        sample_fraction = 1.0 if not args.synthetic_dir else min(1.0, scale / max_scale)

        for scenario in selected_scenarios:
            scenario_key = scenario.name
            results[scale_key][scenario_key] = {}

            baseline_metrics = None
            for aggregator in args.aggregators:
                global_w = np.zeros(feature_count, dtype=np.float32)
                global_b = 0.0
                history = {"round": [], "auc": [], "auprc": [], "recall": [], "fpr": [], "loss": []}

                for round_idx in range(1, args.rounds + 1):
                    deltas_w: list[np.ndarray] = []
                    deltas_b: list[float] = []
                    sizes: list[int] = []
                    round_losses: list[float] = []

                    for bank_id in range(args.banks):
                        rng = np.random.default_rng(SEED + scale * 10_000 + round_idx * 101 + bank_id)
                        if scenario.dropout_rate > 0 and rng.random() < scenario.dropout_rate:
                            continue

                        if args.synthetic_dir:
                            X_base, y_base = loaded_banks[bank_id]
                            X_sub, y_sub = subset_bank_data(
                                X_base,
                                y_base,
                                fraction=sample_fraction,
                                seed=SEED + scale * 1000 + bank_id * 37,
                            )
                            X_c, y_c = apply_scenario_to_loaded_data(
                                X_sub,
                                y_sub,
                                scenario=scenario,
                                bank_id=bank_id,
                                n_banks=args.banks,
                                round_idx=round_idx,
                                seed=SEED + scale * 1000 + bank_id * 37 + round_idx * 13,
                            )
                            samples = len(y_c)
                        else:
                            samples, fraud_rate = make_bank_profile(
                                bank_id=bank_id,
                                n_banks=args.banks,
                                base_samples=args.base_samples,
                                scale=scale,
                                base_fraud_rate=args.fraud_rate,
                                scenario=scenario,
                            )
                            X_c, y_c = generate_bank_data(
                                bank_id=bank_id,
                                n_banks=args.banks,
                                n_samples=samples,
                                fraud_rate=fraud_rate,
                                n_features=feature_count,
                                round_idx=round_idx,
                                scenario=scenario,
                                seed=SEED + scale * 1000 + bank_id * 37 + round_idx * 13,
                            )
                        local_w, local_b, loss = local_train(
                            X_c,
                            y_c,
                            global_w,
                            global_b,
                            epochs=args.epochs,
                            lr=args.lr,
                            batch_size=args.batch_size,
                            rng=rng,
                        )
                        delta_w = local_w - global_w
                        delta_b = float(local_b - global_b)
                        update_ws, update_bs = apply_attack_to_update(
                            bank_id,
                            args.malicious_bank,
                            scenario,
                            delta_w,
                            delta_b,
                            rng,
                        )
                        deltas_w.extend(update_ws)
                        deltas_b.extend(update_bs)
                        sizes.extend([samples] * len(update_ws))
                        round_losses.append(loss)

                    global_w, global_b = aggregate_updates(
                        global_w,
                        global_b,
                        deltas_w,
                        deltas_b,
                        sizes,
                        aggregator,
                    )

                    if args.synthetic_dir:
                        X_test_base, y_test_base = loaded_test
                        X_test, y_test = subset_bank_data(
                            X_test_base,
                            y_test_base,
                            fraction=sample_fraction,
                            seed=SEED + 555 + scale * 11 + round_idx,
                        )
                        X_test, y_test = apply_scenario_to_loaded_data(
                            X_test,
                            y_test,
                            scenario=Scenario(
                                "eval",
                                prior_shift=min(scenario.prior_shift, 0.3),
                                feature_shift=min(scenario.feature_shift, 0.4),
                                missing_rate=min(scenario.missing_rate, 0.02),
                                outlier_rate=min(scenario.outlier_rate, 0.005),
                                concept_drift=scenario.concept_drift,
                            ),
                            bank_id=args.banks // 2,
                            n_banks=args.banks,
                            round_idx=round_idx,
                            seed=SEED + 999 + round_idx,
                        )
                    else:
                        X_test, y_test = make_test_set(
                            n_samples=max(8000, args.base_samples * 2),
                            n_features=feature_count,
                            fraud_rate=args.fraud_rate,
                            round_idx=round_idx,
                            n_banks=args.banks,
                            scenario=scenario,
                        )
                    metrics = compute_metrics(X_test, y_test, global_w, global_b)
                    history["round"].append(round_idx)
                    history["auc"].append(metrics["auc"])
                    history["auprc"].append(metrics["auprc"])
                    history["recall"].append(metrics["recall"])
                    history["fpr"].append(metrics["fpr"])
                    history["loss"].append(float(np.mean(round_losses)) if round_losses else 0.0)

                final_metrics = {
                    "auc": history["auc"][-1],
                    "auprc": history["auprc"][-1],
                    "recall": history["recall"][-1],
                    "fpr": history["fpr"][-1],
                    "loss": history["loss"][-1],
                }
                if aggregator == "fedavg":
                    baseline_metrics = final_metrics

                results[scale_key][scenario_key][aggregator] = {
                    "history": history,
                    "final": final_metrics,
                }

            if baseline_metrics is None:
                baseline_metrics = next(iter(results[scale_key][scenario_key].values()))["final"]

            for aggregator in args.aggregators:
                final_metrics = results[scale_key][scenario_key][aggregator]["final"]
                results[scale_key][scenario_key][aggregator]["verdict"] = pass_fail(final_metrics, baseline_metrics)

    return results


def save_outputs(results: dict[str, dict], args: argparse.Namespace) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    json_path = OUTPUT_DIR / f"{args.output_prefix}_results.json"
    png_path = OUTPUT_DIR / f"{args.output_prefix}_dashboard.png"
    md_path = OUTPUT_DIR / f"{args.output_prefix}_summary.md"

    with json_path.open("w", encoding="ascii") as fh:
        json.dump(results, fh, indent=2)

    scales = list(results.keys())
    scenario_names = list(next(iter(results.values())).keys())

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Phase 6 — Synthetic Red-Team Stress Test", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    first_scale = scales[0]
    for aggregator in args.aggregators:
        history = results[first_scale]["full_red_team"][aggregator]["history"]
        ax.plot(history["round"], history["auc"], marker="o", lw=2, label=aggregator)
    ax.set_title(f"Full Red-Team AUC Convergence ({first_scale})", fontweight="bold")
    ax.set_xlabel("Round")
    ax.set_ylabel("AUC-ROC")
    ax.legend()

    ax = axes[0, 1]
    x = np.arange(len(scenario_names))
    width = 0.8 / len(args.aggregators)
    for idx, aggregator in enumerate(args.aggregators):
        aucs = [results[first_scale][scenario][aggregator]["final"]["auc"] for scenario in scenario_names]
        ax.bar(x + idx * width, aucs, width=width, label=aggregator)
    ax.set_xticks(x + width * (len(args.aggregators) - 1) / 2)
    ax.set_xticklabels(scenario_names, rotation=25, ha="right")
    ax.set_ylabel("Final AUC-ROC")
    ax.set_title(f"Scenario Comparison ({first_scale})", fontweight="bold")
    ax.legend()

    ax = axes[1, 0]
    for aggregator in args.aggregators:
        aucs = [results[scale]["full_red_team"][aggregator]["final"]["auc"] for scale in scales]
        ax.plot(scales, aucs, marker="s", lw=2, label=aggregator)
    ax.set_title("Scale Sensitivity Under Full Red-Team", fontweight="bold")
    ax.set_xlabel("Scale")
    ax.set_ylabel("Final AUC-ROC")
    ax.legend()

    ax = axes[1, 1]
    verdict_map = {"PASS": 2, "WARN": 1, "FAIL": 0}
    heat = np.zeros((len(args.aggregators), len(scenario_names)))
    for i, aggregator in enumerate(args.aggregators):
        for j, scenario in enumerate(scenario_names):
            verdict = results[first_scale][scenario][aggregator]["verdict"]
            heat[i, j] = verdict_map[verdict]
    im = ax.imshow(heat, cmap="RdYlGn", aspect="auto", vmin=0, vmax=2)
    ax.set_yticks(np.arange(len(args.aggregators)))
    ax.set_yticklabels(args.aggregators)
    ax.set_xticks(np.arange(len(scenario_names)))
    ax.set_xticklabels(scenario_names, rotation=25, ha="right")
    ax.set_title(f"Deployment Verdicts ({first_scale})", fontweight="bold")
    for i in range(len(args.aggregators)):
        for j in range(len(scenario_names)):
            text = ("FAIL", "WARN", "PASS")[int(heat[i, j])]
            ax.text(j, i, text, ha="center", va="center", fontsize=9, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# Phase 6 Synthetic Red-Team Stress Test",
        "",
        f"Scales: {args.scales}",
        f"Aggregators: {args.aggregators}",
        f"Banks: {args.banks}, features: {args.features}, rounds: {args.rounds}",
        "",
    ]
    for scale in scales:
        lines.append(f"## {scale}")
        lines.append("")
        lines.append("| Scenario | Aggregator | AUC | AUPRC | Recall | FPR | Verdict |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for scenario in scenario_names:
            for aggregator in args.aggregators:
                row = results[scale][scenario][aggregator]
                final = row["final"]
                lines.append(
                    f"| {scenario} | {aggregator} | {final['auc']:.4f} | {final['auprc']:.4f} | "
                    f"{final['recall']:.4f} | {final['fpr']:.4f} | {row['verdict']} |"
                )
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="ascii")


def main() -> None:
    args = parse_args()
    results = run_experiment(args)
    save_outputs(results, args)

    print("=" * 72)
    print("PHASE 6 SYNTHETIC RED-TEAM STRESS TEST COMPLETE")
    print("=" * 72)
    print(f"Saved: {OUTPUT_DIR / (args.output_prefix + '_results.json')}")
    print(f"Saved: {OUTPUT_DIR / (args.output_prefix + '_dashboard.png')}")
    print(f"Saved: {OUTPUT_DIR / (args.output_prefix + '_summary.md')}")


if __name__ == "__main__":
    main()
