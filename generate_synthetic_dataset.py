"""
Fast synthetic fraud dataset generator.

This script generates larger synthetic federated datasets than the original
credit-card CSV and writes them directly to disk with a consistent folder
structure. It is optimized for CPU + SSD throughput; GPU/VRAM does not matter
for this workload because generation and CSV writing are not GPU-bound.

Examples:
  python3 generate_synthetic_dataset.py
  python3 generate_synthetic_dataset.py --scale 25 --save-combined
  python3 generate_synthetic_dataset.py --scale 100 --skip-stressed --test-size 100000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount", "Class"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fast synthetic fraud datasets")
    parser.add_argument("--scale", type=int, default=20, help="Multiplier over the 284,807-row base dataset")
    parser.add_argument("--banks", type=int, default=5, help="Number of bank files to generate")
    parser.add_argument("--fraud-ratio", type=float, default=0.0017, help="Base fraud rate")
    parser.add_argument("--test-size", type=int, default=75_000, help="Rows in the shared test set")
    parser.add_argument("--output-dir", default="synthetic_data", help="Output directory")
    parser.add_argument("--save-combined", action="store_true", help="Write combined_all_banks.csv")
    parser.add_argument("--skip-stressed", action="store_true", help="Skip stressed variants for speed")
    parser.add_argument("--csv-chunksize", type=int, default=200_000, help="Chunk size for CSV writes")
    parser.add_argument("--float-format", default="%.6f", help="CSV float format to reduce size and write time")
    return parser.parse_args()


def synthetic_bank_frame(
    n_samples: int,
    fraud_ratio: float,
    bank_id: int,
    n_banks: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = max(1, int(n_samples * fraud_ratio))
    n_legit = max(1, n_samples - n_fraud)

    bank_pos = (bank_id - (n_banks - 1) / 2) / max(1, n_banks - 1)
    legit = rng.normal(0.0, 1.0 + 0.15 * bank_id, size=(n_legit, 28)).astype(np.float32)
    legit += (bank_pos * 0.35).astype(np.float32) if isinstance(bank_pos, np.ndarray) else np.float32(bank_pos * 0.35)

    fraud = rng.normal(0.0, 1.2, size=(n_fraud, 28)).astype(np.float32)
    fraud_signal = [4, 9, 11, 13, 16, 17]
    fraud[:, fraud_signal[: min(len(fraud_signal), fraud.shape[1])]] += np.array([3.0, 3.0, 2.5, 2.5, -3.0, -2.5], dtype=np.float32)[: min(len(fraud_signal), fraud.shape[1])]

    legit_time = rng.uniform(0, 48 * 3600, size=(n_legit, 1)).astype(np.float32)
    fraud_time = rng.uniform(0, 48 * 3600, size=(n_fraud, 1)).astype(np.float32)
    legit_amount = (rng.lognormal(4.0, 1.4, size=(n_legit, 1)) * (1 + 0.25 * bank_id)).astype(np.float32)
    fraud_amount = rng.lognormal(3.6, 1.9, size=(n_fraud, 1)).astype(np.float32)

    legit_y = np.zeros((n_legit, 1), dtype=np.float32)
    fraud_y = np.ones((n_fraud, 1), dtype=np.float32)

    legit_full = np.hstack([legit, legit_time, legit_amount, legit_y])
    fraud_full = np.hstack([fraud, fraud_time, fraud_amount, fraud_y])
    data = np.vstack([legit_full, fraud_full]).astype(np.float32)

    perm = rng.permutation(len(data))
    data = data[perm]
    return pd.DataFrame(data, columns=FEATURE_COLUMNS)


def apply_stress(df: pd.DataFrame, stress_type: str, bank_id: int, n_banks: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    stressed = df.copy()

    if stress_type == "label_noise":
        flip_ratio = 0.20
        n_flip = int(len(stressed) * flip_ratio)
        flip_idx = rng.choice(len(stressed), size=n_flip, replace=False)
        stressed.iloc[flip_idx, stressed.columns.get_loc("Class")] = 1.0 - stressed.iloc[
            flip_idx, stressed.columns.get_loc("Class")
        ]
        return stressed

    if stress_type == "feature_shift":
        shift = (bank_id - (n_banks - 1) / 2) * 0.45
        for col in FEATURE_COLUMNS[:10]:
            stressed[col] = stressed[col].astype(np.float32) + np.float32(shift)
        return stressed

    if stress_type == "extreme_noniid":
        if bank_id == 0:
            stressed = stressed[stressed["Class"] == 0.0]
        elif bank_id == 1:
            stressed = stressed[stressed["Class"] == 1.0]
            if len(stressed) > 0 and len(stressed) < len(df):
                stressed = stressed.sample(n=len(df), replace=True, random_state=seed)
        return stressed.reset_index(drop=True)

    raise ValueError(f"Unknown stress type: {stress_type}")


def write_csv(df: pd.DataFrame, path: Path, chunksize: int, float_format: str, mode: str = "w", header: bool = True) -> None:
    df.to_csv(
        path,
        index=False,
        mode=mode,
        header=header,
        chunksize=chunksize,
        float_format=float_format,
    )


def total_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    base_rows = 284_807
    total_rows = base_rows * args.scale
    rows_per_bank = total_rows // args.banks
    combined_path = out_dir / "combined_all_banks.csv"

    stress_dirs = {}
    if not args.skip_stressed:
        for stress_name in ("label_noise", "feature_shift", "extreme_noniid"):
            stress_dir = out_dir / f"stressed_{stress_name}"
            stress_dir.mkdir(exist_ok=True)
            stress_dirs[stress_name] = stress_dir

    print("=" * 72)
    print("FAST SYNTHETIC FRAUD DATA GENERATOR")
    print("=" * 72)
    print(f"Scale            : {args.scale}x")
    print(f"Total rows        : {total_rows:,}")
    print(f"Banks             : {args.banks}")
    print(f"Rows per bank     : {rows_per_bank:,}")
    print(f"Fraud ratio       : {args.fraud_ratio:.4%}")
    print(f"Output dir        : {out_dir}")
    print(f"Save combined     : {args.save_combined}")
    print(f"Save stressed     : {not args.skip_stressed}")
    print(f"CSV chunk size    : {args.csv_chunksize:,}")
    print("=" * 72)

    if args.save_combined and combined_path.exists():
        combined_path.unlink()

    amount_sum = 0.0
    amount_max = 0.0
    fraud_sum = 0.0
    row_sum = 0
    feature_corr_samples = []

    start = time.time()

    for bank_id in range(args.banks):
        bank_start = time.time()
        bank_df = synthetic_bank_frame(
            n_samples=rows_per_bank,
            fraud_ratio=args.fraud_ratio,
            bank_id=bank_id,
            n_banks=args.banks,
            seed=SEED + bank_id,
        )

        bank_file = out_dir / f"bank_{bank_id}.csv"
        write_csv(bank_df, bank_file, args.csv_chunksize, args.float_format)

        if args.save_combined:
            write_csv(
                bank_df,
                combined_path,
                args.csv_chunksize,
                args.float_format,
                mode="a" if combined_path.exists() else "w",
                header=not combined_path.exists(),
            )

        if not args.skip_stressed:
            for stress_name, stress_dir in stress_dirs.items():
                stressed_df = apply_stress(
                    bank_df,
                    stress_type=stress_name,
                    bank_id=bank_id,
                    n_banks=args.banks,
                    seed=SEED + 1000 + bank_id,
                )
                stressed_file = stress_dir / f"bank_{bank_id}.csv"
                write_csv(stressed_df, stressed_file, args.csv_chunksize, args.float_format)

        amount_sum += float(bank_df["Amount"].sum())
        amount_max = max(amount_max, float(bank_df["Amount"].max()))
        fraud_sum += float(bank_df["Class"].sum())
        row_sum += len(bank_df)
        feature_corr_samples.append(bank_df.sample(min(20_000, len(bank_df)), random_state=SEED + bank_id))

        elapsed = time.time() - bank_start
        print(
            f"Bank {bank_id}: wrote {len(bank_df):,} rows "
            f"({bank_df['Class'].mean() * 100:.3f}% fraud) in {elapsed:.1f}s"
        )

    test_df = synthetic_bank_frame(
        n_samples=args.test_size,
        fraud_ratio=args.fraud_ratio,
        bank_id=args.banks // 2,
        n_banks=args.banks,
        seed=SEED + 999,
    )
    write_csv(test_df, out_dir / "test_set.csv", args.csv_chunksize, args.float_format)

    corr_df = pd.concat(feature_corr_samples, ignore_index=True)
    correlations = corr_df[FEATURE_COLUMNS].corr(numeric_only=True)["Class"].abs().sort_values(ascending=False)

    duration = time.time() - start
    disk_usage = total_bytes(out_dir) / (1024 ** 3)

    print("\n" + "=" * 72)
    print("GENERATION COMPLETE")
    print("=" * 72)
    print(f"Runtime           : {duration:.1f}s")
    print(f"Disk usage        : {disk_usage:.2f} GB")
    print(f"Total rows        : {row_sum + len(test_df):,}")
    print(f"Overall fraud     : {(fraud_sum / row_sum) * 100:.4f}%")
    print(f"Amount mean       : {amount_sum / row_sum:.2f}")
    print(f"Amount max        : {amount_max:.2f}")
    print(f"Test set rows     : {len(test_df):,}")
    print("\nTop feature correlations with Class:")
    shown = 0
    for feature, corr in correlations.items():
        if feature == "Class":
            continue
        print(f"  {feature:<8} {corr:.4f}")
        shown += 1
        if shown == 5:
            break


if __name__ == "__main__":
    main()
