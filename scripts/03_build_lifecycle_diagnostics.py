#!/usr/bin/env python3
"""Build regime-specific AR lifecycle summaries.

The input is a final regime-labeled lifecycle table with one row per AR track
record/time step. The script bins normalized lifecycle phase, summarizes median
and mean values by regime and phase bin, and optionally estimates bootstrap
confidence intervals by resampling unique tracks within each regime/bin.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

REGIME_ORDER = ["High-moisture", "Coupled-cyclonic", "Frontal", "Ridge"]
DEFAULT_VARIABLES = ["ivt", "tcwv", "precip", "precip_efficiency"]


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if path.name.endswith(".csv.gz") or path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path}")


def infer_lifecycle_phase(df: pd.DataFrame, phase_col: str) -> pd.Series:
    if phase_col in df.columns:
        return pd.to_numeric(df[phase_col], errors="coerce")
    candidates = ["lifecycle_phase", "normalized_lifecycle_phase", "phase_norm"]
    for c in candidates:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    raise ValueError(
        f"Could not find lifecycle phase column. Provide --phase-col or include one of {candidates}."
    )


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    boots = np.empty(n_boot, dtype=float)
    n = len(values)
    for i in range(n_boot):
        boots[i] = np.nanmedian(rng.choice(values, size=n, replace=True))
    return tuple(np.nanpercentile(boots, [2.5, 97.5]))


def summarize(df: pd.DataFrame, variables: Sequence[str], regime_col: str, track_col: str, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (regime, phase_bin), grp in df.groupby([regime_col, "phase_bin"], observed=True):
        for var in variables:
            vals = pd.to_numeric(grp[var], errors="coerce").to_numpy(dtype=float)
            valid = vals[np.isfinite(vals)]
            row = {
                "regime": regime,
                "phase_bin": int(phase_bin),
                "phase_center": float(grp["phase_center"].iloc[0]),
                "variable": var,
                "n_obs": int(len(valid)),
                "n_tracks": int(grp[track_col].nunique()) if track_col in grp.columns else np.nan,
                "median": float(np.nanmedian(valid)) if len(valid) else np.nan,
                "mean": float(np.nanmean(valid)) if len(valid) else np.nan,
            }
            if n_boot > 0 and len(valid) >= 5:
                lo, hi = bootstrap_ci(valid, rng, n_boot)
                row["median_ci_low"] = lo
                row["median_ci_high"] = hi
            else:
                row["median_ci_low"] = np.nan
                row["median_ci_high"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lifecycle-table", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--regime-col", default="gmm_regime_named")
    parser.add_argument("--track-col", default="track_uid")
    parser.add_argument("--phase-col", default="lifecycle_phase_norm")
    parser.add_argument("--variables", nargs="*", default=DEFAULT_VARIABLES)
    parser.add_argument("--n-bins", type=int, default=21)
    parser.add_argument("--bootstrap", type=int, default=0, help="Number of bootstrap replicates for median CI; 0 disables.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = read_table(args.lifecycle_table)

    missing = [c for c in [args.regime_col] + list(args.variables) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    phase = infer_lifecycle_phase(df, args.phase_col)
    df = df.copy()
    df["phase"] = phase.clip(0, 1)
    bins = np.linspace(0, 1, args.n_bins + 1)
    df["phase_bin"] = pd.cut(df["phase"], bins=bins, include_lowest=True, labels=False)
    centers = 0.5 * (bins[:-1] + bins[1:])
    df["phase_center"] = df["phase_bin"].map({i: c for i, c in enumerate(centers)})
    df = df[df[args.regime_col].isin(REGIME_ORDER)].dropna(subset=["phase_bin"])

    out = summarize(df, args.variables, args.regime_col, args.track_col, args.bootstrap, args.seed)
    out_file = args.out_dir / "lifecycle_summary_by_regime_phase.csv"
    out.to_csv(out_file, index=False)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
