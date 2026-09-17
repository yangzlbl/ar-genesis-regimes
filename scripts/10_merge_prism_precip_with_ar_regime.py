#!/usr/bin/env python3
"""
Merge PRISM daily precipitation metrics with the new same-box daily AR/regime
attribution table.

This script is intended for the U.S. West Coast PRISM sensitivity analysis.
It uses PRISM precipitation to define wet/dry days, but uses the new ERA5/BARD
same-box AR landfall and SH-corrected regime attribution table to define
AR/non-AR and regime labels.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

REGIME_ORDER = ["High-moisture", "Coupled-cyclonic", "Frontal", "Ridge"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prism-precip-file", required=True,
                   help="PRISM daily precipitation summary CSV.GZ")
    p.add_argument("--daily-ar-regime-file", required=True,
                   help="New same-box daily AR regime file, e.g. region_daily_AR_regime_sh_corrected_1970_2024.csv.gz")
    p.add_argument("--region", default="U.S. West Coast")
    p.add_argument("--start-year", type=int, default=1981)
    p.add_argument("--end-year", type=int, default=2024)
    p.add_argument("--wet-metric", default="precip_mean_land_mm_day")
    p.add_argument("--wet-threshold-mm-day", type=float, default=2.0)
    p.add_argument("--ratio-metric", default="precip_mean_land_mm_day")
    p.add_argument("--out-file", required=True)
    p.add_argument("--summary-file", required=True)
    return p.parse_args()


def classify_precip_day(row, wet_metric, wet_threshold):
    val = row[wet_metric]
    if not np.isfinite(val) or val <= wet_threshold:
        return "dry_or_weak_precip"

    status = row.get("daily_AR_regime_status", np.nan)
    regime = row.get("daily_AR_regime", np.nan)
    tar_day = row.get("Tar_day", np.nan)

    if status == "no_ar" or regime == "No_AR" or tar_day == 0:
        return "non_ar_precip"
    return "ar_precip"


def summarize(df, ratio_metric):
    rows = []
    nonar = df[df["precip_day_type"].eq("non_ar_precip")]
    nonar_med = nonar[ratio_metric].median()

    for reg in REGIME_ORDER:
        ar = df[
            df["precip_day_type"].eq("ar_precip")
            & df["daily_AR_regime_status"].eq("single_known_regime")
            & df["daily_AR_regime"].eq(reg)
        ]
        ar_med = ar[ratio_metric].median()
        rows.append({
            "precip_source": "PRISM",
            "region": df["region"].iloc[0] if len(df) else np.nan,
            "regime": reg,
            "metric": ratio_metric,
            "n_ar_days": len(ar),
            "n_nonar_days": len(nonar),
            "ar_median": ar_med,
            "nonar_median": nonar_med,
            "ratio_vs_nonar": ar_med / nonar_med if pd.notna(nonar_med) and nonar_med > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def main():
    args = parse_args()

    prism = pd.read_csv(args.prism_precip_file)
    prism["date"] = pd.to_datetime(prism["date"]).dt.normalize()
    prism = prism[
        prism["region"].eq(args.region)
        & prism["date"].dt.year.between(args.start_year, args.end_year)
    ].copy()

    ar = pd.read_csv(args.daily_ar_regime_file)
    ar["date"] = pd.to_datetime(ar["date"]).dt.normalize()
    ar = ar[
        ar["region"].eq(args.region)
        & ar["date"].dt.year.between(args.start_year, args.end_year)
    ].copy()

    keep_ar_cols = [
        c for c in [
            "date", "region", "Tar_day", "n_Tar_hours",
            "daily_AR_regime", "daily_AR_regime_status",
            "has_unmatched_hours", "n_known_regime_Tar_hours", "n_unmatched_Tar_hours",
            "known_regime_list", "known_regime_hour_counts",
        ]
        if c in ar.columns
    ]
    ar = ar[keep_ar_cols].copy()

    merged = prism.merge(ar, on=["region", "date"], how="left", validate="one_to_one")

    if args.wet_metric not in merged.columns:
        raise KeyError(f"wet metric {args.wet_metric!r} not in merged columns")
    if args.ratio_metric not in merged.columns:
        raise KeyError(f"ratio metric {args.ratio_metric!r} not in merged columns")

    # Fill missing AR labels conservatively as no_ar only if the AR/regime table has no row.
    # In normal use the same region/date grid should be complete.
    merged["daily_AR_regime_status"] = merged["daily_AR_regime_status"].fillna("no_ar")
    merged["daily_AR_regime"] = merged["daily_AR_regime"].fillna("No_AR")

    merged["precip_day_type"] = merged.apply(
        lambda r: classify_precip_day(r, args.wet_metric, args.wet_threshold_mm_day),
        axis=1,
    )

    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_file, index=False, compression="gzip")

    summary = summarize(merged, args.ratio_metric)
    summary_file = Path(args.summary_file)
    summary.to_csv(summary_file, index=False, compression="gzip")

    print(f"[SAVE] {out_file} rows={len(merged)}")
    print(f"[SAVE] {summary_file} rows={len(summary)}")
    print("\n[INFO] precip_day_type counts:")
    print(merged["precip_day_type"].value_counts(dropna=False).to_string())
    print("\n[INFO] wet AR regime-status counts:")
    print(merged.loc[merged["precip_day_type"].eq("ar_precip"), "daily_AR_regime_status"].value_counts(dropna=False).to_string())
    print("\n[INFO] ratio summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
