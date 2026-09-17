#!/usr/bin/env python
"""
Concatenate yearly precip-regime products and write full-period files.

This script expects yearly outputs from run_precip_regime_pipeline_one_year.py:
  region_daily_precip_{YEAR}.csv.gz
  region_hourly_Tar_object_centroids_{YEAR}.csv.gz
  region_hourly_AR_regime_sh_corrected_{YEAR}.csv.gz
  region_daily_AR_regime_sh_corrected_{YEAR}.csv.gz
  region_daily_precip_AR_regime_merged_{YEAR}.csv.gz
  regime_precip_ratio_summary_{YEAR}.csv.gz
  ar_regime_linkage_diagnostics_{YEAR}.csv
  track_uid_to_sh_corrected_regime_{YEAR}.csv.gz

It writes 1970-2024 full-period versions by default.
"""

import argparse
import gzip
from pathlib import Path

import pandas as pd


PRODUCTS = {
    "daily_precip": {
        "pattern": "region_daily_precip_{year}.csv.gz",
        "out": "region_daily_precip_{start}_{end}.csv.gz",
        "kind": "gz_csv",
    },
    "hourly_tar": {
        "pattern": "region_hourly_Tar_object_centroids_{year}.csv.gz",
        "out": "region_hourly_Tar_object_centroids_{start}_{end}.csv.gz",
        "kind": "gz_csv",
    },
    "hourly_regime": {
        "pattern": "region_hourly_AR_regime_sh_corrected_{year}.csv.gz",
        "out": "region_hourly_AR_regime_sh_corrected_{start}_{end}.csv.gz",
        "kind": "gz_csv",
    },
    "daily_regime": {
        "pattern": "region_daily_AR_regime_sh_corrected_{year}.csv.gz",
        "out": "region_daily_AR_regime_sh_corrected_{start}_{end}.csv.gz",
        "kind": "gz_csv",
    },
    "merged": {
        "pattern": "region_daily_precip_AR_regime_merged_{year}.csv.gz",
        "out": "region_daily_precip_AR_regime_merged_{start}_{end}.csv.gz",
        "kind": "gz_csv",
    },
    "summary": {
        "pattern": "regime_precip_ratio_summary_{year}.csv.gz",
        "out": "regime_precip_ratio_summary_{start}_{end}.csv.gz",
        "kind": "gz_csv",
    },
    "diagnostics": {
        "pattern": "ar_regime_linkage_diagnostics_{year}.csv",
        "out": "ar_regime_linkage_diagnostics_{start}_{end}.csv",
        "kind": "csv",
    },
    "track_map": {
        "pattern": "track_uid_to_sh_corrected_regime_{year}.csv.gz",
        "out": "track_uid_to_sh_corrected_regime_{start}_{end}.csv.gz",
        "kind": "gz_csv",
    },
}


def gzip_has_header(path: Path) -> bool:
    if (not path.exists()) or path.stat().st_size == 0:
        return False
    try:
        with gzip.open(path, "rt") as f:
            return bool(f.readline().strip())
    except Exception:
        return False


def csv_has_header(path: Path) -> bool:
    if (not path.exists()) or path.stat().st_size == 0:
        return False
    try:
        with open(path, "rt") as f:
            return bool(f.readline().strip())
    except Exception:
        return False


def read_year_file(path: Path, kind: str) -> pd.DataFrame:
    if kind == "gz_csv":
        if not gzip_has_header(path):
            raise RuntimeError(f"Missing/empty/bad gzip file: {path}")
    else:
        if not csv_has_header(path):
            raise RuntimeError(f"Missing/empty/bad csv file: {path}")
    return pd.read_csv(path)


def concatenate_product(out_dir: Path, start_year: int, end_year: int, product: str, overwrite: bool = False) -> Path:
    spec = PRODUCTS[product]
    out_path = out_dir / spec["out"].format(start=start_year, end=end_year)

    if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
        print(f"[SKIP] {product}: {out_path} exists")
        return out_path

    frames = []
    for year in range(start_year, end_year + 1):
        path = out_dir / spec["pattern"].format(year=year)
        print(f"[READ] {product} {year}: {path}")
        df = read_year_file(path, spec["kind"])
        if "year" not in df.columns:
            df["year"] = year
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)

    # Normalize common date column, then sort if possible.
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    if "time_value" in out.columns:
        # preserve full timestamp text after sorting
        out["_time_sort"] = pd.to_datetime(out["time_value"], errors="coerce")

    sort_cols = []
    if "date" in out.columns:
        sort_cols.append("date")
    if "_time_sort" in out.columns:
        sort_cols.append("_time_sort")
    if "region" in out.columns:
        sort_cols.append("region")
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    if "_time_sort" in out.columns:
        out = out.drop(columns=["_time_sort"])

    print(f"[SAVE] {out_path} rows={len(out)} cols={len(out.columns)}")
    if spec["kind"] == "gz_csv":
        out.to_csv(out_path, index=False, compression="gzip")
    else:
        out.to_csv(out_path, index=False)

    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="/pscratch/sd/y/yangzhou/ARgenesis/precip_regime")
    p.add_argument("--start-year", type=int, default=1970)
    p.add_argument("--end-year", type=int, default=2024)
    p.add_argument(
        "--products",
        nargs="+",
        default=["daily_precip", "daily_regime", "merged", "summary", "diagnostics", "track_map"],
        choices=list(PRODUCTS),
        help="Products to concatenate. Hourly products are large; include hourly_tar/hourly_regime only if needed.",
    )
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)

    written = []
    for product in args.products:
        written.append(concatenate_product(out_dir, args.start_year, args.end_year, product, overwrite=args.overwrite))

    print("\n[DONE] wrote/checked:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
