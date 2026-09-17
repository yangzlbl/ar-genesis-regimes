#!/usr/bin/env python
"""
Match hourly regional AR landfall flags to final SH-corrected genesis regimes.

Workflow:
  1. Read hourly regional Tar/object file from build_region_hourly_ar_landfall_object_centroids.py.
  2. For Tar_hour == 1, link regional AR object to raw DART steps by
       time_value + ar_object_label_t == time_value + label_t.
  3. Attach final SH-corrected genesis regime using the cleaned lifecycle table
       features_1970_2024_clean_regime_matched_sh_corrected.csv.gz
     by track_uid.
  4. Aggregate hourly labels to daily regional AR-regime labels.

This script intentionally does NOT force unmatched AR-mask hours into a regime.
It preserves explicit diagnostic statuses:
  - matched_raw_step_with_sh_corrected_regime
  - matched_raw_step_but_no_sh_corrected_regime
  - no_raw_step_label_match

Daily regime-specific analysis should use daily_AR_regime_status ==
'single_known_regime'.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


REGIME_ORDER = ["High-moisture", "Coupled-cyclonic", "Frontal", "Ridge"]

DEFAULT_TRACK_DIR = "/pscratch/sd/y/yangzhou/ARgenesis/out_tracks_1970_2024"
DEFAULT_LIFECYCLE_FILE = (
    "/pscratch/sd/y/yangzhou/ARgenesis/lifecycle_features/clean/"
    "features_1970_2024_clean_regime_matched_sh_corrected.csv.gz"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Match regional hourly AR landfall objects to SH-corrected genesis regimes."
    )
    p.add_argument("--hourly-tar-file", required=True,
                   help="Hourly Tar/object centroid CSV produced by Step 2.")
    p.add_argument("--track-dir", default=DEFAULT_TRACK_DIR,
                   help="Directory containing raw DART steps_YYYY.csv.gz files.")
    p.add_argument("--lifecycle-file", default=DEFAULT_LIFECYCLE_FILE,
                   help="SH-corrected lifecycle/regime file.")
    p.add_argument("--start-year", type=int, required=True)
    p.add_argument("--end-year", type=int, required=True)
    p.add_argument("--regions", nargs="*", default=None,
                   help="Optional subset of region names to process.")
    p.add_argument("--out-hourly-file", required=True)
    p.add_argument("--out-daily-file", required=True)
    p.add_argument("--out-track-map-file", default=None,
                   help="Optional output track_uid-to-regime lookup used by this run.")
    p.add_argument("--out-diagnostics-file", default=None,
                   help="Optional output diagnostics summary CSV.")
    p.add_argument("--chunksize", type=int, default=None,
                   help="Optional chunksize for reading lifecycle file. Useful for memory-limited runs.")
    return p.parse_args()


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def normalize_time(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.floor("h")


def read_tar_file(path: str | Path, regions: Optional[List[str]] = None) -> pd.DataFrame:
    print(f"[READ] hourly Tar/object file: {path}", flush=True)
    tar = pd.read_csv(path)

    required = ["region", "time_value", "date", "Tar_hour", "ar_object_label_t"]
    missing = [c for c in required if c not in tar.columns]
    if missing:
        raise KeyError(f"Hourly Tar file is missing required columns: {missing}")

    if regions:
        tar = tar[tar["region"].isin(regions)].copy()
        print(f"[INFO] restricted to regions: {regions}", flush=True)

    tar["time_value"] = normalize_time(tar["time_value"])
    tar["date"] = pd.to_datetime(tar["date"], errors="coerce").dt.floor("D")
    tar["Tar_hour"] = pd.to_numeric(tar["Tar_hour"], errors="coerce").fillna(0).astype(int)
    tar["ar_object_label_t"] = pd.to_numeric(tar["ar_object_label_t"], errors="coerce").astype("Int64")

    print(f"[INFO] Tar rows: {tar.shape}", flush=True)
    print("[INFO] Tar_hour counts:", flush=True)
    print(tar["Tar_hour"].value_counts(dropna=False), flush=True)
    return tar


def read_raw_steps(track_dir: str | Path, start_year: int, end_year: int) -> pd.DataFrame:
    cols = [
        "track_id", "time_value", "label_t", "area_cells", "area_km2",
        "centroid_lat", "centroid_lon",
    ]
    pieces = []
    for year in range(start_year, end_year + 1):
        f = Path(track_dir) / f"steps_{year}.csv.gz"
        if not f.exists():
            raise FileNotFoundError(f"Missing raw DART step file: {f}")
        print(f"[READ] raw DART steps {year}: {f}", flush=True)
        s = pd.read_csv(f, usecols=lambda c: c in cols)
        missing = [c for c in ["track_id", "time_value", "label_t"] if c not in s.columns]
        if missing:
            raise KeyError(f"{f} missing required columns: {missing}")
        s["year"] = year
        s["time_value"] = normalize_time(s["time_value"])
        s["label_t"] = pd.to_numeric(s["label_t"], errors="coerce").astype("Int64")
        s["track_uid"] = s["year"].astype(str) + "_" + s["track_id"].astype(str)
        pieces.append(s)

    steps = pd.concat(pieces, ignore_index=True)
    print(f"[INFO] raw steps rows before key dedupe: {steps.shape}", flush=True)

    # For a connected-component label at a given timestamp, there should be only one retained step.
    dup = steps.duplicated(["time_value", "label_t"], keep=False)
    ndup = int(dup.sum())
    if ndup:
        print(
            f"[WARN] found {ndup} rows with duplicate (time_value, label_t); "
            "keeping first after sorting by area_cells descending.",
            flush=True,
        )
        sort_cols = ["time_value", "label_t"]
        if "area_cells" in steps.columns:
            steps = steps.sort_values(sort_cols + ["area_cells"], ascending=[True, True, False])
        else:
            steps = steps.sort_values(sort_cols)
        steps = steps.drop_duplicates(["time_value", "label_t"], keep="first")

    print(f"[INFO] raw steps rows used: {steps.shape}", flush=True)
    return steps


def _finalize_life_track_map(life: pd.DataFrame) -> pd.DataFrame:
    required = ["track_uid", "gmm_regime_named"]
    missing = [c for c in required if c not in life.columns]
    if missing:
        raise KeyError(f"Lifecycle file missing required columns: {missing}")

    life = life[life["track_uid"].notna()].copy()

    # One final regime should be attached to each track_uid. Check and report.
    consistency = (
        life.groupby("track_uid")["gmm_regime_named"]
        .nunique(dropna=True)
        .sort_values(ascending=False)
    )
    if not consistency.empty:
        print("[INFO] max unique regimes per track_uid:", int(consistency.iloc[0]), flush=True)
        if int(consistency.iloc[0]) > 1:
            bad = consistency[consistency > 1].head(10)
            print("[WARN] track_uid with multiple regimes, first 10:", flush=True)
            print(bad, flush=True)

    sort_cols = ["track_uid"]
    if "step_index" in life.columns:
        sort_cols.append("step_index")
    elif "time_value" in life.columns:
        life["time_value"] = pd.to_datetime(life["time_value"], errors="coerce")
        sort_cols.append("time_value")

    track_map = life.sort_values(sort_cols).drop_duplicates("track_uid", keep="first").copy()

    keep = [
        "track_uid", "global_track_id", "connected_track_id",
        "gmm_regime_named", "gmm_regime", "gmm_component_raw",
        "gmm_probability", "gmm_max_prob", "match_status",
    ]
    keep = [c for c in keep if c in track_map.columns]
    track_map = track_map[keep].copy()

    print("[INFO] SH-corrected track-level regime counts:", flush=True)
    print(track_map["gmm_regime_named"].value_counts(dropna=False), flush=True)
    return track_map


def read_lifecycle_track_map(
    lifecycle_file: str | Path,
    start_year: int,
    end_year: int,
    chunksize: Optional[int] = None,
) -> pd.DataFrame:
    print(f"[READ] SH-corrected lifecycle file: {lifecycle_file}", flush=True)

    usecols = [
        "track_uid", "global_track_id", "connected_track_id", "year",
        "gmm_regime_named", "gmm_regime", "gmm_component_raw",
        "gmm_probability", "gmm_max_prob", "match_status",
        "step_index", "time_value",
    ]

    if chunksize is None:
        life = pd.read_csv(lifecycle_file, usecols=lambda c: c in usecols)
        if "year" not in life.columns:
            raise KeyError("Lifecycle file must contain a 'year' column.")
        life = life[(life["year"] >= start_year) & (life["year"] <= end_year)].copy()
        print(f"[INFO] lifecycle rows in year range: {life.shape}", flush=True)
        return _finalize_life_track_map(life)

    pieces = []
    for chunk in pd.read_csv(lifecycle_file, usecols=lambda c: c in usecols, chunksize=chunksize):
        if "year" not in chunk.columns:
            raise KeyError("Lifecycle file must contain a 'year' column.")
        chunk = chunk[(chunk["year"] >= start_year) & (chunk["year"] <= end_year)].copy()
        if not chunk.empty:
            pieces.append(chunk)
    life = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=usecols)
    print(f"[INFO] lifecycle rows in year range: {life.shape}", flush=True)
    return _finalize_life_track_map(life)


def match_hourly(tar: pd.DataFrame, steps: pd.DataFrame, track_map: pd.DataFrame) -> pd.DataFrame:
    tar1 = tar[tar["Tar_hour"] == 1].copy().reset_index(drop=True)
    print(f"[INFO] matching Tar=1 rows: {len(tar1)}", flush=True)

    step_cols = [
        "track_id", "track_uid", "time_value", "label_t",
        "centroid_lat", "centroid_lon", "area_cells", "area_km2",
    ]
    step_cols = [c for c in step_cols if c in steps.columns]

    linked = tar1.merge(
        steps[step_cols],
        left_on=["time_value", "ar_object_label_t"],
        right_on=["time_value", "label_t"],
        how="left",
        suffixes=("", "_rawstep"),
    )

    # Add raw step status before adding regimes.
    linked["raw_step_status"] = "no_raw_step_label_match"
    linked.loc[linked["track_uid"].notna(), "raw_step_status"] = "matched_raw_step"

    linked = linked.merge(
        track_map,
        on="track_uid",
        how="left",
        suffixes=("", "_life"),
    )

    linked["matched_regime"] = linked["gmm_regime_named"]
    linked["final_assignment_status"] = "no_raw_step_label_match"
    linked.loc[
        linked["track_uid"].notna() & linked["matched_regime"].notna(),
        "final_assignment_status",
    ] = "matched_raw_step_with_sh_corrected_regime"
    linked.loc[
        linked["track_uid"].notna() & linked["matched_regime"].isna(),
        "final_assignment_status",
    ] = "matched_raw_step_but_no_sh_corrected_regime"

    # Keep old-compatible name too.
    linked["match_status"] = linked["final_assignment_status"]

    print("[INFO] hourly final assignment status:", flush=True)
    print(linked["final_assignment_status"].value_counts(dropna=False), flush=True)
    print("[INFO] hourly final regime counts:", flush=True)
    print(linked["matched_regime"].value_counts(dropna=False), flush=True)

    return linked


def aggregate_to_daily(tar: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly Tar/regime assignments to one row per region-day.

    Daily regime rule:
      - one known regime among all known hourly assignments -> single_known_regime
        even if some hours are unmatched;
      - two or more known regimes -> multiple_known_regimes / Mixed;
      - no known regime -> unmatched_only.
    """
    tar = tar.copy()
    tar["date"] = pd.to_datetime(tar["date"], errors="coerce").dt.floor("D")

    # Older pandas can be finicky with named aggregation + as_index=False.
    # Use explicit reset_index() so region/date are guaranteed columns.
    agg_dict = {
        "Tar_day": ("Tar_hour", "max"),
        "n_Tar_hours": ("Tar_hour", "sum"),
    }
    if "ar_mask_area_km2" in tar.columns:
        agg_dict["ar_mask_area_km2_max"] = ("ar_mask_area_km2", "max")
        agg_dict["ar_mask_area_km2_sum_hours"] = ("ar_mask_area_km2", "sum")
    else:
        agg_dict["ar_mask_area_km2_max"] = ("Tar_hour", "max")
        agg_dict["ar_mask_area_km2_sum_hours"] = ("Tar_hour", "sum")

    tar_daily = (
        tar.groupby(["region", "date"])
        .agg(**agg_dict)
        .reset_index()
    )

    rows = []
    if hourly is not None and not hourly.empty:
        h = hourly.copy()
        h["date"] = pd.to_datetime(h["date"], errors="coerce").dt.floor("D")

        for (region, date), g in h.groupby(["region", "date"]):
            regs = sorted([
                r for r in g["matched_regime"].dropna().unique()
                if r in REGIME_ORDER
            ], key=lambda x: REGIME_ORDER.index(x))

            has_no_raw_step = (g["final_assignment_status"] == "no_raw_step_label_match").any()
            has_raw_no_regime = (
                g["final_assignment_status"] == "matched_raw_step_but_no_sh_corrected_regime"
            ).any()

            n_known_hours = int(g["matched_regime"].isin(REGIME_ORDER).sum())
            n_no_raw_step_hours = int((g["final_assignment_status"] == "no_raw_step_label_match").sum())
            n_raw_no_regime_hours = int((g["final_assignment_status"] == "matched_raw_step_but_no_sh_corrected_regime").sum())
            n_tar_hours = int(len(g))

            if len(regs) == 1:
                daily_regime = regs[0]
                status = "single_known_regime"
            elif len(regs) >= 2:
                daily_regime = "Mixed"
                status = "multiple_known_regimes"
            else:
                daily_regime = "Unmatched"
                status = "unmatched_only"

            rows.append({
                "region": region,
                "date": date,
                "daily_AR_regime": daily_regime,
                "daily_AR_regime_status": status,
                "known_regimes_on_day": ",".join(regs),
                "n_known_regime_hours": n_known_hours,
                "n_no_raw_step_hours": n_no_raw_step_hours,
                "n_raw_step_no_regime_hours": n_raw_no_regime_hours,
                "n_matched_Tar_hours": n_tar_hours,
                "has_no_raw_step_hours": bool(has_no_raw_step),
                "has_raw_step_no_regime_hours": bool(has_raw_no_regime),
            })

    reg_daily = pd.DataFrame(rows)
    if reg_daily.empty:
        reg_daily = pd.DataFrame(columns=[
            "region", "date", "daily_AR_regime", "daily_AR_regime_status",
            "known_regimes_on_day", "n_known_regime_hours", "n_no_raw_step_hours",
            "n_raw_step_no_regime_hours", "n_matched_Tar_hours",
            "has_no_raw_step_hours", "has_raw_step_no_regime_hours",
        ])
    else:
        reg_daily["date"] = pd.to_datetime(reg_daily["date"], errors="coerce").dt.floor("D")

    # Safety check; avoids cryptic KeyError if a future edit breaks grouping.
    for name, df_check in [("tar_daily", tar_daily), ("reg_daily", reg_daily)]:
        missing = [c for c in ["region", "date"] if c not in df_check.columns]
        if missing:
            raise KeyError(f"{name} is missing merge keys {missing}; columns={df_check.columns.tolist()}")

    out = tar_daily.merge(reg_daily, on=["region", "date"], how="left")

    out.loc[out["Tar_day"] == 0, "daily_AR_regime"] = "No_AR"
    out.loc[out["Tar_day"] == 0, "daily_AR_regime_status"] = "no_ar"

    out["daily_AR_regime"] = out["daily_AR_regime"].fillna("Unmatched")
    out["daily_AR_regime_status"] = out["daily_AR_regime_status"].fillna("unmatched_only")
    out["known_regimes_on_day"] = out["known_regimes_on_day"].fillna("")

    fill_zero_cols = [
        "n_known_regime_hours", "n_no_raw_step_hours", "n_raw_step_no_regime_hours",
        "n_matched_Tar_hours",
    ]
    for c in fill_zero_cols:
        if c in out.columns:
            out[c] = out[c].fillna(0).astype(int)

    for c in ["has_no_raw_step_hours", "has_raw_step_no_regime_hours"]:
        if c in out.columns:
            out[c] = out[c].fillna(False).astype(bool)

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    print("[INFO] daily AR regime status:", flush=True)
    print(out["daily_AR_regime_status"].value_counts(dropna=False), flush=True)
    print("[INFO] daily AR regime counts:", flush=True)
    print(out["daily_AR_regime"].value_counts(dropna=False), flush=True)

    return out


def build_diagnostics(tar: pd.DataFrame, hourly: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(metric, category, count):
        rows.append({"metric": metric, "category": category, "count": int(count)})

    add("hourly_Tar", "all_rows", len(tar))
    add("hourly_Tar", "Tar_hour_1", int((tar["Tar_hour"] == 1).sum()))
    add("hourly_Tar", "Tar_hour_0", int((tar["Tar_hour"] == 0).sum()))

    for k, v in hourly["final_assignment_status"].value_counts(dropna=False).items():
        add("hourly_assignment_status", str(k), v)
    for k, v in hourly["matched_regime"].value_counts(dropna=False).items():
        add("hourly_matched_regime", str(k), v)

    for k, v in daily["daily_AR_regime_status"].value_counts(dropna=False).items():
        add("daily_AR_regime_status", str(k), v)
    for k, v in daily["daily_AR_regime"].value_counts(dropna=False).items():
        add("daily_AR_regime", str(k), v)

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    tar = read_tar_file(args.hourly_tar_file, regions=args.regions)
    steps = read_raw_steps(args.track_dir, args.start_year, args.end_year)
    track_map = read_lifecycle_track_map(
        args.lifecycle_file,
        args.start_year,
        args.end_year,
        chunksize=args.chunksize,
    )

    if args.out_track_map_file:
        ensure_parent(args.out_track_map_file)
        track_map.to_csv(args.out_track_map_file, index=False, compression="gzip")
        print(f"[SAVE] {args.out_track_map_file} rows={len(track_map)}", flush=True)

    hourly = match_hourly(tar, steps, track_map)
    daily = aggregate_to_daily(tar, hourly)

    ensure_parent(args.out_hourly_file)
    hourly.to_csv(args.out_hourly_file, index=False, compression="gzip")
    print(f"[SAVE] {args.out_hourly_file} rows={len(hourly)}", flush=True)

    ensure_parent(args.out_daily_file)
    daily.to_csv(args.out_daily_file, index=False, compression="gzip")
    print(f"[SAVE] {args.out_daily_file} rows={len(daily)}", flush=True)

    if args.out_diagnostics_file:
        diag = build_diagnostics(tar, hourly, daily)
        ensure_parent(args.out_diagnostics_file)
        diag.to_csv(args.out_diagnostics_file, index=False)
        print(f"[SAVE] {args.out_diagnostics_file} rows={len(diag)}", flush=True)


if __name__ == "__main__":
    main()
