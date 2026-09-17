#!/usr/bin/env python
"""
Run the modular regional precipitation / AR-regime attribution workflow for one year.

This driver is designed for embarrassingly parallel year-by-year execution.
For each focused year it runs:
  1. build_region_daily_precip_timeseries.py
  2. build_region_hourly_ar_landfall_object_centroids.py
  3. match_region_hourly_ar_to_regime_sh_corrected_final.py
  4. merge_daily_precip_ar_regime.py

The daily regime rule in Step 3 is:
  - single_known_regime: all known AR-regime hours on the region-day have one regime
    even if some other AR hours are unmatched;
  - multiple_known_regimes: two or more known regimes occur on the region-day;
  - unmatched_only: no known regime-attributed AR hours occur on the region-day.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


DEFAULT_OUT_DIR = "/pscratch/sd/y/yangzhou/ARgenesis/precip_regime"
DEFAULT_TRACK_DIR = "/pscratch/sd/y/yangzhou/ARgenesis/out_tracks_1970_2024"
DEFAULT_LIFECYCLE_FILE = (
    "/pscratch/sd/y/yangzhou/ARgenesis/lifecycle_features/clean/"
    "features_1970_2024_clean_regime_matched_sh_corrected.csv.gz"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run one year of the precip/AR-regime attribution pipeline."
    )
    p.add_argument("--year", type=int, required=True, help="Focused year to process, e.g. 1970.")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--scripts-dir", default=".", help="Directory containing the four workflow scripts.")
    p.add_argument("--python", default=sys.executable, help="Python executable to use.")

    # Script names can be overridden if needed.
    p.add_argument("--precip-script", default="build_region_daily_precip_timeseries.py")
    p.add_argument("--tar-script", default="build_region_hourly_ar_landfall_object_centroids.py")
    p.add_argument("--match-script", default="match_region_hourly_ar_to_regime_sh_corrected_final.py")
    p.add_argument("--merge-script", default="merge_daily_precip_ar_regime.py")

    # Core data paths.
    p.add_argument("--track-dir", default=DEFAULT_TRACK_DIR)
    p.add_argument("--lifecycle-file", default=DEFAULT_LIFECYCLE_FILE)

    # Data variable options.
    p.add_argument("--precip-var", default="tp")
    p.add_argument("--precip-units", default="kg m-2 s-1")
    p.add_argument("--mask-var", default="ar_binary_tag")
    p.add_argument("--mask-file-pattern", default="*.nc4*")

    # Thresholds.
    p.add_argument("--wet-threshold-mm-day", type=float, default=2.0)
    p.add_argument("--wet-metric", default="precip_mean_land_mm_day")
    p.add_argument("--ratio-metric", default="precip_mean_land_mm_day")

    # Optional subset / performance.
    p.add_argument("--regions", nargs="*", default=None, help="Optional subset of regions.")
    p.add_argument("--chunksize", type=int, default=None, help="Chunksize for Step 3 lifecycle read.")
    p.add_argument("--chunks-time", type=int, default=None, help="Optional time chunk size for Step 1/2 scripts if supported.")
    p.add_argument("--n-boot", type=int, default=1000, help="Bootstrap samples used by merge summary.")

    # Execution controls.
    p.add_argument("--skip-existing", action="store_true", help="Skip each step if its main output already exists.")
    p.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    p.add_argument("--stop-after", choices=["precip", "tar", "match", "merge"], default=None)
    return p.parse_args()


def script_path(scripts_dir: str | Path, name: str) -> str:
    p = Path(name)
    if p.is_absolute():
        return str(p)
    return str(Path(scripts_dir) / name)


def run_cmd(cmd: List[str], output_file: Optional[str], skip_existing: bool, dry_run: bool) -> None:
    if output_file and skip_existing and Path(output_file).exists():
        print(f"[SKIP] {output_file} exists", flush=True)
        return
    print("[CMD]", " ".join(shlex.quote(str(x)) for x in cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def add_regions(cmd: List[str], regions: Optional[List[str]]) -> List[str]:
    if regions:
        cmd += ["--regions"] + list(regions)
    return cmd


def maybe_add(cmd: List[str], flag: str, value) -> List[str]:
    if value is not None:
        cmd += [flag, str(value)]
    return cmd


def main() -> None:
    args = parse_args()
    year = args.year
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    precip_file = out_dir / f"region_daily_precip_{year}.csv.gz"
    tar_file = out_dir / f"region_hourly_Tar_object_centroids_{year}.csv.gz"
    hourly_regime_file = out_dir / f"region_hourly_AR_regime_sh_corrected_{year}.csv.gz"
    daily_regime_file = out_dir / f"region_daily_AR_regime_sh_corrected_{year}.csv.gz"
    track_map_file = out_dir / f"track_uid_to_sh_corrected_regime_{year}.csv.gz"
    diag_file = out_dir / f"ar_regime_linkage_diagnostics_{year}.csv"
    merged_file = out_dir / f"region_daily_precip_AR_regime_merged_{year}.csv.gz"
    summary_file = out_dir / f"regime_precip_ratio_summary_{year}.csv.gz"

    # Step 1: daily precipitation.
    cmd1 = [
        args.python,
        script_path(args.scripts_dir, args.precip_script),
        "--start-year", str(year),
        "--end-year", str(year),
        "--precip-var", args.precip_var,
        "--precip-units", args.precip_units,
        "--wet-threshold-mm-day", str(args.wet_threshold_mm_day),
        "--out-file", str(precip_file),
    ]
    cmd1 = add_regions(cmd1, args.regions)
    cmd1 = maybe_add(cmd1, "--chunks-time", args.chunks_time)
    run_cmd(cmd1, str(precip_file), args.skip_existing, args.dry_run)
    if args.stop_after == "precip":
        return

    # Step 2: hourly AR landfall/object centroids.
    cmd2 = [
        args.python,
        script_path(args.scripts_dir, args.tar_script),
        "--start-year", str(year),
        "--end-year", str(year),
        "--mask-var", args.mask_var,
        "--mask-file-pattern", args.mask_file_pattern,
        "--out-file", str(tar_file),
    ]
    cmd2 = add_regions(cmd2, args.regions)
    cmd2 = maybe_add(cmd2, "--chunks-time", args.chunks_time)
    run_cmd(cmd2, str(tar_file), args.skip_existing, args.dry_run)
    if args.stop_after == "tar":
        return

    # Step 3: Tar/object -> raw steps -> SH-corrected regime.
    cmd3 = [
        args.python,
        script_path(args.scripts_dir, args.match_script),
        "--hourly-tar-file", str(tar_file),
        "--track-dir", args.track_dir,
        "--lifecycle-file", args.lifecycle_file,
        "--start-year", str(year),
        "--end-year", str(year),
        "--out-hourly-file", str(hourly_regime_file),
        "--out-daily-file", str(daily_regime_file),
        "--out-track-map-file", str(track_map_file),
        "--out-diagnostics-file", str(diag_file),
    ]
    cmd3 = add_regions(cmd3, args.regions)
    cmd3 = maybe_add(cmd3, "--chunksize", args.chunksize)
    run_cmd(cmd3, str(daily_regime_file), args.skip_existing, args.dry_run)
    if args.stop_after == "match":
        return

    # Step 4: merge daily precip + daily AR regime labels.
    cmd4 = [
        args.python,
        script_path(args.scripts_dir, args.merge_script),
        "--daily-precip-file", str(precip_file),
        "--daily-ar-regime-file", str(daily_regime_file),
        "--wet-metric", args.wet_metric,
        "--wet-threshold-mm-day", str(args.wet_threshold_mm_day),
        "--ratio-metric", args.ratio_metric,
        "--n-boot", str(args.n_boot),
        "--out-file", str(merged_file),
        "--summary-file", str(summary_file),
    ]
    run_cmd(cmd4, str(merged_file), args.skip_existing, args.dry_run)

    print("[DONE]", year, flush=True)
    print("[OUTPUT]", merged_file, flush=True)
    print("[SUMMARY]", summary_file, flush=True)


if __name__ == "__main__":
    main()
