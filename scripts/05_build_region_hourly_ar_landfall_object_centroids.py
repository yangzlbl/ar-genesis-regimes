#!/usr/bin/env python3
"""
Build hourly regional AR landfall flags and same-hour full-object centroids.

This script uses the SAME regional boxes for AR landfall and precipitation.
For each region x hour:
  - Tar_hour = 1 if the TECA-BARD binary AR mask overlaps land in the region.
  - If Tar_hour = 1, identify the connected AR object label(s) overlapping
    the regional land box.
  - Choose the dominant overlapping object by regional land-overlap area.
  - Record BOTH:
      (a) centroid of the regional AR footprint inside the box, and
      (b) centroid of the FULL connected AR object at that hour.

The full-object centroid is the appropriate quantity to compare with DART
steps_YYYY.csv.gz centroid_lat/centroid_lon. The object label is also saved as
ar_object_label_t, which can be matched directly to steps.label_t at the same
hour when the same labeling convention is used.
"""

from pathlib import Path
import argparse
import re
import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage as ndi

REGION_BOXES = {
    "U.S. West Coast":        (-130, -117,  32,  50),
    "Alaska / NE Pacific":    (-160, -130,  52,  66),
    "Western Europe":         ( -12,    8,  36,  60),
    "East Asia":              ( 118,  148,  24,  46),
    "Mediterranean":          (  -6,   36,  32,  44),
    "Greenland / Iceland":    ( -55,  -12,  62,  82),
    "Chile":                  ( -78,  -68, -55, -25),
    "SE South America":       ( -60,  -40, -42, -22),
    "SE Australia / NZ":      ( 145,  178, -48, -28),
    "South Africa":           (  15,   35, -36, -22),
    "Antarctic":              (-180,  180, -90, -66),
}

BARD_DIR = (
    "/global/cfs/projectdirs/m4374/user_work_directories/taobrien/"
    "teca_bard_era5_redo/teca_bard_era5"
)

LSM_FILE = (
    "/global/cfs/projectdirs/m3522/cmip6/ERA5/e5.oper.invariant/197901/"
    "e5.oper.invariant.128_172_lsm.ll025sc.1979010100_1979010100.nc"
)


def extract_year_from_name(path):
    name = Path(path).name
    for m in re.finditer(r"(19\d{2}|20\d{2})(\d{2})?", name):
        y = int(m.group(1))
        if 1900 <= y <= 2100:
            return y
    return None


def discover_files(root, pattern, start_year, end_year):
    files = sorted(Path(root).rglob(pattern))
    out = []
    for f in files:
        y = extract_year_from_name(f)
        if y is not None and start_year <= y <= end_year:
            out.append(f)
    if not out:
        raise FileNotFoundError(f"No mask files found under {root} pattern={pattern} years={start_year}-{end_year}")
    return out


def find_data_var(ds, preferred=None, candidates=None):
    data_vars = list(ds.data_vars)
    candidates = candidates or []
    if preferred and preferred in ds.variables:
        return preferred
    if preferred:
        lower_map = {v.lower(): v for v in data_vars}
        if preferred.lower() in lower_map:
            chosen = lower_map[preferred.lower()]
            print(f"[INFO] variable {preferred!r} not found; using {chosen!r}", flush=True)
            return chosen
    lower_map = {v.lower(): v for v in data_vars}
    for c in candidates:
        if c in ds.variables:
            return c
        if c.lower() in lower_map:
            chosen = lower_map[c.lower()]
            print(f"[INFO] using inferred variable {chosen!r}", flush=True)
            return chosen
    if len(data_vars) == 1:
        chosen = data_vars[0]
        print(f"[INFO] using only data variable {chosen!r}", flush=True)
        return chosen
    raise KeyError(f"Could not find variable {preferred!r}; data_vars={data_vars}")


def infer_lat_lon_names(ds):
    lat_candidates = ["lat", "latitude", "LAT", "Latitude"]
    lon_candidates = ["lon", "longitude", "LON", "Longitude"]
    lat = next((c for c in lat_candidates if c in ds.coords or c in ds.dims), None)
    lon = next((c for c in lon_candidates if c in ds.coords or c in ds.dims), None)
    if lat is None or lon is None:
        raise ValueError(f"Could not infer lat/lon names from {list(ds.variables)}")
    return lat, lon


def normalize_lon_180(lon):
    lon = np.asarray(lon, dtype=float)
    return ((lon + 180.0) % 360.0) - 180.0


def circular_mean_lon(lons, weights=None):
    lon_rad = np.deg2rad(normalize_lon_180(lons))
    if weights is None:
        weights = np.ones_like(lon_rad, dtype=float)
    x = np.nansum(weights * np.cos(lon_rad))
    y = np.nansum(weights * np.sin(lon_rad))
    if x == 0 and y == 0:
        return np.nan
    return float(normalize_lon_180(np.rad2deg(np.arctan2(y, x))))


def subset_box_da(da, lat_name, lon_name, box):
    lon_min, lon_max, lat_min, lat_max = box
    lat_vals = da[lat_name].values
    lon_vals = da[lon_name].values
    if lat_vals[0] < lat_vals[-1]:
        out = da.sel({lat_name: slice(lat_min, lat_max)})
    else:
        out = da.sel({lat_name: slice(lat_max, lat_min)})
    if (lon_max - lon_min) >= 359.999:
        return out
    ds_lon_min = float(np.nanmin(lon_vals))
    ds_lon_max = float(np.nanmax(lon_vals))
    if ds_lon_min >= 0 and ds_lon_max > 180:
        a = lon_min % 360
        b = lon_max % 360
    else:
        a = lon_min
        b = lon_max
    if a <= b:
        out = out.sel({lon_name: slice(a, b)})
    else:
        out1 = out.sel({lon_name: slice(a, ds_lon_max)})
        out2 = out.sel({lon_name: slice(ds_lon_min, b)})
        out = xr.concat([out1, out2], dim=lon_name)
    return out


def box_bool_mask(lat_vals, lon_vals, box):
    lon_min, lon_max, lat_min, lat_max = box
    lat_ok = (lat_vals >= lat_min) & (lat_vals <= lat_max)
    lon_norm = normalize_lon_180(lon_vals)
    if (lon_max - lon_min) >= 359.999:
        lon_ok = np.ones_like(lon_vals, dtype=bool)
    else:
        a = ((lon_min + 180.0) % 360.0) - 180.0
        b = ((lon_max + 180.0) % 360.0) - 180.0
        if a <= b:
            lon_ok = (lon_norm >= a) & (lon_norm <= b)
        else:
            lon_ok = (lon_norm >= a) | (lon_norm <= b)
    return lat_ok[:, None] & lon_ok[None, :]


def gridcell_area_km2(lat, dlon_deg, dlat_deg):
    r = 6371.0
    return (r ** 2) * np.deg2rad(dlat_deg) * np.deg2rad(dlon_deg) * np.cos(np.deg2rad(lat))


def prepare_lsm(lsm, target_sub, lat_lsm, lon_lsm, lat_t, lon_t):
    lsm_i = lsm.interp({lat_lsm: target_sub[lat_t], lon_lsm: target_sub[lon_t]}, method="nearest")
    ren = {}
    if lat_lsm != lat_t:
        ren[lat_lsm] = lat_t
    if lon_lsm != lon_t:
        ren[lon_lsm] = lon_t
    if ren:
        lsm_i = lsm_i.rename(ren)
    return lsm_i


def label_ar_objects(binary_2d):
    """8-connected labeling plus simple dateline merge, matching ARTRACK logic."""
    structure = np.ones((3, 3), dtype=int)
    labeled_array, _ = ndi.label(binary_2d.astype(bool), structure=structure)
    new_labeled_array = labeled_array.copy()
    set_0 = {0}
    west = labeled_array[:, 0].copy()
    east = labeled_array[:, -1].copy()
    branches = list(set(west) - set_0)
    flag_branch = np.zeros(len(west), dtype=new_labeled_array.dtype)
    for branch in branches:
        ab_intersect = list(set(east[west == branch]) - set_0)
        if len(ab_intersect) > 0:
            for i_intersect in ab_intersect:
                if np.all(flag_branch[east == i_intersect] == 0):
                    new_labeled_array[labeled_array == i_intersect] = branch
                    flag_branch[east == i_intersect] = branch
                else:
                    new_labeled_array[labeled_array == i_intersect] = flag_branch[east == i_intersect][0]
    return new_labeled_array


def weighted_centroid(mask, lat2d, lon2d, area2d):
    n = int(np.count_nonzero(mask))
    if n == 0:
        return np.nan, np.nan, 0, 0.0
    w = area2d[mask]
    lat = float(np.average(lat2d[mask], weights=w))
    lon = circular_mean_lon(lon2d[mask], weights=w)
    area = float(np.nansum(w))
    return lon, lat, n, area


def compute_rows_for_file(ar, lsm, lat_name, lon_name, lat_lsm, lon_lsm, boxes, args):
    lat_vals = ar[lat_name].values
    lon_vals = ar[lon_name].values
    dlat = float(np.nanmedian(np.abs(np.diff(lat_vals)))) if len(lat_vals) > 1 else 0.25
    dlon = float(np.nanmedian(np.abs(np.diff(lon_vals)))) if len(lon_vals) > 1 else 0.25
    lat_area = gridcell_area_km2(lat_vals, dlon, dlat)
    area2d = np.ones((len(lat_vals), len(lon_vals)), dtype=float) * lat_area[:, None]
    lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)

    # LSM on the full AR grid.
    lsm_i = prepare_lsm(lsm, ar.isel(time=0), lat_lsm, lon_lsm, lat_name, lon_name).load()
    land = (lsm_i.values >= args.land_threshold)

    # Precompute regional masks on full grid.
    region_masks = {region: box_bool_mask(lat_vals, lon_vals, box) & land for region, box in boxes.items()}

    arr = ar.values
    times = pd.to_datetime(ar["time"].values)
    rows = []
    for it, t in enumerate(times):
        if it == 0 or (it + 1) % 200 == 0 or (it + 1) == len(times):
            print(f"    time {it+1}/{len(times)}: {t}", flush=True)
        binary = np.asarray(arr[it, :, :] >= args.ar_threshold, dtype=bool)
        labels = label_ar_objects(binary)
        for region, reg_mask in region_masks.items():
            regional_ar = binary & reg_mask
            reg_lon, reg_lat, reg_n, reg_area = weighted_centroid(regional_ar, lat2d, lon2d, area2d)
            base = {
                "time_value": t,
                "date": t.floor("D"),
                "year": t.year,
                "month": t.month,
                "day": t.day,
                "hour": t.hour,
                "region": region,
            }
            if reg_n == 0:
                base.update({
                    "Tar_hour": 0,
                    "ar_mask_centroid_lon": np.nan,
                    "ar_mask_centroid_lat": np.nan,
                    "ar_mask_n_cells": 0,
                    "ar_mask_area_km2": 0.0,
                    "ar_region_centroid_lon": np.nan,
                    "ar_region_centroid_lat": np.nan,
                    "ar_region_n_cells": 0,
                    "ar_region_area_km2": 0.0,
                    "ar_object_label_t": np.nan,
                    "ar_object_centroid_lon": np.nan,
                    "ar_object_centroid_lat": np.nan,
                    "ar_object_n_cells": 0,
                    "ar_object_area_km2": 0.0,
                    "ar_object_regional_overlap_cells": 0,
                    "ar_object_regional_overlap_area_km2": 0.0,
                    "ar_object_regional_overlap_frac": np.nan,
                    "n_ar_object_labels_in_region": 0,
                })
                rows.append(base)
                continue

            labels_in_region = labels[regional_ar]
            labels_in_region = labels_in_region[labels_in_region != 0]
            uniq, counts = np.unique(labels_in_region, return_counts=True)
            if len(uniq) == 0:
                # Should not happen if binary/labels are consistent, but keep robust.
                base.update({
                    "Tar_hour": 1,
                    "ar_mask_centroid_lon": reg_lon,
                    "ar_mask_centroid_lat": reg_lat,
                    "ar_mask_n_cells": reg_n,
                    "ar_mask_area_km2": reg_area,
                    "ar_region_centroid_lon": reg_lon,
                    "ar_region_centroid_lat": reg_lat,
                    "ar_region_n_cells": reg_n,
                    "ar_region_area_km2": reg_area,
                    "ar_object_label_t": np.nan,
                    "ar_object_centroid_lon": np.nan,
                    "ar_object_centroid_lat": np.nan,
                    "ar_object_n_cells": 0,
                    "ar_object_area_km2": 0.0,
                    "ar_object_regional_overlap_cells": reg_n,
                    "ar_object_regional_overlap_area_km2": reg_area,
                    "ar_object_regional_overlap_frac": np.nan,
                    "n_ar_object_labels_in_region": 0,
                })
                rows.append(base)
                continue

            # Choose dominant object by regional overlap area, not just cells.
            overlap_areas = []
            for lab in uniq:
                overlap_areas.append(float(np.nansum(area2d[regional_ar & (labels == lab)])))
            overlap_areas = np.asarray(overlap_areas)
            j = int(np.nanargmax(overlap_areas))
            dom_label = int(uniq[j])
            dom_overlap_area = float(overlap_areas[j])
            dom_overlap_cells = int(counts[j])

            obj_mask = labels == dom_label
            obj_lon, obj_lat, obj_n, obj_area = weighted_centroid(obj_mask, lat2d, lon2d, area2d)
            base.update({
                "Tar_hour": 1,
                # Backward-compatible names now hold FULL OBJECT centroid, matching DART step centroid convention.
                "ar_mask_centroid_lon": obj_lon,
                "ar_mask_centroid_lat": obj_lat,
                "ar_mask_n_cells": obj_n,
                "ar_mask_area_km2": obj_area,
                # Explicit regional-footprint diagnostics retained separately.
                "ar_region_centroid_lon": reg_lon,
                "ar_region_centroid_lat": reg_lat,
                "ar_region_n_cells": reg_n,
                "ar_region_area_km2": reg_area,
                "ar_object_label_t": dom_label,
                "ar_object_centroid_lon": obj_lon,
                "ar_object_centroid_lat": obj_lat,
                "ar_object_n_cells": obj_n,
                "ar_object_area_km2": obj_area,
                "ar_object_regional_overlap_cells": dom_overlap_cells,
                "ar_object_regional_overlap_area_km2": dom_overlap_area,
                "ar_object_regional_overlap_frac": dom_overlap_area / obj_area if obj_area > 0 else np.nan,
                "n_ar_object_labels_in_region": int(len(uniq)),
            })
            rows.append(base)
    return rows


def process_file(mask_file, args, lsm, lat_lsm, lon_lsm, boxes, out_rows):
    print(f"[READ] {mask_file}", flush=True)
    ds = xr.open_dataset(mask_file, chunks={"time": args.chunks_time} if args.chunks_time else None)
    var = find_data_var(ds, preferred=args.mask_var, candidates=["ar_binary_tag", "ar_mask", "AR", "ar", "mask"])
    lat, lon = infer_lat_lon_names(ds)
    ar = ds[var]
    time_index = pd.to_datetime(ar["time"].values)
    keep = (time_index.year >= args.start_year) & (time_index.year <= args.end_year)
    if not keep.any():
        ds.close()
        return
    ar = ar.isel(time=np.where(keep)[0]).load()
    out_rows.extend(compute_rows_for_file(ar, lsm, lat, lon, lat_lsm, lon_lsm, boxes, args))
    ds.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bard-dir", default=BARD_DIR)
    p.add_argument("--mask-files", nargs="*", default=None)
    p.add_argument("--mask-file-pattern", default="*.nc4*")
    p.add_argument("--mask-var", required=True)
    p.add_argument("--lsm-file", default=LSM_FILE)
    p.add_argument("--regions", nargs="*", default=None)
    p.add_argument("--start-year", type=int, default=1970)
    p.add_argument("--end-year", type=int, default=2024)
    p.add_argument("--ar-threshold", type=float, default=0.5)
    p.add_argument("--land-threshold", type=float, default=0.5)
    p.add_argument("--out-file", required=True)
    p.add_argument("--chunks-time", type=int, default=None)
    args = p.parse_args()

    boxes = REGION_BOXES.copy()
    if args.regions:
        boxes = {r: boxes[r] for r in args.regions}
    files = [Path(f) for f in args.mask_files] if args.mask_files else discover_files(args.bard_dir, args.mask_file_pattern, args.start_year, args.end_year)
    print(f"[INFO] mask files: {len(files)}", flush=True)
    print(f"[INFO] first: {files[0]}", flush=True)
    print(f"[INFO] last : {files[-1]}", flush=True)

    ds_lsm = xr.open_dataset(args.lsm_file)
    lsm_var = find_data_var(ds_lsm, preferred="lsm", candidates=["lsm", "LSM", "var_172", "VAR_172"])
    lat_lsm, lon_lsm = infer_lat_lon_names(ds_lsm)
    lsm = ds_lsm[lsm_var]
    if "time" in lsm.dims:
        lsm = lsm.isel(time=0)

    rows = []
    for f in files:
        process_file(f, args, lsm, lat_lsm, lon_lsm, boxes, rows)
        tmp = str(args.out_file).replace(".csv.gz", "_partial.csv.gz")
        Path(tmp).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(tmp, index=False, compression="gzip")
        print(f"[PARTIAL SAVE] {tmp} rows={len(rows)}", flush=True)

    out = pd.DataFrame(rows)
    out["time_value"] = pd.to_datetime(out["time_value"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_file, index=False, compression="gzip")
    print(f"[SAVE] {args.out_file} rows={len(out)}", flush=True)
    if len(out):
        print(out.loc[out["Tar_hour"].astype(int) == 1].groupby("region").size().to_string(), flush=True)
        print("[INFO] overlapping-object count per Tar hour:", flush=True)
        print(out.loc[out["Tar_hour"].astype(int) == 1, "n_ar_object_labels_in_region"].value_counts(dropna=False).sort_index().to_string(), flush=True)


if __name__ == "__main__":
    main()
