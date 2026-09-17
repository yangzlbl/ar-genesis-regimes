#!/usr/bin/env python3
"""
Build region-day precipitation time series from hourly ERA5 precipitation.

Design:
  - Uses one common REGION_BOXES definition for both precipitation impact and
    AR landfall attribution in the reset workflow.
  - Computes daily accumulated gridded precipitation first, then region metrics.
  - Does not use AR masks or track IDs.

Outputs one row per region x day.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import xarray as xr

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

TP_DIRS = [
    "/pscratch/sd/y/yangzhou/ARgenesis/e5.accumulated_tp_1h_1970_1978",
    "/pscratch/sd/y/yangzhou/ARgenesis/e5.accumulated_tp_1h_2023_2024",
    "/global/cfs/projectdirs/m3522/cmip6/ERA5/e5.accumulated_tp_1h",
]

LSM_FILE = (
    "/global/cfs/projectdirs/m3522/cmip6/ERA5/e5.oper.invariant/197901/"
    "e5.oper.invariant.128_172_lsm.ll025sc.1979010100_1979010100.nc"
)


def find_data_var(ds, preferred=None, candidates=None):
    data_vars = list(ds.data_vars)
    candidates = candidates or []
    if preferred is not None:
        if preferred in ds.variables:
            return preferred
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


def normalize_units(units):
    return str(units or "").lower().replace(" ", "").replace("_", "")


def precip_to_mm_per_hour(da, user_units=None):
    units = user_units if user_units is not None else da.attrs.get("units", "")
    u = normalize_units(units)
    print(f"[INFO] precipitation units detected/used: {units!r}", flush=True)
    if u in ["m", "meter", "meters"]:
        print("[INFO] converting accumulated depth m to mm: x1000", flush=True)
        return da * 1000.0
    if u in ["mm", "millimeter", "millimeters"]:
        return da
    if u in ["kgm-2s-1", "kgm**-2s**-1", "kg/m2/s", "kgm^-2s^-1", "kgm-2sec-1"]:
        print("[INFO] converting rate kg m-2 s-1 to mm h-1: x3600", flush=True)
        return da * 3600.0
    if u in ["ms-1", "m/s", "msec-1", "msecond-1"]:
        print("[INFO] converting rate m s-1 to mm h-1: x1000*x3600", flush=True)
        return da * 1000.0 * 3600.0
    raise ValueError(
        f"Unknown precip units {units!r}. Pass --precip-units as 'm', 'mm', 'kg m-2 s-1', or 'm s-1'."
    )


def resolve_tp_file(year, month, tp_dirs):
    fname = f"e5.accumulated_tp_1h.{year:04d}{month:02d}.nc"
    tried = []
    for d in tp_dirs:
        p = Path(d) / fname
        tried.append(str(p))
        if p.exists() and p.stat().st_size > 0:
            return p
    raise FileNotFoundError("Could not find TP file. Tried:\n" + "\n".join(tried))


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


def gridcell_area_km2(lat, dlon_deg, dlat_deg):
    r = 6371.0
    return (r ** 2) * np.deg2rad(dlat_deg) * np.deg2rad(dlon_deg) * np.cos(np.deg2rad(lat))


def prepare_lsm(lsm, tp_sub, lat_lsm, lon_lsm, lat_tp, lon_tp):
    lsm_i = lsm.interp({lat_lsm: tp_sub[lat_tp], lon_lsm: tp_sub[lon_tp]}, method="nearest")
    ren = {}
    if lat_lsm != lat_tp:
        ren[lat_lsm] = lat_tp
    if lon_lsm != lon_tp:
        ren[lon_lsm] = lon_tp
    if ren:
        lsm_i = lsm_i.rename(ren)
    return lsm_i


def compute_metrics(tp_day_mm, lsm_sub, lat_name, lon_name,
                    wet_threshold=2.0, heavy_threshold=20.0, extreme_threshold=50.0):
    lat_vals = tp_day_mm[lat_name].values
    lon_vals = tp_day_mm[lon_name].values
    dlat = float(np.nanmedian(np.abs(np.diff(lat_vals)))) if len(lat_vals) > 1 else 0.25
    dlon = float(np.nanmedian(np.abs(np.diff(lon_vals)))) if len(lon_vals) > 1 else 0.25
    lat_area = xr.DataArray(gridcell_area_km2(tp_day_mm[lat_name], dlon, dlat),
                            dims=(lat_name,), coords={lat_name: tp_day_mm[lat_name]})
    area_2d = xr.ones_like(tp_day_mm) * lat_area
    land_weight = area_2d * lsm_sub
    all_weight = area_2d
    land_area = float(land_weight.sum(skipna=True).item())
    total_area = float(all_weight.sum(skipna=True).item())
    if not np.isfinite(land_area) or land_area <= 0:
        return None
    mean_land = float((tp_day_mm * land_weight).sum(skipna=True).item() / land_area)
    mean_all = float((tp_day_mm * all_weight).sum(skipna=True).item() / total_area)
    volume_land = float((tp_day_mm * land_weight).sum(skipna=True).item() * 1e-6)
    tp_land = tp_day_mm.where(lsm_sub >= 0.5)
    vals = tp_land.values
    out = {
        "precip_mean_land_mm_day": mean_land,
        "precip_mean_all_mm_day": mean_all,
        "precip_sum_land_mm_grid": float(tp_land.sum(skipna=True).item()),
        "precip_volume_land_km3_day": volume_land,
        "precip_max_land_mm_day": float(tp_land.max(skipna=True).item()),
        "precip_p95_land_mm_day": float(np.nanpercentile(vals, 95)),
        "precip_p99_land_mm_day": float(np.nanpercentile(vals, 99)),
        "land_area_km2": land_area,
        "total_area_km2": total_area,
        "land_fraction_area": land_area / total_area,
    }
    for name, thr in [("wet", wet_threshold), ("heavy", heavy_threshold), ("extreme", extreme_threshold)]:
        a = float(land_weight.where(tp_day_mm >= thr, 0.0).sum(skipna=True).item())
        out[f"{name}_area_frac_land"] = a / land_area
    return out


def process_month(year, month, args, lsm, lat_lsm, lon_lsm, rows, boxes):
    tp_file = resolve_tp_file(year, month, args.tp_dirs)
    print(f"[MONTH] {year:04d}{month:02d} {tp_file}", flush=True)
    ds = xr.open_dataset(tp_file, chunks={"time": args.chunks_time} if args.chunks_time else None)
    var = find_data_var(ds, preferred=args.precip_var,
                        candidates=["tp", "TP", "total_precipitation", "precip", "precipitation", "pr", "mtpr", "var228", "VAR_228"])
    lat, lon = infer_lat_lon_names(ds)
    pr_mm_h = precip_to_mm_per_hour(ds[var], args.precip_units)
    pr_daily = pr_mm_h.resample(time="1D").sum()
    for date_value in pr_daily["time"].values:
        date = pd.Timestamp(date_value)
        if date.year != year:
            continue
        pr_day = pr_daily.sel(time=date_value)
        for region, box in boxes.items():
            tp_sub = subset_box_da(pr_day, lat, lon, box)
            lsm_box = subset_box_da(lsm, lat_lsm, lon_lsm, box)
            lsm_sub = prepare_lsm(lsm_box, tp_sub, lat_lsm, lon_lsm, lat, lon)
            met = compute_metrics(tp_sub, lsm_sub, lat, lon,
                                  wet_threshold=args.wet_threshold_mm_day,
                                  heavy_threshold=args.heavy_threshold_mm_day,
                                  extreme_threshold=args.extreme_threshold_mm_day)
            if met is None:
                continue
            row = {"date": date.strftime("%Y-%m-%d"), "year": date.year,
                   "month": date.month, "day": date.day, "region": region}
            row.update(met)
            rows.append(row)
    ds.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start-year", type=int, default=1970)
    p.add_argument("--end-year", type=int, default=2024)
    p.add_argument("--regions", nargs="*", default=None)
    p.add_argument("--tp-dirs", nargs="+", default=TP_DIRS)
    p.add_argument("--precip-var", default="tp")
    p.add_argument("--precip-units", default=None)
    p.add_argument("--lsm-file", default=LSM_FILE)
    p.add_argument("--out-file", required=True)
    p.add_argument("--wet-threshold-mm-day", type=float, default=2.0)
    p.add_argument("--heavy-threshold-mm-day", type=float, default=20.0)
    p.add_argument("--extreme-threshold-mm-day", type=float, default=50.0)
    p.add_argument("--chunks-time", type=int, default=None)
    args = p.parse_args()

    boxes = REGION_BOXES.copy()
    if args.regions:
        boxes = {r: boxes[r] for r in args.regions}

    ds_lsm = xr.open_dataset(args.lsm_file)
    lsm_var = find_data_var(ds_lsm, preferred="lsm", candidates=["lsm", "LSM", "var_172", "VAR_172"])
    lat_lsm, lon_lsm = infer_lat_lon_names(ds_lsm)
    lsm = ds_lsm[lsm_var]
    if "time" in lsm.dims:
        lsm = lsm.isel(time=0)

    rows = []
    for year in range(args.start_year, args.end_year + 1):
        for month in range(1, 13):
            process_month(year, month, args, lsm, lat_lsm, lon_lsm, rows, boxes)
            tmp = str(args.out_file).replace(".csv.gz", "_partial.csv.gz")
            Path(tmp).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(tmp, index=False, compression="gzip")
            print(f"[PARTIAL SAVE] {tmp} rows={len(rows)}", flush=True)

    ds_lsm.close()
    out = pd.DataFrame(rows)
    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_file, index=False, compression="gzip")
    print(f"[SAVE] {args.out_file} rows={len(out)}", flush=True)
    print(out.groupby("region").size().to_string(), flush=True)

if __name__ == "__main__":
    main()
