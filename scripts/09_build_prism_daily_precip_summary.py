#!/usr/bin/env python3
"""
Build domain-daily precipitation summary time series from PRISM daily PPT files.

This is the PRISM analogue of build_domain_daily_precip_timeseries_v2.py.
Key differences from ERA5 version:
  - Input is yearly PRISM_PPT_YYYY.nc files with daily PPT already in mm/day.
  - PRISM is land-only / CONUS-focused, so missing pixels are masked by _FillValue/NaN.
  - No ERA5 land-sea mask is used. Area weighting is applied over finite PRISM pixels.

Example:
python build_domain_daily_prism_precip_summary.py \
  --start-year 1981 --end-year 2024 \
  --prism-dir /global/cfs/projectdirs/m3522/cmip6/PRISM \
  --regions "U.S. West Coast" \
  --out-file /pscratch/sd/y/yangzhou/ARgenesis/cutouts_3000km_monthly/gmm_analysis_sh_corrected/prism_daily_precip_summary_impact_1981_2024.csv.gz
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import xarray as xr


# Keep the same impact boxes as the ERA5 impact analysis where possible.
# PRISM is CONUS-focused; most non-US domains will simply have no valid pixels.
IMPACT_BOXES = {
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

COASTAL_BOXES = {
    "U.S. West Coast":        (-145, -117,  28,  52),
    "Alaska / NE Pacific":    (-170, -125,  50,  72),
    "Western Europe":         ( -25,   10,  35,  62),
    "East Asia":              ( 120,  155,  22,  50),
    "Mediterranean":          ( -10,   38,  30,  46),
    "Greenland / Iceland":    ( -60,  -10,  60,  85),
    "Chile":                  ( -90,  -68, -58, -20),
    "SE South America":       ( -65,  -35, -45, -22),
    "SE Australia / NZ":      ( 140,  180, -50, -25),
    "South Africa":           (  10,   40, -38, -20),
    "Antarctic":              (-180,  180, -90, -66),
}


def find_var(ds, candidates):
    for c in candidates:
        if c in ds.variables:
            return c
    lower_map = {v.lower(): v for v in ds.variables}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    data_vars = list(ds.data_vars)
    if len(data_vars) == 1:
        return data_vars[0]
    raise ValueError(f"Could not identify precipitation variable. data_vars={data_vars}")


def infer_lat_lon_names(ds):
    lat_candidates = ["lat", "latitude", "LAT", "Latitude", "y"]
    lon_candidates = ["lon", "longitude", "LON", "Longitude", "x"]
    lat_name = next((c for c in lat_candidates if c in ds.coords or c in ds.dims), None)
    lon_name = next((c for c in lon_candidates if c in ds.coords or c in ds.dims), None)
    if lat_name is None or lon_name is None:
        raise ValueError(f"Could not infer lat/lon names from: {list(ds.variables)}")
    return lat_name, lon_name


def subset_box_da(da, lat_name, lon_name, box):
    """Subset DataArray using a box whose longitudes are in [-180, 180]."""
    lon_min, lon_max, lat_min, lat_max = box

    lat_vals = da[lat_name].values
    lon_vals = da[lon_name].values

    # Latitude can be ascending or descending.
    if lat_vals[0] < lat_vals[-1]:
        out = da.sel({lat_name: slice(lat_min, lat_max)})
    else:
        out = da.sel({lat_name: slice(lat_max, lat_min)})

    full_lon = (lon_max - lon_min) >= 359.999
    if full_lon:
        return out

    ds_lon_min = float(np.nanmin(lon_vals))
    ds_lon_max = float(np.nanmax(lon_vals))

    # Convert requested lon range to dataset convention if needed.
    if ds_lon_min >= 0 and ds_lon_max > 180:
        lon_min_data = lon_min % 360
        lon_max_data = lon_max % 360
        if lon_min_data <= lon_max_data:
            out = out.sel({lon_name: slice(lon_min_data, lon_max_data)})
        else:
            out1 = out.sel({lon_name: slice(lon_min_data, ds_lon_max)})
            out2 = out.sel({lon_name: slice(ds_lon_min, lon_max_data)})
            out = xr.concat([out1, out2], dim=lon_name)
    else:
        if lon_min <= lon_max:
            out = out.sel({lon_name: slice(lon_min, lon_max)})
        else:
            out1 = out.sel({lon_name: slice(lon_min, ds_lon_max)})
            out2 = out.sel({lon_name: slice(ds_lon_min, lon_max)})
            out = xr.concat([out1, out2], dim=lon_name)

    return out


def gridcell_area_km2(lat, lon_resolution_deg, lat_resolution_deg):
    r_km = 6371.0
    dlat = np.deg2rad(lat_resolution_deg)
    dlon = np.deg2rad(lon_resolution_deg)
    lat_rad = np.deg2rad(lat)
    return (r_km ** 2) * dlat * dlon * np.cos(lat_rad)


def compute_domain_daily_metrics(ppt_day_mm, lat_name, lon_name,
                                 wet_threshold=1.0,
                                 heavy_threshold=20.0,
                                 extreme_threshold=50.0,
                                 min_valid_pixels=10):
    """Compute PRISM daily metrics over finite land pixels inside a domain."""
    if ppt_day_mm.size == 0:
        return None

    # xarray should decode _FillValue to NaN, but make this explicit.
    ppt = ppt_day_mm.where(np.isfinite(ppt_day_mm))
    valid = np.isfinite(ppt)
    n_valid = int(valid.sum(skipna=True).item())
    if n_valid < min_valid_pixels:
        return None

    lat_vals = ppt[lat_name].values
    lon_vals = ppt[lon_name].values
    dlat = float(np.nanmedian(np.abs(np.diff(lat_vals)))) if len(lat_vals) > 1 else 0.0416667
    dlon = float(np.nanmedian(np.abs(np.diff(lon_vals)))) if len(lon_vals) > 1 else 0.0416667

    lat_area = xr.DataArray(
        gridcell_area_km2(ppt[lat_name], dlon, dlat),
        dims=(lat_name,),
        coords={lat_name: ppt[lat_name]},
    )
    area_2d = xr.ones_like(ppt) * lat_area
    land_weight = area_2d.where(valid, 0.0)

    land_area_km2 = land_weight.sum(skipna=True).item()
    if not np.isfinite(land_area_km2) or land_area_km2 <= 0:
        return None

    weighted_sum = (ppt.fillna(0.0) * land_weight).sum(skipna=True).item()
    mean_land = weighted_sum / land_area_km2
    volume_land_km3 = weighted_sum * 1e-6  # mm * km2 -> km3

    vals = ppt.values[np.isfinite(ppt.values)]
    max_land = float(np.nanmax(vals))
    p95_land = float(np.nanpercentile(vals, 95))
    p99_land = float(np.nanpercentile(vals, 99))

    wet_area = land_weight.where(ppt >= wet_threshold, 0.0).sum(skipna=True).item()
    heavy_area = land_weight.where(ppt >= heavy_threshold, 0.0).sum(skipna=True).item()
    extreme_area = land_weight.where(ppt >= extreme_threshold, 0.0).sum(skipna=True).item()

    return {
        "precip_mean_land_mm_day": mean_land,
        "precip_sum_land_mm_grid": float(np.nansum(vals)),
        "precip_volume_land_km3_day": volume_land_km3,
        "precip_max_land_mm_day": max_land,
        "precip_p95_land_mm_day": p95_land,
        "precip_p99_land_mm_day": p99_land,
        "wet_area_frac_land": wet_area / land_area_km2,
        "heavy_area_frac_land": heavy_area / land_area_km2,
        "extreme_area_frac_land": extreme_area / land_area_km2,
        "land_area_km2": land_area_km2,
        "total_area_km2": land_area_km2,
        "land_fraction_area": 1.0,
        "n_valid_prism_pixels": n_valid,
    }


def resolve_prism_file(prism_dir, year):
    p = Path(prism_dir) / f"PRISM_PPT_{year}.nc"
    if not p.exists() or p.stat().st_size == 0:
        raise FileNotFoundError(f"Missing PRISM file: {p}")
    return p


def process_one_year(year, prism_dir, domain_boxes, out_rows, args):
    fn = resolve_prism_file(prism_dir, year)
    print(f"[YEAR] {year} | {fn}", flush=True)

    ds = xr.open_dataset(fn, decode_times=True, mask_and_scale=True)
    ppt_name = find_var(ds, ["PPT", "ppt", "precip", "pr", "precipitation"])
    lat_name, lon_name = infer_lat_lon_names(ds)

    ppt = ds[ppt_name]
    # PRISM header says units = mm. Keep a safety hook for unusual files.
    units = str(ppt.attrs.get("units", "")).lower()
    if units in ["m", "meter", "meters"]:
        ppt = ppt * 1000.0

    if "time" not in ppt.dims:
        raise ValueError(f"Expected time dimension in {fn}; dims={ppt.dims}")

    for date_value in ppt["time"].values:
        date = pd.Timestamp(date_value)
        ppt_day = ppt.sel(time=date_value)

        for region, box in domain_boxes.items():
            ppt_sub = subset_box_da(ppt_day, lat_name, lon_name, box)
            metrics = compute_domain_daily_metrics(
                ppt_sub,
                lat_name=lat_name,
                lon_name=lon_name,
                wet_threshold=args.wet_threshold,
                heavy_threshold=args.heavy_threshold,
                extreme_threshold=args.extreme_threshold,
                min_valid_pixels=args.min_valid_pixels,
            )
            if metrics is None:
                continue

            row = {
                "date": date.strftime("%Y-%m-%d"),
                "year": int(date.year),
                "month": int(date.month),
                "day": int(date.day),
                "region": region,
                "precip_source": "PRISM",
            }
            row.update(metrics)
            out_rows.append(row)

    ds.close()


def main():
    parser = argparse.ArgumentParser(
        description="Build domain-daily precipitation summaries from PRISM daily PPT files."
    )
    parser.add_argument("--prism-dir", default="/global/cfs/projectdirs/m3522/cmip6/PRISM")
    parser.add_argument("--start-year", type=int, default=1981)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--box-set", choices=["impact", "coastal"], default="impact")
    parser.add_argument(
        "--regions",
        nargs="*",
        default=["U.S. West Coast"],
        help="Regions to compute. Default only U.S. West Coast because PRISM is CONUS-focused.",
    )
    parser.add_argument("--out-file", default=None)
    parser.add_argument("--wet-threshold", type=float, default=1.0)
    parser.add_argument("--heavy-threshold", type=float, default=20.0)
    parser.add_argument("--extreme-threshold", type=float, default=50.0)
    parser.add_argument("--min-valid-pixels", type=int, default=10)
    args = parser.parse_args()

    all_boxes = IMPACT_BOXES if args.box_set == "impact" else COASTAL_BOXES
    if args.regions == ["all"]:
        domain_boxes = all_boxes
    else:
        missing = [r for r in args.regions if r not in all_boxes]
        if missing:
            raise KeyError(f"Unknown regions: {missing}. Available: {list(all_boxes)}")
        domain_boxes = {r: all_boxes[r] for r in args.regions}

    if args.out_file is None:
        region_tag = "all" if args.regions == ["all"] else "_".join(
            r.replace("/", "").replace(" ", "_").replace("-", "_") for r in domain_boxes
        )
        out_file = Path(
            "/pscratch/sd/y/yangzhou/ARgenesis/cutouts_3000km_monthly/"
            "gmm_analysis_sh_corrected/"
            f"prism_daily_precip_summary_{args.box_set}_{region_tag}_{args.start_year}_{args.end_year}.csv.gz"
        )
    else:
        out_file = Path(args.out_file)

    print(f"[CONFIG] PRISM dir = {args.prism_dir}")
    print(f"[CONFIG] years = {args.start_year}-{args.end_year}")
    print(f"[CONFIG] box_set = {args.box_set}")
    print(f"[CONFIG] output = {out_file}")
    print("[CONFIG] regions:")
    for region, box in domain_boxes.items():
        print(f"  {region:25s} lon=[{box[0]}, {box[1]}] lat=[{box[2]}, {box[3]}]")

    out_rows = []
    for year in range(args.start_year, args.end_year + 1):
        process_one_year(year, args.prism_dir, domain_boxes, out_rows, args)
        if out_rows:
            tmp = Path(str(out_file).replace(".csv.gz", "_partial.csv.gz"))
            tmp.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(out_rows).to_csv(tmp, index=False, compression="gzip")
            print(f"[PARTIAL SAVE] {tmp} rows={len(out_rows)}", flush=True)

    out_df = pd.DataFrame(out_rows)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_file, index=False, compression="gzip")

    print(f"\n[SAVE] {out_file}")
    print(f"  rows: {len(out_df)}")
    if not out_df.empty:
        print(f"  regions: {out_df['region'].nunique()}")
        print(f"  date range: {out_df['date'].min()} to {out_df['date'].max()}")
        print("\n  Rows per region:")
        print(out_df.groupby("region").size().to_string())
        print("\n  Land area per region (km2):")
        print(out_df.groupby("region")["land_area_km2"].first().to_string())


if __name__ == "__main__":
    main()
