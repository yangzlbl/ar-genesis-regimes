#!/usr/bin/env python3
"""Validate frozen GMM labels used for manuscript analysis.

This script does not refit PCA or GMM. It loads the frozen label table, checks
that required columns are present, compares regime counts against the expected
manuscript counts in ``config/regimes.json``, and prints display-oriented PC
centroids for Fig. 1 consistency checks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = ["PC1", "PC2", "PC3", "gmm_regime_named", "gmm_max_prob"]


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if path.name.endswith(".csv.gz") or path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path, help="Frozen GMM label table (.pkl, .csv, .csv.gz).")
    parser.add_argument("--config", required=True, type=Path, help="config/regimes.json")
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())
    order = cfg["regime_order"]
    expected = pd.Series(cfg.get("expected_final_counts", {}), dtype="int64").reindex(order)

    df = read_table(args.labels)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in frozen label table: {missing}")

    counts = df["gmm_regime_named"].value_counts().reindex(order).astype("int64")
    print("Regime counts:")
    print(counts)

    if expected.notna().all():
        print("\nExpected counts:")
        print(expected)
        if counts.equals(expected):
            print("\nOK: counts match expected manuscript values.")
        else:
            diff = counts - expected
            print("\nWARNING: counts differ from expected manuscript values:")
            print(diff)

    cent_raw = df.groupby("gmm_regime_named")[["PC1", "PC2", "PC3"]].median().reindex(order)

    # Display convention only: orient PC1 so positive values represent stronger
    # deep ascent in Fig. 1-style centroid summaries. Raw PC values are unchanged.
    cent_display = cent_raw.copy()
    cent_display["PC1"] = -cent_display["PC1"]

    cent_display_std = (cent_display - cent_display.mean(axis=0)) / cent_display.std(axis=0, ddof=0)

    print("\nRaw PC centroids:")
    print(cent_raw.round(3))
    print("\nDisplay-oriented PC centroids, standardized across regimes:")
    print(cent_display_std.round(3))

    print("\nAssignment-confidence summary:")
    print(df.groupby("gmm_regime_named")["gmm_max_prob"].describe().reindex(order).round(3))


if __name__ == "__main__":
    main()
