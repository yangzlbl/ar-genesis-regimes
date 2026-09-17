#!/usr/bin/env python3
"""
Classify atmospheric-river genesis environments using PCA and Gaussian mixture models.

This script reproduces the genesis-regime classification used in the manuscript.
It applies the Southern Hemisphere sign correction for cyclonic variables, imputes
rare missing feature values, standardizes the feature matrix, projects events onto
the first three principal components, and fits a K-component Gaussian mixture model.
Optional K-means diagnostics are included for comparison only; the manuscript uses
GMM labels.

Recommended manuscript workflow:
  1. Use this script to document how the frozen labels were generated.
  2. For figure regeneration, load the archived/frozen label table rather than
     rerunning GMM, unless intentionally regenerating the classification.

Example:
  python scripts/01_classify_genesis_regimes_pca_gmm.py \
    --features /path/to/ar_genesis_event_features.pkl \
    --out-dir /path/to/gmm_analysis_sh_corrected \
    --k 4 --random-state 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

PCA_FEATURES: List[str] = [
    "pv300_center_mean",
    "msl_center_anom",
    "msl_grad_mean",
    "t850_grad_mean",
    "shear_250_850_mean",
    "d250_center_mean",
    "d850_center_mean",
    "w500_center_mean",
    "vo850_mean",
]

REGIME_ORDER = ["High-moisture", "Coupled-cyclonic", "Frontal", "Ridge"]

# Mapping for the final frozen solution used in the manuscript. Component numbers
# are not physically meaningful and may change if the GMM is refit.
DEFAULT_COMPONENT_NAME_MAP: Dict[int, str] = {
    0: "Frontal",
    1: "Ridge",
    2: "Coupled-cyclonic",
    3: "High-moisture",
}


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if path.suffix.lower() == ".gz" and path.name.endswith(".csv.gz"):
        return pd.read_csv(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path}")


def infer_hemisphere(df: pd.DataFrame) -> pd.Series:
    if "hemisphere" in df.columns:
        return df["hemisphere"].astype(str).str.upper()
    if "event_genesis_lat" not in df.columns:
        raise ValueError("Need either 'hemisphere' or 'event_genesis_lat' to apply SH sign correction.")
    return np.where(df["event_genesis_lat"].to_numpy() < 0, "SH", "NH")


def parse_component_map(path: Path | None) -> Dict[int, str]:
    if path is None:
        return DEFAULT_COMPONENT_NAME_MAP
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if "gmm_component_name_map_for_final_solution" in raw:
        raw = raw["gmm_component_name_map_for_final_solution"]
    return {int(k): str(v) for k, v in raw.items()}


def apply_sh_sign_correction(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    hemi = infer_hemisphere(out)
    sh = np.asarray(hemi == "SH")
    for col in ["vo850_mean", "pv300_center_mean"]:
        if col not in out.columns:
            raise ValueError(f"Missing column needed for SH sign correction: {col}")
        out.loc[sh, col] = -out.loc[sh, col]
    out["hemisphere_for_sign"] = hemi
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True, type=Path, help="Genesis-event feature table (.pkl, .csv, or .csv.gz).")
    p.add_argument("--out-dir", required=True, type=Path, help="Output directory.")
    p.add_argument("--k", type=int, default=4, help="Number of GMM components for final labels.")
    p.add_argument("--k-min", type=int, default=2, help="Minimum K for diagnostic table.")
    p.add_argument("--k-max", type=int, default=8, help="Maximum K for diagnostic table.")
    p.add_argument("--random-state", type=int, default=42, help="Random seed for GMM/K-means.")
    p.add_argument("--n-init", type=int, default=50, help="Number of GMM initializations.")
    p.add_argument("--component-map", type=Path, default=None, help="Optional JSON component-to-regime map.")
    p.add_argument("--no-sh-correction", action="store_true", help="Disable Southern Hemisphere cyclonic sign correction.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = read_table(args.features)
    missing_features = [c for c in PCA_FEATURES if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing PCA feature columns: {missing_features}")

    if args.no_sh_correction:
        df_work = df.copy()
    else:
        df_work = apply_sh_sign_correction(df)

    X_raw = df_work[PCA_FEATURES].copy()

    missing_summary = pd.DataFrame({
        "feature": PCA_FEATURES,
        "n_missing": X_raw.isna().sum().reindex(PCA_FEATURES).to_numpy(),
        "fraction_missing": X_raw.isna().mean().reindex(PCA_FEATURES).to_numpy(),
    })
    missing_summary.to_csv(args.out_dir / "missing_value_summary.csv", index=False)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imp = imputer.fit_transform(X_raw)
    X_std = scaler.fit_transform(X_imp)

    pca = PCA(n_components=3, random_state=args.random_state)
    pcs = pca.fit_transform(X_std)

    df_out = df_work.copy()
    df_out["PC1"] = pcs[:, 0]
    df_out["PC2"] = pcs[:, 1]
    df_out["PC3"] = pcs[:, 2]

    diagnostics = []
    for k in range(args.k_min, args.k_max + 1):
        g = GaussianMixture(
            n_components=k,
            covariance_type="full",
            n_init=args.n_init,
            random_state=args.random_state,
        )
        labels = g.fit_predict(pcs)
        diagnostics.append({
            "method": "GMM",
            "K": k,
            "AIC": g.aic(pcs),
            "BIC": g.bic(pcs),
            "silhouette": silhouette_score(pcs, labels) if len(np.unique(labels)) > 1 else np.nan,
        })

        km = KMeans(n_clusters=k, n_init=50, random_state=args.random_state)
        km_labels = km.fit_predict(pcs)
        diagnostics.append({
            "method": "KMeans",
            "K": k,
            "AIC": np.nan,
            "BIC": np.nan,
            "silhouette": silhouette_score(pcs, km_labels) if len(np.unique(km_labels)) > 1 else np.nan,
        })

    pd.DataFrame(diagnostics).to_csv(args.out_dir / "pca_gmm_kmeans_model_selection.csv", index=False)

    gmm = GaussianMixture(
        n_components=args.k,
        covariance_type="full",
        n_init=args.n_init,
        random_state=args.random_state,
    )
    comp = gmm.fit_predict(pcs)
    prob = gmm.predict_proba(pcs)

    df_out["gmm_component_raw"] = comp
    df_out["gmm_max_prob"] = prob.max(axis=1)
    df_out["gmm_regime_named"] = pd.Series(comp).map(parse_component_map(args.component_map)).to_numpy()

    if df_out["gmm_regime_named"].isna().any():
        raise ValueError("Component map did not assign all GMM components to regime names.")

    labels_csv = args.out_dir / f"ar_genesis_gmm_labels_K{args.k}_sh_corrected.csv"
    labels_pkl = args.out_dir / f"ar_genesis_gmm_labels_K{args.k}_sh_corrected.pkl"
    df_out.to_csv(labels_csv, index=False)
    df_out.to_pickle(labels_pkl)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=PCA_FEATURES,
        columns=["PC1", "PC2", "PC3"],
    )
    loadings.to_csv(args.out_dir / "pca_loadings_raw.csv")

    evr = pd.DataFrame({
        "PC": ["PC1", "PC2", "PC3"],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
    })
    evr.to_csv(args.out_dir / "pca_explained_variance.csv", index=False)

    counts = df_out["gmm_regime_named"].value_counts().reindex(REGIME_ORDER)
    print("Saved:")
    print(labels_csv)
    print(labels_pkl)
    print("\nRegime counts:")
    print(counts)
    print("\nExplained variance:")
    print(evr)


if __name__ == "__main__":
    main()
