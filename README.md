# AR genesis regime analysis code

Analysis code for the manuscript:

**Distinct atmospheric-river genesis environments shape storm evolution and landfall impacts**

This repository contains the cleaned analysis scripts used to classify atmospheric-river (AR) genesis environments, assign final genesis-regime labels, compute lifecycle diagnostics, attribute regional landfall precipitation impacts, and perform the PRISM precipitation-product sensitivity analysis.

The release intentionally focuses on the reproducible **analysis workflow**. It does not include notebook-specific figure-polishing or panel-layout code. Final manuscript figures can be generated from the analysis outputs using local plotting notebooks/scripts.

## Recommended publication workflow

Use GitHub as the working repository and Zenodo as the permanent archive.

1. Create a GitHub repository, e.g. `ar-genesis-regimes`.
2. Commit the cleaned analysis scripts, configuration file, README, license, and citation metadata.
3. Create a versioned GitHub release, e.g. `v1.0.0`.
4. Enable the repository in Zenodo and archive the GitHub release to mint a DOI.
5. Cite the Zenodo DOI in the manuscript Code availability section.

## Repository contents

```text
config/
  regimes.json
      Regime order, colors, PCA features, SH sign-correction variables, expected
      frozen-label counts, and regional landfall-impact boxes.

scripts/
  01_classify_genesis_regimes_pca_gmm.py
      Applies Southern Hemisphere sign correction, median-imputes rare missing
      feature values, standardizes genesis-environment features, performs PCA,
      evaluates GMM and K-means K-selection diagnostics, and generates GMM regime
      labels. The manuscript uses the final GMM labels, not K-means labels.

  02_validate_frozen_gmm_labels.py
      Loads the frozen manuscript label table, verifies regime counts and required
      columns, and prints display-oriented PCA centroids for checking consistency
      with Fig. 1.

  03_build_lifecycle_diagnostics.py
      Builds regime-specific lifecycle summaries for IVT, TCWV, precipitation,
      and precipitation efficiency from a final regime-labeled lifecycle table.

  04_build_region_daily_precip_timeseries.py
      Builds regional daily ERA5 land precipitation summaries.

  05_build_region_hourly_ar_landfall_object_centroids.py
      Identifies hourly regional AR landfall objects from the TECA-BARD binary AR mask.

  06_match_region_hourly_ar_to_regime.py
      Links hourly regional landfall objects to raw DART track steps and final
      SH-corrected genesis-regime labels.

  07_merge_daily_precip_ar_regime.py
      Merges regional daily precipitation metrics with daily AR/regime attribution
      and computes precipitation-enhancement summaries.

  08_concatenate_precip_regime_years.py
      Concatenates annual landfall-regime outputs across the analysis period.

  09_build_prism_daily_precip_summary.py
      Builds regional daily PRISM precipitation summaries for the U.S. West Coast
      precipitation-product sensitivity analysis.

  10_merge_prism_precip_with_ar_regime.py
      Merges PRISM precipitation with the same daily AR/regime attribution used
      for ERA5 and produces PRISM sensitivity summaries.

  run_landfall_pipeline_one_year.py
      Example driver for the modular annual landfall/regime workflow.
```

## Data requirements

The scripts assume local access to the following data products:

- ERA5 atmospheric fields and hourly precipitation.
- TECA-BARD hourly binary AR mask files.
- DART AR lifecycle track and step tables.
- PRISM daily precipitation for the U.S. West Coast sensitivity analysis.
- A generated AR genesis-event feature table containing the PCA/GMM variables listed in `config/regimes.json`.
- A final regime-labeled lifecycle table for lifecycle and landfall-regime linkage.

Large raw data products and intermediate analysis outputs are not included in this code release. File paths should be supplied through command-line arguments or edited in the example driver script for the local compute environment.

## Frozen manuscript labels

For manuscript figure regeneration, use the frozen final Southern-Hemisphere-corrected GMM label table rather than rerunning the stochastic GMM fit. Rerunning PCA/GMM may change component numbering or slightly alter event-level labels depending on software versions, random seeds, and preprocessing details.

Expected final manuscript counts:

```text
High-moisture        12659
Coupled-cyclonic      5075
Frontal              12245
Ridge                18757
```

Validate a frozen label table with:

```bash
python scripts/02_validate_frozen_gmm_labels.py \
  --labels /path/to/ar_genesis_gmm_labels_K4_sh_corrected.pkl \
  --config config/regimes.json
```

## Example: classify genesis events

```bash
python scripts/01_classify_genesis_regimes_pca_gmm.py \
  --features /path/to/ar_genesis_event_features.pkl \
  --out-dir /path/to/gmm_analysis_sh_corrected \
  --k 4 \
  --random-state 42 \
  --n-init 50
```

## Example: lifecycle diagnostics

```bash
python scripts/03_build_lifecycle_diagnostics.py \
  --lifecycle-table /path/to/features_1970_2024_clean_regime_matched_sh_corrected.csv.gz \
  --out-dir /path/to/lifecycle_summaries \
  --n-bins 21 \
  --bootstrap 500
```

## Example: annual landfall/regime workflow

The workflow is modular: first build daily precipitation summaries, then hourly AR landfall-object tables, then match AR objects to regime labels, then merge daily precipitation with regime attribution. The script `run_landfall_pipeline_one_year.py` provides an example driver for one year.

## License

This draft uses the BSD 3-Clause License. Confirm the preferred license with coauthors and the institution before public release.
