# High-Throughput Occupational Exposure Model

## Overview

This repository transforms raw industrial-hygiene air measurement data into an exposure target variable and fits two-stage machine-learning models:

- **Stage 1 – Classifier**: Predict detect / non-detect.
- **Stage 2 – Regressor**: Predict the continuous exposure concentration for detected samples only.

The final target variable (mg/m³) represents the “typical” exposure for each *(chemical, [NAICS](https://www.census.gov/naics/))* pair and can be compared directly with human-equivalent points of departure (PODs) for inhalation risk assessment.

The repo supports the [scikit-learn API](https://scikit-learn.org/stable/developers/develop.html) including both built-in and custom estimators via configuration files.

## Data Sources for Citation & Acknowledgment

Please cite the relevant sources if using this repository in your research:

- **OSHA CEHD** (Chemical Exposure Health Data):  
  https://www.osha.gov/opengov/health-samples

- **OSHA USIS** (United States Information Systems):  
  https://github.com/UofMontreal-Multiexpo/uom.usis, developed by Prof. Jérôme Lavoué et al. (University of Montreal)

- **OPERA (v2.9)** – Physical-chemical properties and descriptors (QSAR):  
  https://github.com/kmansouri/OPERA, developed by Dr. Kamel Mansouri (NIEHS)

The CEHD and USIS data were cleaned and combined based on a methodology developed by Prof. Jérôme Lavoué, adapted here in Python. OPERA descriptors were used as predictors in the ML models.

## Repository Structure

- `config_main/`: Main (top-level) configuration files, serving as entry points to modeling workflows.
- `input/`: Input directory including component-level config files, and raw and processed data (provided as downloadable assets).
- `raw_processing/`: Modules for cleaning and transforming CEHD and USIS data into exposure targets.
- `run.py`: Main entry point for modeling. Supports cross-validation or holdout evaluation and persists results.
- `results/`: Output directory for persisted evaluation scores, fitted estimators, and metadata (config settings for each run).

### Example Main Config Files Included

Both example workflows use the same component-level settings, except for the regression estimators:
- `config_ols.json`: Leverages scikit-learn's `LinearRegression` for standard OLS.
- `config_mixedlm.json`: Leverages `MixedLMRegressor`, a custom statsmodels-based mixed model (defined in `mixedlm_estimation.py`)

## Compatibility

- Tested on Windows 11 with Anaconda/Miniconda.
- Other operating systems (macOS/Linux) may require adjustments.

## Installation

### 1. Clone the repository and extract assets

```bash
git clone https://github.com/jkvasnicka/ht-occupational-plus.git
```
Download and extract the asset bundle from the GitHub "Releases" tab.
Place the extracted contents inside the `input/` subdirectory.

### 2. Set up a Conda environment

From the root of the repository:

1. Create a new environment using the provided `environment.yml`:
```bash
conda env create -f environment.yml -n <your-env-name>
```
2. Activate the environment:
```bash
conda activate <your-env-name>
```
Replace `<your-env-name>` with a name of your choice. 

## Usage

The `run.py` module is the entry point for the modeling workflow. You may either specify:
- A single config file (`--config_file`), or
- A directory containing multiple config files (`--config_dir`) for model selection or sensitivity analyses

### Examples

```bash
# 1. Cross-validate several two-stage estimators for model selection
python run.py -d config_main

# 2. Evaluate holdout performance for a selected estimator (e.g., sklearn OLS)
python run.py -c config_main/config_ols.json -t holdout
```

### Command-Line Arguments

| Argument                  | Description                                                                 |
| ------------------------- | --------------------------------------------------------------------------- |
| `-c`, `--config_file`     | Path to a single main config file                                           |
| `-d`, `--config_dir`      | Path to a directory of main config files                                    |
| `-e`, `--encoding`        | Encoding for the configuration files (default: `"utf-8"`)                   |
| `-t`, `--evaluation_type` | Evaluation type: `"cv"` (cross-validation) or `"holdout"` (default: `"cv"`) |

### Notes

To avoid overoptimistic performance scores, use cross-validation (`cv`) for model selection and then use holdout evaluation (`holdout`) for assessing generalization to "unseen" data.
