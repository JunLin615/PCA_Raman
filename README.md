# PCA_Raman
Single-molecule verification using PCA

# Reference

https://www.nature.com/articles/s41467-018-07869-5

# run

## merge_data.py

**Overview**
A lightweight command-line tool that merges multiple `saved_col_n` files from a specified directory into a single `merged.csv`.

**Key Features**

* **Column Merging:** Retains the shared first column and horizontally appends the second column from each matching file.
* **Dynamic Headers:** Automatically renames the top row of each appended column using its source filename (without the extension).
* **Format Priority:** Prioritizes CSV files over TXT files if both exist in the directory.
* **Smart Sorting:** Sorts input files numerically by `n` (e.g., 1, 2, 10) rather than alphabetically.
* **CLI Execution:** Uses a single command-line argument as both the input directory and output destination.

run:

python merge_data.py "C:\Users\YourName\Desktop\DataFolder"

## mpca_sers_analysis.py

The `mpca_sers_analysis.py` script is a dedicated automation pipeline for analyzing **bi-analyte single-molecule SERS (Surface-Enhanced Raman Scattering) data**. It performs two-component unmixing using Modified PCA to calculate analyte fractions and exports publication-quality visualizations and statistical tables.


### Key Parameters

* **Path Arguments**:
* `--data-dir`: The root directory containing classified spectral data organized into folders.
* `--out-dir`: The directory where output figures and CSV data will be saved.


* **Preprocessing Arguments**:
* `--x-range`: Defines the spectral range to analyze (format: `min max`).
* `--n-points`: Resamples spectra onto a fixed-length grid.
* `--baseline`: Specifies the baseline correction method (`none`, `als`, `poly1`, or `poly2`).
* `--smooth-method`: Sets the smoothing method (`none`, `sg`, or `moving_average`).
* `--normalize`: Determines the normalization strategy (`max`, `minmax`, `soft_minmax`, `vector`, etc.).


* **Operational Arguments**:
* `--per-group`: Analyzes each immediate subfolder independently.
* `--pca-exclude-labels`: Specifies labels (e.g., `None`) to be excluded from PCA fitting.
* `--qc`: Exports additional quality control plots similar to Supplementary Fig. S16.

### Example Execution Command

The following command analyzes the spectral data in the `./data` folder, restricts the range to 500–1800 cm⁻¹, applies Savitzky-Golay smoothing, performs `soft_minmax` normalization, and excludes `None` labels from the PCA fitting process:

```bash
python mpca_sers_analysis.py --data-dir "data/260625Alignment" --out-dir "results/260625Alignment_500_1200_soft_minmax" --baseline als --baseline-lam 1000 --baseline-p 0.001 --normalize soft_minmax --soft-minmax-k 3 --per-group --qc --x-range 500 1200 --n-points 2048 --pca-exclude-labels None --smooth-method moving_average --smooth-window 3

```
## generate_synthetic_sers_data.py

The `generate_synthetic_sers_data.py` script is a companion utility designed to create **synthetic, ground-truth labeled bi-analyte SERS datasets**. It generates a structured directory tree compatible with the `mpca_sers_analysis.py` pipeline, allowing users to validate analysis results against known input parameters, including specific signal-to-noise ratios, amplitude variations, and spectral mixtures.


### Key Features

* **Customizable Spectral Signatures**: Generates synthetic spectra using fixed Lorentzian peaks for R6G and 4-MPY templates.
* **Configurable Noise and Baseline**: Supports independent Gaussian noise levels and optional random polynomial baselines to simulate realistic experimental artifacts.
* **Controlled Mixture Modeling**: Mix spectra are generated as linear combinations of the pure templates, with random amplitude variations simulated via lognormal distributions.
* **Standardized Output**: Automatically creates a directory hierarchy with both individual CSV files and grouped `merged.csv` files, alongside a comprehensive metadata file containing ground-truth coefficients for every generated event.


### Example Execution Command

The following command generates a synthetic dataset in the `./synthetic_data` folder with 3 groups, providing specific event counts for R6G, 4-MPY, Mix, and None categories:

```bash
python generate_synthetic_sers_data.py --out-dir "data/synthetic_3" --overwrite --n-groups 1 --n-r6g 44 --n-mpy 51 --n-mix 11 --n-none 300 --x-start 300 --x-end 2000 --n-points 1701 --signal-scale 300 --noise-std 50 --none-noise-std 50

```



