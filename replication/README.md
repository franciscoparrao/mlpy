# Replication Materials - MLPY JSS Paper

**Paper**: MLPY: A Modular Machine Learning Framework for Python with Native Spatial Support
**Authors**: Francisco Parra O. and Giuseppe Cavallieri
**Journal**: Journal of Statistical Software (2025)

## Contents

- `replicate_all.py` - Main replication script
- `requirements.txt` - Required Python packages
- `README.md` - This file

## Requirements

- Python 3.8+
- Required packages (see requirements.txt)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python replicate_all.py
```

**Expected runtime**: < 10 seconds
**Output directory**: `./output/`

## Outputs

### Figures (PDF)
1. `figure1_spatial_distribution.pdf` - Spatial distribution of zinc
2. `figure2_variogram.pdf` - Empirical variogram
3. `figure3_cv_comparison.pdf` - CV method comparison
4. `figure4_spatial_folds.pdf` - Spatial K-fold visualization

### Tables (CSV)
1. `table1_morans_i.csv` - Moran's I statistics
2. `table2_cv_comparison.csv` - CV RMSE comparison

### Session Information
Prints Python version, platform, and package versions.

## Contact

- Francisco Parra O. - francisco.parra.o@usach.cl
- Giuseppe Cavallieri - giuseppe.cavallieri@usach.cl
