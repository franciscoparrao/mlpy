# Replication Materials for MLPY Paper

**Paper**: MLPY: A Python Framework for Spatial Machine Learning in Geosciences

**Submitted to**: Computers & Geosciences

---

## Contents

- `replicate_all.py` - Main replication script
- `sensitivity_analysis.py` - Sensitivity analysis
- `../case_studies/data/meuse.csv` - Meuse dataset

## Installation

```bash
pip install mlpy-geo
pip install numpy pandas scikit-learn matplotlib geopandas
```

## Running

```bash
cd replication
python replicate_all.py
```

Generates all figures and tables from the paper (~2-3 minutes).

## Expected Results

**Table 2**: Standard CV underestimates error by 31%
- sklearn CV: 221.7 ppm RMSE
- MLPY Spatial CV: 319.3 ppm RMSE

## Contact

- Francisco Parra O. (francisco.parra.o@usach.cl)
- Giuseppe Cavallieri (giuseppe.cavallieri@usach.cl)

## Links

- GitHub: https://github.com/franciscoparrao/mlpy
- PyPI: https://pypi.org/project/mlpy-geo/
