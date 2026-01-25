# Case Studies

This directory contains real-world case studies demonstrating MLPY's capabilities in spatial analysis, mineral prospectivity, and geostatistical modeling.

---

## 📁 Directory Structure

### Core Case Studies

| Script                          | Description                                               | Use Case                                        |
| ------------------------------- | --------------------------------------------------------- | ----------------------------------------------- |
| `jura_case_study_mlpy.py`       | Complete spatial analysis workflow using the Jura dataset | Spatial cross-validation, variogram analysis    |
| `jura_mineral_prospectivity.py` | Mineral prospectivity mapping with spatial ML             | Geological modeling, resource estimation        |
| `meuse_spatial_analysis.py`     | Spatial analysis of Meuse dataset (soil contamination)    | Environmental monitoring, spatial interpolation |
| `kriging_comparison.py`         | Compare ML methods vs. traditional kriging                | Geostatistical modeling comparison              |

### AutoML Examples

| Script                     | Description                           |
| -------------------------- | ------------------------------------- |
| `automl_tuning_example.py` | Advanced AutoML hyperparameter tuning |
| `automl_tuning_quick.py`   | Quick AutoML setup and execution      |

### Data Preparation

| Script                     | Description                                    |
| -------------------------- | ---------------------------------------------- |
| `create_jura_dataset.py`   | Download and prepare Jura dataset              |
| `get_meuse_dataset.py`     | Download and prepare Meuse dataset             |
| `extract_mining_region.py` | Extract specific mining regions from USGS data |
| `explore_usgs_dataset.py`  | Explore USGS geospatial datasets               |

### Visualization

| Script                     | Description                          |
| -------------------------- | ------------------------------------ |
| `create_visualizations.py` | Generate publication-quality figures |

---

## 📊 Output Directories

- **`figures/`**: High-quality figures for publications and presentations
  - Spatial distribution maps
  - Cross-validation comparison plots
  - Spatial fold visualizations
  - ML vs. Kriging comparisons

- **`plots/`**: Additional analysis plots
  - Variogram plots
  - Spatial autocorrelation figures

---

## 🚀 How to Use

### 1. Prepare Data

```bash
# Download and prepare Jura dataset
python create_jura_dataset.py

# Download and prepare Meuse dataset
python get_meuse_dataset.py
```

### 2. Run Case Studies

```bash
# Run complete Jura spatial analysis
python jura_case_study_mlpy.py

# Run mineral prospectivity analysis
python jura_mineral_prospectivity.py

# Compare ML with kriging
python kriging_comparison.py
```

### 3. Generate Visualizations

```bash
# Create all publication-quality figures
python create_visualizations.py
```

---

## 📋 Requirements

Spatial analysis dependencies:

```bash
pip install geopandas pysal scikit-gstat
```

For complete functionality:

```bash
pip install -r requirements-dev.txt
```

---

## 📚 Documentation

- `CODE_COMPARISON_AUTOML.md`: Detailed comparison of AutoML approaches
- `VISUALIZATIONS_SUMMARY.md`: Overview of generated visualizations

---

## 💡 Key Features Demonstrated

- **Spatial Cross-Validation**: Account for spatial autocorrelation
- **Variogram Analysis**: Understand spatial structure
- **Geostatistical Modeling**: Compare ML vs. traditional kriging
- **Mineral Prospectivity**: Real-world geological applications
- **AutoML for Spatial Data**: Automated model selection with spatial awareness
