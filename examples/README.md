# MLPY Examples Directory

This directory contains a comprehensive collection of examples demonstrating the usage of MLPY for various machine learning tasks. The examples are organized to help users progressively learn the framework's capabilities.

---

## 📁 Directory Structure

### 📂 `scripts/` — Basic Scripts

Fundamental examples for getting started with MLPY:

| Script                      | Description                              |
| --------------------------- | ---------------------------------------- |
| `basic_classification.py`   | Basic classification workflow using MLPY |
| `basic_regression.py`       | Basic regression workflow using MLPY     |
| `benchmarking.py`           | Model benchmarking and comparison        |
| `cross_validation.py`       | Cross-validation techniques              |
| `large_dataset_handling.py` | Handling large datasets efficiently      |
| `pipeline_example.py`       | Building ML pipelines with MLPY          |

### 📂 `notebooks/` — Jupyter Notebooks

Interactive notebooks for in-depth exploration:

| Notebook                          | Description                       |
| --------------------------------- | --------------------------------- |
| `01_getting_started.ipynb`        | Introduction to MLPY              |
| `02_automl_example.ipynb`         | AutoML capabilities               |
| `03_advanced_pipelines.ipynb`     | Advanced pipeline workflows       |
| `04_simpleautoml_showcase.ipynb`  | SimpleAutoML demonstration        |
| `05_backends_and_bigdata.ipynb`   | Backend integrations and big data |
| `06_visualization_showcase.ipynb` | Visualization features            |

### 📄 Root-Level Examples

Advanced and specialized examples:

| Category                   | Examples                                                                                                            |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **AutoML**                 | `automl_example.py`, `automl_advanced.py`, `automl_demo.py`, `simple_automl_example.py`                             |
| **Big Data**               | `big_data_example.py`, `big_data_nyc_taxi_example.py`, `big_data_airline_example.py`, `big_data_public_datasets.py` |
| **Pipelines**              | `advanced_pipelines_example.py`, `lazy_pipeline_example.py`, `lazy_evaluation_example.py`                           |
| **Model Comparison**       | `benchmark_multimodel_example.py`, `gradient_boosting_comparison.py`, `xgboost_demo.py`                             |
| **H2O Integration**        | `h2o_integration_demo.py`, `h2o_benchmark.py`, `h2o_benchmark_simple.py`                                            |
| **Spatial ML**             | `spatial_cv_example.py`, `spatial_task_example.py`                                                                  |
| **MLOps**                  | `mlops_demo.py`, `api_deployment_example.py`, `feature_store_example.py`                                            |
| **Explainability**         | `xai_demo.py`, `information_theory_filters_demo.py`                                                                 |
| **Reinforcement Learning** | `rl_cartpole_example.py`, `rl_continuous_example.py`                                                                |
| **Serialization**          | `persistence_example.py`, `serialization_example.py`                                                                |
| **Visualization**          | `dashboard_example.py`, `filters_demo.py`                                                                           |
| **Complete Workflows**     | `complete_ml_workflow.py`, `end_to_end_demo.py`, `ml_workflow_simple.py`                                            |

---

## 🚀 How to Use

1. **Navigate** to the desired script or notebook.
2. **Run** Python scripts:
   ```bash
   python <script_name>.py
   ```
3. **Open** Jupyter notebooks:
   ```bash
   jupyter notebook notebooks/<notebook_name>.ipynb
   ```

---

## 📋 Requirements

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

For development dependencies (testing, linting, etc.):

```bash
pip install -r requirements-dev.txt
```

---

## 💡 Recommended Learning Path

1. Start with `scripts/basic_classification.py` and `scripts/basic_regression.py`
2. Explore `notebooks/01_getting_started.ipynb`
3. Try `scripts/pipeline_example.py` for pipeline concepts
4. Advance to AutoML examples
5. Explore specialized topics (spatial, big data, MLOps)
