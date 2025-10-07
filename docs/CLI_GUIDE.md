# MLPY Command Line Interface Guide

MLPY provides a comprehensive command-line interface (CLI) for common machine learning tasks. This guide covers all available commands and their usage.

## Installation

Once MLPY is installed, the CLI is available through:
- `mlpy` command (after installing the package)
- `python -m mlpy` (running as module)

## Basic Usage

```bash
mlpy --help              # Show all commands
mlpy --version           # Show MLPY version
python -m mlpy --help    # Alternative way to run
```

## Available Commands

### 1. Info Command

Display information about your MLPY installation:

```bash
mlpy info
```

Output includes:
- MLPY version
- Python version
- Installation path
- Core and optional dependencies
- Available components count

### 2. Train Command

Train a model with cross-validation:

```bash
mlpy train data.csv -t classif -y target -l rf -k 5 -m acc -m auc
```

Options:
- `data.csv`: Input data file (CSV or Parquet)
- `-t, --task-type`: Task type (`classif` or `regr`)
- `-y, --target`: Target column name
- `-l, --learner`: Learner to use (default: `rf`)
  - `rf`: Random Forest
  - `lr`: Linear/Logistic Regression
  - `dt`: Decision Tree
- `-k, --cv-folds`: Number of CV folds (default: 5)
- `-m, --measure`: Evaluation measures (can specify multiple)
  - Classification: `acc`, `auc`, `f1`
  - Regression: `rmse`, `r2`, `mae`
- `-o, --output`: Output file for results

Example:
```bash
mlpy train iris.csv -t classif -y species -l rf -k 10 -m acc -m f1 -o results.csv
```

### 3. Benchmark Command

Compare multiple learners on a dataset:

```bash
mlpy benchmark data.csv -t classif -y target -l rf -l lr -l dt
```

Options:
- Similar to train command
- `-l, --learners`: Specify multiple learners to compare
- Outputs score table and ranking

Example:
```bash
mlpy benchmark wine.csv -t classif -y quality -l rf -l lr -l dt -k 5 -o benchmark_results.xlsx
```

### 4. Predict Command

Make predictions using a saved model:

```bash
mlpy predict model.pkl test_data.csv -o predictions.csv
```

Options:
- `model.pkl`: Saved model file
- `test_data.csv`: Data to predict on
- `-o, --output`: Output file for predictions
- `--proba`: Output probabilities instead of classes

Example:
```bash
mlpy predict trained_model.joblib new_data.csv -o predictions.csv --proba
```

### 5. Task Commands

Manage and inspect tasks:

```bash
mlpy task info data.csv -y target
```

Shows:
- Dataset shape and memory usage
- Column information (types, nulls, unique values)
- Target distribution

### 6. Learner Commands

List available learners:

```bash
mlpy learner list
```

Shows:
- Native MLPY learners
- Scikit-learn integration options

### 7. Pipeline Commands

Create ML pipelines:

```bash
mlpy pipeline create -o my_pipeline.pkl
```

Interactive mode guides you through:
- Scaling options
- Feature selection
- Dimensionality reduction
- Learner selection

Or use configuration file:
```bash
mlpy pipeline create -c pipeline_config.yaml -o pipeline.pkl
```

### 8. Preprocess Command

Preprocess data with common transformations:

```bash
mlpy preprocess -i raw_data.csv -o clean_data.csv --scale --impute --encode
```

Options:
- `--scale`: Apply standard scaling
- `--impute`: Impute missing values
- `--encode`: Encode categorical variables
- `-p, --pipeline`: Use existing pipeline file

### 9. Experiment Command

Define and run experiments from configuration:

```bash
# Create experiment template
mlpy experiment my_experiment.yaml

# Run experiment
mlpy experiment my_experiment.yaml --run
```

Example experiment configuration:
```yaml
name: "Iris Classification"
data:
  file: "iris.csv"
  target: "species"
  task_type: "classif"
learners:
  - type: "rf"
    params:
      n_estimators: 100
  - type: "lr"
    params:
      max_iter: 1000
resampling:
  method: "cv"
  folds: 5
measures: ["acc", "f1"]
output:
  results: "experiment_results.csv"
```

### 10. Shell Command

Start an interactive MLPY shell:

```bash
mlpy shell
```

Pre-imported modules:
- `mlpy` (full package)
- `pd` (pandas)
- `np` (numpy)
- Common classes: `TaskClassif`, `TaskRegr`, `learner_sklearn`, etc.

Options:
- `-s, --shell`: Choose shell type (`ipython` or `python`)

## Examples

### Complete Workflow Example

```bash
# 1. Inspect your data
mlpy task info mydata.csv -y outcome

# 2. Preprocess if needed
mlpy preprocess -i mydata.csv -o clean_data.csv --scale --impute

# 3. Compare multiple models
mlpy benchmark clean_data.csv -t classif -y outcome -l rf -l lr -l dt -k 10

# 4. Train best model with full evaluation
mlpy train clean_data.csv -t classif -y outcome -l rf -k 10 -m acc -m auc -m f1 -o results.csv

# 5. Save predictions on new data
mlpy predict model.pkl new_data.csv -o predictions.csv
```

### Experiment-Based Workflow

```bash
# 1. Create experiment configuration
mlpy experiment credit_risk.yaml

# 2. Edit the configuration file
# ... edit credit_risk.yaml ...

# 3. Run the experiment
mlpy experiment credit_risk.yaml --run
```

## Tips and Best Practices

1. **Data Formats**: The CLI supports CSV and Parquet files. For large datasets, use Parquet for better performance.

2. **Model Persistence**: Models are saved with metadata including feature names and performance metrics.

3. **Pipeline Creation**: Use pipelines for reproducible preprocessing steps.

4. **Experiments**: Use experiment files for complex setups with multiple configurations.

5. **Interactive Shell**: Great for exploratory analysis and quick prototyping.

## Troubleshooting

1. **Import Errors**: Ensure all dependencies are installed:
   ```bash
   pip install mlpy[all]
   ```

2. **Memory Issues**: For large datasets, consider using Dask backend (if installed).

3. **Performance**: Use fewer CV folds for faster initial exploration.

## Advanced Usage

### Custom Learners in Shell

```python
# In MLPY shell
from sklearn.svm import SVC
svm = learner_sklearn(SVC(kernel='rbf'))
result = resample(task, svm, ResamplingCV(folds=5))
```

### Batch Processing

Create a script to process multiple datasets:

```bash
#!/bin/bash
for file in data/*.csv; do
    mlpy train "$file" -t classif -y label -l rf -o "results/$(basename $file .csv)_results.csv"
done
```

## Conclusion

The MLPY CLI provides a powerful interface for common ML workflows without writing code. It's designed for both quick experiments and production pipelines. For more complex scenarios, use the Python API directly or combine CLI commands with scripts.