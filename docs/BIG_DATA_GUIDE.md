# MLPY Big Data Guide

This guide explains how to work with datasets that don't fit in memory using MLPY's big data backends.

## Overview

MLPY supports two major frameworks for out-of-core computation:

- **Dask**: Parallel computing library that scales from laptops to clusters
- **Vaex**: Memory-mapped DataFrame library for exploring billion-row datasets

Both backends provide:
- ✅ Lazy evaluation
- ✅ Out-of-core computation
- ✅ Efficient memory usage
- ✅ Seamless integration with MLPY tasks

## Installation

```bash
# For Dask support
pip install dask[dataframe]

# For Vaex support
pip install vaex

# Or install both
pip install mlpy[big-data]
```

## Quick Start

### Using Dask

```python
import dask.dataframe as dd
from mlpy.tasks import task_from_dask
from mlpy.learners.sklearn import LearnerRegrRF
from mlpy.resample import resample

# Load large CSV file lazily
df = dd.read_csv('large_dataset.csv')

# Create MLPY task
task = task_from_dask(df, target='y')

# For now, train on a subset (future: streaming learners)
task_subset = task.filter_rows(task.row_ids[:10000])

# Train model
learner = LearnerRegrRF()
result = resample(task_subset, learner, resampling=ResamplingCV())
```

### Using Vaex

```python
import vaex
from mlpy.tasks import task_from_vaex

# Open large HDF5 file (memory-mapped)
df = vaex.open('large_dataset.hdf5')

# Create MLPY task
task = task_from_vaex(df, target='y')

# Vaex computes statistics efficiently
print(f"Dataset size: {task.nrow:,} rows")
print(f"Features: {task.feature_names}")
```

## Backend Features

### DataBackendDask

```python
from mlpy.backends import DataBackendDask
import dask.dataframe as dd

# Create from pandas
df_pandas = pd.DataFrame(...)
backend = DataBackendDask.from_pandas(df_pandas, npartitions=10)

# Read from CSV
backend = DataBackendDask.read_csv('*.csv')  # Supports wildcards

# Read from Parquet
backend = DataBackendDask.read_parquet('data.parquet')

# Key features
backend.persist()  # Keep in distributed memory
backend.compute()  # Convert to pandas (careful with size!)
```

### DataBackendVaex

```python
from mlpy.backends import DataBackendVaex
import vaex

# Create from pandas
backend = DataBackendVaex.from_pandas(df_pandas)

# Open various formats
backend = DataBackendVaex.open('data.hdf5')  # Memory-mapped
backend = DataBackendVaex.open('data.parquet')
backend = DataBackendVaex.open('data.arrow')

# Export to efficient formats
backend.export_hdf5('output.hdf5')
backend.export_parquet('output.parquet')
```

## Helper Functions

### Lazy Loading from Files

```python
from mlpy.tasks import task_from_csv_lazy, task_from_parquet_lazy

# Load CSV with Dask (default)
task = task_from_csv_lazy(
    'large_data.csv',
    target='y',
    backend='dask',
    chunksize=100_000  # Rows per chunk
)

# Load CSV with Vaex (converts to HDF5)
task = task_from_csv_lazy(
    'large_data.csv',
    target='y',
    backend='vaex'
)

# Load Parquet files
task = task_from_parquet_lazy(
    'data_*.parquet',  # Supports patterns
    target='y',
    backend='dask'
)
```

### Task Creation

```python
# Auto-detect task type
task = task_from_dask(df, target='y')  # Infers regression/classification

# Explicit task type
task = task_from_vaex(
    df,
    target='y',
    task_type='classification',
    positive='1'  # For binary classification
)
```

## Best Practices

### 1. Choose the Right Backend

**Use Dask when:**
- Working with distributed clusters
- Need parallel computation
- Data comes from multiple CSV/Parquet files
- Want to scale from laptop to cluster

**Use Vaex when:**
- Working with single large files
- Need interactive exploration
- Want memory-mapped access
- Prefer columnar operations

### 2. Efficient Data Access

```python
# Good: Use slices for row selection
data_slice = backend.data(rows=slice(0, 10000))

# Avoid: Fancy indexing triggers computation
data_fancy = backend.data(rows=[1, 5, 10, 20])  # Slower

# Good: Select columns to reduce memory
data_subset = backend.data(cols=['feature1', 'feature2'])
```

### 3. Lazy Evaluation

```python
# Operations are lazy by default
task = task_from_dask(df, target='y')
task_filtered = task.filter(task.data['feature1'] > 0)  # Not computed yet

# Computation happens when needed
learner.train(task_filtered)  # Triggers computation
```

### 4. Subset for Training

Currently, most MLPY learners expect in-memory data. Use subsets:

```python
# Random subset
n_samples = 100_000
indices = np.random.choice(task.nrow, n_samples)
task_subset = task.filter_rows(indices)

# First n rows
task_head = task.filter_rows(range(100_000))

# Stratified subset (for classification)
task_stratified = task.filter_rows(
    task.stratify_indices(n=100_000, stratify_by='target')
)
```

## Limitations and Future Work

### Current Limitations

1. **Learners**: Most learners expect in-memory data
2. **Resampling**: Not all strategies support lazy evaluation
3. **Measures**: Computed on materialized predictions

### Future Enhancements

1. **Streaming Learners**: Native support for mini-batch training
2. **Distributed Training**: Leverage Dask-ML and similar
3. **Lazy Pipelines**: Full pipeline execution without materialization
4. **Approximate Algorithms**: For massive datasets

## Examples

### Example 1: Large CSV Processing

```python
import dask.dataframe as dd
from mlpy.tasks import task_from_csv_lazy
from mlpy.learners.sklearn import LearnerRegrLM

# Load 10GB CSV file
task = task_from_csv_lazy(
    'sales_data_*.csv',  # Multiple files
    target='revenue',
    backend='dask',
    parse_dates=['date']
)

print(f"Dataset: {task.nrow:,} rows, {task.ncol} columns")

# Train on manageable subset
task_2023 = task.filter(task.data['date'] >= '2023-01-01')
task_sample = task_2023.head(100_000)

learner = LearnerRegrLM()
learner.train(task_sample)
```

### Example 2: Memory-Mapped Analysis

```python
import vaex
from mlpy.tasks import task_from_vaex

# Open 100GB HDF5 file (instant, no loading)
df = vaex.open('sensor_data.hdf5')

# Create task
task = task_from_vaex(df, target='temperature')

# Vaex can compute statistics on full data
print(f"Target mean: {df['temperature'].mean()}")
print(f"Missing values: {task.missings()}")

# Visualize billion rows efficiently
df.plot1d(df.temperature, limits=[0, 100])
```

### Example 3: Distributed Processing

```python
from dask.distributed import Client
from mlpy.tasks import task_from_parquet_lazy

# Setup Dask cluster
client = Client('scheduler-address:8786')

# Load distributed dataset
task = task_from_parquet_lazy(
    's3://bucket/data/*.parquet',
    target='y',
    storage_options={'anon': True}
)

# Persist in cluster memory
task.backend.persist()

# Now operations are fast
for fold in range(5):
    task_fold = task.filter_rows(task.cv_indices(fold))
    # Train models in parallel
```

## Performance Tips

1. **Persist frequently accessed data**
   ```python
   task.backend.persist()  # Keep in memory/cluster
   ```

2. **Use appropriate chunk/partition sizes**
   ```python
   # Dask: ~100MB per partition
   df = dd.read_csv('data.csv', blocksize='100MB')
   
   # Vaex: automatic for file formats
   ```

3. **Filter early, compute late**
   ```python
   # Good: filter before materializing
   task_filtered = task.filter(conditions)
   data_small = task_filtered.head(1000)
   
   # Bad: materialize then filter
   data_all = task.data()  # Loads everything!
   data_filtered = data_all[conditions]
   ```

4. **Use columnar operations**
   ```python
   # Vaex excels at column statistics
   means = {col: task.data[col].mean() 
            for col in task.feature_names}
   ```

## Troubleshooting

### Out of Memory Errors

```python
# Reduce partition size
df = dd.read_csv('data.csv', blocksize='50MB')

# Use sampling
task_sample = task.sample(frac=0.1)

# Process in batches
for batch in task.iter_batches(size=10000):
    process(batch)
```

### Slow Performance

```python
# Check partition count
print(f"Partitions: {task.backend._data.npartitions}")

# Repartition if needed
task.backend._data = task.backend._data.repartition(npartitions=50)

# Use persist for iterative operations
task.backend.persist()
```

### Import Errors

```python
# Check available backends
from mlpy.backends import DASK_AVAILABLE, VAEX_AVAILABLE

if not DASK_AVAILABLE:
    print("Install dask: pip install dask[dataframe]")
    
if not VAEX_AVAILABLE:
    print("Install vaex: pip install vaex")
```

## Further Resources

- [Dask Documentation](https://docs.dask.org/)
- [Vaex Documentation](https://vaex.io/)
- [Dask-ML](https://ml.dask.org/) - Scalable machine learning
- [Dask Tutorial](https://tutorial.dask.org/)
- [Vaex Tutorial](https://vaex.io/tutorial)

---

With MLPY's big data backends, you can explore and model datasets of any size, from gigabytes to terabytes, all with the same familiar MLPY API!