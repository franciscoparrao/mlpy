# MLPY Lazy Evaluation Guide

This guide explains how to use lazy evaluation in MLPY pipelines for efficient processing of large datasets.

## Overview

Lazy evaluation allows you to build complex ML pipelines that defer computation until results are actually needed. This is especially powerful when working with big data backends like Dask and Vaex, enabling you to:

- ✅ Process datasets larger than memory
- ✅ Optimize computation by combining operations
- ✅ Avoid unnecessary intermediate materializations
- ✅ Scale from laptop to cluster seamlessly

## Installation

To use lazy evaluation features, ensure you have the required backends:

```bash
# For Dask support
pip install dask[dataframe]

# For Vaex support  
pip install vaex

# Or install both
pip install mlpy[big-data]
```

## Lazy Pipeline Operations

MLPY provides several lazy-aware pipeline operations:

### LazyPipeOpScale

Scale numeric features without materializing the full dataset:

```python
from mlpy.pipelines import LazyPipeOpScale
from mlpy.tasks import task_from_dask

# Load large dataset
df = dd.read_csv('large_dataset.csv')
task = task_from_dask(df, target='y')

# Scale features lazily
scaler = LazyPipeOpScale(
    method='standard',  # or 'minmax', 'robust'
    columns=['feature1', 'feature2']  # optional: specific columns
)

# Statistics computed lazily, transformation deferred
scaled_result = scaler.train({'input': task})
scaled_task = scaled_result['output']
```

### LazyPipeOpFilter

Filter rows based on conditions without loading all data:

```python
from mlpy.pipelines import LazyPipeOpFilter

# String condition
filter_op = LazyPipeOpFilter(
    condition="age > 18 and income < 100000"
)

# Or callable condition
def custom_filter(df):
    return (df['score'] > df['score'].mean()) & (df['status'] == 'active')

filter_op = LazyPipeOpFilter(condition=custom_filter)

# Apply filter
filtered_result = filter_op.train({'input': task})
```

### LazyPipeOpSample

Sample data for training without materializing the full dataset:

```python
from mlpy.pipelines import LazyPipeOpSample

# Sample fixed number of rows
sample_op = LazyPipeOpSample(
    n=10000,
    random_state=42
)

# Or sample fraction
sample_op = LazyPipeOpSample(
    frac=0.1,  # 10% of data
    replace=False
)

sampled_result = sample_op.train({'input': task})
```

### LazyPipeOpCache

Cache intermediate results to optimize repeated operations:

```python
from mlpy.pipelines import LazyPipeOpCache

# Cache/persist results
cache_op = LazyPipeOpCache()

# For Dask: persists in distributed memory
# For Vaex: already memory-mapped (no-op)
# For Pandas: already in memory (no-op)
cached_result = cache_op.train({'input': task})
```

## Building Lazy Pipelines

### Sequential Pipeline

Chain operations together for lazy execution:

```python
# Build pipeline
filter_op = LazyPipeOpFilter(condition="feature1 > 0")
scale_op = LazyPipeOpScale(method='standard')
sample_op = LazyPipeOpSample(n=50000)

# Execute lazily
result = filter_op.train({'input': task})
result = scale_op.train({'input': result['output']})
result = sample_op.train({'input': result['output']})

# Only materializes when needed
final_task = result['output']
```

### Graph-based Pipeline

Use MLPY's graph API for complex workflows:

```python
from mlpy.pipelines import Graph, PipeOpLearner
from mlpy.learners.sklearn import LearnerRegrRF

# Create operations
filter_op = LazyPipeOpFilter(id="filter", condition="value > 0")
scale_op = LazyPipeOpScale(id="scale")
cache_op = LazyPipeOpCache(id="cache")
sample_train = LazyPipeOpSample(id="train_sample", frac=0.8)
sample_test = LazyPipeOpSample(id="test_sample", frac=0.2)

# Create graph
g = Graph(id="lazy_pipeline")

# Add operations
g.add_pipeop(filter_op)
g.add_pipeop(scale_op)
g.add_pipeop(cache_op)
g.add_pipeop(sample_train)
g.add_pipeop(sample_test)

# Connect operations
g.add_edge("filter", "scale")
g.add_edge("scale", "cache")
g.add_edge("cache", "train_sample")
g.add_edge("cache", "test_sample")

# Execute pipeline
g.train({'filter': task})
```

## Working with Different Backends

### Dask Backend

```python
import dask.dataframe as dd
from mlpy.tasks import task_from_dask

# Load large CSV files
df = dd.read_csv('data_*.csv')  # Multiple files
task = task_from_dask(df, target='y')

# Build lazy pipeline
pipeline = [
    LazyPipeOpFilter(condition="timestamp >= '2023-01-01'"),
    LazyPipeOpScale(method='robust'),
    LazyPipeOpCache(),  # Persist in cluster memory
    LazyPipeOpSample(n=100000)
]

# Execute pipeline
result = task
for op in pipeline:
    result = op.train({'input': result})['output']
```

### Vaex Backend

```python
import vaex
from mlpy.tasks import task_from_vaex

# Open memory-mapped file
df = vaex.open('large_dataset.hdf5')
task = task_from_vaex(df, target='y')

# Vaex operations are inherently lazy
pipeline = [
    LazyPipeOpFilter(condition="x > 0"),  # Virtual filter
    LazyPipeOpScale(method='minmax'),     # Virtual columns
    LazyPipeOpSample(frac=0.01)           # Efficient sampling
]

# Apply pipeline
result = task
for op in pipeline:
    result = op.train({'input': result})['output']
```

## Best Practices

### 1. Order Operations Wisely

```python
# Good: Filter first to reduce data
filter_op = LazyPipeOpFilter(condition="is_valid == True")
scale_op = LazyPipeOpScale()

# Bad: Scale before filtering (processes unnecessary data)
```

### 2. Cache Strategic Points

```python
# Cache after expensive operations
expensive_op = LazyPipeOpComplexTransform()
cache_op = LazyPipeOpCache()

result = expensive_op.train({'input': task})
result = cache_op.train({'input': result['output']})
```

### 3. Sample for Model Training

```python
# Full dataset for preprocessing
task_full = task_from_dask(large_df, target='y')

# Apply transformations
task_processed = apply_lazy_pipeline(task_full)

# Sample for training
sample_op = LazyPipeOpSample(n=50000)
task_train = sample_op.train({'input': task_processed})['output']

# Materialize for sklearn
data_train = task_train.data()  # Now loads 50k rows
```

### 4. Monitor Memory Usage

```python
# For Dask
from dask.diagnostics import ProgressBar

with ProgressBar():
    result = pipeline.train({'input': task})
    
# Check memory usage
print(f"Task size: {task.backend._data.memory_usage_per_partition().sum().compute()}")
```

## Advanced Patterns

### Custom Lazy Operations

Create your own lazy operations:

```python
class LazyPipeOpCustom(LazyPipeOp):
    def __init__(self, id="custom", **kwargs):
        super().__init__(id=id, **kwargs)
        
    @property
    def input(self):
        return {"input": PipeOpInput("input", Task, Task)}
        
    @property
    def output(self):
        return {"output": PipeOpOutput("output", Task, Task)}
        
    def train(self, inputs):
        task = inputs["input"]
        backend_type = self._get_backend_type(task)
        
        if backend_type == 'dask':
            # Dask-specific lazy operation
            df = task.backend._data
            df_transformed = df.map_partitions(custom_transform)
            new_backend = DataBackendDask(df_transformed)
        elif backend_type == 'vaex':
            # Vaex-specific lazy operation
            df = task.backend._data
            df['new_col'] = vaex_expression
            new_backend = DataBackendVaex(df)
        else:
            # Pandas fallback
            data = task.data()
            data_transformed = custom_transform(data)
            new_backend = DataBackendPandas(data_transformed)
            
        # Create new task
        new_task = type(task)(backend=new_backend, ...)
        return {"output": new_task}
```

### Conditional Execution

```python
def build_adaptive_pipeline(task):
    """Build pipeline based on data characteristics."""
    
    # Check data size
    if task.nrow > 1_000_000:
        # Use aggressive sampling for large data
        pipeline = [
            LazyPipeOpFilter(condition="quality_score > 0.5"),
            LazyPipeOpSample(n=100_000),
            LazyPipeOpScale()
        ]
    else:
        # Process full dataset for small data
        pipeline = [
            LazyPipeOpScale(),
            LazyPipeOpCache()
        ]
        
    return pipeline
```

### Parallel Branch Processing

```python
# Process different feature groups in parallel
numeric_pipeline = [
    LazyPipeOpScale(columns=numeric_features),
    LazyPipeOpImpute(strategy='median')
]

categorical_pipeline = [
    LazyPipeOpEncode(columns=categorical_features),
    LazyPipeOpImpute(strategy='most_frequent')
]

# Combine results
combined_task = combine_parallel_results(
    numeric_result, 
    categorical_result
)
```

## Troubleshooting

### Performance Issues

```python
# Profile Dask operations
from dask.diagnostics import Profiler, ResourceProfiler

with Profiler() as prof, ResourceProfiler() as rprof:
    result = pipeline.execute(task)
    
# Visualize results
from dask.diagnostics import visualize
visualize([prof, rprof])
```

### Memory Errors

```python
# Increase sampling aggressiveness
sample_op = LazyPipeOpSample(
    n=10_000,  # Smaller sample
    random_state=42
)

# Or process in batches
for batch in task.iter_batches(size=50_000):
    result = pipeline.process(batch)
```

### Debugging Lazy Operations

```python
# Force materialization for debugging
def debug_pipeline(task, pipeline):
    for i, op in enumerate(pipeline):
        print(f"\nStep {i}: {op.id}")
        result = op.train({'input': task})
        task = result['output']
        
        # Peek at data
        if hasattr(task.backend, '_data'):
            print(f"Shape: {task.nrow} x {task.ncol}")
            print(f"Columns: {task.colnames[:5]}...")
            
    return task
```

## Performance Comparison

Example benchmark on a 10GB dataset:

| Operation | Eager (Pandas) | Lazy (Dask) | Lazy (Vaex) |
|-----------|---------------|-------------|-------------|
| Load Time | 120s | 0.5s | 0.1s |
| Filter | 15s | 0.1s* | 0.1s* |
| Scale | 20s | 0.2s* | 0.1s* |
| Sample | 5s | 2s | 1s |
| Total | 160s | 3s | 1.3s |

\* Operation definition time only, computation deferred

## Summary

Lazy evaluation in MLPY enables:

1. **Efficient Big Data Processing**: Work with datasets larger than memory
2. **Optimized Computation**: Combine operations for better performance
3. **Flexible Pipelines**: Mix lazy and eager operations as needed
4. **Scalability**: Same code works from laptop to cluster
5. **Memory Efficiency**: Process only what's needed

Start with lazy operations when working with large datasets, and materialize only when necessary for model training or final results!