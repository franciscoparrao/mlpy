# MLPY Big Data Examples Guide

This guide describes the big data examples available in MLPY, demonstrating how to work with datasets that don't fit in memory using Dask and Vaex backends.

## Overview

MLPY provides three comprehensive big data examples:

1. **Airline Dataset** (`big_data_airline_example.py`) - Flight delay prediction
2. **NYC Taxi Dataset** (`big_data_nyc_taxi_example.py`) - Fare prediction and analysis  
3. **Public Datasets** (`big_data_public_datasets.py`) - Click prediction, time series, NLP

## Prerequisites

Install big data backends:

```bash
# For Dask (distributed computing)
pip install dask[complete]

# For Vaex (memory-mapped DataFrames)
pip install vaex

# Additional dependencies
pip install scikit-learn matplotlib seaborn
```

## Example 1: Airline Delay Prediction

Demonstrates working with flight data (millions of records):

### Features
- Lazy data loading with Dask
- Time-based feature engineering
- Target encoding for high-cardinality airports/airlines
- Distributed model training
- Performance comparison between backends

### Key Functions

```python
# Dask pipeline for delay prediction
example_dask_airline_pipeline()

# Interactive analysis with Vaex
example_vaex_airline_analysis()

# Backend performance comparison
example_performance_comparison()
```

### Sample Pipeline

```python
pipeline = linear_pipeline(
    # Filter cancelled flights
    LazyPipeOpFilter(
        condition=lambda df: df['ArrDelay'].notna()
    ),
    
    # Sample for experimentation
    LazyPipeOpSample(fraction=0.1),
    
    # Encode categoricals efficiently
    PipeOpTargetEncode(
        columns=['Airline', 'Origin', 'Dest'],
        smoothing=100
    ),
    
    # Time-based features
    PipeOpBin(
        columns=['ScheduledDepTime'],
        n_bins=24,  # Hourly bins
        encode='onehot'
    ),
    
    # Scale and reduce
    LazyPipeOpScale(method="robust"),
    PipeOpPCA(n_components=0.95),
    
    # Learn
    PipeOpLearner(learner_sklearn(SGDClassifier()))
)
```

### Insights
- Peak delays on Fridays and Sundays
- Evening flights have higher delay rates
- Dask provides 2-5x speedup for large datasets
- Target encoding reduces features from 1000s to 10s

## Example 2: NYC Taxi Fare Prediction

Works with taxi trip records (millions of trips):

### Features
- Geospatial feature engineering
- Time series patterns
- Streaming/incremental learning
- Fare prediction at scale

### Key Functions

```python
# Distributed fare prediction
example_dask_taxi_fare_prediction()

# Interactive exploration
example_vaex_taxi_exploration()

# Streaming pipeline
example_streaming_taxi_pipeline()
```

### Feature Engineering

```python
# Haversine distance calculation
distances = haversine_distance(
    pickup_lat, pickup_lon,
    dropoff_lat, dropoff_lon
)

# Time-based features
ddf['hour'] = ddf['pickup_datetime'].dt.hour
ddf['is_weekend'] = (ddf['day_of_week'] >= 5).astype(int)
ddf['morning_rush'] = ((ddf['hour'] >= 7) & (ddf['hour'] <= 9))

# Location binning (creates grid)
PipeOpBin(
    columns=['pickup_latitude', 'pickup_longitude'],
    n_bins=20,
    strategy='quantile'
)
```

### Streaming Example

```python
# SGD for incremental learning
SGDRegressor(
    loss='squared_error',
    penalty='l2',
    alpha=0.0001
)

# Process in batches
for batch in streaming_batches:
    result = pipeline.partial_fit(batch)
    print(f"Batch RMSE: ${result['rmse']:.2f}")
```

### Insights
- Weekend trips are longer but less frequent
- Credit card payments have higher tips
- Location binning captures neighborhood effects
- Incremental learning enables real-time updates

## Example 3: Public Datasets

Demonstrates various big data scenarios:

### 3.1 Click Prediction (Criteo-style)

```python
# High-cardinality categorical handling
pipeline = linear_pipeline(
    # Missing value imputation
    PipeOpImpute(method="constant", value=-1),
    
    # Target encode frequent categories
    PipeOpTargetEncode(
        columns=['C1', 'C2', ...],
        smoothing=100
    ),
    
    # Bin rare categories
    PipeOpBin(
        columns=['C11', 'C12', ...],
        n_bins=50
    ),
    
    # Feature selection
    PipeOpSelect(k=50, score_func="f_classif"),
    
    # Scalable learning
    PipeOpLearner(SGDClassifier())
)
```

### 3.2 Time Series (Wikipedia Traffic)

```python
# Lag features for time series
for lag in [1, 7, 28]:
    df[f'views_lag_{lag}'] = df.groupby('page')['views'].shift(lag)

# Rolling statistics
df['views_ma_7'] = df.groupby('page')['views'].rolling(7).mean()

# Time encoding
PipeOpBin(
    columns=['day_of_week', 'day_of_month'],
    n_bins=[7, 10],
    encode='onehot'
)
```

### 3.3 NLP at Scale (Reddit Comments)

```python
pipeline = linear_pipeline(
    # Text vectorization
    PipeOpTextVectorize(
        columns=['comment'],
        method='tfidf',
        max_features=1000,
        ngram_range=(1, 2)
    ),
    
    # Metadata encoding
    PipeOpTargetEncode(columns=['subreddit']),
    
    # Scale and learn
    LazyPipeOpScale(),
    PipeOpLearner(SGDClassifier())
)
```

## Performance Tips

### 1. Use Appropriate Backends

**Dask** - Best for:
- Distributed computing across machines
- Complex pipelines with many operations
- When you need the full pandas API

**Vaex** - Best for:
- Interactive exploration
- Memory-mapped files
- Visualization of billions of rows

### 2. Optimize Pipeline Order

```python
# Good: Filter early, sample for development
pipeline = linear_pipeline(
    LazyPipeOpFilter(...),      # Reduce data first
    LazyPipeOpSample(0.01),     # Dev on 1%
    PipeOpTargetEncode(...),    # Then transform
    LazyPipeOpScale(...),       # Lazy where possible
    PipeOpLearner(...)
)
```

### 3. Choose Appropriate Algorithms

For big data, prefer:
- **SGD** variants (SGDClassifier, SGDRegressor)
- **Random Forest** with limited depth
- **Target encoding** over one-hot
- **Hashing trick** for very high cardinality

### 4. Monitor Memory Usage

```python
# Dask dashboard
client = dask.distributed.Client()
print(f"Dashboard: {client.dashboard_link}")

# Vaex memory info
print(f"Memory usage: {vdf.memory_usage(deep=True)}")
```

## Running the Examples

### Basic Usage

```bash
# Run all airline examples
python examples/big_data_airline_example.py

# Run specific example
python -c "from big_data_nyc_taxi_example import example_vaex_taxi_exploration; example_vaex_taxi_exploration()"
```

### With Real Data

The examples create synthetic data for demonstration. To use real datasets:

1. **Airline Data**: Download from [Bureau of Transportation Statistics](https://www.transtats.bts.gov/)
2. **NYC Taxi**: Get from [NYC TLC](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
3. **Criteo**: Available at [Criteo Labs](http://labs.criteo.com/)

Modify the data loading sections to point to your downloaded files.

## Common Patterns

### Pattern 1: Lazy Evaluation Pipeline

```python
# Process only what's needed
lazy_pipeline = linear_pipeline(
    LazyPipeOpFilter(condition=lambda df: df['value'] > 0),
    LazyPipeOpSample(fraction=0.1),
    LazyPipeOpScale(),
    PipeOpLearner(model)
)
```

### Pattern 2: High-Cardinality Handling

```python
# Combine target encoding with binning
encode_pipeline = linear_pipeline(
    # Frequent categories: target encode
    PipeOpTargetEncode(
        columns=['frequent_cat'],
        min_samples_leaf=100
    ),
    # Rare categories: bin
    PipeOpBin(
        columns=['rare_cat'],
        n_bins=50
    )
)
```

### Pattern 3: Streaming Updates

```python
# Incremental learning
for batch in data_stream:
    # Update model
    model.partial_fit(batch)
    
    # Evaluate periodically
    if batch_num % 10 == 0:
        score = model.score(test_data)
        print(f"Current score: {score}")
```

## Troubleshooting

### Out of Memory Errors

1. Increase sampling:
   ```python
   LazyPipeOpSample(fraction=0.01)  # Use only 1%
   ```

2. Use chunking:
   ```python
   for chunk in dd.read_csv(file, chunksize=10000):
       process(chunk)
   ```

3. Reduce features early:
   ```python
   PipeOpSelect(k=100)  # Keep top 100
   ```

### Slow Performance

1. Check Dask dashboard for bottlenecks
2. Use more workers: `Client(n_workers=4)`
3. Increase block size: `blocksize='100MB'`
4. Profile with: `%time` in Jupyter

### Data Loading Issues

1. Specify dtypes:
   ```python
   dd.read_csv(file, dtype={'col': 'float32'})
   ```

2. Parse dates efficiently:
   ```python
   dd.read_csv(file, parse_dates=['date'], 
               date_parser=pd.to_datetime)
   ```

## Summary

These examples demonstrate that MLPY can handle:
- **Scale**: Millions to billions of rows
- **Variety**: Structured, time series, text, geospatial
- **Velocity**: Streaming and batch processing
- **Complexity**: Sophisticated feature engineering

The unified API means the same pipeline code works whether your data fits in memory or requires a cluster!