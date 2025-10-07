# MLPY Advanced Pipeline Operators Guide

This guide covers the advanced pipeline operators available in MLPY for sophisticated data preprocessing and feature engineering.

## Overview

MLPY provides advanced operators that go beyond basic preprocessing:

- **Dimensionality Reduction**: PCA and other techniques
- **Outlier Detection**: Multiple algorithms with flexible handling
- **Feature Engineering**: Binning, polynomials, interactions
- **Text Processing**: Vectorization for NLP tasks
- **Advanced Encoding**: Target encoding for high-cardinality features

## Dimensionality Reduction

### PipeOpPCA

Principal Component Analysis for reducing feature dimensions while preserving variance.

```python
from mlpy.pipelines import PipeOpPCA

# Keep specific number of components
pca_fixed = PipeOpPCA(n_components=10)

# Keep components explaining 95% variance
pca_variance = PipeOpPCA(n_components=0.95)

# Whiten components (decorrelate and scale)
pca_white = PipeOpPCA(n_components=20, whiten=True)

# Use in pipeline
pipeline = linear_pipeline(
    PipeOpScale(),  # Important: scale before PCA
    PipeOpPCA(n_components=0.99),
    PipeOpLearner(learner)
)
```

**Parameters:**
- `n_components`: Number of components (int), variance fraction (float), or 'mle'
- `whiten`: Whether to whiten components (decorrelate)
- `svd_solver`: Algorithm - 'auto', 'full', 'arpack', 'randomized'

**When to use:**
- High-dimensional data (many features)
- Multicollinearity issues
- Visualization (2-3 components)
- Noise reduction

**Example: Visualizing high-dimensional data**

```python
# Reduce to 2D for visualization
pca_viz = PipeOpPCA(n_components=2)
result = pca_viz.train({'input': task})
transformed = result['output'].data()

plt.scatter(transformed['PC1'], transformed['PC2'], c=task.truth())
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
```

## Outlier Detection

### PipeOpOutlierDetect

Detect and handle outliers using various algorithms.

```python
from mlpy.pipelines import PipeOpOutlierDetect

# Flag outliers (add indicator column)
outlier_flag = PipeOpOutlierDetect(
    method='isolation',
    contamination=0.1,
    action='flag'
)

# Remove outliers
outlier_remove = PipeOpOutlierDetect(
    method='elliptic',
    contamination='auto',
    action='remove'
)

# Impute outliers with median
outlier_impute = PipeOpOutlierDetect(
    method='lof',
    contamination=0.05,
    action='impute'
)
```

**Parameters:**
- `method`: Detection algorithm
  - `'isolation'`: Isolation Forest (good for high-dimensional data)
  - `'elliptic'`: Elliptic Envelope (assumes Gaussian distribution)
  - `'lof'`: Local Outlier Factor (density-based)
- `contamination`: Expected outlier fraction or 'auto'
- `action`: What to do with outliers
  - `'flag'`: Add binary indicator column
  - `'remove'`: Remove outlier rows
  - `'impute'`: Replace with median values
- `flag_column`: Name for indicator column (if action='flag')

**When to use:**
- Noisy datasets
- Before training sensitive models
- Exploratory data analysis
- Robust modeling

**Example: Comparing outlier strategies**

```python
strategies = ['flag', 'remove', 'impute']
results = {}

for action in strategies:
    pipeline = linear_pipeline(
        PipeOpOutlierDetect(method='isolation', action=action),
        PipeOpScale(),
        PipeOpLearner(learner)
    )
    
    cv_result = resample(task, pipeline, ResamplingCV(5), measure)
    results[action] = cv_result.aggregate()
```

## Feature Engineering

### PipeOpBin

Discretize continuous features into bins.

```python
from mlpy.pipelines import PipeOpBin

# Equal-width bins
bin_uniform = PipeOpBin(
    n_bins=5,
    strategy='uniform',
    encode='ordinal'
)

# Equal-frequency bins (quantiles)
bin_quantile = PipeOpBin(
    n_bins=10,
    strategy='quantile',
    encode='onehot'
)

# K-means based bins
bin_kmeans = PipeOpBin(
    n_bins=4,
    strategy='kmeans',
    columns=['age', 'income']  # Specific columns
)
```

**Parameters:**
- `n_bins`: Number of bins to create
- `strategy`: Binning strategy
  - `'uniform'`: Equal width bins
  - `'quantile'`: Equal frequency bins
  - `'kmeans'`: Cluster-based bins
- `encode`: How to encode bins
  - `'ordinal'`: Integer encoding (0, 1, 2, ...)
  - `'onehot'`: Binary columns for each bin
- `columns`: Specific columns to bin (default: all numeric)

**When to use:**
- Non-linear relationships
- Decision tree-like splits for linear models
- Handling outliers
- Creating categorical features from continuous

### PipeOpPolynomial

Generate polynomial and interaction features.

```python
from mlpy.pipelines import PipeOpPolynomial

# Degree 2 with all terms
poly_full = PipeOpPolynomial(degree=2)

# Only interaction terms (no squares)
poly_interact = PipeOpPolynomial(
    degree=2,
    interaction_only=True
)

# Higher degree for specific columns
poly_custom = PipeOpPolynomial(
    degree=3,
    columns=['x1', 'x2'],
    include_bias=False
)
```

**Parameters:**
- `degree`: Maximum polynomial degree
- `interaction_only`: Only interaction terms (no powers)
- `include_bias`: Include bias/intercept term
- `columns`: Specific columns (default: all numeric)

**When to use:**
- Non-linear relationships
- Feature interactions
- Basis expansion for linear models
- Physics-informed features

**Example: Capturing non-linearity**

```python
# Original linear model fails
linear = linear_pipeline(
    PipeOpScale(),
    PipeOpLearner(Ridge())
)

# Polynomial features capture non-linearity
poly_pipeline = linear_pipeline(
    PipeOpScale(),
    PipeOpPolynomial(degree=3),
    PipeOpLearner(Ridge(alpha=1.0))  # Regularization important!
)
```

## Advanced Encoding

### PipeOpTargetEncode

Encode categorical variables using target statistics.

```python
from mlpy.pipelines import PipeOpTargetEncode

# Basic target encoding
target_enc = PipeOpTargetEncode(
    columns=['city', 'occupation']
)

# With smoothing to prevent overfitting
target_smooth = PipeOpTargetEncode(
    smoothing=20,  # Higher = more regularization
    min_samples_leaf=10  # Minimum samples per category
)

# Auto-detect categorical columns
target_auto = PipeOpTargetEncode(columns=None)
```

**Parameters:**
- `columns`: Columns to encode (None = auto-detect)
- `smoothing`: Smoothing strength (prevents overfitting)
- `min_samples_leaf`: Minimum samples to use category mean

**When to use:**
- High-cardinality categoricals (many unique values)
- Alternative to one-hot encoding
- Tree-based models
- Limited memory

**How it works:**
```
For each category:
encoded_value = (n * category_mean + smoothing * global_mean) / (n + smoothing)

where n = number of samples in category
```

**Example: High-cardinality features**

```python
# Problem: City has 1000 unique values
# One-hot would create 1000 columns!

# Solution: Target encoding
pipeline = linear_pipeline(
    PipeOpTargetEncode(
        columns=['city'],
        smoothing=50  # Strong smoothing for rare cities
    ),
    PipeOpScale(),
    PipeOpLearner(learner)
)
```

## Text Processing

### PipeOpTextVectorize

Convert text to numeric features using bag-of-words or TF-IDF.

```python
from mlpy.pipelines import PipeOpTextVectorize

# TF-IDF vectorization
tfidf = PipeOpTextVectorize(
    columns=['review_text'],
    method='tfidf',
    max_features=1000,
    ngram_range=(1, 2),  # Unigrams and bigrams
    min_df=5,  # Minimum document frequency
    max_df=0.95  # Maximum document frequency
)

# Simple count vectorization
count_vec = PipeOpTextVectorize(
    columns=['description', 'title'],
    method='count',
    max_features=500
)
```

**Parameters:**
- `columns`: Text columns to vectorize
- `method`: Vectorization method
  - `'tfidf'`: Term frequency-inverse document frequency
  - `'count'`: Simple word counts
- `max_features`: Maximum vocabulary size
- `ngram_range`: (min_n, max_n) for n-grams
- `min_df`: Ignore rare terms
- `max_df`: Ignore too common terms

**When to use:**
- Text classification
- Document clustering
- Feature extraction from text
- Sentiment analysis

**Example: Text classification pipeline**

```python
text_pipeline = linear_pipeline(
    # Process text
    PipeOpTextVectorize(
        columns=['review'],
        method='tfidf',
        max_features=5000,
        ngram_range=(1, 3)
    ),
    # Add other features
    PipeOpScale(),  # Scales numeric features
    # Reduce dimensionality
    PipeOpPCA(n_components=100),
    # Learn
    PipeOpLearner(LogisticRegression())
)
```

## Best Practices

### 1. Order Matters

```python
# Good: Scale before PCA
pipeline = linear_pipeline(
    PipeOpScale(),
    PipeOpPCA(n_components=50)
)

# Bad: PCA before scaling
# (PCA will be dominated by features with large scales)
```

### 2. Handle Outliers Early

```python
# Good: Remove outliers before other transformations
pipeline = linear_pipeline(
    PipeOpOutlierDetect(action='remove'),
    PipeOpScale(),
    PipeOpPCA()
)
```

### 3. Combine Operators Intelligently

```python
# Example: Text + numeric features
pipeline = linear_pipeline(
    # Handle text
    PipeOpTextVectorize(columns=['text'], max_features=100),
    
    # Handle categoricals
    PipeOpTargetEncode(columns=['category']),
    
    # Create interactions
    PipeOpPolynomial(columns=['numeric1', 'numeric2'], degree=2),
    
    # Scale everything
    PipeOpScale(),
    
    # Reduce dimensions
    PipeOpPCA(n_components=0.99),
    
    # Learn
    PipeOpLearner(RandomForestClassifier())
)
```

### 4. Monitor Transformations

```python
# Train pipeline
pipeline.train(task)

# Inspect transformations
pca_op = pipeline.pipeops['pca']
print(f"Variance explained: {sum(pca_op.state['explained_variance_ratio'])}")

outlier_op = pipeline.pipeops['outlier']
print(f"Outliers found: {outlier_op.state['n_outliers']}")
```

### 5. Cross-Validation Considerations

```python
# Operators learn from training data only
# This prevents data leakage in CV

result = resample(
    task=task,
    learner=pipeline,  # All operators respect train/test split
    resampling=ResamplingCV(5),
    measure=measure
)
```

## Performance Tips

### Memory Efficiency

```python
# For large datasets, limit features early
pipeline = linear_pipeline(
    PipeOpSelect(k=100),  # Select top 100 features first
    PipeOpPolynomial(degree=2),  # Then create polynomials
    PipeOpPCA(n_components=50)  # Finally reduce
)
```

### Computation Speed

```python
# Use sparse operations where possible
text_pipeline = linear_pipeline(
    PipeOpTextVectorize(
        method='tfidf',
        max_features=1000  # Limit vocabulary size
    ),
    # RandomForest handles sparse matrices well
    PipeOpLearner(RandomForestClassifier())
)
```

### Debugging Pipelines

```python
# Test operators individually
outlier_op = PipeOpOutlierDetect()
result = outlier_op.train({'input': task})
print(f"Shape after: {result['output'].shape}")

# Use smaller data for testing
small_task = task.filter_rows(range(100))
test_result = pipeline.train(small_task)
```

## Common Patterns

### Pattern 1: High-Dimensional Data

```python
high_dim_pipeline = linear_pipeline(
    PipeOpScale(),
    PipeOpPCA(n_components=0.95),
    PipeOpLearner(SVC())
)
```

### Pattern 2: Robust Regression

```python
robust_pipeline = linear_pipeline(
    PipeOpOutlierDetect(action='remove'),
    PipeOpScale(method='robust'),
    PipeOpLearner(Ridge())
)
```

### Pattern 3: Non-Linear Classification

```python
nonlinear_pipeline = linear_pipeline(
    PipeOpBin(n_bins=10, encode='onehot'),
    PipeOpPolynomial(degree=2, interaction_only=True),
    PipeOpLearner(LogisticRegression())
)
```

### Pattern 4: Text + Structured Data

```python
mixed_pipeline = linear_pipeline(
    PipeOpTextVectorize(columns=['text'], max_features=200),
    PipeOpTargetEncode(columns=['category']),
    PipeOpScale(),
    PipeOpPCA(n_components=100),
    PipeOpLearner(GradientBoostingClassifier())
)
```

## Summary

Advanced operators in MLPY enable:

1. **Sophisticated preprocessing**: Handle complex real-world data
2. **Feature engineering**: Create powerful representations
3. **Dimensionality reduction**: Manage high-dimensional data
4. **Robust pipelines**: Handle outliers and noise
5. **Text processing**: Integrate NLP seamlessly

These operators follow MLPY's philosophy of composable, reusable components that work together in pipelines, making it easy to build sophisticated ML workflows!