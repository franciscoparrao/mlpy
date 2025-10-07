"""
Example of using lazy pipeline operations with big data backends.

This example demonstrates how to build ML pipelines that work efficiently
with large datasets using lazy evaluation.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Union

# Check backend availability
try:
    import dask
    import dask.dataframe as dd
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    warnings.warn("Dask not available. Install with: pip install dask[dataframe]")

try:
    import vaex
    VAEX_AVAILABLE = True
except ImportError:
    VAEX_AVAILABLE = False
    warnings.warn("Vaex not available. Install with: pip install vaex")

# Import MLPY components
from mlpy.tasks import TaskRegr, TaskClassif
from mlpy.learners.sklearn import LearnerRegrRF, LearnerClassifRF
from mlpy.resamplings import ResamplingHoldout
from mlpy.measures import MeasureRegrRMSE, MeasureClassifAcc
from mlpy.resample import resample

# Import pipeline components
try:
    from mlpy.pipelines import (
        LazyPipeOpScale, LazyPipeOpFilter, LazyPipeOpSample,
        LazyPipeOpCache, PipeOpLearner, Graph, linear_pipeline
    )
    LAZY_OPS_AVAILABLE = True
except ImportError:
    LAZY_OPS_AVAILABLE = False
    warnings.warn("Lazy pipeline operations not available")


def create_synthetic_data(n_rows: int = 100_000, n_features: int = 20) -> pd.DataFrame:
    """Create synthetic dataset for demonstration."""
    print(f"Creating synthetic dataset with {n_rows:,} rows and {n_features} features...")
    
    np.random.seed(42)
    
    # Create features
    X = np.random.randn(n_rows, n_features)
    
    # Create target with non-linear relationship
    y = (
        2 * X[:, 0] + 
        0.5 * X[:, 1]**2 + 
        0.3 * X[:, 0] * X[:, 1] +
        0.1 * np.random.randn(n_rows)
    )
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Add categorical feature
    df['category'] = np.random.choice(['A', 'B', 'C', 'D'], n_rows)
    
    return df


def demo_lazy_pipeline_pandas():
    """Demonstrate lazy pipeline with pandas backend."""
    print("\n" + "="*60)
    print("DEMO: Lazy Pipeline with Pandas Backend")
    print("="*60)
    
    # Create data
    df = create_synthetic_data(n_rows=10_000)
    
    # Create task
    task = TaskRegr(data=df, target='target', id='pandas_task')
    print(f"\nOriginal task: {task.nrow:,} rows, {task.ncol} columns")
    
    # Build lazy pipeline
    print("\nBuilding lazy pipeline...")
    
    # Step 1: Filter outliers
    filter_op = LazyPipeOpFilter(
        id="filter_outliers",
        condition=lambda df: (df['target'] > df['target'].quantile(0.01)) & 
                           (df['target'] < df['target'].quantile(0.99))
    )
    
    # Step 2: Scale features
    scale_op = LazyPipeOpScale(
        id="scale_features",
        method="standard"
    )
    
    # Step 3: Sample for training
    sample_op = LazyPipeOpSample(
        id="sample_train",
        frac=0.8,
        random_state=42
    )
    
    # Step 4: Train model
    learner = LearnerRegrRF(id="rf", n_estimators=10)
    learner_op = PipeOpLearner(learner=learner)
    
    # Execute pipeline
    print("\nExecuting pipeline...")
    result = filter_op.train({'input': task})
    print(f"After filtering: {result['output'].nrow:,} rows")
    
    result = scale_op.train({'input': result['output']})
    print(f"After scaling: {result['output'].nrow:,} rows")
    
    result = sample_op.train({'input': result['output']})
    print(f"After sampling: {result['output'].nrow:,} rows")
    
    # For learner, we need materialized data
    result = learner_op.train({'input': result['output']})
    predictions = result['output']
    
    # Evaluate
    rmse = MeasureRegrRMSE()
    score = rmse.score(predictions)
    print(f"\nTraining RMSE: {score:.4f}")
    
    # Test on new data
    print("\nTesting on new data...")
    test_df = create_synthetic_data(n_rows=2_000)
    test_task = TaskRegr(data=test_df, target='target')
    
    # Apply same transformations (without sampling)
    result = filter_op.predict({'input': test_task})
    result = scale_op.predict({'input': result['output']})
    result = learner_op.predict({'input': result['output']})
    
    test_predictions = result['output']
    test_score = rmse.score(test_predictions)
    print(f"Test RMSE: {test_score:.4f}")


def demo_lazy_pipeline_dask():
    """Demonstrate lazy pipeline with Dask backend."""
    if not DASK_AVAILABLE:
        print("\nSkipping Dask demo - library not available")
        return
        
    print("\n" + "="*60)
    print("DEMO: Lazy Pipeline with Dask Backend")
    print("="*60)
    
    # Create large dataset
    df = create_synthetic_data(n_rows=1_000_000, n_features=30)
    
    # Convert to Dask
    print("\nConverting to Dask DataFrame...")
    df_dask = dd.from_pandas(df, npartitions=20)
    
    from mlpy.backends import DataBackendDask
    from mlpy.tasks import task_from_dask
    
    task = task_from_dask(df_dask, target='target', task_type='regression')
    print(f"Dask task created: {task.nrow:,} rows across {df_dask.npartitions} partitions")
    
    # Build pipeline with caching
    print("\nBuilding lazy pipeline with caching...")
    
    # Filter extreme values
    filter_op = LazyPipeOpFilter(
        id="filter_extremes",
        condition="(target > -5) & (target < 5)"
    )
    
    # Scale features
    scale_op = LazyPipeOpScale(
        id="scale_robust",
        method="robust"  # More robust to outliers
    )
    
    # Cache intermediate results
    cache_op = LazyPipeOpCache(id="cache_scaled")
    
    # Sample for manageable training
    sample_op = LazyPipeOpSample(
        id="sample_10k",
        n=10_000,
        random_state=42
    )
    
    # Execute pipeline
    print("\nExecuting lazy pipeline...")
    
    # All operations are lazy until we need to materialize
    result = filter_op.train({'input': task})
    print("✓ Filter operation queued (lazy)")
    
    result = scale_op.train({'input': result['output']})
    print("✓ Scale operation queued (lazy)")
    
    result = cache_op.train({'input': result['output']})
    print("✓ Cache operation triggered computation")
    
    result = sample_op.train({'input': result['output']})
    sampled_task = result['output']
    print(f"✓ Sampled {sampled_task.nrow:,} rows for training")
    
    # Train model on sample
    learner = LearnerRegrRF(id="rf_dask", n_estimators=20, max_depth=5)
    learner_op = PipeOpLearner(learner=learner)
    
    # Need to materialize for sklearn
    print("\nMaterializing sample for training...")
    materialized_data = sampled_task.data()
    materialized_task = TaskRegr(data=materialized_data, target='target')
    
    result = learner_op.train({'input': materialized_task})
    print("✓ Model trained on sample")
    
    # Evaluate on another sample
    print("\nEvaluating on test sample...")
    test_sample = LazyPipeOpSample(n=5_000, random_state=123)
    test_result = test_sample.train({'input': task})
    
    # Apply transformations
    test_result = filter_op.predict({'input': test_result['output']})
    test_result = scale_op.predict({'input': test_result['output']})
    
    # Materialize for prediction
    test_data = test_result['output'].data()
    test_task = TaskRegr(data=test_data, target='target')
    
    test_result = learner_op.predict({'input': test_task})
    
    rmse = MeasureRegrRMSE()
    score = rmse.score(test_result['output'])
    print(f"Test RMSE: {score:.4f}")


def demo_graph_pipeline():
    """Demonstrate lazy operations in graph-based pipeline."""
    if not LAZY_OPS_AVAILABLE:
        print("\nLazy operations not available")
        return
        
    print("\n" + "="*60)
    print("DEMO: Graph-based Lazy Pipeline")
    print("="*60)
    
    # Create data
    df = create_synthetic_data(n_rows=50_000)
    task = TaskRegr(data=df, target='target')
    
    # Build graph pipeline
    print("\nBuilding graph pipeline...")
    
    # Create operations
    filter_op = LazyPipeOpFilter(
        id="filter",
        condition="feature_0 > -2"
    )
    
    scale_op = LazyPipeOpScale(
        id="scale",
        method="minmax"
    )
    
    sample_train = LazyPipeOpSample(
        id="sample_train",
        frac=0.7,
        random_state=42
    )
    
    sample_valid = LazyPipeOpSample(
        id="sample_valid", 
        frac=0.3,
        random_state=42
    )
    
    # Create learners
    rf_learner = PipeOpLearner(
        LearnerRegrRF(id="rf", n_estimators=10),
        id="rf_op"
    )
    
    # Build graph
    g = Graph(id="lazy_graph")
    
    # Add nodes
    g.add_pipeop(filter_op)
    g.add_pipeop(scale_op)
    g.add_pipeop(sample_train)
    g.add_pipeop(sample_valid)
    g.add_pipeop(rf_learner)
    
    # Connect nodes
    g.add_edge("filter", "scale", 
               src_channel="output", dst_channel="input")
    g.add_edge("scale", "sample_train",
               src_channel="output", dst_channel="input")
    g.add_edge("scale", "sample_valid",
               src_channel="output", dst_channel="input")
    g.add_edge("sample_train", "rf_op",
               src_channel="output", dst_channel="input")
    
    # Execute pipeline
    print("\nExecuting graph pipeline...")
    g.train({'filter': task})
    
    print("✓ Pipeline executed successfully")
    
    # Get predictions on validation set
    valid_task = g.state['sample_valid']['output']
    predictions = rf_learner.predict({'input': valid_task})['output']
    
    # Evaluate
    rmse = MeasureRegrRMSE()
    score = rmse.score(predictions)
    print(f"\nValidation RMSE: {score:.4f}")


def main():
    """Run all demonstrations."""
    print("MLPY Lazy Pipeline Operations Demo")
    print("==================================")
    
    if not LAZY_OPS_AVAILABLE:
        print("\nLazy operations not available!")
        print("This likely means Dask or Vaex are not installed.")
        return
        
    # Run demos
    demo_lazy_pipeline_pandas()
    demo_lazy_pipeline_dask()
    demo_graph_pipeline()
    
    print("\n" + "="*60)
    print("Demo completed!")
    print("="*60)
    
    print("\nKey takeaways:")
    print("1. Lazy operations defer computation until needed")
    print("2. They work seamlessly with both in-memory and out-of-core backends")
    print("3. Pipelines can mix lazy and eager operations")
    print("4. Caching can optimize repeated operations")
    print("5. Sampling enables training on manageable subsets of big data")


if __name__ == "__main__":
    main()