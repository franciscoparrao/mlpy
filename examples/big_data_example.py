"""
Example of using MLPY with large datasets via Dask and Vaex.

This example demonstrates how to work with datasets that don't fit in memory
using MLPY's big data backends.
"""

import numpy as np
import pandas as pd
import warnings

# Check if big data backends are available
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

from mlpy.tasks import TaskRegr
from mlpy.learners.sklearn import LearnerRegrLM
from mlpy.resamplings import ResamplingHoldout
from mlpy.measures import MeasureRegrRMSE, MeasureRegrMAE
from mlpy.resample import resample


def create_synthetic_large_dataset(n_rows=1_000_000, n_features=20):
    """Create a synthetic dataset for demonstration."""
    print(f"Creating synthetic dataset with {n_rows:,} rows and {n_features} features...")
    
    # Create in chunks to avoid memory issues
    chunk_size = 100_000
    chunks = []
    
    for i in range(0, n_rows, chunk_size):
        size = min(chunk_size, n_rows - i)
        
        # Create features
        X = np.random.randn(size, n_features)
        
        # Create target with linear relationship + noise
        coefficients = np.random.randn(n_features)
        y = X @ coefficients + 0.1 * np.random.randn(size)
        
        # Create DataFrame
        df_chunk = pd.DataFrame(X, columns=[f'feature_{j}' for j in range(n_features)])
        df_chunk['target'] = y
        chunks.append(df_chunk)
        
        print(f"  Created chunk {i//chunk_size + 1}/{(n_rows + chunk_size - 1)//chunk_size}")
    
    # Combine chunks
    df = pd.concat(chunks, ignore_index=True)
    print(f"Dataset created: {df.shape}")
    return df


def demo_dask_backend():
    """Demonstrate using Dask backend for large datasets."""
    if not DASK_AVAILABLE:
        print("Skipping Dask demo - library not available")
        return
        
    print("\n" + "="*60)
    print("DEMO: Using Dask Backend for Large Datasets")
    print("="*60)
    
    # Create synthetic data
    df_pandas = create_synthetic_large_dataset(n_rows=500_000, n_features=10)
    
    # Convert to Dask DataFrame
    print("\nConverting to Dask DataFrame...")
    from mlpy.backends import DataBackendDask
    
    # Create Dask DataFrame with partitions
    df_dask = dd.from_pandas(df_pandas, npartitions=10)
    print(f"Dask DataFrame created with {df_dask.npartitions} partitions")
    
    # Create Task with Dask backend
    print("\nCreating Task with Dask backend...")
    from mlpy.tasks import task_from_dask
    
    task = task_from_dask(df_dask, target='target', task_type='regression')
    print(f"Task created: {task}")
    
    # Note: Most learners expect data in memory, so we'll use a subset
    print("\nFor this demo, we'll train on a subset of data...")
    
    # Sample data for training (in practice, you'd use learners that support streaming)
    train_indices = np.random.choice(len(df_pandas), size=10_000, replace=False)
    task_subset = TaskRegr(
        id="dask_subset",
        data=df_pandas.iloc[train_indices],
        target="target"
    )
    
    # Train a simple model
    print("\nTraining Linear Regression on subset...")
    learner = LearnerRegrLM(id="lm")
    learner.train(task_subset)
    
    # Make predictions on another subset
    test_indices = np.random.choice(len(df_pandas), size=5_000, replace=False)
    task_test = TaskRegr(
        id="dask_test",
        data=df_pandas.iloc[test_indices],
        target="target"
    )
    
    predictions = learner.predict(task_test)
    
    # Evaluate
    rmse = MeasureRegrRMSE()
    mae = MeasureRegrMAE()
    
    print(f"\nResults on test subset:")
    print(f"  RMSE: {rmse.score(predictions):.4f}")
    print(f"  MAE: {mae.score(predictions):.4f}")
    
    # Clean up
    del df_pandas, df_dask


def demo_vaex_backend():
    """Demonstrate using Vaex backend for large datasets."""
    if not VAEX_AVAILABLE:
        print("Skipping Vaex demo - library not available")
        return
        
    print("\n" + "="*60)
    print("DEMO: Using Vaex Backend for Large Datasets")
    print("="*60)
    
    # Create synthetic data
    df_pandas = create_synthetic_large_dataset(n_rows=500_000, n_features=10)
    
    # Convert to Vaex DataFrame
    print("\nConverting to Vaex DataFrame...")
    from mlpy.backends import DataBackendVaex
    
    df_vaex = vaex.from_pandas(df_pandas)
    print(f"Vaex DataFrame created: {len(df_vaex):,} rows")
    
    # Create Task with Vaex backend
    print("\nCreating Task with Vaex backend...")
    from mlpy.tasks import task_from_vaex
    
    task = task_from_vaex(df_vaex, target='target', task_type='regression')
    print(f"Task created: {task}")
    
    # For demo, we'll use a subset
    print("\nFor this demo, we'll train on a subset of data...")
    
    # Sample data for training
    train_indices = np.random.choice(len(df_pandas), size=10_000, replace=False)
    task_subset = TaskRegr(
        id="vaex_subset",
        data=df_pandas.iloc[train_indices],
        target="target"
    )
    
    # Train a simple model
    print("\nTraining Linear Regression on subset...")
    learner = LearnerRegrLM(id="lm")
    learner.train(task_subset)
    
    # Make predictions
    test_indices = np.random.choice(len(df_pandas), size=5_000, replace=False)
    task_test = TaskRegr(
        id="vaex_test",
        data=df_pandas.iloc[test_indices],
        target="target"
    )
    
    predictions = learner.predict(task_test)
    
    # Evaluate
    rmse = MeasureRegrRMSE()
    mae = MeasureRegrMAE()
    
    print(f"\nResults on test subset:")
    print(f"  RMSE: {rmse.score(predictions):.4f}")
    print(f"  MAE: {mae.score(predictions):.4f}")
    
    # Clean up
    del df_pandas, df_vaex


def demo_lazy_loading():
    """Demonstrate lazy loading from files."""
    print("\n" + "="*60)
    print("DEMO: Lazy Loading from Files")
    print("="*60)
    
    # Create a sample CSV file
    print("\nCreating sample CSV file...")
    df_sample = create_synthetic_large_dataset(n_rows=100_000, n_features=5)
    df_sample.to_csv('large_dataset_demo.csv', index=False)
    print("Sample CSV file created: large_dataset_demo.csv")
    
    if DASK_AVAILABLE:
        print("\nLoading with Dask backend...")
        from mlpy.tasks import task_from_csv_lazy
        
        try:
            task_dask = task_from_csv_lazy(
                'large_dataset_demo.csv',
                target='target',
                backend='dask',
                task_type='regression'
            )
            print(f"Task created with Dask: {task_dask}")
            
            # Show backend info
            print(f"Backend type: {type(task_dask.backend)}")
            print(f"Number of rows: {task_dask.nrow}")
            print(f"Number of features: {task_dask.ncol - 1}")  # Exclude target
            
        except Exception as e:
            print(f"Error creating Dask task: {e}")
    
    if VAEX_AVAILABLE:
        print("\nLoading with Vaex backend...")
        try:
            task_vaex = task_from_csv_lazy(
                'large_dataset_demo.csv',
                target='target',
                backend='vaex',
                task_type='regression'
            )
            print(f"Task created with Vaex: {task_vaex}")
            
            # Show backend info
            print(f"Backend type: {type(task_vaex.backend)}")
            print(f"Number of rows: {task_vaex.nrow}")
            print(f"Number of features: {task_vaex.ncol - 1}")  # Exclude target
            
        except Exception as e:
            print(f"Error creating Vaex task: {e}")
    
    # Clean up
    import os
    if os.path.exists('large_dataset_demo.csv'):
        os.remove('large_dataset_demo.csv')
        print("\nCleaned up temporary file")


def main():
    """Run all demos."""
    print("MLPY Big Data Backends Demo")
    print("===========================")
    print(f"Dask available: {DASK_AVAILABLE}")
    print(f"Vaex available: {VAEX_AVAILABLE}")
    
    if not (DASK_AVAILABLE or VAEX_AVAILABLE):
        print("\nNo big data backends available!")
        print("Install with:")
        print("  pip install dask[dataframe]")
        print("  pip install vaex")
        return
    
    # Run demos
    demo_dask_backend()
    demo_vaex_backend()
    demo_lazy_loading()
    
    print("\n" + "="*60)
    print("Demo completed!")
    print("="*60)
    print("\nKey takeaways:")
    print("1. MLPY supports Dask and Vaex for datasets that don't fit in memory")
    print("2. Use task_from_dask() or task_from_vaex() for DataFrame conversion")
    print("3. Use task_from_csv_lazy() or task_from_parquet_lazy() for file loading")
    print("4. Most learners still expect in-memory data (subset or stream)")
    print("5. Future work: learners with native out-of-core support")


if __name__ == "__main__":
    main()