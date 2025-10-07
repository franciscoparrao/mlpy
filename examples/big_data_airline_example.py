"""
Example using MLPY with large airline dataset.

This example demonstrates working with the US airline on-time performance dataset,
which contains millions of flight records. It showcases:
- Loading large datasets with Dask/Vaex
- Lazy evaluation pipelines
- Advanced operators on big data
- Performance comparison

Dataset: https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp
"""

import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# MLPY imports
from mlpy.tasks import TaskClassif
from mlpy.backends import DataBackendDask, DataBackendVaex
from mlpy.tasks.big_data import create_dask_task, create_vaex_task
from mlpy.learners.sklearn import learner_sklearn
from mlpy.pipelines import (
    # Basic operators
    PipeOpScale, PipeOpEncode, PipeOpSelect,
    # Advanced operators
    PipeOpPCA, PipeOpTargetEncode, PipeOpOutlierDetect,
    PipeOpBin, PipeOpPolynomial,
    # Lazy operators
    LazyPipeOpScale, LazyPipeOpFilter, LazyPipeOpSample,
    # Pipeline utilities
    PipeOpLearner, linear_pipeline
)
from mlpy.resamplings import ResamplingHoldout
from mlpy.measures import MeasureClassifAcc, MeasureClassifAUC
from mlpy.resample import resample

# Optional imports
try:
    import dask.dataframe as dd
    import dask.distributed
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    print("Warning: Dask not available. Install with: pip install dask[complete]")

try:
    import vaex
    VAEX_AVAILABLE = True
except ImportError:
    VAEX_AVAILABLE = False
    print("Warning: Vaex not available. Install with: pip install vaex")

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import SGDClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available")


def download_airline_data(year=2019, month=1):
    """
    Download airline on-time performance data.
    
    Note: This is a simplified example. In practice, you would download
    the full dataset from the Bureau of Transportation Statistics.
    """
    # For this example, we'll create synthetic data that mimics airline data
    print(f"Creating synthetic airline data for {year}-{month:02d}...")
    
    n_flights = 500000  # Half million flights
    
    # Create synthetic data
    import numpy as np
    np.random.seed(42)
    
    # Airlines
    airlines = ['AA', 'DL', 'UA', 'WN', 'B6', 'AS', 'NK', 'F9']
    airline_weights = [0.2, 0.18, 0.17, 0.2, 0.08, 0.07, 0.05, 0.05]
    
    # Airports (major hubs)
    airports = ['ATL', 'ORD', 'LAX', 'DFW', 'DEN', 'JFK', 'SFO', 'SEA', 
                'LAS', 'MCO', 'PHX', 'IAH', 'BOS', 'MSP', 'DTW', 'CLT']
    
    # Day of week effect on delays
    dow_delay_prob = [0.15, 0.12, 0.13, 0.14, 0.20, 0.25, 0.18]  # Higher on Fri/Sun
    
    data = {
        'Year': year,
        'Month': month,
        'DayOfMonth': np.random.randint(1, 29, n_flights),
        'DayOfWeek': np.random.randint(1, 8, n_flights),
        'Airline': np.random.choice(airlines, n_flights, p=airline_weights),
        'Origin': np.random.choice(airports, n_flights),
        'Dest': np.random.choice(airports, n_flights),
        'ScheduledDepTime': np.random.randint(0, 2400, n_flights),
        'Distance': np.random.exponential(800, n_flights),
        'ScheduledElapsedTime': np.random.normal(150, 60, n_flights),
    }
    
    # Create delay based on various factors
    delay_prob = np.zeros(n_flights)
    
    # Day of week effect
    for i, dow in enumerate(data['DayOfWeek']):
        delay_prob[i] += dow_delay_prob[dow - 1]
    
    # Time of day effect (more delays in evening)
    evening_mask = data['ScheduledDepTime'] > 1700
    delay_prob[evening_mask] += 0.1
    
    # Distance effect (longer flights more likely to recover from delays)
    long_flight_mask = data['Distance'] > 1500
    delay_prob[long_flight_mask] -= 0.05
    
    # Airline effect
    airline_delay_rates = {'NK': 0.25, 'F9': 0.22, 'B6': 0.18}
    for airline, rate in airline_delay_rates.items():
        airline_mask = data['Airline'] == airline
        delay_prob[airline_mask] += rate - 0.15  # Adjust relative to average
    
    # Add some randomness
    delay_prob += np.random.normal(0, 0.05, n_flights)
    delay_prob = np.clip(delay_prob, 0, 1)
    
    # Create delay flag (15+ minutes)
    data['ArrDelay'] = np.where(
        np.random.random(n_flights) < delay_prob,
        np.random.exponential(25, n_flights),  # Delayed flights
        np.random.normal(0, 5, n_flights)  # On-time flights
    )
    data['IsDelayed'] = (data['ArrDelay'] > 15).astype(int)
    
    # Additional features
    data['CarrierDelay'] = np.where(data['IsDelayed'], np.random.exponential(10, n_flights), 0)
    data['WeatherDelay'] = np.where(data['IsDelayed'], np.random.exponential(5, n_flights), 0)
    data['NASDelay'] = np.where(data['IsDelayed'], np.random.exponential(8, n_flights), 0)
    data['TaxiOut'] = np.random.exponential(15, n_flights)
    data['TaxiIn'] = np.random.exponential(7, n_flights)
    
    df = pd.DataFrame(data)
    
    # Save as CSV
    filename = f"airline_data_{year}_{month:02d}.csv"
    df.to_csv(filename, index=False)
    print(f"Created {filename} with {len(df):,} flights")
    
    return filename


def example_dask_airline_pipeline():
    """Example: Large-scale airline delay prediction with Dask."""
    if not DASK_AVAILABLE:
        print("Dask not available. Skipping example.")
        return
        
    print("\n" + "="*60)
    print("EXAMPLE: Airline Delay Prediction with Dask")
    print("="*60)
    
    # Create/load data
    filename = download_airline_data(2019, 1)
    
    # Start Dask client for parallel processing
    print("\nStarting Dask client...")
    client = dask.distributed.Client(n_workers=2, threads_per_worker=2)
    print(f"Dashboard: {client.dashboard_link}")
    
    try:
        # Load data with Dask
        print(f"\nLoading data with Dask...")
        start_time = time.time()
        
        ddf = dd.read_csv(filename, blocksize='50MB')
        
        # Create task
        task = create_dask_task(
            data=ddf,
            target='IsDelayed',
            task_type='classif',
            task_id='airline_delays'
        )
        
        load_time = time.time() - start_time
        print(f"Data loaded in {load_time:.2f} seconds")
        print(f"Dataset shape: {task.nrow:,} flights, {task.ncol} features")
        
        # Build pipeline with lazy operations
        print("\nBuilding lazy evaluation pipeline...")
        pipeline = linear_pipeline(
            # 1. Filter out cancelled flights (if any)
            LazyPipeOpFilter(
                id="filter",
                condition=lambda df: df['ArrDelay'].notna()
            ),
            
            # 2. Sample for faster experimentation
            LazyPipeOpSample(
                id="sample",
                fraction=0.1,  # Use 10% for faster demo
                random_state=42
            ),
            
            # 3. Target encode high-cardinality features
            PipeOpTargetEncode(
                id="target_enc",
                columns=['Airline', 'Origin', 'Dest'],
                smoothing=100  # More smoothing for stability
            ),
            
            # 4. Create time-based features
            PipeOpBin(
                id="time_bins",
                columns=['ScheduledDepTime'],
                n_bins=24,  # Hourly bins
                strategy='uniform',
                encode='onehot'
            ),
            
            # 5. Create polynomial features for key numerics
            PipeOpPolynomial(
                id="poly",
                columns=['Distance', 'ScheduledElapsedTime'],
                degree=2,
                interaction_only=True
            ),
            
            # 6. Scale features
            LazyPipeOpScale(
                id="scale",
                method="robust"  # Robust to outliers
            ),
            
            # 7. Reduce dimensionality
            PipeOpPCA(
                id="pca",
                n_components=0.95  # Keep 95% variance
            ),
            
            # 8. Learn with SGD (supports partial_fit for streaming)
            PipeOpLearner(
                learner_sklearn(
                    SGDClassifier(
                        loss='log',
                        penalty='elasticnet',
                        max_iter=1000,
                        random_state=42
                    )
                ),
                id="sgd"
            )
        )
        
        # Train and evaluate
        print("\nTraining pipeline (this may take a moment)...")
        start_time = time.time()
        
        result = resample(
            task=task,
            learner=pipeline,
            resampling=ResamplingHoldout(ratio=0.8),
            measure=[MeasureClassifAcc(), MeasureClassifAUC()]
        )
        
        train_time = time.time() - start_time
        
        # Results
        metrics = result.aggregate()
        print(f"\nTraining completed in {train_time:.2f} seconds")
        print(f"Accuracy: {metrics['acc'][0]:.3f}")
        print(f"AUC: {metrics['auc'][0]:.3f}")
        
        # Analyze feature importance
        print("\nAnalyzing pipeline...")
        pipeline.train(task)
        
        # Get PCA info
        pca_op = pipeline.pipeops['pca']
        n_components = pca_op.state['n_components']
        variance_explained = sum(pca_op.state['explained_variance_ratio'])
        
        print(f"PCA reduced features to {n_components} components")
        print(f"Variance explained: {variance_explained:.1%}")
        
    finally:
        client.close()
        # Clean up
        if os.path.exists(filename):
            os.remove(filename)


def example_vaex_airline_analysis():
    """Example: Interactive airline data analysis with Vaex."""
    if not VAEX_AVAILABLE:
        print("Vaex not available. Skipping example.")
        return
        
    print("\n" + "="*60)
    print("EXAMPLE: Interactive Airline Analysis with Vaex")
    print("="*60)
    
    # Create data for multiple months
    print("\nCreating multi-month dataset...")
    filenames = []
    for month in range(1, 4):  # Q1 data
        filename = download_airline_data(2019, month)
        filenames.append(filename)
    
    try:
        # Convert to HDF5 for Vaex (memory-mapped)
        print("\nConverting to Vaex format (HDF5)...")
        
        # Read all CSVs and concatenate
        dfs = []
        for filename in filenames:
            df = pd.read_csv(filename)
            dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Save as HDF5
        hdf5_file = "airline_data_q1_2019.hdf5"
        combined_df.to_hdf(hdf5_file, key='data', mode='w')
        
        # Open with Vaex
        vdf = vaex.open(hdf5_file)
        print(f"Loaded {len(vdf):,} flights with Vaex")
        
        # Create task
        task = create_vaex_task(
            data=vdf,
            target='IsDelayed',
            task_type='classif',
            task_id='airline_q1'
        )
        
        # Build analysis pipeline
        print("\nBuilding analysis pipeline...")
        pipeline = linear_pipeline(
            # 1. Remove outliers in delays
            PipeOpOutlierDetect(
                id="outlier",
                method="isolation",
                contamination=0.01,
                action="flag"
            ),
            
            # 2. Encode categoricals efficiently
            PipeOpTargetEncode(
                id="encode",
                columns=['Airline', 'Origin', 'Dest', 'DayOfWeek'],
                smoothing=50
            ),
            
            # 3. Create delay risk features
            PipeOpBin(
                id="risk_bins",
                columns=['ScheduledDepTime', 'Distance'],
                n_bins=10,
                strategy='quantile',
                encode='ordinal'
            ),
            
            # 4. Scale
            PipeOpScale(id="scale"),
            
            # 5. Fast random forest
            PipeOpLearner(
                learner_sklearn(
                    RandomForestClassifier(
                        n_estimators=50,
                        max_depth=10,
                        n_jobs=-1,
                        random_state=42
                    )
                ),
                id="rf"
            )
        )
        
        # Train on sample for speed
        print("\nTraining on data sample...")
        start_time = time.time()
        
        # Vaex can efficiently sample
        sample_indices = vdf.sample(n=100000, random_state=42).index.values
        sample_task = task.filter_rows(sample_indices)
        
        result = resample(
            task=sample_task,
            learner=pipeline,
            resampling=ResamplingHoldout(ratio=0.8),
            measure=MeasureClassifAcc()
        )
        
        train_time = time.time() - start_time
        
        print(f"Training completed in {train_time:.2f} seconds")
        print(f"Sample accuracy: {result.aggregate()['acc'][0]:.3f}")
        
        # Analyze delay patterns
        print("\n" + "-"*40)
        print("DELAY PATTERN ANALYSIS")
        print("-"*40)
        
        # By airline
        print("\nDelay rate by airline:")
        for airline in ['AA', 'DL', 'UA', 'WN', 'B6']:
            mask = vdf.Airline == airline
            delay_rate = vdf[mask].IsDelayed.mean()
            n_flights = mask.sum()
            print(f"  {airline}: {delay_rate:.1%} ({n_flights:,} flights)")
        
        # By day of week
        print("\nDelay rate by day of week:")
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        for i, day in enumerate(days, 1):
            mask = vdf.DayOfWeek == i
            delay_rate = vdf[mask].IsDelayed.mean()
            print(f"  {day}: {delay_rate:.1%}")
        
        # By time of day
        print("\nDelay rate by time of day:")
        time_bins = [(0, 600, "Early Morning"), 
                     (600, 1200, "Morning"),
                     (1200, 1800, "Afternoon"), 
                     (1800, 2400, "Evening")]
        
        for start, end, period in time_bins:
            mask = (vdf.ScheduledDepTime >= start) & (vdf.ScheduledDepTime < end)
            delay_rate = vdf[mask].IsDelayed.mean()
            print(f"  {period}: {delay_rate:.1%}")
            
    finally:
        # Clean up
        for filename in filenames:
            if os.path.exists(filename):
                os.remove(filename)
        if os.path.exists(hdf5_file):
            os.remove(hdf5_file)


def example_performance_comparison():
    """Compare performance of different backends."""
    print("\n" + "="*60)
    print("EXAMPLE: Backend Performance Comparison")
    print("="*60)
    
    # Create test data
    filename = download_airline_data(2019, 1)
    
    try:
        results = {}
        
        # 1. Pandas (baseline)
        print("\n1. Testing Pandas backend...")
        start_time = time.time()
        
        df = pd.read_csv(filename)
        # Sample for fairness (Pandas loads everything)
        df_sample = df.sample(n=50000, random_state=42)
        
        task = TaskClassif(data=df_sample, target='IsDelayed')
        
        # Simple pipeline
        pipeline = linear_pipeline(
            PipeOpEncode(id="encode", method="target"),
            PipeOpScale(id="scale"),
            PipeOpLearner(
                learner_sklearn(RandomForestClassifier(n_estimators=10)),
                id="rf"
            )
        )
        
        result = resample(
            task=task,
            learner=pipeline,
            resampling=ResamplingHoldout(ratio=0.8),
            measure=MeasureClassifAcc()
        )
        
        pandas_time = time.time() - start_time
        results['Pandas'] = {
            'time': pandas_time,
            'accuracy': result.aggregate()['acc'][0]
        }
        
        # 2. Dask
        if DASK_AVAILABLE:
            print("\n2. Testing Dask backend...")
            client = dask.distributed.Client(
                n_workers=2, 
                threads_per_worker=2,
                silence_logs=50
            )
            
            try:
                start_time = time.time()
                
                ddf = dd.read_csv(filename, blocksize='25MB')
                ddf_sample = ddf.sample(frac=0.1, random_state=42)
                
                task = create_dask_task(
                    data=ddf_sample,
                    target='IsDelayed',
                    task_type='classif'
                )
                
                # Use lazy operations
                pipeline = linear_pipeline(
                    PipeOpTargetEncode(id="encode"),
                    LazyPipeOpScale(id="scale"),
                    PipeOpLearner(
                        learner_sklearn(RandomForestClassifier(n_estimators=10)),
                        id="rf"
                    )
                )
                
                result = resample(
                    task=task,
                    learner=pipeline,
                    resampling=ResamplingHoldout(ratio=0.8),
                    measure=MeasureClassifAcc()
                )
                
                dask_time = time.time() - start_time
                results['Dask'] = {
                    'time': dask_time,
                    'accuracy': result.aggregate()['acc'][0]
                }
                
            finally:
                client.close()
        
        # 3. Vaex
        if VAEX_AVAILABLE:
            print("\n3. Testing Vaex backend...")
            start_time = time.time()
            
            # Convert to HDF5 for Vaex
            df = pd.read_csv(filename)
            hdf5_file = "temp_airline.hdf5"
            df.to_hdf(hdf5_file, key='data', mode='w')
            
            vdf = vaex.open(hdf5_file)
            vdf_sample = vdf.sample(n=50000, random_state=42)
            
            task = create_vaex_task(
                data=vdf_sample,
                target='IsDelayed',
                task_type='classif'
            )
            
            pipeline = linear_pipeline(
                PipeOpTargetEncode(id="encode"),
                PipeOpScale(id="scale"),
                PipeOpLearner(
                    learner_sklearn(RandomForestClassifier(n_estimators=10)),
                    id="rf"
                )
            )
            
            result = resample(
                task=task,
                learner=pipeline,
                resampling=ResamplingHoldout(ratio=0.8),
                measure=MeasureClassifAcc()
            )
            
            vaex_time = time.time() - start_time
            results['Vaex'] = {
                'time': vaex_time,
                'accuracy': result.aggregate()['acc'][0]
            }
            
            os.remove(hdf5_file)
        
        # Display results
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON RESULTS")
        print("="*60)
        
        print(f"\n{'Backend':<10} {'Time (s)':<12} {'Accuracy':<12} {'Speedup':<10}")
        print("-" * 45)
        
        baseline_time = results['Pandas']['time']
        for backend, metrics in results.items():
            speedup = baseline_time / metrics['time']
            print(f"{backend:<10} {metrics['time']:<12.2f} "
                  f"{metrics['accuracy']:<12.3f} {speedup:<10.2f}x")
        
        # Visualization
        if len(results) > 1:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Time comparison
            backends = list(results.keys())
            times = [results[b]['time'] for b in backends]
            
            ax1.bar(backends, times)
            ax1.set_xlabel('Backend')
            ax1.set_ylabel('Time (seconds)')
            ax1.set_title('Processing Time Comparison')
            
            # Accuracy comparison
            accuracies = [results[b]['accuracy'] for b in backends]
            
            ax2.bar(backends, accuracies)
            ax2.set_xlabel('Backend')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Model Accuracy Comparison')
            ax2.set_ylim(min(accuracies) * 0.95, max(accuracies) * 1.02)
            
            plt.tight_layout()
            plt.show()
            
    finally:
        if os.path.exists(filename):
            os.remove(filename)


def main():
    """Run all big data examples."""
    print("MLPY Big Data Examples: Airline Dataset")
    print("="*40)
    
    if not SKLEARN_AVAILABLE:
        print("\nError: scikit-learn required for these examples")
        return
    
    if not DASK_AVAILABLE and not VAEX_AVAILABLE:
        print("\nError: At least one big data backend (Dask or Vaex) required")
        print("Install with:")
        print("  pip install dask[complete]")
        print("  pip install vaex")
        return
    
    # Run examples
    if DASK_AVAILABLE:
        example_dask_airline_pipeline()
    
    if VAEX_AVAILABLE:
        example_vaex_airline_analysis()
    
    # Performance comparison
    example_performance_comparison()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)
    
    print("\nKey takeaways:")
    print("1. Dask enables distributed processing of datasets larger than memory")
    print("2. Vaex provides lightning-fast exploration of billion-row datasets")
    print("3. Lazy evaluation reduces memory usage and improves performance")
    print("4. MLPY's unified API works seamlessly across different backends")
    print("5. Advanced operators (PCA, target encoding) work with big data")
    print("\nUse Dask for complex distributed ML pipelines")
    print("Use Vaex for interactive exploration and visualization")


if __name__ == "__main__":
    # Set plotting style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    main()