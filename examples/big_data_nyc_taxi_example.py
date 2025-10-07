"""
Example using MLPY with NYC Taxi dataset.

This example demonstrates working with the NYC Taxi trip record dataset,
which contains millions of taxi trips. It showcases:
- Memory-efficient data loading
- Geospatial feature engineering  
- Time series patterns
- Fare prediction at scale

Dataset info: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
"""

import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# MLPY imports
from mlpy.tasks import TaskRegr
from mlpy.backends import DataBackendDask, DataBackendVaex
from mlpy.tasks.big_data import create_dask_task, create_vaex_task
from mlpy.learners.sklearn import learner_sklearn
from mlpy.pipelines import (
    # Basic operators
    PipeOpScale, PipeOpSelect, PipeOpImpute,
    # Advanced operators  
    PipeOpOutlierDetect, PipeOpBin, PipeOpPolynomial,
    PipeOpTargetEncode,
    # Lazy operators
    LazyPipeOpFilter, LazyPipeOpSample, LazyPipeOpScale,
    # Pipeline utilities
    PipeOpLearner, linear_pipeline
)
from mlpy.resamplings import ResamplingHoldout, ResamplingCV
from mlpy.measures import MeasureRegrRMSE, MeasureRegrMAE, MeasureRegrR2
from mlpy.resample import resample

# Optional imports
try:
    import dask.dataframe as dd
    import dask.distributed
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

try:
    import vaex
    VAEX_AVAILABLE = True
except ImportError:
    VAEX_AVAILABLE = False

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def create_synthetic_taxi_data(n_trips=1000000):
    """
    Create synthetic NYC taxi data for demonstration.
    
    In practice, you would download the real dataset from:
    https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
    """
    print(f"Creating synthetic NYC taxi data with {n_trips:,} trips...")
    
    np.random.seed(42)
    
    # Time range (one month)
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 31)
    
    # Generate pickup times
    pickup_times = []
    for _ in range(n_trips):
        # More trips during rush hours and weekends
        hour = np.random.choice(24, p=get_hourly_distribution())
        day_offset = np.random.randint(0, 31)
        pickup_time = start_date + timedelta(days=day_offset, hours=hour, 
                                           minutes=np.random.randint(0, 60))
        pickup_times.append(pickup_time)
    
    # NYC bounding box (approximate)
    lat_min, lat_max = 40.5, 40.9
    lon_min, lon_max = -74.3, -73.7
    
    # Generate locations (concentrated around Manhattan)
    pickup_lat = np.random.normal(40.75, 0.05, n_trips)
    pickup_lon = np.random.normal(-73.98, 0.05, n_trips)
    dropoff_lat = np.random.normal(40.75, 0.05, n_trips)
    dropoff_lon = np.random.normal(-73.98, 0.05, n_trips)
    
    # Clip to NYC bounds
    pickup_lat = np.clip(pickup_lat, lat_min, lat_max)
    pickup_lon = np.clip(pickup_lon, lon_min, lon_max)
    dropoff_lat = np.clip(dropoff_lat, lat_min, lat_max)
    dropoff_lon = np.clip(dropoff_lon, lon_min, lon_max)
    
    # Calculate distances (Haversine approximation)
    distances = haversine_distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
    
    # Trip duration based on distance and time of day
    base_duration = distances * 180  # 3 minutes per km base
    
    # Add traffic effects
    traffic_multiplier = np.ones(n_trips)
    for i, dt in enumerate(pickup_times):
        if dt.weekday() < 5:  # Weekday
            if 7 <= dt.hour <= 9 or 17 <= dt.hour <= 19:  # Rush hour
                traffic_multiplier[i] = np.random.uniform(1.5, 2.5)
            elif 10 <= dt.hour <= 16:  # Daytime
                traffic_multiplier[i] = np.random.uniform(1.2, 1.5)
        else:  # Weekend
            if 12 <= dt.hour <= 20:  # Busy weekend times
                traffic_multiplier[i] = np.random.uniform(1.3, 1.8)
    
    trip_duration = base_duration * traffic_multiplier + np.random.normal(0, 60, n_trips)
    trip_duration = np.maximum(trip_duration, 60)  # Minimum 1 minute
    
    # Calculate fare
    base_fare = 2.50
    per_km_rate = 2.50
    per_minute_rate = 0.50
    
    fare_amount = (base_fare + 
                   distances * per_km_rate + 
                   (trip_duration / 60) * per_minute_rate)
    
    # Add surge pricing for high-demand times
    surge_multiplier = np.ones(n_trips)
    for i, dt in enumerate(pickup_times):
        if dt.weekday() >= 5 and 22 <= dt.hour:  # Weekend nights
            surge_multiplier[i] = np.random.uniform(1.2, 2.0)
        elif dt.weekday() < 5 and (7 <= dt.hour <= 9):  # Weekday morning rush
            surge_multiplier[i] = np.random.uniform(1.1, 1.5)
    
    fare_amount = fare_amount * surge_multiplier
    
    # Add tips (correlated with fare amount)
    tip_percentage = np.random.beta(2, 5, n_trips) * 0.3  # 0-30% tips
    tip_amount = fare_amount * tip_percentage
    
    # Total amount
    total_amount = fare_amount + tip_amount + np.random.uniform(0, 3, n_trips)  # Small fees
    
    # Create DataFrame
    data = pd.DataFrame({
        'pickup_datetime': pickup_times,
        'pickup_latitude': pickup_lat,
        'pickup_longitude': pickup_lon,
        'dropoff_latitude': dropoff_lat,
        'dropoff_longitude': dropoff_lon,
        'trip_distance': distances,
        'trip_duration': trip_duration,
        'passenger_count': np.random.choice([1, 2, 3, 4], n_trips, p=[0.7, 0.2, 0.08, 0.02]),
        'payment_type': np.random.choice(['Credit', 'Cash', 'App'], n_trips, p=[0.7, 0.2, 0.1]),
        'fare_amount': fare_amount,
        'tip_amount': tip_amount,
        'total_amount': total_amount
    })
    
    # Save to CSV
    filename = 'nyc_taxi_trips.csv'
    data.to_csv(filename, index=False)
    print(f"Created {filename} with {len(data):,} trips")
    
    return filename


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points on Earth (in km)."""
    R = 6371  # Earth's radius in km
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    
    a = (np.sin(delta_lat/2)**2 + 
         np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c


def get_hourly_distribution():
    """Get probability distribution for taxi trips by hour."""
    # Approximate NYC taxi demand pattern
    hourly_probs = np.array([
        0.02, 0.01, 0.01, 0.01, 0.02, 0.03,  # 0-5 AM
        0.04, 0.06, 0.08, 0.07, 0.06, 0.05,  # 6-11 AM
        0.05, 0.05, 0.05, 0.04, 0.04, 0.05,  # 12-5 PM
        0.07, 0.08, 0.06, 0.05, 0.04, 0.03   # 6-11 PM
    ])
    return hourly_probs / hourly_probs.sum()


def example_dask_taxi_fare_prediction():
    """Example: Taxi fare prediction with Dask."""
    if not DASK_AVAILABLE:
        print("Dask not available. Skipping example.")
        return
        
    print("\n" + "="*60)
    print("EXAMPLE: NYC Taxi Fare Prediction with Dask")
    print("="*60)
    
    # Create data
    filename = create_synthetic_taxi_data(n_trips=500000)
    
    # Start Dask client
    print("\nStarting Dask client...")
    client = dask.distributed.Client(n_workers=2, threads_per_worker=2, silence_logs=50)
    
    try:
        # Load with Dask
        print("Loading data with Dask...")
        
        # Parse dates while reading
        ddf = dd.read_csv(
            filename,
            parse_dates=['pickup_datetime'],
            blocksize='25MB'
        )
        
        # Create engineered features
        print("Engineering features...")
        
        # Time-based features
        ddf['hour'] = ddf['pickup_datetime'].dt.hour
        ddf['day_of_week'] = ddf['pickup_datetime'].dt.dayofweek
        ddf['is_weekend'] = (ddf['day_of_week'] >= 5).astype(int)
        
        # Rush hour flags
        ddf['morning_rush'] = ((ddf['hour'] >= 7) & (ddf['hour'] <= 9)).astype(int)
        ddf['evening_rush'] = ((ddf['hour'] >= 17) & (ddf['hour'] <= 19)).astype(int)
        
        # Create task (predict total amount)
        task = create_dask_task(
            data=ddf,
            target='total_amount',
            task_type='regr',
            task_id='taxi_fare'
        )
        
        # Build pipeline
        print("Building ML pipeline...")
        pipeline = linear_pipeline(
            # 1. Filter invalid trips
            LazyPipeOpFilter(
                id="filter",
                condition=lambda df: (
                    (df['trip_distance'] > 0) & 
                    (df['trip_distance'] < 100) &  # Remove outliers
                    (df['total_amount'] > 0) & 
                    (df['total_amount'] < 500)
                )
            ),
            
            # 2. Remove extreme outliers
            PipeOpOutlierDetect(
                id="outliers",
                method="isolation",
                contamination=0.01,
                action="remove"
            ),
            
            # 3. Target encode categorical features
            PipeOpTargetEncode(
                id="encode",
                columns=['payment_type'],
                smoothing=100
            ),
            
            # 4. Bin continuous features
            PipeOpBin(
                id="location_bins",
                columns=['pickup_latitude', 'pickup_longitude', 
                        'dropoff_latitude', 'dropoff_longitude'],
                n_bins=20,  # 20x20 grid for each
                strategy='quantile',
                encode='ordinal'
            ),
            
            # 5. Create polynomial features for key interactions
            PipeOpPolynomial(
                id="poly",
                columns=['trip_distance', 'hour', 'is_weekend'],
                degree=2,
                interaction_only=True
            ),
            
            # 6. Scale features
            LazyPipeOpScale(
                id="scale",
                method="robust"
            ),
            
            # 7. Learn with gradient boosting
            PipeOpLearner(
                learner_sklearn(
                    GradientBoostingRegressor(
                        n_estimators=100,
                        max_depth=5,
                        learning_rate=0.1,
                        subsample=0.8,
                        random_state=42
                    )
                ),
                id="gbr"
            )
        )
        
        # Sample for faster training
        print("\nSampling data for training...")
        pipeline_sample = linear_pipeline(
            LazyPipeOpSample(id="sample", fraction=0.2, random_state=42),
            pipeline
        )
        
        # Train and evaluate
        print("Training model...")
        start_time = time.time()
        
        result = resample(
            task=task,
            learner=pipeline_sample,
            resampling=ResamplingHoldout(ratio=0.8),
            measure=[MeasureRegrRMSE(), MeasureRegrMAE(), MeasureRegrR2()]
        )
        
        train_time = time.time() - start_time
        
        # Results
        metrics = result.aggregate()
        print(f"\nTraining completed in {train_time:.2f} seconds")
        print(f"RMSE: ${metrics['rmse'][0]:.2f}")
        print(f"MAE: ${metrics['mae'][0]:.2f}")
        print(f"R²: {metrics['r2'][0]:.3f}")
        
        # Feature importance analysis
        print("\nAnalyzing feature importance...")
        pipeline_sample.train(task)
        
        gbr = pipeline_sample.pipeops['gbr']._trained_learner._model
        if hasattr(gbr, 'feature_importances_'):
            feature_names = pipeline_sample.pipeops['gbr']._trained_learner._feature_names
            importances = gbr.feature_importances_
            
            # Get top 10 features
            top_indices = np.argsort(importances)[-10:][::-1]
            
            print("\nTop 10 most important features:")
            for i, idx in enumerate(top_indices):
                print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.3f}")
                
    finally:
        client.close()
        if os.path.exists(filename):
            os.remove(filename)


def example_vaex_taxi_exploration():
    """Example: Interactive taxi data exploration with Vaex."""
    if not VAEX_AVAILABLE:
        print("Vaex not available. Skipping example.")
        return
        
    print("\n" + "="*60)
    print("EXAMPLE: NYC Taxi Data Exploration with Vaex")
    print("="*60)
    
    # Create larger dataset for exploration
    filename = create_synthetic_taxi_data(n_trips=2000000)
    
    try:
        # Convert to HDF5 for Vaex
        print("\nConverting to Vaex format...")
        df = pd.read_csv(filename, parse_dates=['pickup_datetime'])
        
        # Add computed columns
        df['hour'] = df['pickup_datetime'].dt.hour
        df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        hdf5_file = 'nyc_taxi_trips.hdf5'
        df.to_hdf(hdf5_file, key='data', mode='w')
        
        # Open with Vaex
        vdf = vaex.open(hdf5_file)
        print(f"Loaded {len(vdf):,} trips with Vaex")
        
        # Explore patterns
        print("\n" + "-"*40)
        print("DATA EXPLORATION")
        print("-"*40)
        
        # Basic statistics
        print("\nTrip statistics:")
        print(f"Average distance: {vdf.trip_distance.mean():.2f} km")
        print(f"Average duration: {vdf.trip_duration.mean()/60:.1f} minutes")
        print(f"Average fare: ${vdf.total_amount.mean():.2f}")
        
        # Patterns by time
        print("\nAverage fare by hour of day:")
        for hour in [0, 6, 12, 18]:
            mask = vdf.hour == hour
            avg_fare = vdf[mask].total_amount.mean()
            n_trips = mask.sum()
            print(f"  {hour:02d}:00 - ${avg_fare:.2f} ({n_trips:,} trips)")
        
        # Weekend vs weekday
        print("\nWeekend vs Weekday comparison:")
        weekend_mask = vdf.is_weekend == 1
        weekday_mask = vdf.is_weekend == 0
        
        print(f"  Weekend: ${vdf[weekend_mask].total_amount.mean():.2f} avg fare")
        print(f"  Weekday: ${vdf[weekday_mask].total_amount.mean():.2f} avg fare")
        
        # Payment type analysis
        print("\nPayment type distribution:")
        for ptype in ['Credit', 'Cash', 'App']:
            mask = vdf.payment_type == ptype
            pct = (mask.sum() / len(vdf)) * 100
            avg_tip = vdf[mask].tip_amount.mean()
            print(f"  {ptype}: {pct:.1f}% (avg tip: ${avg_tip:.2f})")
        
        # Create task for modeling
        print("\nCreating prediction task...")
        task = create_vaex_task(
            data=vdf,
            target='tip_amount',  # Predict tips
            task_type='regr',
            task_id='taxi_tips'
        )
        
        # Build exploration pipeline
        pipeline = linear_pipeline(
            # Focus on credit card payments (have tip data)
            LazyPipeOpFilter(
                id="filter",
                condition=lambda df: df['payment_type'] == 'Credit'
            ),
            
            # Remove zero tips (likely missing data)
            LazyPipeOpFilter(
                id="filter2", 
                condition=lambda df: df['tip_amount'] > 0
            ),
            
            # Encode time features
            PipeOpBin(
                id="time_bins",
                columns=['hour'],
                n_bins=6,  # 4-hour blocks
                strategy='uniform',
                encode='onehot'
            ),
            
            # Location encoding
            PipeOpBin(
                id="location",
                columns=['pickup_latitude', 'pickup_longitude'],
                n_bins=10,
                strategy='kmeans',  # Cluster-based
                encode='ordinal'
            ),
            
            # Scale
            PipeOpScale(id="scale"),
            
            # Random Forest for interpretability
            PipeOpLearner(
                learner_sklearn(
                    RandomForestRegressor(
                        n_estimators=50,
                        max_depth=10,
                        n_jobs=-1,
                        random_state=42
                    )
                ),
                id="rf"
            )
        )
        
        # Train on sample
        print("\nTraining tip prediction model...")
        sample_indices = vdf.sample(n=50000, random_state=42).index.values
        sample_task = task.filter_rows(sample_indices)
        
        result = resample(
            task=sample_task,
            learner=pipeline,
            resampling=ResamplingCV(folds=3),
            measure=[MeasureRegrRMSE(), MeasureRegrR2()]
        )
        
        metrics = result.aggregate()
        print(f"\nTip prediction performance:")
        print(f"RMSE: ${metrics['rmse'][0]:.2f}")
        print(f"R²: {metrics['r2'][0]:.3f}")
        
        # Visualizations
        print("\nCreating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Trips by hour
        hours = list(range(24))
        trip_counts = [vdf[vdf.hour == h].count() for h in hours]
        
        axes[0, 0].bar(hours, trip_counts)
        axes[0, 0].set_xlabel('Hour of Day')
        axes[0, 0].set_ylabel('Number of Trips')
        axes[0, 0].set_title('Trip Volume by Hour')
        
        # 2. Average fare by distance bins
        dist_bins = [0, 2, 5, 10, 20, 50]
        avg_fares = []
        for i in range(len(dist_bins)-1):
            mask = (vdf.trip_distance >= dist_bins[i]) & (vdf.trip_distance < dist_bins[i+1])
            avg_fares.append(vdf[mask].total_amount.mean())
        
        axes[0, 1].bar(range(len(avg_fares)), avg_fares)
        axes[0, 1].set_xlabel('Distance Range (km)')
        axes[0, 1].set_ylabel('Average Fare ($)')
        axes[0, 1].set_title('Fare by Distance')
        axes[0, 1].set_xticklabels([f'{dist_bins[i]}-{dist_bins[i+1]}' 
                                    for i in range(len(dist_bins)-1)])
        
        # 3. Tip percentage distribution
        # Sample for histogram
        sample_tips = vdf.sample(n=10000, random_state=42)
        tip_pct = (sample_tips['tip_amount'] / sample_tips['fare_amount'] * 100).values
        tip_pct = tip_pct[(tip_pct >= 0) & (tip_pct <= 50)]  # Remove outliers
        
        axes[1, 0].hist(tip_pct, bins=50, edgecolor='black')
        axes[1, 0].set_xlabel('Tip Percentage')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Tip Percentage Distribution')
        
        # 4. Weekend vs weekday patterns
        categories = ['Weekday', 'Weekend']
        avg_distance = [
            vdf[vdf.is_weekend == 0].trip_distance.mean(),
            vdf[vdf.is_weekend == 1].trip_distance.mean()
        ]
        avg_fare = [
            vdf[vdf.is_weekend == 0].total_amount.mean(),
            vdf[vdf.is_weekend == 1].total_amount.mean()
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, avg_distance, width, label='Avg Distance (km)')
        axes[1, 1].bar(x + width/2, avg_fare, width, label='Avg Fare ($)')
        axes[1, 1].set_xlabel('Day Type')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(categories)
        axes[1, 1].legend()
        axes[1, 1].set_title('Weekend vs Weekday Patterns')
        
        plt.tight_layout()
        plt.show()
        
    finally:
        if os.path.exists(filename):
            os.remove(filename)
        if os.path.exists(hdf5_file):
            os.remove(hdf5_file)


def example_streaming_taxi_pipeline():
    """Example: Streaming/incremental learning with taxi data."""
    if not DASK_AVAILABLE:
        print("Dask not available. Skipping example.")
        return
        
    print("\n" + "="*60)
    print("EXAMPLE: Streaming Taxi Data Pipeline")
    print("="*60)
    
    # Simulate streaming data
    print("Simulating streaming taxi data...")
    
    # Create initial batch
    filename = create_synthetic_taxi_data(n_trips=100000)
    
    client = dask.distributed.Client(n_workers=2, threads_per_worker=2, silence_logs=50)
    
    try:
        # Load initial data
        ddf = dd.read_csv(filename, parse_dates=['pickup_datetime'], blocksize='10MB')
        
        # Add features
        ddf['hour'] = ddf['pickup_datetime'].dt.hour
        ddf['day_of_week'] = ddf['pickup_datetime'].dt.dayofweek
        
        # Create task
        task = create_dask_task(
            data=ddf,
            target='total_amount',
            task_type='regr'
        )
        
        # Build streaming-friendly pipeline
        print("Building streaming pipeline...")
        
        # Use SGD for incremental learning
        from sklearn.linear_model import SGDRegressor
        
        pipeline = linear_pipeline(
            # Quick filtering
            LazyPipeOpFilter(
                id="filter",
                condition=lambda df: (
                    (df['trip_distance'] > 0) & 
                    (df['total_amount'] > 0) & 
                    (df['total_amount'] < 200)
                )
            ),
            
            # Simple encoding
            PipeOpTargetEncode(
                id="encode",
                columns=['payment_type'],
                smoothing=1000  # High smoothing for stability
            ),
            
            # Binning for non-linearity
            PipeOpBin(
                id="bins",
                columns=['trip_distance', 'hour'],
                n_bins=10,
                encode='onehot'
            ),
            
            # Scale
            LazyPipeOpScale(id="scale"),
            
            # SGD for streaming
            PipeOpLearner(
                learner_sklearn(
                    SGDRegressor(
                        loss='squared_error',
                        penalty='l2',
                        alpha=0.0001,
                        max_iter=1000,
                        random_state=42
                    )
                ),
                id="sgd"
            )
        )
        
        # Simulate streaming batches
        print("\nSimulating streaming batches...")
        
        batch_results = []
        cumulative_samples = 0
        
        for batch_num in range(5):
            print(f"\nBatch {batch_num + 1}/5")
            
            # Sample a batch
            batch_start = batch_num * 20000
            batch_end = (batch_num + 1) * 20000
            
            if batch_end > task.nrow:
                break
                
            batch_task = task.filter_rows(range(batch_start, batch_end))
            cumulative_samples += batch_task.nrow
            
            # Evaluate on this batch
            result = resample(
                task=batch_task,
                learner=pipeline,
                resampling=ResamplingHoldout(ratio=0.8),
                measure=MeasureRegrRMSE()
            )
            
            rmse = result.aggregate()['rmse'][0]
            batch_results.append({
                'batch': batch_num + 1,
                'samples': cumulative_samples,
                'rmse': rmse
            })
            
            print(f"  Samples seen: {cumulative_samples:,}")
            print(f"  Batch RMSE: ${rmse:.2f}")
        
        # Plot learning curve
        if len(batch_results) > 1:
            plt.figure(figsize=(10, 6))
            
            batches = [r['batch'] for r in batch_results]
            rmses = [r['rmse'] for r in batch_results]
            samples = [r['samples'] for r in batch_results]
            
            plt.subplot(1, 2, 1)
            plt.plot(batches, rmses, 'bo-')
            plt.xlabel('Batch Number')
            plt.ylabel('RMSE ($)')
            plt.title('Model Performance Over Time')
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            plt.plot(samples, rmses, 'ro-')
            plt.xlabel('Cumulative Samples')
            plt.ylabel('RMSE ($)')
            plt.title('Learning Curve')
            plt.grid(True)
            
            plt.tight_layout()
            plt.show()
            
    finally:
        client.close()
        if os.path.exists(filename):
            os.remove(filename)


def main():
    """Run NYC taxi examples."""
    print("MLPY Big Data Examples: NYC Taxi Dataset")
    print("="*40)
    
    if not SKLEARN_AVAILABLE:
        print("\nError: scikit-learn required for these examples")
        return
    
    if not DASK_AVAILABLE and not VAEX_AVAILABLE:
        print("\nError: At least one big data backend required")
        return
    
    # Run examples
    if DASK_AVAILABLE:
        example_dask_taxi_fare_prediction()
        example_streaming_taxi_pipeline()
    
    if VAEX_AVAILABLE:
        example_vaex_taxi_exploration()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)
    
    print("\nKey insights from taxi data:")
    print("1. Time-of-day and day-of-week strongly affect fares")
    print("2. Location binning captures geographic patterns efficiently")
    print("3. Target encoding handles high-cardinality features well")
    print("4. Streaming pipelines enable real-time model updates")
    print("5. Vaex excels at interactive exploration of millions of trips")


if __name__ == "__main__":
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    main()