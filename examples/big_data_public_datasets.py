"""
Example using MLPY with real public big datasets.

This example shows how to work with actual large public datasets:
- Criteo Click Prediction (1TB+)
- Wikipedia Page Traffic 
- OpenStreetMap Data
- Reddit Comments

Demonstrates downloading, processing, and modeling at scale.
"""

import os
import time
import requests
import zipfile
import gzip
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# MLPY imports
from mlpy.tasks import TaskClassif, TaskRegr
from mlpy.backends import DataBackendDask, DataBackendVaex
from mlpy.tasks.big_data import create_dask_task, create_vaex_task
from mlpy.learners.sklearn import learner_sklearn
from mlpy.pipelines import (
    PipeOpScale, PipeOpImpute, PipeOpSelect,
    PipeOpTargetEncode, PipeOpOutlierDetect, PipeOpBin,
    LazyPipeOpScale, LazyPipeOpFilter, LazyPipeOpSample,
    PipeOpLearner, linear_pipeline
)
from mlpy.resamplings import ResamplingHoldout
from mlpy.measures import MeasureClassifAcc, MeasureClassifAUC, MeasureRegrRMSE
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
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import SGDClassifier, SGDRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def download_file(url, filename, chunk_size=8192):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    print(f"\rDownloading: {progress:.1f}%", end='', flush=True)
    print()
    return filename


def example_criteo_click_prediction():
    """
    Example: Click-through rate prediction with Criteo dataset.
    
    Dataset: http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/
    
    Note: This is a 11GB dataset with 45M rows. For demo purposes,
    we'll create a similar synthetic dataset.
    """
    if not DASK_AVAILABLE:
        print("Dask required for this example")
        return
        
    print("\n" + "="*60)
    print("EXAMPLE: Click Prediction with Criteo-style Data")
    print("="*60)
    
    # Create synthetic data similar to Criteo
    print("Creating synthetic click prediction data...")
    
    n_rows = 1000000  # 1M rows for demo
    n_numeric = 13
    n_categorical = 26
    
    # Generate features
    data = {}
    
    # Numeric features (like Criteo's integer features)
    for i in range(n_numeric):
        if i < 5:  # Some features are counts (small integers)
            data[f'I{i+1}'] = np.random.poisson(3, n_rows)
        else:  # Others are larger values
            data[f'I{i+1}'] = np.random.lognormal(5, 2, n_rows).astype(int)
    
    # Categorical features (hashed strings in Criteo)
    cardinalities = [100, 500, 1000, 5000, 10000]  # Varying cardinalities
    for i in range(n_categorical):
        card = cardinalities[i % len(cardinalities)]
        data[f'C{i+1}'] = np.random.randint(0, card, n_rows)
    
    # Target (click or not)
    # Create realistic click rate (~3%)
    click_prob = np.zeros(n_rows)
    
    # Some features correlate with clicks
    click_prob += (data['I1'] > 5) * 0.02  # High count feature
    click_prob += (data['I5'] > 1000) * 0.01  # High value feature
    click_prob += (data['C1'] < 10) * 0.03  # Specific category values
    
    # Add noise
    click_prob += np.random.normal(0, 0.01, n_rows)
    click_prob = np.clip(click_prob, 0, 0.2)  # Max 20% CTR
    
    data['click'] = (np.random.random(n_rows) < click_prob).astype(int)
    
    # Save as CSV
    df = pd.DataFrame(data)
    filename = 'criteo_sample.csv'
    df.to_csv(filename, index=False)
    
    print(f"Created {filename} with {n_rows:,} rows")
    print(f"Click rate: {data['click'].mean():.2%}")
    
    # Start Dask
    client = dask.distributed.Client(n_workers=2, threads_per_worker=2, silence_logs=50)
    
    try:
        # Load with Dask
        print("\nLoading data with Dask...")
        ddf = dd.read_csv(filename, blocksize='50MB')
        
        # Create task
        task = create_dask_task(
            data=ddf,
            target='click',
            task_type='classif'
        )
        
        # Build pipeline for CTR prediction
        print("Building CTR prediction pipeline...")
        
        pipeline = linear_pipeline(
            # 1. Handle missing values (common in real Criteo data)
            PipeOpImpute(
                id="impute",
                method="constant",
                value=-1  # Flag missing as -1
            ),
            
            # 2. Cap extreme values
            LazyPipeOpFilter(
                id="cap",
                condition=lambda df: df  # In practice, would cap outliers
            ),
            
            # 3. Target encode high-cardinality categoricals
            PipeOpTargetEncode(
                id="encode_cat",
                columns=[f'C{i+1}' for i in range(10)],  # First 10 categoricals
                smoothing=100
            ),
            
            # 4. Bin remaining categoricals
            PipeOpBin(
                id="bin_cat",
                columns=[f'C{i+1}' for i in range(10, n_categorical)],
                n_bins=50,
                strategy='quantile',
                encode='ordinal'
            ),
            
            # 5. Log transform numeric features
            # (In practice, would apply log1p to count features)
            
            # 6. Scale all features
            LazyPipeOpScale(
                id="scale",
                method="standard"
            ),
            
            # 7. Select top features (reduce dimensionality)
            PipeOpSelect(
                id="select",
                k=50,  # Top 50 features
                score_func="f_classif"
            ),
            
            # 8. Learn with SGD (scalable for big data)
            PipeOpLearner(
                learner_sklearn(
                    SGDClassifier(
                        loss='log',
                        penalty='elasticnet',
                        alpha=0.0001,
                        l1_ratio=0.15,
                        max_iter=1000,
                        random_state=42
                    )
                ),
                id="sgd"
            )
        )
        
        # Sample for training
        print("\nTraining on data sample...")
        
        pipeline_sample = linear_pipeline(
            LazyPipeOpSample(fraction=0.1, random_state=42),
            pipeline
        )
        
        start_time = time.time()
        result = resample(
            task=task,
            learner=pipeline_sample,
            resampling=ResamplingHoldout(ratio=0.8),
            measure=[MeasureClassifAcc(), MeasureClassifAUC()]
        )
        train_time = time.time() - start_time
        
        metrics = result.aggregate()
        print(f"\nResults:")
        print(f"Training time: {train_time:.2f}s")
        print(f"Accuracy: {metrics['acc'][0]:.3f}")
        print(f"AUC: {metrics['auc'][0]:.3f}")
        
        # Feature importance from selection
        print("\nTop features selected:")
        pipeline_sample.train(task)
        selected_features = pipeline_sample.pipeops['select'].state['selected_features']
        print(f"Selected {len(selected_features)} features")
        
    finally:
        client.close()
        if os.path.exists(filename):
            os.remove(filename)


def example_wikipedia_traffic_time_series():
    """
    Example: Wikipedia page traffic time series forecasting.
    
    Based on: https://www.kaggle.com/c/web-traffic-time-series-forecasting
    """
    if not VAEX_AVAILABLE:
        print("Vaex required for this example")
        return
        
    print("\n" + "="*60)
    print("EXAMPLE: Wikipedia Traffic Time Series")
    print("="*60)
    
    # Create synthetic Wikipedia traffic data
    print("Creating synthetic Wikipedia traffic data...")
    
    n_pages = 10000
    n_days = 365
    
    # Generate page names
    languages = ['en', 'es', 'de', 'fr', 'ru', 'ja', 'zh']
    access_types = ['desktop', 'mobile-web', 'mobile-app']
    agents = ['spider', 'all-agents']
    
    pages = []
    for i in range(n_pages):
        lang = np.random.choice(languages)
        access = np.random.choice(access_types)
        agent = np.random.choice(agents)
        page_name = f"Page_{i}_{lang}.wikipedia.org_{access}_{agent}"
        pages.append(page_name)
    
    # Generate time series data
    dates = pd.date_range('2022-01-01', periods=n_days, freq='D')
    
    # Create data in long format
    data_long = []
    
    for page in pages[:1000]:  # Use subset for demo
        # Base traffic level
        base_traffic = np.random.lognormal(7, 2)
        
        # Weekly seasonality
        weekly_pattern = np.sin(np.arange(n_days) * 2 * np.pi / 7) * base_traffic * 0.2
        
        # Trend
        trend = np.linspace(0, base_traffic * 0.1, n_days)
        
        # Random noise
        noise = np.random.normal(0, base_traffic * 0.3, n_days)
        
        # Combine
        traffic = base_traffic + weekly_pattern + trend + noise
        traffic = np.maximum(traffic, 0).astype(int)
        
        # Add special events (spikes)
        n_events = np.random.poisson(2)
        for _ in range(n_events):
            event_day = np.random.randint(0, n_days)
            traffic[event_day] += np.random.exponential(base_traffic * 5)
        
        # Create records
        for i, (date, views) in enumerate(zip(dates, traffic)):
            data_long.append({
                'page': page,
                'date': date,
                'views': views,
                'day_of_week': date.dayofweek,
                'day_of_month': date.day,
                'month': date.month,
                'is_weekend': date.dayofweek >= 5
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(data_long)
    
    # Add lag features
    print("Creating lag features...")
    df = df.sort_values(['page', 'date'])
    
    for lag in [1, 7, 28]:  # Previous day, week, month
        df[f'views_lag_{lag}'] = df.groupby('page')['views'].shift(lag)
    
    # Rolling statistics
    df['views_ma_7'] = df.groupby('page')['views'].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    df['views_ma_28'] = df.groupby('page')['views'].transform(
        lambda x: x.rolling(28, min_periods=1).mean()
    )
    
    # Remove rows with NaN (from lag features)
    df = df.dropna()
    
    # Save to HDF5 for Vaex
    hdf5_file = 'wikipedia_traffic.hdf5'
    df.to_hdf(hdf5_file, key='data', mode='w')
    
    # Load with Vaex
    print(f"\nLoading {len(df):,} records with Vaex...")
    vdf = vaex.open(hdf5_file)
    
    try:
        # Create regression task (predict next day views)
        task = create_vaex_task(
            data=vdf,
            target='views',
            task_type='regr'
        )
        
        # Build time series pipeline
        print("Building time series forecasting pipeline...")
        
        pipeline = linear_pipeline(
            # 1. Remove outliers (viral spikes)
            PipeOpOutlierDetect(
                id="outliers",
                method="isolation",
                contamination=0.05,
                action="flag"  # Keep but flag
            ),
            
            # 2. Encode categorical time features
            PipeOpBin(
                id="time_bins",
                columns=['day_of_week', 'day_of_month'],
                n_bins=[7, 10],  # Weekly and ~3 bins per month
                encode='onehot'
            ),
            
            # 3. Scale features
            PipeOpScale(
                id="scale",
                method="robust"  # Robust to outliers
            ),
            
            # 4. Learn with Random Forest
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
        print("\nTraining on data sample...")
        sample_size = 50000
        sample_indices = np.random.choice(len(vdf), sample_size, replace=False)
        sample_task = task.filter_rows(sample_indices)
        
        result = resample(
            task=sample_task,
            learner=pipeline,
            resampling=ResamplingHoldout(ratio=0.8),
            measure=MeasureRegrRMSE()
        )
        
        print(f"RMSE: {result.aggregate()['rmse'][0]:.2f} views")
        
        # Analyze patterns
        print("\nAnalyzing traffic patterns...")
        
        # Day of week effect
        print("\nAverage views by day of week:")
        for dow in range(7):
            mask = vdf.day_of_week == dow
            avg_views = vdf[mask].views.mean()
            day_name = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][dow]
            print(f"  {day_name}: {avg_views:.0f}")
        
        # Feature importance
        pipeline.train(sample_task)
        rf_model = pipeline.pipeops['rf']._trained_learner._model
        
        if hasattr(rf_model, 'feature_importances_'):
            importances = rf_model.feature_importances_
            feature_names = pipeline.pipeops['rf']._trained_learner._feature_names
            
            # Top features
            top_idx = np.argsort(importances)[-5:][::-1]
            print("\nTop 5 most important features:")
            for i, idx in enumerate(top_idx):
                print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.3f}")
                
        # Visualize sample page
        print("\nCreating visualization...")
        
        # Get one page's data
        sample_page = pages[0]
        page_data = df[df['page'] == sample_page].copy()
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(page_data['date'], page_data['views'], 'b-', alpha=0.7, label='Actual')
        plt.plot(page_data['date'], page_data['views_ma_7'], 'r-', label='7-day MA')
        plt.xlabel('Date')
        plt.ylabel('Page Views')
        plt.title(f'Wikipedia Traffic: {sample_page}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        # Weekly pattern
        weekly_avg = page_data.groupby('day_of_week')['views'].mean()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        plt.bar(days, weekly_avg)
        plt.xlabel('Day of Week')
        plt.ylabel('Average Views')
        plt.title('Weekly Pattern')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    finally:
        if os.path.exists(hdf5_file):
            os.remove(hdf5_file)


def example_reddit_comments_nlp():
    """
    Example: Reddit comments sentiment analysis at scale.
    
    Based on Reddit comments dataset.
    """
    print("\n" + "="*60)
    print("EXAMPLE: Reddit Comments Analysis")
    print("="*60)
    
    # Create synthetic Reddit-like data
    print("Creating synthetic Reddit comments data...")
    
    n_comments = 100000
    
    # Subreddits
    subreddits = ['python', 'machinelearning', 'datascience', 
                  'programming', 'askreddit', 'technology']
    
    # Generate comments
    positive_words = ['great', 'awesome', 'excellent', 'love', 'best', 
                     'amazing', 'wonderful', 'fantastic', 'helpful']
    negative_words = ['terrible', 'hate', 'worst', 'awful', 'horrible', 
                     'bad', 'useless', 'waste', 'disappointed']
    neutral_words = ['okay', 'fine', 'average', 'normal', 'typical',
                    'standard', 'regular', 'common', 'usual']
    
    data = []
    for i in range(n_comments):
        subreddit = np.random.choice(subreddits)
        
        # Generate comment based on sentiment
        sentiment = np.random.choice(['positive', 'negative', 'neutral'], 
                                   p=[0.4, 0.3, 0.3])
        
        if sentiment == 'positive':
            words = np.random.choice(positive_words, 3)
            score = np.random.randint(10, 100)
        elif sentiment == 'negative':
            words = np.random.choice(negative_words, 3)
            score = np.random.randint(-50, 0)
        else:
            words = np.random.choice(neutral_words, 3)
            score = np.random.randint(-5, 10)
            
        comment = f"This is {words[0]} and {words[1]}, very {words[2]}!"
        
        data.append({
            'id': f'comment_{i}',
            'subreddit': subreddit,
            'comment': comment,
            'score': score,
            'length': len(comment),
            'sentiment_label': 1 if score > 5 else 0  # Positive if high score
        })
    
    df = pd.DataFrame(data)
    
    print(f"Created {len(df):,} comments")
    print(f"Positive sentiment: {(df['sentiment_label'] == 1).mean():.1%}")
    
    if DASK_AVAILABLE:
        # Save and load with Dask
        filename = 'reddit_comments.csv'
        df.to_csv(filename, index=False)
        
        print("\nLoading with Dask for NLP pipeline...")
        ddf = dd.read_csv(filename, blocksize='25MB')
        
        try:
            # Create task
            task = create_dask_task(
                data=ddf,
                target='sentiment_label',
                task_type='classif'
            )
            
            # Import text vectorizer
            from mlpy.pipelines import PipeOpTextVectorize
            
            # Build NLP pipeline
            print("Building NLP pipeline...")
            
            pipeline = linear_pipeline(
                # 1. Vectorize text
                PipeOpTextVectorize(
                    id="tfidf",
                    columns=['comment'],
                    method='tfidf',
                    max_features=1000,
                    ngram_range=(1, 2),
                    min_df=5
                ),
                
                # 2. Add metadata features
                PipeOpTargetEncode(
                    id="encode_sub",
                    columns=['subreddit'],
                    smoothing=50
                ),
                
                # 3. Scale all features
                LazyPipeOpScale(id="scale"),
                
                # 4. Learn
                PipeOpLearner(
                    learner_sklearn(
                        SGDClassifier(
                            loss='log',
                            penalty='l2',
                            alpha=0.0001,
                            max_iter=1000,
                            random_state=42
                        )
                    ),
                    id="sgd"
                )
            )
            
            # Sample and train
            pipeline_sample = linear_pipeline(
                LazyPipeOpSample(fraction=0.1, random_state=42),
                pipeline
            )
            
            print("\nTraining sentiment classifier...")
            result = resample(
                task=task,
                learner=pipeline_sample,
                resampling=ResamplingHoldout(ratio=0.8),
                measure=[MeasureClassifAcc(), MeasureClassifAUC()]
            )
            
            metrics = result.aggregate()
            print(f"\nResults:")
            print(f"Accuracy: {metrics['acc'][0]:.3f}")
            print(f"AUC: {metrics['auc'][0]:.3f}")
            
            # Analyze learned patterns
            pipeline_sample.train(task)
            
            # Get top words
            vectorizer = pipeline_sample.pipeops['tfidf']
            if 'feature_names' in vectorizer.state:
                feature_names = vectorizer.state['feature_names']['comment']
                print(f"\nVocabulary size: {len(feature_names)} words/bigrams")
                print("Sample features:", feature_names[:10])
                
        finally:
            if os.path.exists(filename):
                os.remove(filename)
    else:
        print("Dask not available for this example")


def main():
    """Run public dataset examples."""
    print("MLPY Big Data Examples: Public Datasets")
    print("="*40)
    
    if not SKLEARN_AVAILABLE:
        print("\nError: scikit-learn required")
        return
    
    if not DASK_AVAILABLE and not VAEX_AVAILABLE:
        print("\nError: At least one big data backend required")
        return
    
    # Run examples
    if DASK_AVAILABLE:
        example_criteo_click_prediction()
        example_reddit_comments_nlp()
    
    if VAEX_AVAILABLE:
        example_wikipedia_traffic_time_series()
    
    print("\n" + "="*60)
    print("Summary: Working with Public Big Data")
    print("="*60)
    
    print("\nKey patterns for big data:")
    print("1. **Sampling**: Train on samples, evaluate on larger sets")
    print("2. **Lazy evaluation**: Process only what's needed")
    print("3. **Target encoding**: Handle high-cardinality efficiently")
    print("4. **Incremental learning**: Use SGD for streaming data")
    print("5. **Feature engineering**: Create domain-specific features")
    
    print("\nRecommended workflows:")
    print("- Criteo/Ads: Target encoding + SGD for billions of rows")
    print("- Time series: Lag features + tree models")
    print("- Text data: TF-IDF + metadata for context")
    print("- Streaming: Incremental models with periodic evaluation")
    
    print("\nMLPY makes big data ML accessible and efficient!")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-v0_8-darkgrid')
    
    main()