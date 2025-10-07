"""
Example of using advanced pipeline operators in MLPY.

This example demonstrates dimensionality reduction, outlier detection,
feature engineering, and text processing in ML pipelines.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, fetch_20newsgroups

from mlpy.tasks import TaskClassif, TaskRegr
from mlpy.learners.sklearn import learner_sklearn
from mlpy.pipelines import (
    # Basic operators
    PipeOpScale, PipeOpEncode,
    # Advanced operators
    PipeOpPCA, PipeOpTargetEncode, PipeOpOutlierDetect,
    PipeOpBin, PipeOpTextVectorize, PipeOpPolynomial,
    # Pipeline utilities
    PipeOpLearner, linear_pipeline
)
from mlpy.resamplings import ResamplingCV
from mlpy.measures import MeasureClassifAcc, MeasureRegrRMSE
from mlpy.resample import resample

# Check for optional dependencies
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Some examples will be skipped.")


def create_sample_datasets():
    """Create various datasets for demonstrating operators."""
    np.random.seed(42)
    
    # 1. High-dimensional data for PCA
    X_high, y_high = make_classification(
        n_samples=500,
        n_features=100,
        n_informative=20,
        n_redundant=30,
        n_classes=3,
        random_state=42
    )
    df_high = pd.DataFrame(X_high, columns=[f'feat_{i}' for i in range(100)])
    df_high['target'] = y_high
    
    # 2. Data with outliers
    n_samples = 300
    X_clean = np.random.randn(n_samples, 5)
    y_clean = 2 * X_clean[:, 0] + X_clean[:, 1] + np.random.randn(n_samples) * 0.5
    
    # Add outliers
    n_outliers = 30
    X_outliers = np.random.uniform(-10, 10, (n_outliers, 5))
    y_outliers = np.random.uniform(-20, 20, n_outliers)
    
    X_with_outliers = np.vstack([X_clean, X_outliers])
    y_with_outliers = np.hstack([y_clean, y_outliers])
    
    df_outliers = pd.DataFrame(X_with_outliers, columns=[f'x{i}' for i in range(5)])
    df_outliers['target'] = y_with_outliers
    
    # 3. Mixed data with high-cardinality categoricals
    n_samples = 1000
    df_mixed = pd.DataFrame({
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.lognormal(10, 1, n_samples),
        'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix', 
                                 'Philly', 'San Antonio', 'San Diego', 'Dallas'], n_samples),
        'occupation': np.random.choice(['Engineer', 'Teacher', 'Doctor', 'Sales', 
                                       'Manager', 'Other'] * 10, n_samples),  # Repeated for variety
        'score': np.random.randn(n_samples),
        'target': np.random.choice([0, 1], n_samples)
    })
    
    # 4. Text data (simple example)
    df_text = pd.DataFrame({
        'review': [
            "This product is amazing and works great",
            "Terrible quality, very disappointed",
            "Good value for money, recommended",
            "Not worth the price, poor quality",
            "Excellent service and fast delivery",
            "Product broke after one week",
            "Very satisfied with my purchase",
            "Waste of money, do not buy",
            "High quality product, five stars",
            "Average product, nothing special"
        ] * 10,  # Repeat for more samples
        'rating': [5, 1, 4, 2, 5, 1, 5, 1, 5, 3] * 10,
        'verified': np.random.choice([0, 1], 100),
        'helpful_votes': np.random.randint(0, 50, 100)
    })
    df_text['sentiment'] = (df_text['rating'] >= 4).astype(int)
    
    return df_high, df_outliers, df_mixed, df_text


def example_pca_dimensionality_reduction():
    """Example: Using PCA for dimensionality reduction."""
    print("\n" + "="*60)
    print("EXAMPLE 1: PCA for Dimensionality Reduction")
    print("="*60)
    
    df_high, _, _, _ = create_sample_datasets()
    task = TaskClassif(data=df_high, target='target', id='high_dim')
    
    print(f"Original data: {task.nrow} samples, {len(task.feature_names)} features")
    
    # Build pipeline with PCA
    pipeline = linear_pipeline(
        PipeOpScale(id="scale"),  # Important to scale before PCA
        PipeOpPCA(id="pca", n_components=0.95),  # Keep 95% variance
        PipeOpLearner(
            learner_sklearn(RandomForestClassifier(n_estimators=50, random_state=42)),
            id="rf"
        )
    )
    
    # Train and evaluate
    print("\nTraining pipeline with PCA...")
    result = resample(
        task=task,
        learner=pipeline,
        resampling=ResamplingCV(folds=3),
        measure=MeasureClassifAcc()
    )
    
    print(f"Cross-validation accuracy: {result.aggregate()['acc'][0]:.3f}")
    
    # Check how many components were kept
    pipeline.train(task)
    pca_op = pipeline.pipeops['pca']
    n_components = pca_op.state['n_components']
    explained_var = pca_op.state['explained_variance_ratio']
    
    print(f"\nPCA reduced features from {len(task.feature_names)} to {n_components}")
    print(f"Cumulative variance explained: {sum(explained_var):.3f}")
    
    # Visualize explained variance
    if len(explained_var) <= 20:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.bar(range(1, len(explained_var) + 1), explained_var)
        plt.xlabel('Principal Component')
        plt.ylabel('Variance Explained')
        plt.title('Variance Explained by Each PC')
        
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(explained_var) + 1), np.cumsum(explained_var), 'bo-')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Variance Explained')
        plt.title('Cumulative Variance Explained')
        plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
        plt.legend()
        
        plt.tight_layout()
        plt.show()


def example_outlier_detection():
    """Example: Outlier detection and handling."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Outlier Detection and Handling")
    print("="*60)
    
    _, df_outliers, _, _ = create_sample_datasets()
    task = TaskRegr(data=df_outliers, target='target', id='outliers')
    
    print(f"Dataset with outliers: {task.nrow} samples")
    
    # Compare different outlier handling strategies
    strategies = {
        'No outlier handling': linear_pipeline(
            PipeOpScale(id="scale"),
            PipeOpLearner(learner_sklearn(Ridge()), id="ridge")
        ),
        'Flag outliers': linear_pipeline(
            PipeOpOutlierDetect(id="outlier", method="isolation", action="flag"),
            PipeOpScale(id="scale"),
            PipeOpLearner(learner_sklearn(Ridge()), id="ridge")
        ),
        'Remove outliers': linear_pipeline(
            PipeOpOutlierDetect(id="outlier", method="isolation", action="remove"),
            PipeOpScale(id="scale"),
            PipeOpLearner(learner_sklearn(Ridge()), id="ridge")
        ),
        'Impute outliers': linear_pipeline(
            PipeOpOutlierDetect(id="outlier", method="isolation", action="impute"),
            PipeOpScale(id="scale"),
            PipeOpLearner(learner_sklearn(Ridge()), id="ridge")
        )
    }
    
    results = {}
    for name, pipeline in strategies.items():
        print(f"\nTesting: {name}")
        result = resample(
            task=task,
            learner=pipeline,
            resampling=ResamplingCV(folds=3),
            measure=MeasureRegrRMSE()
        )
        rmse = result.aggregate()['rmse'][0]
        results[name] = rmse
        print(f"  RMSE: {rmse:.3f}")
        
    # Visualize results
    plt.figure(figsize=(8, 5))
    plt.bar(results.keys(), results.values())
    plt.xlabel('Strategy')
    plt.ylabel('RMSE')
    plt.title('Impact of Outlier Handling on Model Performance')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # Show outlier detection visualization
    pipeline_detect = linear_pipeline(
        PipeOpOutlierDetect(id="outlier", method="isolation", action="flag")
    )
    result = pipeline_detect.train(task)
    outlier_flags = result['output'].data()['is_outlier']
    
    plt.figure(figsize=(10, 5))
    colors = ['blue' if flag == 0 else 'red' for flag in outlier_flags]
    plt.scatter(df_outliers['x0'], df_outliers['target'], c=colors, alpha=0.6)
    plt.xlabel('Feature x0')
    plt.ylabel('Target')
    plt.title('Outlier Detection Results')
    plt.legend(['Inlier', 'Outlier'])
    plt.show()


def example_target_encoding():
    """Example: Target encoding for high-cardinality categoricals."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Target Encoding for Categorical Features")
    print("="*60)
    
    _, _, df_mixed, _ = create_sample_datasets()
    task = TaskClassif(data=df_mixed, target='target', id='mixed')
    
    print(f"Dataset: {task.nrow} samples")
    print(f"Categorical features: city ({df_mixed['city'].nunique()} unique), "
          f"occupation ({df_mixed['occupation'].nunique()} unique)")
    
    # Compare encoding strategies
    strategies = {
        'One-hot encoding': linear_pipeline(
            PipeOpEncode(id="encode", method="onehot"),
            PipeOpScale(id="scale"),
            PipeOpLearner(
                learner_sklearn(LogisticRegression(max_iter=1000)),
                id="logreg"
            )
        ),
        'Target encoding': linear_pipeline(
            PipeOpTargetEncode(id="target_enc", smoothing=5.0),
            PipeOpScale(id="scale"),
            PipeOpLearner(
                learner_sklearn(LogisticRegression(max_iter=1000)),
                id="logreg"
            )
        )
    }
    
    for name, pipeline in strategies.items():
        print(f"\n{name}:")
        
        # Train to see feature count
        result = pipeline.train(task)
        n_features = len(result['logreg'].data()[0].feature_names)
        print(f"  Number of features after encoding: {n_features}")
        
        # Evaluate
        cv_result = resample(
            task=task,
            learner=pipeline,
            resampling=ResamplingCV(folds=5),
            measure=MeasureClassifAcc()
        )
        print(f"  CV Accuracy: {cv_result.aggregate()['acc'][0]:.3f}")


def example_feature_binning():
    """Example: Binning continuous features."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Feature Binning")
    print("="*60)
    
    # Create data with non-linear relationships
    np.random.seed(42)
    n_samples = 500
    X = np.random.randn(n_samples, 3)
    
    # Non-linear relationship
    y = (
        np.sin(2 * X[:, 0]) +  # Sinusoidal
        (X[:, 1] > 0).astype(int) * 2 +  # Step function
        np.exp(-X[:, 2]**2) +  # Gaussian-like
        np.random.randn(n_samples) * 0.2
    )
    
    df = pd.DataFrame(X, columns=['x1_sin', 'x2_step', 'x3_gauss'])
    df['target'] = y
    task = TaskRegr(data=df, target='target')
    
    # Compare linear model with and without binning
    pipelines = {
        'Linear (no binning)': linear_pipeline(
            PipeOpScale(id="scale"),
            PipeOpLearner(learner_sklearn(Ridge()), id="ridge")
        ),
        'Linear + Binning': linear_pipeline(
            PipeOpBin(id="bin", n_bins=10, strategy='quantile', encode='onehot'),
            PipeOpLearner(learner_sklearn(Ridge()), id="ridge")
        ),
        'Linear + Polynomial': linear_pipeline(
            PipeOpScale(id="scale"),
            PipeOpPolynomial(id="poly", degree=3),
            PipeOpLearner(learner_sklearn(Ridge(alpha=1.0)), id="ridge")
        )
    }
    
    results = {}
    for name, pipeline in pipelines.items():
        result = resample(
            task=task,
            learner=pipeline,
            resampling=ResamplingCV(folds=5),
            measure=MeasureRegrRMSE()
        )
        results[name] = result.aggregate()['rmse'][0]
        print(f"{name}: RMSE = {results[name]:.3f}")
        
    # Visualize feature transformations
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, (col, ax) in enumerate(zip(df.columns[:3], axes)):
        # Original relationship
        ax.scatter(df[col], df['target'], alpha=0.5, s=20)
        ax.set_xlabel(col)
        ax.set_ylabel('Target')
        ax.set_title(f'Feature {i+1} vs Target')
        
    plt.tight_layout()
    plt.show()


def example_text_processing():
    """Example: Text vectorization in pipelines."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Text Processing Pipeline")
    print("="*60)
    
    _, _, _, df_text = create_sample_datasets()
    task = TaskClassif(data=df_text, target='sentiment', id='text')
    
    print(f"Text classification task: {task.nrow} reviews")
    print(f"Sample review: '{df_text['review'].iloc[0]}'")
    
    # Build text processing pipeline
    pipeline = linear_pipeline(
        # Vectorize text
        PipeOpTextVectorize(
            id="tfidf",
            columns=['review'],
            method='tfidf',
            max_features=100,
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=2
        ),
        # Add other features
        PipeOpScale(id="scale"),  # Scales numeric features
        # Learn
        PipeOpLearner(
            learner_sklearn(LogisticRegression(max_iter=1000)),
            id="logreg"
        )
    )
    
    # Evaluate
    print("\nEvaluating text classification pipeline...")
    result = resample(
        task=task,
        learner=pipeline,
        resampling=ResamplingCV(folds=5),
        measure=MeasureClassifAcc()
    )
    
    print(f"CV Accuracy: {result.aggregate()['acc'][0]:.3f}")
    
    # Train and inspect important words
    pipeline.train(task)
    
    # Get feature importances (coefficients for logistic regression)
    learner = pipeline.pipeops['logreg']._trained_learner
    if hasattr(learner._model, 'coef_'):
        coefficients = learner._model.coef_[0]
        feature_names = [f for f in pipeline.pipeops['tfidf'].state['feature_names']['review']]
        
        # Get top positive and negative words
        word_importance = list(zip(feature_names, coefficients))
        word_importance.sort(key=lambda x: x[1], reverse=True)
        
        print("\nMost positive words:")
        for word, coef in word_importance[:5]:
            print(f"  {word}: {coef:.3f}")
            
        print("\nMost negative words:")
        for word, coef in word_importance[-5:]:
            print(f"  {word}: {coef:.3f}")


def example_complex_pipeline():
    """Example: Complex pipeline combining multiple advanced operators."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Complex Pipeline with Multiple Operators")
    print("="*60)
    
    # Create complex dataset
    np.random.seed(42)
    n_samples = 1000
    
    df = pd.DataFrame({
        # Numeric features
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.lognormal(10, 1, n_samples),
        'credit_score': np.random.normal(700, 100, n_samples),
        
        # Categorical
        'education': np.random.choice(['HS', 'Bachelor', 'Master', 'PhD'], n_samples),
        'employment': np.random.choice(['Full-time', 'Part-time', 'Self', 'None'], n_samples),
        
        # Text
        'notes': [f"Customer type {i % 5} with history {i % 3}" for i in range(n_samples)],
        
        # Target (loan approval)
        'approved': np.random.choice([0, 1], n_samples)
    })
    
    task = TaskClassif(data=df, target='approved', id='complex')
    
    # Build sophisticated pipeline
    pipeline = linear_pipeline(
        # 1. Handle outliers in income
        PipeOpOutlierDetect(
            id="outlier",
            method="isolation",
            contamination=0.05,
            action="impute"
        ),
        
        # 2. Bin continuous features
        PipeOpBin(
            id="bin_age",
            columns=['age'],
            n_bins=5,
            strategy='quantile',
            encode='onehot'
        ),
        
        # 3. Target encode high-cardinality categoricals
        PipeOpTargetEncode(
            id="target_enc",
            columns=['education', 'employment'],
            smoothing=10
        ),
        
        # 4. Create polynomial features for numeric
        PipeOpPolynomial(
            id="poly",
            columns=['income', 'credit_score'],
            degree=2,
            interaction_only=True
        ),
        
        # 5. Process text
        PipeOpTextVectorize(
            id="text",
            columns=['notes'],
            method='count',
            max_features=20
        ),
        
        # 6. Scale all features
        PipeOpScale(id="scale"),
        
        # 7. Reduce dimensionality
        PipeOpPCA(
            id="pca",
            n_components=0.99  # Keep 99% variance
        ),
        
        # 8. Learn
        PipeOpLearner(
            learner_sklearn(
                RandomForestClassifier(n_estimators=100, random_state=42)
            ),
            id="rf"
        )
    )
    
    print("Pipeline structure:")
    print("1. Outlier detection (impute)")
    print("2. Age binning (5 bins)")
    print("3. Target encoding (education, employment)")
    print("4. Polynomial features (interactions)")
    print("5. Text vectorization (count)")
    print("6. Feature scaling")
    print("7. PCA (99% variance)")
    print("8. Random Forest classifier")
    
    # Evaluate
    print("\nEvaluating complex pipeline...")
    result = resample(
        task=task,
        learner=pipeline,
        resampling=ResamplingCV(folds=5),
        measure=MeasureClassifAcc()
    )
    
    print(f"\nFinal CV Accuracy: {result.aggregate()['acc'][0]:.3f}")
    
    # Analyze pipeline stages
    pipeline.train(task)
    
    print("\nPipeline analysis:")
    print(f"- Outliers detected: {pipeline.pipeops['outlier'].state['n_outliers']}")
    print(f"- Features after polynomial: {len(pipeline.pipeops['poly'].state['poly_feature_names'])}")
    print(f"- Text features created: {pipeline.pipeops['text'].state['n_features_total']}")
    print(f"- Final features after PCA: {pipeline.pipeops['pca'].state['n_components']}")


def main():
    """Run all examples."""
    print("MLPY Advanced Pipeline Operators Examples")
    print("=========================================")
    
    if not SKLEARN_AVAILABLE:
        print("\nWarning: scikit-learn not available. Examples cannot run.")
        return
        
    # Run examples
    example_pca_dimensionality_reduction()
    example_outlier_detection()
    example_target_encoding()
    example_feature_binning()
    example_text_processing()
    example_complex_pipeline()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)
    
    print("\nKey takeaways:")
    print("1. PCA effectively reduces dimensionality while preserving information")
    print("2. Outlier detection can significantly improve model robustness")
    print("3. Target encoding handles high-cardinality categoricals efficiently")
    print("4. Binning captures non-linear relationships for linear models")
    print("5. Text vectorization integrates NLP into ML pipelines")
    print("6. Complex pipelines can combine multiple preprocessing steps")
    print("\nThese operators make MLPY pipelines powerful and flexible!")


if __name__ == "__main__":
    # Set plotting style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    main()