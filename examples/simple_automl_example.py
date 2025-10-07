"""
Simple AutoML Example for MLPY

This example demonstrates how to use MLPY's SimpleAutoML
for automated machine learning workflows.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
from mlpy.automl import SimpleAutoML

def classification_example():
    """Example with breast cancer classification dataset."""
    print("=" * 60)
    print("🎯 CLASSIFICATION EXAMPLE - Breast Cancer Dataset")
    print("=" * 60)
    
    # Load data
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    
    print(f"📊 Dataset shape: {df.shape}")
    print(f"📋 Target distribution:")
    print(df['target'].value_counts())
    
    # Create AutoML instance
    automl = SimpleAutoML(
        time_limit=120,  # 2 minutes
        max_models=20,
        feature_engineering=True,
        feature_selection=True,
        cross_validation=3,  # Faster for demo
        verbose=True
    )
    
    # Fit AutoML
    result = automl.fit(df, target='target')
    
    # Display results
    print(f"\n📊 RESULTS SUMMARY:")
    print(f"⭐ Best Score: {result.best_score:.4f}")
    print(f"⏱️  Training Time: {result.training_time:.1f}s")
    print(f"🔍 Models Tried: {len(result.leaderboard)}")
    
    print(f"\n🏆 TOP 5 MODELS:")
    print(result.leaderboard.head().to_string(index=False))
    
    if result.feature_importance is not None:
        print(f"\n🔍 TOP 10 FEATURES:")
        top_features = result.feature_importance.nlargest(10)
        for feature, importance in top_features.items():
            print(f"  {feature}: {importance:.4f}")
    
    return result

def regression_example():
    """Example with diabetes regression dataset."""
    print("=" * 60)
    print("📈 REGRESSION EXAMPLE - Diabetes Dataset")
    print("=" * 60)
    
    # Load data
    data = load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    
    print(f"📊 Dataset shape: {df.shape}")
    print(f"📋 Target stats:")
    print(f"  Mean: {df['target'].mean():.2f}")
    print(f"  Std:  {df['target'].std():.2f}")
    print(f"  Range: [{df['target'].min():.1f}, {df['target'].max():.1f}]")
    
    # Create AutoML instance
    automl = SimpleAutoML(
        time_limit=120,  # 2 minutes
        max_models=20,
        feature_engineering=True,
        feature_selection=False,  # Small dataset
        cross_validation=3,  # Faster for demo
        verbose=True
    )
    
    # Fit AutoML
    result = automl.fit(df, target='target')
    
    # Display results
    print(f"\n📊 RESULTS SUMMARY:")
    print(f"⭐ Best Score (MSE): {result.best_score:.2f}")
    print(f"📏 RMSE: {np.sqrt(result.best_score):.2f}")
    print(f"⏱️  Training Time: {result.training_time:.1f}s")
    print(f"🔍 Models Tried: {len(result.leaderboard)}")
    
    print(f"\n🏆 TOP 5 MODELS:")
    leaderboard_display = result.leaderboard.copy()
    leaderboard_display['rmse'] = np.sqrt(leaderboard_display['score'])
    print(leaderboard_display[['model', 'score', 'rmse']].head().to_string(index=False))
    
    if result.feature_importance is not None:
        print(f"\n🔍 ALL FEATURES IMPORTANCE:")
        for feature, importance in result.feature_importance.items():
            print(f"  {feature}: {importance:.4f}")
    
    return result

def custom_data_example():
    """Example with synthetic custom data."""
    print("=" * 60)
    print("🎲 CUSTOM DATA EXAMPLE - Synthetic Dataset")
    print("=" * 60)
    
    # Create synthetic dataset
    np.random.seed(42)
    n_samples = 1000
    n_features = 15
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Create target with some feature interactions
    y = (
        3 * X[:, 0] +           # Linear term
        -2 * X[:, 1] +          # Linear term  
        1.5 * X[:, 0] * X[:, 1] +  # Interaction
        0.8 * X[:, 2]**2 +      # Non-linear
        0.5 * np.random.randn(n_samples)  # Noise
    )
    
    # Convert to DataFrame
    feature_names = [f'feature_{i:02d}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    print(f"📊 Dataset shape: {df.shape}")
    print(f"📋 Target stats:")
    print(f"  Mean: {y.mean():.3f}")
    print(f"  Std:  {y.std():.3f}")
    print(f"  Range: [{y.min():.2f}, {y.max():.2f}]")
    
    # Create AutoML instance  
    automl = SimpleAutoML(
        time_limit=180,  # 3 minutes
        max_models=30,
        feature_engineering=True,  # Should help with interactions
        feature_selection=True,
        cross_validation=5,
        verbose=True
    )
    
    # Fit AutoML
    result = automl.fit(df, target='target')
    
    # Display results
    print(f"\n📊 RESULTS SUMMARY:")
    print(f"⭐ Best Score (MSE): {result.best_score:.4f}")
    print(f"📏 RMSE: {np.sqrt(result.best_score):.4f}")
    print(f"⏱️  Training Time: {result.training_time:.1f}s")
    print(f"🔍 Models Tried: {len(result.leaderboard)}")
    
    print(f"\n🏆 TOP 5 MODELS:")
    leaderboard_display = result.leaderboard.copy()
    leaderboard_display['rmse'] = np.sqrt(leaderboard_display['score'])
    print(leaderboard_display[['model', 'score', 'rmse']].head().to_string(index=False))
    
    if result.feature_importance is not None:
        print(f"\n🔍 TOP 10 FEATURES:")
        top_features = result.feature_importance.nlargest(10)
        for feature, importance in top_features.items():
            print(f"  {feature}: {importance:.4f}")
            
        # Check if it found the important features
        print(f"\n🎯 KEY INSIGHTS:")
        important_original = ['feature_00', 'feature_01', 'feature_02']
        found_important = [f for f in important_original if f in top_features.index[:5]]
        print(f"  Found {len(found_important)}/3 truly important features in top 5")
        
        # Look for interaction features
        interaction_features = [f for f in top_features.index if '_x_' in f]
        if interaction_features:
            print(f"  Found interaction features: {interaction_features[:3]}")
    
    return result

def main():
    """Run all examples."""
    print("🚀 MLPY SimpleAutoML Examples")
    print("This will run three examples demonstrating AutoML capabilities\n")
    
    try:
        # Run classification example
        result1 = classification_example()
        
        print("\n" + "="*60 + "\n")
        
        # Run regression example  
        result2 = regression_example()
        
        print("\n" + "="*60 + "\n")
        
        # Run custom data example
        result3 = custom_data_example()
        
        print("\n" + "="*60)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return result1, result2, result3
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()