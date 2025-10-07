"""
Example of using Transport Gaussian Process (TGPY) with MLPY.

This example demonstrates how to use the TGPY wrapper for
Gaussian Process regression with transport maps.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Import MLPY components
from mlpy.tasks import TaskRegr
from mlpy.measures import MeasureRegrRMSE, MeasureRegrMAE, MeasureRegrR2
from mlpy.benchmark import benchmark
from mlpy.resamplings import ResamplingCV, ResamplingHoldout

# Try to import TGPY learner
try:
    from mlpy.learners import LearnerTGPRegressor
    TGPY_AVAILABLE = True
except ImportError:
    TGPY_AVAILABLE = False
    print("Warning: TGPY learner not available. Install TGPY from tgpy-master/")


def generate_1d_data(n_samples=100, noise=0.1):
    """Generate 1D regression data for GP testing."""
    np.random.seed(42)
    X = np.random.uniform(-3, 3, n_samples).reshape(-1, 1)
    # True function: sin(x) + 0.5*x
    y = np.sin(X).ravel() + 0.5 * X.ravel() + noise * np.random.randn(n_samples)
    return X, y


def test_tgpy_simple():
    """Simple test of TGPY learner."""
    print("=" * 80)
    print("TGPY Simple Test")
    print("=" * 80)
    
    if not TGPY_AVAILABLE:
        print("TGPY not available. Skipping test.")
        return
        
    # Generate 1D data
    X, y = generate_1d_data(n_samples=50)
    
    # Create dataframe
    data = pd.DataFrame(X, columns=['x'])
    data['y'] = y
    
    # Create task
    task = TaskRegr(
        id="tgpy_1d",
        data=data,
        target="y"
    )
    
    # Create and train TGPY learner
    print("\n1. Training TGPY model...")
    learner = LearnerTGPRegressor(
        kernel='SE',
        lengthscale=1.0,
        variance=1.0,
        noise=0.1,
        n_iterations=50,
        learning_rate=0.01
    )
    
    # Train on subset
    train_ids = list(range(40))
    test_ids = list(range(40, 50))
    
    try:
        learner.train(task, row_ids=train_ids)
        print("   Training completed!")
        
        # Make predictions
        pred = learner.predict(task, row_ids=test_ids)
        
        # Calculate metrics
        rmse = MeasureRegrRMSE()
        mae = MeasureRegrMAE()
        
        print(f"\n2. Test Results:")
        print(f"   RMSE: {rmse.score(pred):.4f}")
        print(f"   MAE: {mae.score(pred):.4f}")
        
        # Get predictions with uncertainty
        learner.predict_type = "se"
        pred_se = learner.predict(task, row_ids=test_ids)
        
        print(f"   Mean Standard Error: {np.mean(pred_se.se):.4f}")
        
        # Plot results
        plt.figure(figsize=(10, 6))
        
        # Training data
        plt.scatter(data.iloc[train_ids]['x'], data.iloc[train_ids]['y'], 
                   alpha=0.5, label='Training data')
        
        # Test data
        plt.scatter(data.iloc[test_ids]['x'], data.iloc[test_ids]['y'], 
                   alpha=0.5, color='red', label='Test data')
        
        # Predictions
        X_test = data.iloc[test_ids]['x'].values
        y_pred = pred.response
        
        # Sort for plotting
        sort_idx = np.argsort(X_test)
        plt.plot(X_test[sort_idx], y_pred[sort_idx], 'g-', label='TGPY predictions')
        
        # Uncertainty bands if available
        if hasattr(pred_se, 'se') and pred_se.se is not None:
            y_std = pred_se.se
            plt.fill_between(X_test[sort_idx], 
                           y_pred[sort_idx] - 2*y_std[sort_idx],
                           y_pred[sort_idx] + 2*y_std[sort_idx],
                           alpha=0.2, color='green', label='95% confidence')
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('TGPY Regression with Transport Maps')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    except Exception as e:
        print(f"   Error during training/prediction: {str(e)}")
        print("   Note: TGPY may require additional dependencies (IPython, torch, etc.)")


def compare_with_baselines():
    """Compare TGPY with other regression methods."""
    print("\n" + "=" * 80)
    print("TGPY Comparison with Baselines")
    print("=" * 80)
    
    # Generate regression data
    X, y = make_regression(
        n_samples=200,
        n_features=5,
        n_informative=3,
        noise=10,
        random_state=42
    )
    
    # Create dataframe
    feature_cols = [f"feat_{i}" for i in range(X.shape[1])]
    data = pd.DataFrame(X, columns=feature_cols)
    data['target'] = y
    
    # Create task
    task = TaskRegr(
        id="comparison",
        data=data,
        target="target"
    )
    
    # Define learners
    learners = []
    
    # Always include baseline
    from mlpy.learners import LearnerRegrFeatureless
    learners.append(LearnerRegrFeatureless(id="featureless"))
    
    # Add native learners
    try:
        from mlpy.learners.native import LearnerLinearRegression, LearnerKNNRegressor
        learners.extend([
            LearnerLinearRegression(id="linear_native"),
            LearnerKNNRegressor(id="knn_native", n_neighbors=5)
        ])
    except ImportError:
        print("Native learners not available")
    
    # Add TGPY if available
    if TGPY_AVAILABLE:
        learners.append(
            LearnerTGPRegressor(
                id="tgpy",
                kernel='SE',
                n_iterations=50,
                transport='covariance'
            )
        )
    
    # Add sklearn learners if available
    try:
        from mlpy.learners import LearnerLinearRegression as LearnerLinearRegressionSK
        from mlpy.learners import LearnerRandomForestRegressor
        learners.extend([
            LearnerLinearRegressionSK(id="linear_sklearn"),
            LearnerRandomForestRegressor(id="rf_sklearn", n_estimators=50)
        ])
    except ImportError:
        print("Sklearn learners not available")
    
    # Run benchmark
    print(f"\nBenchmarking {len(learners)} learners...")
    bench = benchmark(
        tasks=[task],
        learners=learners,
        resampling=ResamplingHoldout(ratio=0.8),
        measures=[MeasureRegrRMSE(), MeasureRegrMAE(), MeasureRegrR2()]
    )
    
    # Display results
    results = bench.aggregate('RMSE')
    print("\nResults (sorted by RMSE):")
    print(results[['learner_id', 'test_RMSE', 'test_MAE', 'test_R2']].sort_values('test_RMSE'))


def test_tgpy_advanced():
    """Test advanced TGPY features."""
    print("\n" + "=" * 80)
    print("TGPY Advanced Features")
    print("=" * 80)
    
    if not TGPY_AVAILABLE:
        print("TGPY not available. Skipping test.")
        return
        
    # Generate non-linear data
    np.random.seed(42)
    n_samples = 100
    X = np.random.uniform(-5, 5, (n_samples, 2))
    # Complex non-linear function
    y = np.sin(X[:, 0]) * np.cos(X[:, 1]) + 0.1 * X[:, 0] * X[:, 1] + 0.1 * np.random.randn(n_samples)
    
    # Create dataframe
    data = pd.DataFrame(X, columns=['x1', 'x2'])
    data['y'] = y
    
    # Create task
    task = TaskRegr(
        id="tgpy_2d",
        data=data,
        target="y"
    )
    
    # Test different transport types
    transport_types = ['marginal', 'covariance']
    
    for transport in transport_types:
        print(f"\n Testing transport type: {transport}")
        
        learner = LearnerTGPRegressor(
            id=f"tgpy_{transport}",
            kernel='SE',
            transport=transport,
            n_iterations=100,
            batch_size=0.5  # Use mini-batches
        )
        
        try:
            # Use cross-validation
            cv = ResamplingCV(folds=3)
            bench = benchmark(
                tasks=[task],
                learners=[learner],
                resampling=cv,
                measures=[MeasureRegrRMSE()]
            )
            
            results = bench.aggregate('RMSE')
            mean_rmse = results['mean_test_RMSE'].iloc[0]
            std_rmse = results['std_test_RMSE'].iloc[0]
            
            print(f"   CV RMSE: {mean_rmse:.4f} (+/- {std_rmse:.4f})")
            
            # Get kernel parameters after training
            learner.train(task)
            params = learner.get_kernel_params()
            print(f"   Learned parameters: {params}")
            
        except Exception as e:
            print(f"   Error: {str(e)}")


if __name__ == "__main__":
    # Run all tests
    test_tgpy_simple()
    compare_with_baselines()
    test_tgpy_advanced()
    
    print("\n" + "=" * 80)
    print("TGPY Integration Test Complete!")
    print("=" * 80)
    
    if not TGPY_AVAILABLE:
        print("\nNote: To use TGPY with MLPY, install TGPY from the tgpy-master directory:")
        print("  1. Install IPython: pip install ipython")
        print("  2. Install TGPY: cd tgpy-master && pip install -e .")
    else:
        print("\nTGPY is successfully integrated with MLPY!")