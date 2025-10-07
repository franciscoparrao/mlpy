"""
Example of Spatial Cross-Validation in MLPY.

This example demonstrates various spatial resampling strategies
to handle spatial autocorrelation in geographic machine learning.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

from mlpy.tasks import TaskClassifSpatial
from mlpy.resamplings import (
    SpatialKFold,
    SpatialBlockCV,
    SpatialBufferCV,
    SpatialEnvironmentalCV
)


def create_spatial_data(n_samples=500, spatial_pattern='clusters'):
    """
    Create synthetic spatial data with different patterns.
    
    Parameters
    ----------
    n_samples : int
        Number of samples
    spatial_pattern : str
        Type of spatial pattern: 'clusters', 'gradient', 'random'
    """
    np.random.seed(42)
    
    if spatial_pattern == 'clusters':
        # Create clustered spatial data
        n_clusters = 5
        cluster_centers = np.random.uniform(0, 100, (n_clusters, 2))
        
        x, y = [], []
        for _ in range(n_samples):
            cluster = np.random.choice(n_clusters)
            center = cluster_centers[cluster]
            point = center + np.random.randn(2) * 10
            x.append(point[0])
            y.append(point[1])
        
        x = np.array(x)
        y = np.array(y)
        
    elif spatial_pattern == 'gradient':
        # Create data with spatial gradient
        x = np.random.uniform(0, 100, n_samples)
        y = np.random.uniform(0, 100, n_samples)
        
    else:  # random
        x = np.random.uniform(0, 100, n_samples)
        y = np.random.uniform(0, 100, n_samples)
    
    # Create features with spatial autocorrelation
    elevation = 1000 + x * 10 + y * 5 + np.random.randn(n_samples) * 50
    slope = np.abs(np.gradient(elevation)[0]) + np.random.randn(n_samples) * 2
    ndvi = 0.7 - (elevation - 1000) / 2000 + np.random.randn(n_samples) * 0.1
    
    # Create target with spatial structure
    if spatial_pattern == 'clusters':
        # Class depends on which cluster
        target_prob = np.zeros(n_samples)
        for i in range(n_samples):
            min_dist = np.min([np.linalg.norm([x[i] - c[0], y[i] - c[1]]) 
                              for c in cluster_centers])
            target_prob[i] = 1 / (1 + np.exp(-0.1 * (30 - min_dist)))
    else:
        # Gradient pattern
        target_prob = 1 / (1 + np.exp(-(0.02 * x + 0.01 * y - 2 + np.random.randn(n_samples) * 0.5)))
    
    target = (target_prob > 0.5).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'x': x,
        'y': y,
        'elevation': elevation,
        'slope': slope,
        'ndvi': ndvi,
        'landslide': target
    })
    
    return df


def visualize_cv_splits(task, cv, title="Spatial CV Splits"):
    """
    Visualize spatial cross-validation splits.
    
    Parameters
    ----------
    task : TaskClassifSpatial
        Spatial task with data
    cv : SpatialResampling
        Spatial CV strategy
    title : str
        Plot title
    """
    # Instantiate CV with task
    cv.instantiate(task)
    
    # Get coordinates
    coords = task.coordinates()
    
    # Create figure with subplots for each fold
    n_folds = cv.iters
    n_cols = min(3, n_folds)
    n_rows = int(np.ceil(n_folds / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_folds == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else axes
    
    for fold_idx in range(n_folds):
        ax = axes[fold_idx] if n_folds > 1 else axes[0]
        
        # Get train and test indices
        train_idx = cv.train_set(fold_idx)
        test_idx = cv.test_set(fold_idx)
        
        # Plot points
        if len(train_idx) > 0:
            ax.scatter(coords[train_idx, 0], coords[train_idx, 1], 
                      c='blue', alpha=0.5, s=10, label='Train')
        if len(test_idx) > 0:
            ax.scatter(coords[test_idx, 0], coords[test_idx, 1], 
                      c='red', alpha=0.8, s=20, label='Test')
        
        ax.set_title(f'Fold {fold_idx + 1}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        ax.set_aspect('equal')
    
    # Hide extra subplots
    for idx in range(n_folds, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def example_spatial_kfold():
    """Example of Spatial K-Fold CV."""
    print("=" * 60)
    print("SPATIAL K-FOLD CROSS-VALIDATION")
    print("=" * 60)
    
    # Create clustered spatial data
    df = create_spatial_data(n_samples=300, spatial_pattern='clusters')
    
    # Create spatial task
    task = TaskClassifSpatial(
        data=df,
        target='landslide',
        coordinate_names=['x', 'y'],
        crs='EPSG:32719'
    )
    
    print(f"\nTask created with {task.nrow} samples")
    print(f"Spatial extent: {task.spatial_extent}")
    
    # Create Spatial K-Fold CV with K-Means clustering
    cv_kmeans = SpatialKFold(
        n_folds=5,
        clustering_method='kmeans',
        random_state=42
    )
    
    print("\n1. K-Means Clustering Method:")
    cv_kmeans.instantiate(task)
    
    for i in range(3):  # Show first 3 folds
        train = cv_kmeans.train_set(i)
        test = cv_kmeans.test_set(i)
        print(f"  Fold {i+1}: Train={len(train)}, Test={len(test)}")
    
    # Visualize
    visualize_cv_splits(task, cv_kmeans, "Spatial K-Fold (K-Means)")
    
    # Create Spatial K-Fold CV with Grid clustering
    cv_grid = SpatialKFold(
        n_folds=4,
        clustering_method='grid'
    )
    
    print("\n2. Grid Clustering Method:")
    cv_grid.instantiate(task)
    
    for i in range(cv_grid.iters):
        train = cv_grid.train_set(i)
        test = cv_grid.test_set(i)
        print(f"  Fold {i+1}: Train={len(train)}, Test={len(test)}")
    
    # Visualize
    visualize_cv_splits(task, cv_grid, "Spatial K-Fold (Grid)")


def example_spatial_block():
    """Example of Spatial Block CV."""
    print("\n" + "=" * 60)
    print("SPATIAL BLOCK CROSS-VALIDATION")
    print("=" * 60)
    
    # Create gradient spatial data
    df = create_spatial_data(n_samples=400, spatial_pattern='gradient')
    
    # Create spatial task
    task = TaskClassifSpatial(
        data=df,
        target='landslide',
        coordinate_names=['x', 'y']
    )
    
    # Create Spatial Block CV with 3x3 grid
    cv_block = SpatialBlockCV(
        n_blocks=(3, 3),
        method='systematic'
    )
    
    print(f"\nBlock CV with 3x3 grid (systematic assignment)")
    cv_block.instantiate(task)
    
    for i in range(min(5, cv_block.iters)):
        train = cv_block.train_set(i)
        test = cv_block.test_set(i)
        print(f"  Fold {i+1}: Train={len(train)}, Test={len(test)}")
    
    # Visualize
    visualize_cv_splits(task, cv_block, "Spatial Block CV (3x3 Systematic)")
    
    # Create random block assignment
    cv_block_random = SpatialBlockCV(
        n_blocks=9,
        method='random',
        random_state=42
    )
    
    print(f"\nBlock CV with random assignment")
    cv_block_random.instantiate(task)
    
    # Visualize
    visualize_cv_splits(task, cv_block_random, "Spatial Block CV (Random)")


def example_spatial_buffer():
    """Example of Spatial Buffer CV."""
    print("\n" + "=" * 60)
    print("SPATIAL BUFFER CROSS-VALIDATION")
    print("=" * 60)
    
    # Create random spatial data
    df = create_spatial_data(n_samples=200, spatial_pattern='random')
    
    # Create spatial task
    task = TaskClassifSpatial(
        data=df,
        target='landslide',
        coordinate_names=['x', 'y']
    )
    
    # Create Spatial Buffer CV
    cv_buffer = SpatialBufferCV(
        buffer_distance=15.0,  # 15 units buffer
        test_size=0.2,
        n_folds=3,
        random_state=42
    )
    
    print(f"\nBuffer CV with distance=15.0, test_size=0.2")
    cv_buffer.instantiate(task)
    
    for i in range(cv_buffer.iters):
        train = cv_buffer.train_set(i)
        test = cv_buffer.test_set(i)
        print(f"  Fold {i+1}: Train={len(train)}, Test={len(test)}")
        print(f"    Note: Training points are >15 units from test points")
    
    # Visualize
    visualize_cv_splits(task, cv_buffer, "Spatial Buffer CV (15 unit buffer)")


def example_spatial_environmental():
    """Example of Spatial-Environmental CV."""
    print("\n" + "=" * 60)
    print("SPATIAL-ENVIRONMENTAL CROSS-VALIDATION")
    print("=" * 60)
    
    # Create data with environmental gradients
    df = create_spatial_data(n_samples=300, spatial_pattern='gradient')
    
    # Create spatial task
    task = TaskClassifSpatial(
        data=df,
        target='landslide',
        coordinate_names=['x', 'y']
    )
    
    # Create Spatial-Environmental CV
    cv_env = SpatialEnvironmentalCV(
        n_folds=4,
        spatial_weight=0.6,  # 60% spatial, 40% environmental
        environmental_cols=['elevation', 'slope', 'ndvi'],
        method='kmeans',
        random_state=42
    )
    
    print(f"\nEnvironmental CV with spatial_weight=0.6")
    print(f"Environmental features: elevation, slope, ndvi")
    cv_env.instantiate(task)
    
    for i in range(cv_env.iters):
        train = cv_env.train_set(i)
        test = cv_env.test_set(i)
        print(f"  Fold {i+1}: Train={len(train)}, Test={len(test)}")
    
    # Visualize
    visualize_cv_splits(task, cv_env, 
                       "Spatial-Environmental CV (60% spatial, 40% environmental)")


def compare_cv_strategies():
    """Compare different CV strategies on the same data."""
    print("\n" + "=" * 60)
    print("COMPARISON OF CV STRATEGIES")
    print("=" * 60)
    
    # Create data
    df = create_spatial_data(n_samples=200, spatial_pattern='clusters')
    
    # Create spatial task
    task = TaskClassifSpatial(
        data=df,
        target='landslide',
        coordinate_names=['x', 'y']
    )
    
    # Define CV strategies
    strategies = {
        'Spatial K-Fold': SpatialKFold(n_folds=4, clustering_method='kmeans'),
        'Spatial Block': SpatialBlockCV(n_blocks=(2, 2)),
        'Spatial Buffer': SpatialBufferCV(buffer_distance=20, test_size=0.25, n_folds=4),
        'Spatial-Environmental': SpatialEnvironmentalCV(
            n_folds=4, spatial_weight=0.7, environmental_cols=['elevation']
        )
    }
    
    # Compare strategies
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (name, cv) in enumerate(strategies.items()):
        ax = axes[idx]
        cv.instantiate(task)
        
        # Get first fold
        train_idx = cv.train_set(0)
        test_idx = cv.test_set(0)
        
        coords = task.coordinates()
        
        # Plot
        ax.scatter(coords[train_idx, 0], coords[train_idx, 1], 
                  c='blue', alpha=0.5, s=20, label='Train')
        ax.scatter(coords[test_idx, 0], coords[test_idx, 1], 
                  c='red', alpha=0.8, s=30, label='Test')
        
        ax.set_title(name)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        ax.set_aspect('equal')
        
        # Add statistics
        train_pct = len(train_idx) / task.nrow * 100
        test_pct = len(test_idx) / task.nrow * 100
        ax.text(0.02, 0.98, f'Train: {train_pct:.1f}%\nTest: {test_pct:.1f}%',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Comparison of Spatial CV Strategies (First Fold)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def main():
    """Run all spatial CV examples."""
    
    # Run individual examples
    example_spatial_kfold()
    example_spatial_block()
    example_spatial_buffer()
    example_spatial_environmental()
    
    # Compare strategies
    compare_cv_strategies()
    
    print("\n" + "=" * 60)
    print("SPATIAL CV EXAMPLES COMPLETED!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. Spatial K-Fold: Creates spatially contiguous folds using clustering")
    print("2. Spatial Block: Divides space into rectangular blocks")
    print("3. Spatial Buffer: Ensures minimum distance between train/test")
    print("4. Spatial-Environmental: Considers both space and environment")
    print("\nChoose based on your data's spatial structure and autocorrelation!")


if __name__ == "__main__":
    main()