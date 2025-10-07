"""
Example of using Spatial Tasks in MLPY.

This example demonstrates how to create and use spatial tasks
for geographic machine learning problems.
"""

import numpy as np
import pandas as pd
from mlpy.tasks import TaskClassifSpatial, TaskRegrSpatial, create_spatial_task


def create_landslide_data():
    """Create synthetic landslide susceptibility data."""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate spatial coordinates (UTM Zone 19S - Chile)
    x = np.random.uniform(300000, 400000, n_samples)
    y = np.random.uniform(7400000, 7500000, n_samples)
    
    # Generate terrain features
    elevation = np.random.uniform(0, 3000, n_samples)
    slope = np.random.uniform(0, 45, n_samples)
    aspect = np.random.uniform(0, 360, n_samples)
    curvature = np.random.normal(0, 0.1, n_samples)
    twi = np.random.uniform(4, 12, n_samples)  # Topographic Wetness Index
    
    # Generate vegetation indices
    ndvi = np.random.uniform(-0.2, 0.8, n_samples)
    evi = np.random.uniform(-0.1, 0.6, n_samples)
    
    # Generate landslide susceptibility based on features
    # Higher slope and lower NDVI increase landslide probability
    landslide_prob = 1 / (1 + np.exp(-(
        0.05 * slope - 
        2 * ndvi + 
        0.001 * elevation - 
        0.1 * twi + 
        np.random.normal(0, 0.5, n_samples)
    )))
    
    landslide = (landslide_prob > 0.5).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'x': x,
        'y': y,
        'elevation': elevation,
        'slope': slope,
        'aspect': aspect,
        'curvature': curvature,
        'twi': twi,
        'ndvi': ndvi,
        'evi': evi,
        'landslide': landslide,
        'susceptibility': landslide_prob  # For regression example
    })
    
    return df


def example_classification():
    """Example of spatial classification task."""
    print("=" * 60)
    print("SPATIAL CLASSIFICATION TASK EXAMPLE")
    print("=" * 60)
    
    # Create data
    df = create_landslide_data()
    
    # Create spatial classification task
    task = TaskClassifSpatial(
        data=df,
        target='landslide',
        coordinate_names=['x', 'y'],
        crs='EPSG:32719',  # UTM Zone 19S
        coords_as_features=False,
        id='landslide_classification',
        label='Landslide Susceptibility Classification'
    )
    
    # Display task information
    print(f"\nTask: {task}")
    print(f"Task type: {task.task_type}")
    print(f"Number of samples: {task.nrow}")
    print(f"Number of features: {len(task.feature_names)}")
    print(f"Features: {task.feature_names[:5]}...")
    print(f"Target: {task.target_names}")
    print(f"Classes: {task.class_names}")
    print(f"Class distribution: {task.n_classes} classes")
    
    # Display spatial information
    print(f"\nSpatial Information:")
    print(f"Coordinate columns: {task.coordinate_names}")
    print(f"CRS: {task.crs}")
    print(f"Coords as features: {task.coords_as_features}")
    print(f"Spatial extent: {task.spatial_extent}")
    
    # Get coordinates
    coords = task.coordinates()
    print(f"\nCoordinates shape: {coords.shape}")
    print(f"First 3 coordinates:\n{coords[:3]}")
    
    # Find spatial neighbors
    neighbors = task.spatial_neighbors(n_neighbors=3)
    print(f"\nNeighbors of first point: {neighbors[0]}")
    
    # Calculate distance matrix for a subset
    distances = task.distance_matrix(rows=list(range(5)))
    print(f"\nDistance matrix (first 5 points):\n{distances}")
    
    return task


def example_regression():
    """Example of spatial regression task."""
    print("\n" + "=" * 60)
    print("SPATIAL REGRESSION TASK EXAMPLE")
    print("=" * 60)
    
    # Create data
    df = create_landslide_data()
    
    # Create spatial regression task
    task = TaskRegrSpatial(
        data=df,
        target='susceptibility',
        coordinate_names=['x', 'y'],
        crs=32719,  # Can also use integer EPSG code
        coords_as_features=False,
        id='landslide_regression',
        label='Landslide Susceptibility Regression'
    )
    
    # Display task information
    print(f"\nTask: {task}")
    print(f"Task type: {task.task_type}")
    print(f"Target statistics:")
    truth = task.truth()
    print(f"  - Mean: {truth.mean():.3f}")
    print(f"  - Std: {truth.std():.3f}")
    print(f"  - Min: {truth.min():.3f}")
    print(f"  - Max: {truth.max():.3f}")
    
    return task


def example_auto_detection():
    """Example of automatic task type detection."""
    print("\n" + "=" * 60)
    print("AUTOMATIC TASK TYPE DETECTION")
    print("=" * 60)
    
    df = create_landslide_data()
    
    # Automatically detect classification task
    task_classif = create_spatial_task(
        data=df,
        target='landslide',
        coordinate_names=['x', 'y'],
        crs='EPSG:32719'
    )
    print(f"\nAuto-detected for 'landslide': {type(task_classif).__name__}")
    
    # Automatically detect regression task
    task_regr = create_spatial_task(
        data=df,
        target='susceptibility',
        coordinate_names=['x', 'y'],
        crs='EPSG:32719'
    )
    print(f"Auto-detected for 'susceptibility': {type(task_regr).__name__}")


def example_coords_as_features():
    """Example using coordinates as features."""
    print("\n" + "=" * 60)
    print("COORDINATES AS FEATURES")
    print("=" * 60)
    
    df = create_landslide_data()
    
    # Task without coordinates as features
    task_no_coords = TaskClassifSpatial(
        data=df,
        target='landslide',
        coordinate_names=['x', 'y'],
        coords_as_features=False
    )
    
    # Task with coordinates as features
    task_with_coords = TaskClassifSpatial(
        data=df,
        target='landslide',
        coordinate_names=['x', 'y'],
        coords_as_features=True
    )
    
    print(f"\nWithout coordinates as features:")
    print(f"  Features: {task_no_coords.feature_names}")
    
    print(f"\nWith coordinates as features:")
    print(f"  Features: {task_with_coords.feature_names}")


def example_geopandas_export():
    """Example of exporting to GeoPandas."""
    print("\n" + "=" * 60)
    print("GEOPANDAS EXPORT")
    print("=" * 60)
    
    df = create_landslide_data()
    
    task = TaskClassifSpatial(
        data=df,
        target='landslide',
        coordinate_names=['x', 'y'],
        crs='EPSG:32719'
    )
    
    try:
        # Convert to GeoPandas
        gdf = task.to_geopandas(rows=list(range(10)))
        print(f"\nGeoPandas DataFrame created:")
        print(f"  Shape: {gdf.shape}")
        print(f"  CRS: {gdf.crs}")
        print(f"  Geometry type: {gdf.geometry.type.unique()}")
        print(f"\nFirst 3 rows:")
        print(gdf[['x', 'y', 'landslide', 'geometry']].head(3))
    except ImportError as e:
        print(f"\nGeoPandas not available: {e}")
        print("Install with: pip install geopandas")


def main():
    """Run all examples."""
    # Run examples
    task_classif = example_classification()
    task_regr = example_regression()
    example_auto_detection()
    example_coords_as_features()
    example_geopandas_export()
    
    print("\n" + "=" * 60)
    print("EXAMPLES COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return task_classif, task_regr


if __name__ == "__main__":
    main()