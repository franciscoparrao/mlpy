"""
Tests for spatial tasks in MLPY.
"""

import pytest
import numpy as np
import pandas as pd
from mlpy.tasks.spatial import (
    TaskClassifSpatial, 
    TaskRegrSpatial, 
    create_spatial_task
)


@pytest.fixture
def spatial_data():
    """Create sample spatial data for testing."""
    np.random.seed(42)
    n = 100
    
    df = pd.DataFrame({
        'x': np.random.uniform(0, 100, n),
        'y': np.random.uniform(0, 100, n),
        'lon': np.random.uniform(-180, 180, n),
        'lat': np.random.uniform(-90, 90, n),
        'feature1': np.random.randn(n),
        'feature2': np.random.randn(n),
        'feature3': np.random.randn(n),
        'class_target': np.random.choice(['A', 'B', 'C'], n),
        'binary_target': np.random.choice([0, 1], n),
        'numeric_target': np.random.randn(n) * 10 + 50
    })
    
    return df


class TestTaskClassifSpatial:
    """Tests for TaskClassifSpatial."""
    
    def test_basic_creation(self, spatial_data):
        """Test basic task creation."""
        task = TaskClassifSpatial(
            data=spatial_data,
            target='class_target',
            coordinate_names=['x', 'y'],
            crs='EPSG:4326'
        )
        
        assert task.task_type == 'classif_spatial'
        assert task.coordinate_names == ['x', 'y']
        assert task.crs == 'EPSG:4326'
        assert task.n_classes == 3
        assert set(task.class_names) == {'A', 'B', 'C'}
    
    def test_binary_classification(self, spatial_data):
        """Test binary classification task."""
        task = TaskClassifSpatial(
            data=spatial_data,
            target='binary_target',
            coordinate_names=['lon', 'lat'],
            positive='1'
        )
        
        assert task.n_classes == 2
        assert task.positive == '1'
        assert task.negative == '0'
    
    def test_coords_not_as_features(self, spatial_data):
        """Test that coordinates are excluded from features by default."""
        task = TaskClassifSpatial(
            data=spatial_data,
            target='class_target',
            coordinate_names=['x', 'y'],
            coords_as_features=False
        )
        
        assert 'x' not in task.feature_names
        assert 'y' not in task.feature_names
        assert 'feature1' in task.feature_names
    
    def test_coords_as_features(self, spatial_data):
        """Test including coordinates as features."""
        task = TaskClassifSpatial(
            data=spatial_data,
            target='class_target',
            coordinate_names=['x', 'y'],
            coords_as_features=True
        )
        
        assert 'x' in task.feature_names
        assert 'y' in task.feature_names
    
    def test_coordinates_retrieval(self, spatial_data):
        """Test getting coordinates."""
        task = TaskClassifSpatial(
            data=spatial_data,
            target='class_target',
            coordinate_names=['x', 'y']
        )
        
        coords = task.coordinates()
        assert coords.shape == (100, 2)
        assert coords.dtype == np.float64
    
    def test_spatial_extent(self, spatial_data):
        """Test spatial extent calculation."""
        task = TaskClassifSpatial(
            data=spatial_data,
            target='class_target',
            coordinate_names=['x', 'y']
        )
        
        extent = task.spatial_extent
        assert 'xmin' in extent
        assert 'xmax' in extent
        assert 'ymin' in extent
        assert 'ymax' in extent
        assert extent['xmin'] < extent['xmax']
        assert extent['ymin'] < extent['ymax']
    
    def test_distance_matrix(self, spatial_data):
        """Test distance matrix calculation."""
        task = TaskClassifSpatial(
            data=spatial_data[:10],  # Use subset for speed
            target='class_target',
            coordinate_names=['x', 'y']
        )
        
        distances = task.distance_matrix()
        assert distances.shape == (10, 10)
        assert np.allclose(distances, distances.T)  # Should be symmetric
        assert np.allclose(np.diag(distances), 0)  # Diagonal should be zeros
    
    def test_spatial_neighbors(self, spatial_data):
        """Test finding spatial neighbors."""
        task = TaskClassifSpatial(
            data=spatial_data[:20],  # Use subset
            target='class_target',
            coordinate_names=['x', 'y']
        )
        
        neighbors = task.spatial_neighbors(n_neighbors=3)
        assert len(neighbors) == 20
        assert all(len(n) == 3 for n in neighbors.values())
        
        # Test with max_distance
        neighbors_dist = task.spatial_neighbors(n_neighbors=5, max_distance=10)
        assert all(len(n) <= 5 for n in neighbors_dist.values())
    
    def test_integer_crs(self, spatial_data):
        """Test using integer EPSG code for CRS."""
        task = TaskClassifSpatial(
            data=spatial_data,
            target='class_target',
            coordinate_names=['lon', 'lat'],
            crs=4326
        )
        
        assert task.crs == 4326
    
    def test_missing_coordinates_error(self, spatial_data):
        """Test error when coordinate columns don't exist."""
        with pytest.warns(UserWarning):
            task = TaskClassifSpatial(
                data=spatial_data,
                target='class_target',
                coordinate_names=['missing_x', 'missing_y']
            )
    
    def test_repr(self, spatial_data):
        """Test string representation."""
        task = TaskClassifSpatial(
            data=spatial_data,
            target='class_target',
            coordinate_names=['x', 'y'],
            crs='EPSG:4326'
        )
        
        repr_str = repr(task)
        assert 'TaskClassifSpatial' in repr_str
        assert '[coords: ' in repr_str
        assert '[crs: ' in repr_str


class TestTaskRegrSpatial:
    """Tests for TaskRegrSpatial."""
    
    def test_basic_creation(self, spatial_data):
        """Test basic regression task creation."""
        task = TaskRegrSpatial(
            data=spatial_data,
            target='numeric_target',
            coordinate_names=['x', 'y'],
            crs='EPSG:32719'
        )
        
        assert task.task_type == 'regr_spatial'
        assert task.coordinate_names == ['x', 'y']
        assert task.crs == 'EPSG:32719'
    
    def test_multi_target(self, spatial_data):
        """Test multi-target regression."""
        spatial_data['target2'] = np.random.randn(100)
        
        task = TaskRegrSpatial(
            data=spatial_data,
            target=['numeric_target', 'target2'],
            coordinate_names=['lon', 'lat']
        )
        
        assert len(task.target_names) == 2
        assert 'numeric_target' in task.target_names
        assert 'target2' in task.target_names
    
    def test_truth_values(self, spatial_data):
        """Test getting true target values."""
        task = TaskRegrSpatial(
            data=spatial_data,
            target='numeric_target',
            coordinate_names=['x', 'y']
        )
        
        truth = task.truth()
        assert len(truth) == 100
        assert truth.dtype == np.float64


class TestCreateSpatialTask:
    """Tests for create_spatial_task convenience function."""
    
    def test_auto_detect_classification(self, spatial_data):
        """Test automatic detection of classification task."""
        task = create_spatial_task(
            data=spatial_data,
            target='class_target',
            coordinate_names=['x', 'y']
        )
        
        assert isinstance(task, TaskClassifSpatial)
    
    def test_auto_detect_regression(self, spatial_data):
        """Test automatic detection of regression task."""
        task = create_spatial_task(
            data=spatial_data,
            target='numeric_target',
            coordinate_names=['x', 'y']
        )
        
        assert isinstance(task, TaskRegrSpatial)
    
    def test_explicit_type(self, spatial_data):
        """Test explicit task type specification."""
        task_classif = create_spatial_task(
            data=spatial_data,
            target='numeric_target',  # Numeric but force classification
            task_type='classif',
            coordinate_names=['x', 'y']
        )
        
        assert isinstance(task_classif, TaskClassifSpatial)
        
        task_regr = create_spatial_task(
            data=spatial_data,
            target='binary_target',  # Binary but force regression
            task_type='regr',
            coordinate_names=['x', 'y']
        )
        
        assert isinstance(task_regr, TaskRegrSpatial)
    
    def test_dict_input(self):
        """Test creating task from dictionary."""
        data_dict = {
            'x': [1, 2, 3, 4, 5],
            'y': [1, 2, 3, 4, 5],
            'feature': [0.1, 0.2, 0.3, 0.4, 0.5],
            'target': ['A', 'B', 'A', 'B', 'A']
        }
        
        task = create_spatial_task(
            data=data_dict,
            target='target',
            coordinate_names=['x', 'y']
        )
        
        assert isinstance(task, TaskClassifSpatial)
        assert task.nrow == 5


class TestGeoPandasIntegration:
    """Tests for GeoPandas integration."""
    
    def test_to_geopandas(self, spatial_data):
        """Test conversion to GeoPandas DataFrame."""
        task = TaskClassifSpatial(
            data=spatial_data,
            target='class_target',
            coordinate_names=['lon', 'lat'],
            crs='EPSG:4326'
        )
        
        try:
            import geopandas as gpd
            gdf = task.to_geopandas(rows=list(range(10)))
            
            assert isinstance(gdf, gpd.GeoDataFrame)
            assert len(gdf) == 10
            assert gdf.crs == 'EPSG:4326'
            assert all(gdf.geometry.type == 'Point')
        except ImportError:
            pytest.skip("GeoPandas not installed")
    
    def test_geopandas_error_without_coords(self, spatial_data):
        """Test error when trying to convert without enough coordinates."""
        task = TaskClassifSpatial(
            data=spatial_data,
            target='class_target',
            coordinate_names=['x'],  # Only one coordinate
            crs='EPSG:4326'
        )
        
        try:
            import geopandas as gpd
            with pytest.raises(ValueError):
                task.to_geopandas()
        except ImportError:
            pytest.skip("GeoPandas not installed")


class TestSpatialProperties:
    """Tests for spatial properties and methods."""
    
    def test_empty_coordinates(self, spatial_data):
        """Test behavior with no coordinates specified."""
        task = TaskClassifSpatial(
            data=spatial_data,
            target='class_target'
        )
        
        assert task.coordinate_names == []
        assert task.spatial_extent is None
        
        with pytest.raises(ValueError):
            task.coordinates()
    
    def test_custom_spatial_extent(self, spatial_data):
        """Test providing custom spatial extent."""
        custom_extent = {
            'xmin': 0,
            'xmax': 100,
            'ymin': -50,
            'ymax': 50
        }
        
        task = TaskClassifSpatial(
            data=spatial_data,
            target='class_target',
            coordinate_names=['x', 'y'],
            spatial_extent=custom_extent
        )
        
        assert task.spatial_extent == custom_extent


if __name__ == "__main__":
    pytest.main([__file__, "-v"])