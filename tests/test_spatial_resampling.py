"""
Tests for spatial resampling strategies in MLPY.
"""

import pytest
import numpy as np
import pandas as pd
from mlpy.tasks.spatial import TaskClassifSpatial
from mlpy.resamplings.spatial import (
    SpatialKFold,
    SpatialBlockCV,
    SpatialBufferCV,
    SpatialEnvironmentalCV
)


@pytest.fixture
def spatial_task():
    """Create a spatial task for testing."""
    np.random.seed(42)
    n = 100
    
    # Create spatial data
    x = np.random.uniform(0, 100, n)
    y = np.random.uniform(0, 100, n)
    
    # Create features
    feature1 = x * 0.1 + y * 0.05 + np.random.randn(n)
    feature2 = np.random.randn(n)
    
    # Create spatially structured target
    target = ((x > 50) & (y > 50)).astype(int)
    
    df = pd.DataFrame({
        'x': x,
        'y': y,
        'feature1': feature1,
        'feature2': feature2,
        'target': target
    })
    
    task = TaskClassifSpatial(
        data=df,
        target='target',
        coordinate_names=['x', 'y'],
        crs='EPSG:4326'
    )
    
    return task


@pytest.fixture
def spatial_task_with_env():
    """Create a spatial task with environmental features."""
    np.random.seed(42)
    n = 100
    
    x = np.random.uniform(0, 100, n)
    y = np.random.uniform(0, 100, n)
    
    # Environmental features
    elevation = 1000 + x * 10 + y * 5 + np.random.randn(n) * 50
    slope = np.abs(np.random.randn(n) * 10 + 15)
    ndvi = np.random.uniform(0.2, 0.8, n)
    
    target = ((elevation > 1500) & (slope > 20)).astype(int)
    
    df = pd.DataFrame({
        'x': x,
        'y': y,
        'elevation': elevation,
        'slope': slope,
        'ndvi': ndvi,
        'target': target
    })
    
    task = TaskClassifSpatial(
        data=df,
        target='target',
        coordinate_names=['x', 'y']
    )
    
    return task


class TestSpatialKFold:
    """Tests for SpatialKFold."""
    
    def test_basic_instantiation(self, spatial_task):
        """Test basic instantiation and fold creation."""
        cv = SpatialKFold(n_folds=5, clustering_method='kmeans', random_state=42)
        cv.instantiate(spatial_task)
        
        assert cv.iters == 5
        assert cv.is_instantiated
        
        # Check all folds
        for i in range(cv.iters):
            train_idx = cv.train_set(i)
            test_idx = cv.test_set(i)
            
            # Check no overlap
            assert len(np.intersect1d(train_idx, test_idx)) == 0
            
            # Check all indices are covered
            all_idx = np.concatenate([train_idx, test_idx])
            assert len(all_idx) == spatial_task.nrow
            assert len(np.unique(all_idx)) == spatial_task.nrow
    
    def test_kmeans_clustering(self, spatial_task):
        """Test K-means clustering method."""
        cv = SpatialKFold(n_folds=3, clustering_method='kmeans', random_state=42)
        cv.instantiate(spatial_task)
        
        # Check clusters were created
        assert 'clusters' in cv._instance
        clusters = cv._instance['clusters']
        assert len(clusters) == spatial_task.nrow
        assert len(np.unique(clusters)) == 3
    
    def test_grid_clustering(self, spatial_task):
        """Test grid clustering method."""
        cv = SpatialKFold(n_folds=4, clustering_method='grid')
        cv.instantiate(spatial_task)
        
        # Check clusters were created
        assert 'clusters' in cv._instance
        clusters = cv._instance['clusters']
        assert len(clusters) == spatial_task.nrow
        assert len(np.unique(clusters)) <= 4
    
    def test_invalid_clustering_method(self, spatial_task):
        """Test error with invalid clustering method."""
        cv = SpatialKFold(n_folds=3, clustering_method='invalid')
        
        with pytest.raises(ValueError, match="Unknown clustering method"):
            cv.instantiate(spatial_task)
    
    def test_fold_sizes(self, spatial_task):
        """Test that fold sizes are roughly balanced."""
        cv = SpatialKFold(n_folds=5, clustering_method='kmeans', random_state=42)
        cv.instantiate(spatial_task)
        
        test_sizes = []
        for i in range(cv.iters):
            test_idx = cv.test_set(i)
            test_sizes.append(len(test_idx))
        
        # Check that test sizes are somewhat balanced
        mean_size = np.mean(test_sizes)
        for size in test_sizes:
            assert abs(size - mean_size) / mean_size < 0.5  # Within 50% of mean
    
    def test_reproducibility(self, spatial_task):
        """Test that results are reproducible with same random state."""
        cv1 = SpatialKFold(n_folds=3, clustering_method='kmeans', random_state=42)
        cv1.instantiate(spatial_task)
        
        cv2 = SpatialKFold(n_folds=3, clustering_method='kmeans', random_state=42)
        cv2.instantiate(spatial_task)
        
        for i in range(cv1.iters):
            train1 = cv1.train_set(i)
            train2 = cv2.train_set(i)
            assert np.array_equal(train1, train2)


class TestSpatialBlockCV:
    """Tests for SpatialBlockCV."""
    
    def test_basic_instantiation(self, spatial_task):
        """Test basic instantiation with integer blocks."""
        cv = SpatialBlockCV(n_blocks=9)
        cv.instantiate(spatial_task)
        
        assert cv.is_instantiated
        assert cv.iters <= 9
    
    def test_grid_specification(self, spatial_task):
        """Test specifying grid dimensions."""
        cv = SpatialBlockCV(n_blocks=(3, 3))
        cv.instantiate(spatial_task)
        
        assert cv.n_rows == 3
        assert cv.n_cols == 3
        
        # Check blocks were created
        assert 'blocks' in cv._instance
        blocks = cv._instance['blocks']
        assert len(blocks) == spatial_task.nrow
    
    def test_systematic_assignment(self, spatial_task):
        """Test systematic block assignment."""
        cv = SpatialBlockCV(n_blocks=(2, 2), method='systematic')
        cv.instantiate(spatial_task)
        
        # Check fold assignments follow checkerboard pattern
        assignments = cv._instance['fold_assignments']
        assert len(assignments) == 4  # 2x2 grid
    
    def test_random_assignment(self, spatial_task):
        """Test random block assignment."""
        cv = SpatialBlockCV(n_blocks=4, method='random', random_state=42)
        cv.instantiate(spatial_task)
        
        # Check reproducibility
        cv2 = SpatialBlockCV(n_blocks=4, method='random', random_state=42)
        cv2.instantiate(spatial_task)
        
        for i in range(cv.iters):
            test1 = cv.test_set(i)
            test2 = cv2.test_set(i)
            assert np.array_equal(test1, test2)
    
    def test_invalid_method(self, spatial_task):
        """Test error with invalid assignment method."""
        cv = SpatialBlockCV(n_blocks=4, method='invalid')
        
        with pytest.raises(ValueError, match="Unknown method"):
            cv.instantiate(spatial_task)
    
    def test_all_points_assigned(self, spatial_task):
        """Test that all points are assigned to blocks."""
        cv = SpatialBlockCV(n_blocks=(3, 3))
        cv.instantiate(spatial_task)
        
        all_indices = set()
        for i in range(cv.iters):
            train_idx = cv.train_set(i)
            test_idx = cv.test_set(i)
            all_indices.update(train_idx)
            all_indices.update(test_idx)
        
        assert len(all_indices) == spatial_task.nrow


class TestSpatialBufferCV:
    """Tests for SpatialBufferCV."""
    
    def test_basic_instantiation(self, spatial_task):
        """Test basic instantiation."""
        cv = SpatialBufferCV(
            buffer_distance=10.0,
            test_size=0.2,
            n_folds=3,
            random_state=42
        )
        cv.instantiate(spatial_task)
        
        assert cv.iters == 3
        assert cv.is_instantiated
    
    def test_buffer_enforcement(self, spatial_task):
        """Test that buffer distance is enforced."""
        buffer_dist = 20.0
        cv = SpatialBufferCV(
            buffer_distance=buffer_dist,
            test_size=10,  # Fixed number of test points
            n_folds=2,
            random_state=42
        )
        cv.instantiate(spatial_task)
        
        coords = spatial_task.coordinates()
        
        for i in range(cv.iters):
            train_idx = cv.train_set(i)
            test_idx = cv.test_set(i)
            
            if len(train_idx) > 0 and len(test_idx) > 0:
                # Check minimum distance between train and test
                train_coords = coords[train_idx]
                test_coords = coords[test_idx]
                
                for test_pt in test_coords:
                    distances = np.linalg.norm(train_coords - test_pt, axis=1)
                    assert np.min(distances) > buffer_dist
    
    def test_test_size_float(self, spatial_task):
        """Test with test_size as proportion."""
        cv = SpatialBufferCV(
            buffer_distance=5.0,
            test_size=0.3,
            n_folds=2
        )
        cv.instantiate(spatial_task)
        
        test_idx = cv.test_set(0)
        expected_size = int(spatial_task.nrow * 0.3)
        assert len(test_idx) == expected_size
    
    def test_test_size_int(self, spatial_task):
        """Test with test_size as absolute number."""
        cv = SpatialBufferCV(
            buffer_distance=5.0,
            test_size=15,
            n_folds=2
        )
        cv.instantiate(spatial_task)
        
        test_idx = cv.test_set(0)
        assert len(test_idx) == 15


class TestSpatialEnvironmentalCV:
    """Tests for SpatialEnvironmentalCV."""
    
    def test_basic_instantiation(self, spatial_task_with_env):
        """Test basic instantiation."""
        cv = SpatialEnvironmentalCV(
            n_folds=4,
            spatial_weight=0.5,
            environmental_cols=['elevation', 'slope'],
            method='kmeans',
            random_state=42
        )
        cv.instantiate(spatial_task_with_env)
        
        assert cv.iters == 4
        assert cv.is_instantiated
    
    def test_spatial_only(self, spatial_task):
        """Test with spatial distance only (no environmental)."""
        cv = SpatialEnvironmentalCV(
            n_folds=3,
            spatial_weight=1.0,  # Only spatial
            environmental_cols=None,
            method='kmeans',
            random_state=42
        )
        cv.instantiate(spatial_task)
        
        # Should work like spatial k-fold
        assert 'clusters' in cv._instance
        assert len(cv._instance['clusters']) == spatial_task.nrow
    
    def test_environmental_included(self, spatial_task_with_env):
        """Test that environmental features are used."""
        cv = SpatialEnvironmentalCV(
            n_folds=3,
            spatial_weight=0.6,
            environmental_cols=['elevation', 'slope', 'ndvi'],
            method='kmeans',
            random_state=42
        )
        cv.instantiate(spatial_task_with_env)
        
        # Check combined features were created
        assert 'combined_features' in cv._instance
        combined = cv._instance['combined_features']
        
        # Should have more dimensions than just spatial (2)
        assert combined.shape[1] > 2
    
    def test_hierarchical_clustering(self, spatial_task):
        """Test hierarchical clustering method."""
        cv = SpatialEnvironmentalCV(
            n_folds=3,
            spatial_weight=0.7,
            method='hierarchical'
        )
        cv.instantiate(spatial_task)
        
        assert 'clusters' in cv._instance
        clusters = cv._instance['clusters']
        assert len(np.unique(clusters)) == 3
    
    def test_invalid_method(self, spatial_task):
        """Test error with invalid clustering method."""
        cv = SpatialEnvironmentalCV(
            n_folds=3,
            method='invalid'
        )
        
        with pytest.raises(ValueError, match="Unknown method"):
            cv.instantiate(spatial_task)
    
    def test_weight_bounds(self, spatial_task):
        """Test spatial weight boundaries."""
        # Weight = 0 (only environmental, but no env cols specified)
        cv1 = SpatialEnvironmentalCV(
            n_folds=3,
            spatial_weight=0.0,
            environmental_cols=None
        )
        cv1.instantiate(spatial_task)
        
        # Weight = 1 (only spatial)
        cv2 = SpatialEnvironmentalCV(
            n_folds=3,
            spatial_weight=1.0
        )
        cv2.instantiate(spatial_task)
        
        # Both should work
        assert cv1.is_instantiated
        assert cv2.is_instantiated


class TestSpatialValidation:
    """Test validation of spatial tasks."""
    
    def test_non_spatial_task_error(self):
        """Test error when using non-spatial task."""
        from mlpy.tasks import TaskClassif
        
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.choice(['A', 'B'], 100)
        })
        
        task = TaskClassif(data=df, target='target')
        
        cv = SpatialKFold(n_folds=3)
        
        with pytest.raises(ValueError, match="Task must be a spatial task"):
            cv.instantiate(task)
    
    def test_missing_coordinates_error(self):
        """Test error when coordinates are missing."""
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'target': np.random.choice([0, 1], 100)
        })
        
        # Create task without coordinates
        task = TaskClassifSpatial(
            data=df,
            target='target',
            coordinate_names=[]  # No coordinates
        )
        
        cv = SpatialKFold(n_folds=3)
        
        with pytest.raises(ValueError, match="Task must be a spatial task"):
            cv.instantiate(task)


class TestIntegration:
    """Integration tests for spatial resampling."""
    
    def test_all_methods_compatible(self, spatial_task):
        """Test that all spatial CV methods work with same task."""
        methods = [
            SpatialKFold(n_folds=3),
            SpatialBlockCV(n_blocks=4),
            SpatialBufferCV(buffer_distance=10, test_size=0.2, n_folds=3),
            SpatialEnvironmentalCV(n_folds=3)
        ]
        
        for cv in methods:
            cv.instantiate(spatial_task)
            assert cv.is_instantiated
            
            # Check first fold
            train = cv.train_set(0)
            test = cv.test_set(0)
            assert len(train) > 0
            assert len(test) > 0
            assert len(np.intersect1d(train, test)) == 0
    
    def test_iteration_interface(self, spatial_task):
        """Test iteration through folds."""
        cv = SpatialKFold(n_folds=3)
        cv.instantiate(spatial_task)
        
        # Test accessing all folds
        all_test_indices = []
        for i in range(cv.iters):
            train = cv.train_set(i)
            test = cv.test_set(i)
            all_test_indices.extend(test)
            
            assert len(train) > 0
            assert len(test) > 0
        
        # All indices should be covered
        assert len(np.unique(all_test_indices)) == spatial_task.nrow
    
    def test_error_before_instantiation(self):
        """Test error when accessing folds before instantiation."""
        cv = SpatialKFold(n_folds=3)
        
        with pytest.raises(RuntimeError, match="must be instantiated"):
            cv.train_set(0)
        
        with pytest.raises(RuntimeError, match="must be instantiated"):
            cv.test_set(0)
    
    def test_out_of_range_fold(self, spatial_task):
        """Test error when accessing invalid fold index."""
        cv = SpatialKFold(n_folds=3)
        cv.instantiate(spatial_task)
        
        with pytest.raises(IndexError):
            cv.train_set(3)  # Only 0, 1, 2 are valid
        
        with pytest.raises(IndexError):
            cv.test_set(-1)  # Negative index


if __name__ == "__main__":
    pytest.main([__file__, "-v"])