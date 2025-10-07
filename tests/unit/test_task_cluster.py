"""
Tests for TaskCluster implementation.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs

from mlpy.tasks import TaskCluster
from mlpy.backends.pandas_backend import DataBackendPandas


class TestTaskCluster:
    """Test TaskCluster functionality."""
    
    @pytest.fixture
    def simple_data(self):
        """Create simple clustering dataset."""
        np.random.seed(42)
        X, _ = make_blobs(
            n_samples=100, 
            centers=3, 
            n_features=2, 
            random_state=42,
            cluster_std=1.0
        )
        df = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
        df['id'] = range(len(df))
        return df
    
    @pytest.fixture 
    def mixed_data(self):
        """Create dataset with mixed feature types."""
        np.random.seed(42)
        X, _ = make_blobs(n_samples=50, centers=2, n_features=3, random_state=42)
        df = pd.DataFrame(X, columns=['num1', 'num2', 'num3'])
        df['category'] = np.random.choice(['A', 'B', 'C'], len(df))
        df['text'] = [f'item_{i}' for i in range(len(df))]
        return df
    
    def test_task_creation_basic(self, simple_data):
        """Test basic task creation."""
        task = TaskCluster(simple_data)
        
        assert task.task_type == "cluster"
        assert task.nrow == 100
        assert len(task.feature_names) == 3  # feature_1, feature_2, id
        assert task.n_clusters is None
    
    def test_task_creation_with_exclude(self, simple_data):
        """Test task creation with excluded columns."""
        task = TaskCluster(simple_data, exclude=['id'])
        
        assert len(task.feature_names) == 2  # Only feature_1, feature_2
        assert 'id' not in task.feature_names
    
    def test_task_creation_with_n_clusters(self, simple_data):
        """Test task creation with specified n_clusters."""
        task = TaskCluster(simple_data, n_clusters=3, exclude=['id'])
        
        assert task.n_clusters == 3
        assert "k_3" in task._properties
    
    def test_numeric_categorical_features(self, mixed_data):
        """Test identification of numeric vs categorical features."""
        task = TaskCluster(mixed_data)
        
        numeric_features = task.get_numeric_features()
        categorical_features = task.get_categorical_features()
        
        assert len(numeric_features) == 3  # num1, num2, num3
        assert len(categorical_features) == 2  # category, text
        assert task.n_numeric_features == 3
        assert task.n_categorical_features == 2
    
    def test_suggest_n_clusters(self, simple_data):
        """Test cluster number suggestion."""
        task = TaskCluster(simple_data, exclude=['id'])
        
        suggestion = task.suggest_n_clusters(max_k=10)
        
        assert 'primary' in suggestion
        assert 'range' in suggestion
        assert 'heuristics' in suggestion
        assert 'reasoning' in suggestion
        
        # Should suggest reasonable range
        assert suggestion['primary'] >= 2
        assert len(suggestion['range']) > 0
        assert all(k >= 2 for k in suggestion['range'])
    
    def test_preprocess_features(self, mixed_data):
        """Test feature preprocessing."""
        task = TaskCluster(mixed_data)
        
        # Test standardization
        processed = task.preprocess_features(standardize=True)
        numeric_features = task.get_numeric_features()
        
        for col in numeric_features:
            if col in processed.columns:
                # Should be approximately standardized (mean~0, std~1)
                assert abs(processed[col].mean()) < 0.1
                assert abs(processed[col].std() - 1.0) < 0.1
    
    def test_distance_matrix(self, simple_data):
        """Test distance matrix computation.""" 
        task = TaskCluster(simple_data, exclude=['id'])
        
        # Test different metrics
        for metric in ['euclidean', 'manhattan', 'cosine']:
            distances = task.distance_matrix(metric=metric)
            
            assert distances.shape == (100, 100)
            assert np.allclose(distances, distances.T)  # Symmetric
            assert np.allclose(np.diag(distances), 0)   # Zero diagonal
            assert np.all(distances >= 0)               # Non-negative
    
    def test_evaluate_clustering(self, simple_data):
        """Test clustering evaluation."""
        task = TaskCluster(simple_data, exclude=['id'])
        
        # Create dummy cluster labels
        cluster_labels = np.array([0, 1, 2] * 33 + [0])  # 100 points, 3 clusters
        
        # Test silhouette score
        silhouette = task.evaluate_clustering(cluster_labels, metric='silhouette')
        assert isinstance(silhouette, float)
        assert -1 <= silhouette <= 1
        
        # Test inertia
        inertia = task.evaluate_clustering(cluster_labels, metric='inertia')
        assert isinstance(inertia, float)
        assert inertia <= 0  # Negative (we return -inertia)
    
    def test_validation_errors(self):
        """Test validation errors."""
        # Too few observations
        with pytest.raises(ValueError, match="at least 2 observations"):
            TaskCluster(pd.DataFrame({'x': [1]}))
        
        # No features
        with pytest.raises(ValueError, match="at least one feature"):
            TaskCluster(
                pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]}), 
                exclude=['x', 'y']
            )
        
        # Invalid n_clusters
        with pytest.raises(ValueError, match="Number of clusters must be >= 2"):
            TaskCluster(
                pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]}),
                n_clusters=1
            )
        
        # n_clusters >= observations
        with pytest.raises(ValueError, match="must be less than observations"):
            TaskCluster(
                pd.DataFrame({'x': [1, 2], 'y': [4, 5]}),
                n_clusters=3
            )
    
    def test_from_dict(self):
        """Test task creation from dictionary."""
        data_dict = {
            'x': [1, 2, 3, 4, 5],
            'y': [2, 4, 1, 3, 5],
            'z': ['a', 'b', 'c', 'd', 'e']
        }
        
        task = TaskCluster(data_dict)
        
        assert task.nrow == 5
        assert len(task.feature_names) == 3
        assert task.task_type == "cluster"
    
    def test_from_backend(self, simple_data):
        """Test task creation from DataBackend."""
        backend = DataBackendPandas(simple_data)
        task = TaskCluster(backend, exclude=['id'])
        
        assert task.nrow == 100
        assert len(task.feature_names) == 2
        assert task._backend is backend
    
    def test_properties(self, mixed_data):
        """Test task properties."""
        task = TaskCluster(mixed_data)
        
        properties = task._properties
        assert "cluster" in properties
        assert "unsupervised" in properties
        assert "numeric" in properties      # Has numeric features
        assert "categorical" in properties  # Has categorical features
    
    def test_task_registry(self):
        """Test that TaskCluster is registered."""
        from mlpy.utils.registry import mlpy_tasks
        
        assert "cluster" in mlpy_tasks
        assert "clustering" in mlpy_tasks  # Alias
        
        # Test creating task via registry
        data = pd.DataFrame({
            'x': [1, 2, 3, 4],
            'y': [2, 1, 4, 3]
        })
        
        task = mlpy_tasks.get("cluster")(data)
        assert isinstance(task, TaskCluster)
    
    def test_warnings(self):
        """Test warnings for problematic data."""
        # Non-numeric features should trigger warning
        data = pd.DataFrame({
            'text1': ['a', 'b', 'c', 'd'],
            'text2': ['x', 'y', 'z', 'w']
        })
        
        with pytest.warns(UserWarning, match="Non-numeric features found"):
            TaskCluster(data)
        
        # Few observations relative to features should trigger warning
        data = pd.DataFrame({
            'x1': [1, 2],
            'x2': [3, 4], 
            'x3': [5, 6],
            'x4': [7, 8],
            'x5': [9, 10]
        })
        
        with pytest.warns(UserWarning, match="Few observations.*relative to features"):
            TaskCluster(data)
    
    def test_evaluate_clustering_errors(self, simple_data):
        """Test clustering evaluation errors."""
        task = TaskCluster(simple_data, exclude=['id'])
        
        # Wrong number of labels
        with pytest.raises(ValueError, match="cluster_labels length.*must match"):
            task.evaluate_clustering([0, 1, 2])  # Only 3 labels for 100 points
        
        # No numeric features for evaluation
        text_data = pd.DataFrame({'text': ['a', 'b', 'c', 'd']})
        text_task = TaskCluster(text_data)
        
        with pytest.raises(ValueError, match="requires numeric features"):
            text_task.evaluate_clustering([0, 1, 0, 1])
    
    def test_distance_matrix_errors(self, simple_data):
        """Test distance matrix computation errors."""
        # No numeric features
        text_data = pd.DataFrame({'text': ['a', 'b', 'c']})
        text_task = TaskCluster(text_data)
        
        with pytest.raises(ValueError, match="requires numeric features"):
            text_task.distance_matrix()
        
        # Unknown metric
        task = TaskCluster(simple_data, exclude=['id'])
        with pytest.raises(ValueError, match="Unknown metric"):
            task.distance_matrix(metric='unknown')
    
    def test_evaluate_clustering_unknown_metric(self, simple_data):
        """Test evaluation with unknown metric."""
        task = TaskCluster(simple_data, exclude=['id'])
        cluster_labels = [0] * 100
        
        with pytest.raises(ValueError, match="Unknown evaluation metric"):
            task.evaluate_clustering(cluster_labels, metric='unknown')