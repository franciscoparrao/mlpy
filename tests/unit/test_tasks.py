"""Tests for Task classes."""

import pytest
import numpy as np
import pandas as pd

from mlpy.tasks import Task, TaskClassif, TaskRegr
from mlpy.backends import DataBackendPandas


class TestTaskClassif:
    """Test TaskClassif functionality."""
    
    @pytest.fixture
    def iris_data(self):
        """Create iris-like data."""
        np.random.seed(42)
        n = 150
        return pd.DataFrame({
            "sepal_length": np.random.normal(5.8, 0.8, n),
            "sepal_width": np.random.normal(3.0, 0.4, n),
            "petal_length": np.random.normal(3.8, 1.8, n),
            "petal_width": np.random.normal(1.2, 0.8, n),
            "species": np.random.choice(["setosa", "versicolor", "virginica"], n),
        })
    
    def test_create_task(self, iris_data):
        """Test creating a classification task."""
        task = TaskClassif(iris_data, target="species", id="iris", label="Iris Dataset")
        
        assert task.id == "iris"
        assert task.label == "Iris Dataset"
        assert task.task_type == "classif"
        assert task.nrow == 150
        assert task.ncol == 5  # 4 features + 1 target
        
        # Check roles
        assert task.target_names == ["species"]
        assert set(task.feature_names) == {"sepal_length", "sepal_width", "petal_length", "petal_width"}
    
    def test_class_properties(self, iris_data):
        """Test classification-specific properties."""
        task = TaskClassif(iris_data, target="species")
        
        assert task.n_classes == 3
        assert set(task.class_names) == {"setosa", "versicolor", "virginica"}
        assert "multiclass" in task._properties
    
    def test_binary_classification(self):
        """Test binary classification task."""
        data = pd.DataFrame({
            "x1": [1, 2, 3, 4, 5],
            "x2": [5, 4, 3, 2, 1],
            "y": [0, 1, 0, 1, 0],
        })
        
        task = TaskClassif(data, target="y")
        
        assert task.n_classes == 2
        assert task.class_names == ["0", "1"]
        assert task.positive == "1"  # Default positive class
        assert task.negative == "0"
        assert "binary" in task._properties
    
    def test_positive_class_specification(self):
        """Test specifying positive class."""
        data = pd.DataFrame({
            "feature": [1, 2, 3, 4],
            "outcome": ["healthy", "sick", "healthy", "sick"],
        })
        
        task = TaskClassif(data, target="outcome", positive="sick")
        
        assert task.positive == "sick"
        assert task.negative == "healthy"
    
    def test_data_access(self, iris_data):
        """Test accessing task data."""
        task = TaskClassif(iris_data, target="species")
        
        # Get all data
        df = task.data()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 150
        assert len(df.columns) == 5
        
        # Get subset of rows
        df_subset = task.data(rows=[0, 1, 2])
        assert len(df_subset) == 3
        
        # Get as array
        arr = task.data(data_format="array")
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (150, 5)
    
    def test_head(self, iris_data):
        """Test head method."""
        task = TaskClassif(iris_data, target="species")
        
        head = task.head(10)
        assert len(head) == 10
        assert "species" in head.columns
    
    def test_filter(self, iris_data):
        """Test filtering rows."""
        task = TaskClassif(iris_data, target="species")
        
        # Filter to first 50 rows
        task_filtered = task.filter(rows=list(range(50)))
        
        assert task_filtered.nrow == 50
        assert task_filtered.ncol == task.ncol
        
        # Original task unchanged
        assert task.nrow == 150
    
    def test_select(self, iris_data):
        """Test selecting features."""
        task = TaskClassif(iris_data, target="species")
        
        # Select only sepal features
        task_selected = task.select(cols=["sepal_length", "sepal_width"])
        
        assert len(task_selected.feature_names) == 2
        assert set(task_selected.feature_names) == {"sepal_length", "sepal_width"}
        assert task_selected.target_names == ["species"]  # Target unchanged
    
    def test_truth(self, iris_data):
        """Test getting true labels."""
        task = TaskClassif(iris_data, target="species")
        
        truth = task.truth()
        assert len(truth) == 150
        assert all(t in ["setosa", "versicolor", "virginica"] for t in truth)
        
        # Get truth for subset
        truth_subset = task.truth(rows=[0, 1, 2])
        assert len(truth_subset) == 3
    
    def test_validation_errors(self):
        """Test task validation."""
        data = pd.DataFrame({
            "x1": [1, 2, 3],
            "x2": [4, 5, 6],
        })
        
        # Multiple targets not allowed
        with pytest.raises(ValueError, match="exactly one target"):
            TaskClassif(data, target=["x1", "x2"])
        
        # Single class
        data["y"] = "A"
        with pytest.raises(ValueError, match="at least 2 classes"):
            TaskClassif(data, target="y")
        
        # Invalid positive class
        data["y"] = ["A", "B", "A"]
        with pytest.raises(ValueError, match="not found in target"):
            TaskClassif(data, target="y", positive="C")


class TestTaskRegr:
    """Test TaskRegr functionality."""
    
    @pytest.fixture
    def boston_data(self):
        """Create Boston housing-like data."""
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            "crim": np.random.exponential(3.6, n),
            "rm": np.random.normal(6.3, 0.7, n),
            "age": np.random.uniform(0, 100, n),
            "dis": np.random.exponential(3.8, n),
            "price": np.random.normal(22.5, 9.2, n),
        })
    
    def test_create_task(self, boston_data):
        """Test creating a regression task."""
        task = TaskRegr(boston_data, target="price", id="boston", label="Boston Housing")
        
        assert task.id == "boston"
        assert task.label == "Boston Housing"
        assert task.task_type == "regr"
        assert task.nrow == 100
        assert task.ncol == 5  # 4 features + 1 target
        
        # Check roles
        assert task.target_names == ["price"]
        assert set(task.feature_names) == {"crim", "rm", "age", "dis"}
    
    def test_numeric_target_validation(self):
        """Test that regression requires numeric target."""
        data = pd.DataFrame({
            "x1": [1, 2, 3, 4],
            "x2": [4, 3, 2, 1],
            "y": ["a", "b", "c", "d"],  # Non-numeric
        })
        
        with pytest.raises(ValueError, match="must be numeric"):
            TaskRegr(data, target="y")
    
    def test_truth(self, boston_data):
        """Test getting true values."""
        task = TaskRegr(boston_data, target="price")
        
        truth = task.truth()
        assert len(truth) == 100
        assert isinstance(truth, np.ndarray)
        assert truth.dtype == np.float64
    
    def test_cbind(self, boston_data):
        """Test adding columns."""
        task = TaskRegr(boston_data, target="price")
        
        # Add new features
        new_features = pd.DataFrame({
            "tax": np.random.uniform(200, 800, 100),
            "ptratio": np.random.normal(18.5, 2.2, 100),
        })
        
        task_extended = task.cbind(new_features)
        
        assert task_extended.nrow == task.nrow
        assert len(task_extended.feature_names) == len(task.feature_names) + 2
        assert "tax" in task_extended.feature_names
        assert "ptratio" in task_extended.feature_names
    
    def test_rbind(self, boston_data):
        """Test adding rows."""
        task = TaskRegr(boston_data[:50], target="price")
        
        # Add more data
        task_extended = task.rbind(boston_data[50:])
        
        assert task_extended.nrow == 100
        assert task_extended.ncol == task.ncol
    
    def test_formula(self, boston_data):
        """Test formula representation."""
        task = TaskRegr(boston_data, target="price")
        
        formula = task.formula
        assert "price ~" in formula
        assert "..." in formula  # Too many features to show all
    
    def test_col_roles(self, boston_data):
        """Test setting column roles."""
        task = TaskRegr(boston_data, target="price")
        
        # Set weight column
        boston_data["weights"] = np.random.uniform(0.5, 1.5, 100)
        task_weighted = TaskRegr(boston_data, target="price")
        task_weighted.set_col_roles({
            "weight": "weights",
            "feature": ["crim", "rm", "age", "dis"],  # Exclude weights from features
        })
        
        assert "weights" in task_weighted._col_roles["weight"]
        assert "weights" not in task_weighted.feature_names
        assert "weights" in task_weighted._properties
    
    def test_from_dict(self):
        """Test creating task from dictionary."""
        data = {
            "x1": np.array([1.0, 2.0, 3.0, 4.0]),
            "x2": np.array([4.0, 3.0, 2.0, 1.0]),
            "y": np.array([2.5, 3.5, 4.5, 5.5]),
        }
        
        task = TaskRegr(data, target="y")
        
        assert task.nrow == 4
        assert task.ncol == 3
        assert set(task.feature_names) == {"x1", "x2"}