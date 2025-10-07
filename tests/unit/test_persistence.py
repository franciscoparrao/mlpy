"""Tests for model persistence functionality."""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

from mlpy.tasks import TaskClassif, TaskRegr
from mlpy.learners import LearnerClassifDebug, LearnerRegrDebug
from mlpy.learners import learner_sklearn
from mlpy.pipelines import PipeOpScale, PipeOpLearner, linear_pipeline
from mlpy.persistence import (
    save_model, load_model, ModelBundle,
    PickleSerializer, JoblibSerializer, JSONSerializer,
    ModelRegistry, export_model_package
)

# Check for optional dependencies
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir)


@pytest.fixture
def classification_task():
    """Create a simple classification task."""
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 5)
    y = np.random.choice([0, 1], n_samples)
    
    data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    data['target'] = y
    
    return TaskClassif(data=data, target='target', id='test_classif')


@pytest.fixture
def regression_task():
    """Create a simple regression task."""
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 5)
    y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n_samples) * 0.1
    
    data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    data['target'] = y
    
    return TaskRegr(data=data, target='target', id='test_regr')


class TestPickleSerializer:
    """Test pickle serialization."""
    
    def test_serialize_deserialize_learner(self, temp_dir, classification_task):
        """Test serializing and deserializing a learner."""
        # Train learner
        learner = LearnerClassifDebug(id='debug_clf')
        learner.train(classification_task)
        
        # Save
        path = temp_dir / "model.pkl"
        serializer = PickleSerializer()
        metadata = serializer.serialize(learner, path)
        
        assert path.exists()
        assert metadata['serializer'] == 'pickle'
        
        # Load
        loaded_learner = serializer.deserialize(path)
        
        assert isinstance(loaded_learner, LearnerClassifDebug)
        assert loaded_learner.id == 'debug_clf'
        assert loaded_learner.is_trained
        
    def test_can_serialize(self):
        """Test can_serialize method."""
        serializer = PickleSerializer()
        
        # Should be able to serialize most objects
        assert serializer.can_serialize({"a": 1})
        assert serializer.can_serialize([1, 2, 3])
        assert serializer.can_serialize(LearnerClassifDebug())
        
    def test_file_extension(self):
        """Test default file extension."""
        serializer = PickleSerializer()
        assert serializer.file_extension == ".pkl"


@pytest.mark.skipif(not JOBLIB_AVAILABLE, reason="joblib not available")
class TestJoblibSerializer:
    """Test joblib serialization."""
    
    def test_serialize_deserialize_with_compression(self, temp_dir, regression_task):
        """Test joblib with compression."""
        # Train learner
        learner = LearnerRegrDebug(id='debug_regr')
        learner.train(regression_task)
        
        # Save with compression
        path = temp_dir / "model.joblib"
        serializer = JoblibSerializer(compression=3)
        metadata = serializer.serialize(learner, path)
        
        assert path.exists()
        assert metadata['compression'] == 3
        
        # Load
        loaded_learner = serializer.deserialize(path)
        
        assert isinstance(loaded_learner, LearnerRegrDebug)
        assert loaded_learner.id == 'debug_regr'
        assert loaded_learner.is_trained
        
    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn not available")
    def test_serialize_sklearn_model(self, temp_dir, classification_task):
        """Test serializing sklearn model."""
        # Train sklearn model
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        learner = learner_sklearn(rf, id='rf_clf')
        learner.train(classification_task)
        
        # Save
        path = temp_dir / "sklearn_model.joblib"
        serializer = JoblibSerializer()
        serializer.serialize(learner, path)
        
        # Load
        loaded_learner = serializer.deserialize(path)
        
        # Make predictions to verify it works
        pred1 = learner.predict(classification_task)
        pred2 = loaded_learner.predict(classification_task)
        
        np.testing.assert_array_equal(pred1.response, pred2.response)


class TestJSONSerializer:
    """Test JSON serialization."""
    
    def test_serialize_metadata(self, temp_dir):
        """Test serializing metadata."""
        metadata = {
            "model_name": "test_model",
            "accuracy": 0.95,
            "params": {"n_estimators": 100, "max_depth": 10}
        }
        
        path = temp_dir / "metadata.json"
        serializer = JSONSerializer()
        serializer.serialize(metadata, path)
        
        # Load and verify
        loaded = serializer.deserialize(path)
        assert loaded == metadata
        
    def test_serialize_bundle_metadata(self, temp_dir):
        """Test serializing ModelBundle metadata."""
        learner = LearnerClassifDebug()
        bundle = ModelBundle(learner, metadata={"test": "value"})
        
        path = temp_dir / "bundle.json"
        serializer = JSONSerializer()
        
        # Can only serialize metadata, not the model itself
        assert serializer.can_serialize(bundle)
        serializer.serialize(bundle, path)
        
        # Load - should get metadata
        with open(path, 'r') as f:
            data = json.load(f)
            
        assert data['metadata']['test'] == 'value'
        assert data['_is_bundle'] is True
        
    def test_file_extension(self):
        """Test default file extension."""
        serializer = JSONSerializer()
        assert serializer.file_extension == ".json"


class TestSaveLoadModel:
    """Test high-level save/load functions."""
    
    def test_save_load_basic(self, temp_dir, classification_task):
        """Test basic save and load."""
        # Train model
        learner = LearnerClassifDebug(id='test_model')
        learner.train(classification_task)
        
        # Save
        path = save_model(learner, temp_dir / "model.pkl")
        assert path.exists()
        
        # Load
        loaded = load_model(path)
        assert isinstance(loaded, LearnerClassifDebug)
        assert loaded.id == 'test_model'
        assert loaded.is_trained
        
    def test_save_with_metadata(self, temp_dir, regression_task):
        """Test saving with metadata."""
        # Train model
        learner = LearnerRegrDebug()
        learner.train(regression_task)
        
        # Save with metadata
        metadata = {
            "experiment": "test_001",
            "rmse": 0.123,
            "dataset": "test_data"
        }
        
        path = save_model(
            learner,
            temp_dir / "model_with_meta.pkl",
            metadata=metadata
        )
        
        # Load bundle to access metadata
        bundle = load_model(path, return_bundle=True)
        assert isinstance(bundle, ModelBundle)
        assert bundle.metadata['experiment'] == 'test_001'
        assert bundle.metadata['rmse'] == 0.123
        
    def test_auto_serializer_selection(self, temp_dir, classification_task):
        """Test automatic serializer selection."""
        learner = LearnerClassifDebug()
        learner.train(classification_task)
        
        # Should automatically use joblib or pickle
        path = save_model(learner, temp_dir / "auto_model", serializer="auto")
        assert path.exists()
        
        loaded = load_model(path, serializer="auto")
        assert isinstance(loaded, LearnerClassifDebug)
        
    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn not available")
    def test_save_pipeline(self, temp_dir, regression_task):
        """Test saving a pipeline."""
        # Create pipeline
        rf = RandomForestRegressor(n_estimators=5, random_state=42)
        learner = learner_sklearn(rf)
        
        pipeline = linear_pipeline(
            PipeOpScale(id="scale"),
            PipeOpLearner(learner, id="learner")
        )
        
        # Train pipeline
        pipeline.train(regression_task)
        
        # Save
        path = save_model(pipeline, temp_dir / "pipeline.pkl")
        
        # Load
        loaded_pipeline = load_model(path)
        
        # Verify predictions are same
        pred1 = pipeline.predict(regression_task)
        pred2 = loaded_pipeline.predict(regression_task)
        
        np.testing.assert_array_almost_equal(
            pred1['output'].response,
            pred2['output'].response
        )


class TestModelRegistry:
    """Test model registry functionality."""
    
    def test_register_and_load_model(self, temp_dir, classification_task):
        """Test registering and loading a model."""
        registry = ModelRegistry(temp_dir / "registry")
        
        # Train model
        learner = LearnerClassifDebug(id='registry_test')
        learner.train(classification_task)
        
        # Register
        path = registry.register_model(
            learner,
            name="test_model",
            tags=["debug", "classification"],
            metadata={"accuracy": 0.95}
        )
        
        assert path.exists()
        
        # Load
        loaded = registry.load_model("test_model")
        assert isinstance(loaded, LearnerClassifDebug)
        assert loaded.id == 'registry_test'
        
        # Load with metadata
        loaded_model, metadata = registry.load_model(
            "test_model",
            return_metadata=True
        )
        assert metadata['tags'] == ["debug", "classification"]
        assert metadata['accuracy'] == 0.95
        
    def test_model_versioning(self, temp_dir):
        """Test model versioning in registry."""
        registry = ModelRegistry(temp_dir / "registry")
        
        # Register multiple versions
        for i in range(3):
            learner = LearnerClassifDebug(id=f'model_v{i}')
            registry.register_model(
                learner,
                name="versioned_model",
                version=f"v{i}"
            )
            
        # List versions
        models = registry.list_models()
        assert "versioned_model" in models
        assert set(models["versioned_model"]) == {"v0", "v1", "v2"}
        
        # Load specific version
        loaded = registry.load_model("versioned_model", version="v1")
        assert loaded.id == 'model_v1'
        
        # Load latest (should be v2)
        loaded_latest = registry.load_model("versioned_model")
        assert loaded_latest.id == 'model_v2'
        
    def test_delete_model(self, temp_dir):
        """Test deleting models from registry."""
        registry = ModelRegistry(temp_dir / "registry")
        
        # Register model
        learner = LearnerRegrDebug()
        registry.register_model(learner, name="to_delete", version="v1")
        registry.register_model(learner, name="to_delete", version="v2")
        
        # Delete specific version
        registry.delete_model("to_delete", version="v1")
        models = registry.list_models()
        assert "v1" not in models["to_delete"]
        assert "v2" in models["to_delete"]
        
        # Delete all versions
        registry.delete_model("to_delete")
        models = registry.list_models()
        assert "to_delete" not in models
        
    def test_get_metadata(self, temp_dir):
        """Test getting model metadata."""
        registry = ModelRegistry(temp_dir / "registry")
        
        # Register with metadata
        learner = LearnerClassifDebug()
        registry.register_model(
            learner,
            name="meta_model",
            version="v1",
            metadata={"note": "test metadata"}
        )
        
        # Get metadata
        metadata = registry.get_metadata("meta_model", version="v1")
        assert metadata['name'] == "meta_model"
        assert metadata['version'] == "v1"
        assert metadata['note'] == "test metadata"


class TestExportModelPackage:
    """Test model package export functionality."""
    
    def test_export_package(self, temp_dir, regression_task):
        """Test exporting a model package."""
        # Train model
        learner = LearnerRegrDebug(id='export_test')
        learner.train(regression_task)
        
        # Export package
        package_path = temp_dir / "model_package.zip"
        result_path = export_model_package(
            learner,
            package_path,
            name="MyModel",
            metadata={"author": "Test User"}
        )
        
        assert result_path.exists()
        assert result_path.suffix == '.zip'
        
        # Verify package contents
        import zipfile
        with zipfile.ZipFile(package_path, 'r') as zf:
            files = zf.namelist()
            assert 'model.pkl' in files
            assert 'requirements.txt' in files
            assert 'example.py' in files
            assert 'README.md' in files
            
            # Check requirements
            with zf.open('requirements.txt') as f:
                requirements = f.read().decode('utf-8')
                assert 'mlpy' in requirements


class TestModelBundleChecksum:
    """Test model checksum functionality."""
    
    def test_bundle_checksum(self):
        """Test that bundle generates checksums."""
        learner = LearnerClassifDebug(id='checksum_test')
        bundle = ModelBundle(learner, metadata={"test": True})
        
        checksum1 = bundle.get_checksum()
        assert isinstance(checksum1, str)
        assert len(checksum1) == 32  # MD5 hash length
        
        # Same bundle should have same checksum
        checksum2 = bundle.get_checksum()
        assert checksum1 == checksum2