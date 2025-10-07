"""
Tests for Model Registry.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from mlpy.registry import (
    ModelRegistry, ModelVersion, ModelMetadata,
    FileSystemRegistry, ModelStage,
    generate_model_id, compare_models, 
    validate_model_name, validate_version_string
)
from mlpy.learners import LearnerClassifSklearn, LearnerRegrSklearn
from mlpy.tasks import TaskClassif, TaskRegr

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.datasets import make_classification, make_regression


class TestModelMetadata:
    """Test ModelMetadata class."""
    
    def test_metadata_creation(self):
        """Test creating model metadata."""
        metadata = ModelMetadata(
            name="test_model",
            version="1.0.0",
            description="Test model for unit tests",
            author="test_user",
            tags={"environment": "test", "framework": "sklearn"},
            metrics={"accuracy": 0.95, "f1_score": 0.93},
            parameters={"n_estimators": 100, "max_depth": 5},
            stage=ModelStage.DEVELOPMENT
        )
        
        assert metadata.name == "test_model"
        assert metadata.version == "1.0.0"
        assert metadata.author == "test_user"
        assert metadata.tags["environment"] == "test"
        assert metadata.metrics["accuracy"] == 0.95
        assert metadata.stage == ModelStage.DEVELOPMENT
    
    def test_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        metadata = ModelMetadata(
            name="test_model",
            version="1.0.0",
            stage=ModelStage.PRODUCTION
        )
        
        data = metadata.to_dict()
        assert data["name"] == "test_model"
        assert data["version"] == "1.0.0"
        assert data["stage"] == "production"
        assert "created_at" in data
        assert "updated_at" in data
    
    def test_metadata_from_dict(self):
        """Test creating metadata from dictionary."""
        data = {
            "name": "test_model",
            "version": "2.0.0",
            "stage": "staging",
            "created_at": "2024-01-01T12:00:00",
            "updated_at": "2024-01-02T12:00:00",
            "metrics": {"rmse": 0.5}
        }
        
        metadata = ModelMetadata.from_dict(data)
        assert metadata.name == "test_model"
        assert metadata.version == "2.0.0"
        assert metadata.stage == ModelStage.STAGING
        assert isinstance(metadata.created_at, datetime)
        assert metadata.metrics["rmse"] == 0.5


class TestModelVersion:
    """Test ModelVersion class."""
    
    @pytest.fixture
    def sample_model(self):
        """Create a sample model."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(10)])
        df["target"] = y
        
        task = TaskClassif(df, target="target")
        learner = LearnerClassifSklearn(LogisticRegression(random_state=42))
        learner.train(task)
        return learner
    
    def test_model_version_creation(self, sample_model):
        """Test creating a model version."""
        metadata = ModelMetadata(
            name="classifier",
            version="1.0.0"
        )
        
        model_version = ModelVersion(
            model=sample_model,
            metadata=metadata
        )
        
        assert model_version.model == sample_model
        assert model_version.metadata.name == "classifier"
        assert model_version.model_id != ""  # Auto-generated
    
    def test_model_version_update_metrics(self, sample_model):
        """Test updating model metrics."""
        metadata = ModelMetadata(name="classifier", version="1.0.0")
        model_version = ModelVersion(model=sample_model, metadata=metadata)
        
        model_version.update_metrics({"accuracy": 0.92, "precision": 0.90})
        
        assert model_version.metadata.metrics["accuracy"] == 0.92
        assert model_version.metadata.metrics["precision"] == 0.90
    
    def test_model_version_add_tags(self, sample_model):
        """Test adding tags to model."""
        metadata = ModelMetadata(name="classifier", version="1.0.0")
        model_version = ModelVersion(model=sample_model, metadata=metadata)
        
        model_version.add_tags({"dataset": "iris", "validated": "true"})
        
        assert model_version.metadata.tags["dataset"] == "iris"
        assert model_version.metadata.tags["validated"] == "true"
    
    def test_model_version_set_stage(self, sample_model):
        """Test setting model stage."""
        metadata = ModelMetadata(name="classifier", version="1.0.0")
        model_version = ModelVersion(model=sample_model, metadata=metadata)
        
        model_version.set_stage(ModelStage.PRODUCTION)
        
        assert model_version.metadata.stage == ModelStage.PRODUCTION


class TestFileSystemRegistry:
    """Test FileSystemRegistry implementation."""
    
    @pytest.fixture
    def temp_registry_path(self):
        """Create temporary directory for registry."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def registry(self, temp_registry_path):
        """Create a FileSystemRegistry instance."""
        return FileSystemRegistry(registry_path=temp_registry_path)
    
    @pytest.fixture
    def sample_models(self):
        """Create sample models for testing."""
        # Classification model
        X_clf, y_clf = make_classification(n_samples=100, n_features=10, random_state=42)
        df_clf = pd.DataFrame(X_clf, columns=[f"feat_{i}" for i in range(10)])
        df_clf["target"] = y_clf
        task_clf = TaskClassif(df_clf, target="target")
        
        learner_clf = LearnerClassifSklearn(LogisticRegression(random_state=42))
        learner_clf.train(task_clf)
        
        # Regression model
        X_reg, y_reg = make_regression(n_samples=100, n_features=10, random_state=42)
        df_reg = pd.DataFrame(X_reg, columns=[f"feat_{i}" for i in range(10)])
        df_reg["target"] = y_reg
        task_reg = TaskRegr(df_reg, target="target")
        
        learner_reg = LearnerRegrSklearn(LinearRegression())
        learner_reg.train(task_reg)
        
        return learner_clf, learner_reg
    
    def test_registry_initialization(self, temp_registry_path):
        """Test registry initialization."""
        registry = FileSystemRegistry(registry_path=temp_registry_path)
        
        assert registry.registry_path == temp_registry_path
        assert (temp_registry_path / "models").exists()
        assert (temp_registry_path / "index.json").exists()
    
    def test_register_model(self, registry, sample_models):
        """Test registering a model."""
        learner_clf, _ = sample_models
        
        model_version = registry.register_model(
            model=learner_clf,
            name="test_classifier",
            description="Test classification model",
            author="test_user",
            metrics={"accuracy": 0.95}
        )
        
        assert model_version.metadata.name == "test_classifier"
        assert model_version.metadata.version == "1.0.0"  # First version
        assert model_version.metadata.author == "test_user"
        assert model_version.metadata.metrics["accuracy"] == 0.95
    
    def test_get_model(self, registry, sample_models):
        """Test retrieving a model."""
        learner_clf, _ = sample_models
        
        # Register model
        registry.register_model(
            model=learner_clf,
            name="test_classifier",
            version="1.0.0"
        )
        
        # Retrieve model
        retrieved = registry.get_model("test_classifier", version="1.0.0")
        
        assert retrieved is not None
        assert retrieved.metadata.name == "test_classifier"
        assert retrieved.metadata.version == "1.0.0"
        assert retrieved.model is not None
    
    def test_get_latest_model(self, registry, sample_models):
        """Test retrieving latest version of a model."""
        learner_clf, _ = sample_models
        
        # Register multiple versions
        registry.register_model(learner_clf, "test_model", version="1.0.0")
        registry.register_model(learner_clf, "test_model", version="1.0.1")
        registry.register_model(learner_clf, "test_model", version="1.0.2")
        
        # Get latest
        latest = registry.get_model("test_model")
        
        assert latest.metadata.version == "1.0.2"
    
    def test_list_models(self, registry, sample_models):
        """Test listing all models."""
        learner_clf, learner_reg = sample_models
        
        registry.register_model(learner_clf, "classifier_1")
        registry.register_model(learner_reg, "regressor_1")
        registry.register_model(learner_clf, "classifier_2")
        
        models = registry.list_models()
        
        assert len(models) == 3
        assert "classifier_1" in models
        assert "regressor_1" in models
        assert "classifier_2" in models
    
    def test_list_versions(self, registry, sample_models):
        """Test listing versions of a model."""
        learner_clf, _ = sample_models
        
        registry.register_model(learner_clf, "test_model", version="1.0.0")
        registry.register_model(learner_clf, "test_model", version="1.1.0")
        registry.register_model(learner_clf, "test_model", version="2.0.0")
        
        versions = registry.list_versions("test_model")
        
        assert len(versions) == 3
        assert "1.0.0" in versions
        assert "1.1.0" in versions
        assert "2.0.0" in versions
    
    def test_delete_model_version(self, registry, sample_models):
        """Test deleting a specific model version."""
        learner_clf, _ = sample_models
        
        registry.register_model(learner_clf, "test_model", version="1.0.0")
        registry.register_model(learner_clf, "test_model", version="1.0.1")
        
        # Delete version 1.0.0
        success = registry.delete_model("test_model", version="1.0.0")
        
        assert success
        versions = registry.list_versions("test_model")
        assert len(versions) == 1
        assert "1.0.1" in versions
        assert "1.0.0" not in versions
    
    def test_delete_all_versions(self, registry, sample_models):
        """Test deleting all versions of a model."""
        learner_clf, _ = sample_models
        
        registry.register_model(learner_clf, "test_model", version="1.0.0")
        registry.register_model(learner_clf, "test_model", version="1.0.1")
        
        # Delete all versions
        success = registry.delete_model("test_model")
        
        assert success
        assert "test_model" not in registry.list_models()
    
    def test_update_model_stage(self, registry, sample_models):
        """Test updating model stage."""
        learner_clf, _ = sample_models
        
        registry.register_model(learner_clf, "test_model", version="1.0.0")
        
        # Update stage
        success = registry.update_model_stage(
            "test_model", "1.0.0", ModelStage.PRODUCTION
        )
        
        assert success
        model = registry.get_model("test_model", version="1.0.0")
        assert model.metadata.stage == ModelStage.PRODUCTION
    
    def test_promote_model(self, registry, sample_models):
        """Test promoting a model to production."""
        learner_clf, _ = sample_models
        
        # Register two versions
        registry.register_model(learner_clf, "test_model", version="1.0.0")
        registry.update_model_stage("test_model", "1.0.0", ModelStage.PRODUCTION)
        
        registry.register_model(learner_clf, "test_model", version="2.0.0")
        
        # Promote version 2.0.0 to production
        success = registry.promote_model("test_model", "2.0.0", ModelStage.PRODUCTION)
        
        assert success
        
        # Check that 1.0.0 was demoted
        v1 = registry.get_model("test_model", version="1.0.0")
        assert v1.metadata.stage == ModelStage.ARCHIVED
        
        # Check that 2.0.0 is production
        v2 = registry.get_model("test_model", version="2.0.0")
        assert v2.metadata.stage == ModelStage.PRODUCTION
    
    def test_search_models(self, registry, sample_models):
        """Test searching models by criteria."""
        learner_clf, learner_reg = sample_models
        
        # Register models with different tags and stages
        v1 = registry.register_model(
            learner_clf, "model_1", 
            tags={"dataset": "iris", "validated": "yes"}
        )
        registry.update_model_stage("model_1", v1.metadata.version, ModelStage.PRODUCTION)
        
        v2 = registry.register_model(
            learner_reg, "model_2",
            tags={"dataset": "boston", "validated": "no"}
        )
        
        v3 = registry.register_model(
            learner_clf, "model_3",
            tags={"dataset": "iris", "validated": "no"}
        )
        
        # Search by tags
        results = registry.search_models(tags={"dataset": "iris"})
        assert len(results) == 2
        
        # Search by stage
        results = registry.search_models(stage=ModelStage.PRODUCTION)
        assert len(results) == 1
        assert results[0].metadata.name == "model_1"
    
    def test_get_model_history(self, registry, sample_models):
        """Test getting model history."""
        learner_clf, _ = sample_models
        
        # Register multiple versions
        registry.register_model(learner_clf, "test_model", version="1.0.0")
        registry.register_model(learner_clf, "test_model", version="1.0.1")
        registry.update_model_stage("test_model", "1.0.1", ModelStage.STAGING)
        registry.register_model(learner_clf, "test_model", version="2.0.0")
        
        history = registry.get_model_history("test_model")
        
        assert len(history) == 3
        assert history[0]["version"] == "1.0.0"
        assert history[1]["version"] == "1.0.1"
        assert history[1]["stage"] == "staging"
        assert history[2]["version"] == "2.0.0"
    
    def test_cleanup_old_versions(self, registry, sample_models):
        """Test cleaning up old model versions."""
        learner_clf, _ = sample_models
        
        # Register 5 versions
        for i in range(5):
            registry.register_model(learner_clf, "test_model", version=f"1.0.{i}")
        
        # Keep only latest 2
        registry.cleanup_old_versions("test_model", keep_latest=2)
        
        versions = registry.list_versions("test_model")
        assert len(versions) == 2
        assert "1.0.3" in versions
        assert "1.0.4" in versions
    
    def test_persistence_across_sessions(self, temp_registry_path, sample_models):
        """Test that registry persists across sessions."""
        learner_clf, _ = sample_models
        
        # First session
        registry1 = FileSystemRegistry(registry_path=temp_registry_path)
        registry1.register_model(
            learner_clf, "persistent_model",
            version="1.0.0",
            tags={"test": "persistence"}
        )
        
        # Second session (new registry instance)
        registry2 = FileSystemRegistry(registry_path=temp_registry_path)
        
        # Should be able to retrieve the model
        model = registry2.get_model("persistent_model", version="1.0.0")
        
        assert model is not None
        assert model.metadata.name == "persistent_model"
        assert model.metadata.tags["test"] == "persistence"


class TestRegistryUtils:
    """Test registry utility functions."""
    
    def test_generate_model_id(self):
        """Test model ID generation."""
        id1 = generate_model_id("model", "1.0.0")
        id2 = generate_model_id("model", "1.0.0")
        
        assert len(id1) == 16
        assert id1 != id2  # Different timestamps
    
    def test_validate_model_name(self):
        """Test model name validation."""
        assert validate_model_name("valid_model_name")
        assert validate_model_name("model-123")
        assert not validate_model_name("")
        assert not validate_model_name("model/with/slash")
        assert not validate_model_name("model:with:colon")
    
    def test_validate_version_string(self):
        """Test version string validation."""
        assert validate_version_string("1.0.0")
        assert validate_version_string("2.1.5")
        assert validate_version_string("20231225.120000")
        assert validate_version_string("v1.2.3-alpha")
        assert not validate_version_string("")
        assert not validate_version_string("...")
    
    def test_compare_models(self):
        """Test comparing models."""
        # Create sample model versions
        models = []
        for i in range(3):
            metadata = ModelMetadata(
                name=f"model_{i}",
                version=f"1.0.{i}",
                metrics={"accuracy": 0.9 + i * 0.01, "f1": 0.85 + i * 0.02}
            )
            models.append(ModelVersion(model=None, metadata=metadata))
        
        # Compare models
        comparison_df = compare_models(models, metrics=["accuracy", "f1"])
        
        assert len(comparison_df) == 3
        assert "accuracy" in comparison_df.columns
        assert "f1" in comparison_df.columns
        assert comparison_df.iloc[0]["accuracy"] == 0.9
        assert comparison_df.iloc[2]["f1"] == 0.89