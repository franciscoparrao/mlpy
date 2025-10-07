"""
Base classes for Model Registry.

This module provides the abstract base classes and data structures
for implementing model registries in MLPY.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json
import pickle
import hashlib
from enum import Enum

from ..learners import Learner
from ..pipelines import GraphLearner


class ModelStage(Enum):
    """Stages in model lifecycle."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass
class ModelMetadata:
    """Metadata for a registered model.
    
    Attributes
    ----------
    name : str
        Name of the model.
    version : str
        Version of the model.
    description : str
        Description of the model.
    created_at : datetime
        When the model was created.
    updated_at : datetime
        When the model was last updated.
    author : str
        Who created the model.
    tags : Dict[str, str]
        Tags associated with the model.
    metrics : Dict[str, float]
        Performance metrics of the model.
    parameters : Dict[str, Any]
        Hyperparameters used for training.
    dataset_info : Dict[str, Any]
        Information about the training dataset.
    stage : ModelStage
        Current stage of the model.
    framework : str
        ML framework used (e.g., 'sklearn', 'mlpy').
    task_type : str
        Type of task (e.g., 'classification', 'regression').
    """
    name: str
    version: str
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    author: str = "unknown"
    tags: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    dataset_info: Dict[str, Any] = field(default_factory=dict)
    stage: ModelStage = ModelStage.DEVELOPMENT
    framework: str = "mlpy"
    task_type: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'author': self.author,
            'tags': self.tags,
            'metrics': self.metrics,
            'parameters': self.parameters,
            'dataset_info': self.dataset_info,
            'stage': self.stage.value,
            'framework': self.framework,
            'task_type': self.task_type
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create metadata from dictionary."""
        # Convert string dates back to datetime
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        # Convert stage string back to enum
        if 'stage' in data and isinstance(data['stage'], str):
            data['stage'] = ModelStage(data['stage'])
            
        return cls(**data)


@dataclass
class ModelVersion:
    """A specific version of a model.
    
    Attributes
    ----------
    model : Union[Learner, Pipeline]
        The actual model object.
    metadata : ModelMetadata
        Metadata about this model version.
    model_id : str
        Unique identifier for this model version.
    parent_version : Optional[str]
        ID of the parent version if this is derived from another.
    """
    model: Union[Learner, GraphLearner]
    metadata: ModelMetadata
    model_id: str = ""
    parent_version: Optional[str] = None
    
    def __post_init__(self):
        """Generate model ID if not provided."""
        if not self.model_id:
            self.model_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique ID for this model version."""
        # Create hash from model name, version, and timestamp
        content = f"{self.metadata.name}_{self.metadata.version}_{self.metadata.created_at}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]
    
    def update_metrics(self, metrics: Dict[str, float]):
        """Update model metrics."""
        self.metadata.metrics.update(metrics)
        self.metadata.updated_at = datetime.now()
    
    def add_tags(self, tags: Dict[str, str]):
        """Add tags to the model."""
        self.metadata.tags.update(tags)
        self.metadata.updated_at = datetime.now()
    
    def set_stage(self, stage: ModelStage):
        """Set the model stage."""
        self.metadata.stage = stage
        self.metadata.updated_at = datetime.now()


class ModelRegistry(ABC):
    """Abstract base class for model registries.
    
    A model registry provides centralized model management including:
    - Model storage and retrieval
    - Version control
    - Metadata management
    - Model lifecycle management
    """
    
    def __init__(self, name: str = "mlpy_registry"):
        """Initialize the registry.
        
        Parameters
        ----------
        name : str
            Name of the registry.
        """
        self.name = name
        self._models: Dict[str, List[ModelVersion]] = {}
    
    @abstractmethod
    def register_model(
        self,
        model: Union[Learner, GraphLearner],
        name: str,
        version: Optional[str] = None,
        **metadata_kwargs
    ) -> ModelVersion:
        """Register a new model.
        
        Parameters
        ----------
        model : Union[Learner, Pipeline]
            The model to register.
        name : str
            Name for the model.
        version : Optional[str]
            Version string. If None, auto-generated.
        **metadata_kwargs
            Additional metadata fields.
            
        Returns
        -------
        ModelVersion
            The registered model version.
        """
        pass
    
    @abstractmethod
    def get_model(
        self,
        name: str,
        version: Optional[str] = None,
        stage: Optional[ModelStage] = None
    ) -> Optional[ModelVersion]:
        """Retrieve a model from the registry.
        
        Parameters
        ----------
        name : str
            Name of the model.
        version : Optional[str]
            Specific version to retrieve. If None, gets latest.
        stage : Optional[ModelStage]
            Filter by stage (e.g., get production model).
            
        Returns
        -------
        Optional[ModelVersion]
            The model version if found, None otherwise.
        """
        pass
    
    @abstractmethod
    def list_models(self) -> List[str]:
        """List all model names in the registry.
        
        Returns
        -------
        List[str]
            List of model names.
        """
        pass
    
    @abstractmethod
    def list_versions(self, name: str) -> List[str]:
        """List all versions of a model.
        
        Parameters
        ----------
        name : str
            Name of the model.
            
        Returns
        -------
        List[str]
            List of version strings.
        """
        pass
    
    @abstractmethod
    def delete_model(
        self,
        name: str,
        version: Optional[str] = None
    ) -> bool:
        """Delete a model from the registry.
        
        Parameters
        ----------
        name : str
            Name of the model.
        version : Optional[str]
            Specific version to delete. If None, deletes all versions.
            
        Returns
        -------
        bool
            True if deletion was successful.
        """
        pass
    
    @abstractmethod
    def update_model_stage(
        self,
        name: str,
        version: str,
        stage: ModelStage
    ) -> bool:
        """Update the stage of a model version.
        
        Parameters
        ----------
        name : str
            Name of the model.
        version : str
            Version to update.
        stage : ModelStage
            New stage for the model.
            
        Returns
        -------
        bool
            True if update was successful.
        """
        pass
    
    @abstractmethod
    def search_models(
        self,
        tags: Optional[Dict[str, str]] = None,
        stage: Optional[ModelStage] = None,
        task_type: Optional[str] = None
    ) -> List[ModelVersion]:
        """Search for models matching criteria.
        
        Parameters
        ----------
        tags : Optional[Dict[str, str]]
            Tags to match.
        stage : Optional[ModelStage]
            Stage to filter by.
        task_type : Optional[str]
            Task type to filter by.
            
        Returns
        -------
        List[ModelVersion]
            List of matching model versions.
        """
        pass
    
    def compare_models(
        self,
        model_ids: List[str],
        metric: str
    ) -> Dict[str, float]:
        """Compare models by a specific metric.
        
        Parameters
        ----------
        model_ids : List[str]
            IDs of models to compare.
        metric : str
            Metric to compare by.
            
        Returns
        -------
        Dict[str, float]
            Model IDs mapped to metric values.
        """
        comparison = {}
        for model_id in model_ids:
            model = self._find_model_by_id(model_id)
            if model and metric in model.metadata.metrics:
                comparison[model_id] = model.metadata.metrics[metric]
        return comparison
    
    def _find_model_by_id(self, model_id: str) -> Optional[ModelVersion]:
        """Find a model by its ID.
        
        Parameters
        ----------
        model_id : str
            The model ID to search for.
            
        Returns
        -------
        Optional[ModelVersion]
            The model if found, None otherwise.
        """
        for versions in self._models.values():
            for version in versions:
                if version.model_id == model_id:
                    return version
        return None
    
    def get_production_model(self, name: str) -> Optional[ModelVersion]:
        """Get the production version of a model.
        
        Parameters
        ----------
        name : str
            Name of the model.
            
        Returns
        -------
        Optional[ModelVersion]
            The production model if exists.
        """
        return self.get_model(name, stage=ModelStage.PRODUCTION)
    
    def promote_model(
        self,
        name: str,
        version: str,
        target_stage: ModelStage
    ) -> bool:
        """Promote a model to a new stage.
        
        Parameters
        ----------
        name : str
            Name of the model.
        version : str
            Version to promote.
        target_stage : ModelStage
            Target stage.
            
        Returns
        -------
        bool
            True if promotion was successful.
        """
        # If promoting to production, demote current production model
        if target_stage == ModelStage.PRODUCTION:
            current_prod = self.get_production_model(name)
            if current_prod:
                self.update_model_stage(
                    name,
                    current_prod.metadata.version,
                    ModelStage.ARCHIVED
                )
        
        return self.update_model_stage(name, version, target_stage)