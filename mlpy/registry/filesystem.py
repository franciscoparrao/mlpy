"""
File system based model registry implementation.

This module provides a file system based implementation of the model registry
that stores models and metadata locally.
"""

import json
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import logging

from .base import ModelRegistry, ModelVersion, ModelMetadata, ModelStage
from ..learners import Learner
from ..pipelines import GraphLearner


logger = logging.getLogger(__name__)


class FileSystemRegistry(ModelRegistry):
    """File system based model registry.
    
    This registry stores models and metadata on the local file system,
    organizing them in a directory structure.
    
    Directory structure:
    registry_path/
        models/
            model_name/
                version_1/
                    model.pkl
                    metadata.json
                version_2/
                    model.pkl
                    metadata.json
        index.json
    
    Parameters
    ----------
    registry_path : Union[str, Path]
        Path to the registry directory.
    name : str
        Name of the registry.
    """
    
    def __init__(
        self,
        registry_path: Union[str, Path] = "./mlpy_models",
        name: str = "filesystem_registry"
    ):
        super().__init__(name)
        self.registry_path = Path(registry_path)
        self._initialize_registry()
        self._load_index()
    
    def _initialize_registry(self):
        """Initialize the registry directory structure."""
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.models_path = self.registry_path / "models"
        self.models_path.mkdir(exist_ok=True)
        self.index_path = self.registry_path / "index.json"
        
        # Create index file if it doesn't exist
        if not self.index_path.exists():
            self._save_index()
    
    def _load_index(self):
        """Load the registry index from disk."""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                index_data = json.load(f)
                
            # Reconstruct models dictionary from index
            self._models = {}
            for model_name, versions_data in index_data.get('models', {}).items():
                self._models[model_name] = []
                for version_data in versions_data:
                    # Load metadata
                    metadata = ModelMetadata.from_dict(version_data['metadata'])
                    # Create placeholder ModelVersion (actual model loaded on demand)
                    model_version = ModelVersion(
                        model=None,  # Will be loaded lazily
                        metadata=metadata,
                        model_id=version_data['model_id'],
                        parent_version=version_data.get('parent_version')
                    )
                    self._models[model_name].append(model_version)
    
    def _save_index(self):
        """Save the registry index to disk."""
        index_data = {
            'registry_name': self.name,
            'updated_at': datetime.now().isoformat(),
            'models': {}
        }
        
        for model_name, versions in self._models.items():
            index_data['models'][model_name] = []
            for version in versions:
                version_data = {
                    'model_id': version.model_id,
                    'metadata': version.metadata.to_dict(),
                    'parent_version': version.parent_version
                }
                index_data['models'][model_name].append(version_data)
        
        with open(self.index_path, 'w') as f:
            json.dump(index_data, f, indent=2)
    
    def _get_model_path(self, name: str, version: str) -> Path:
        """Get the path for a specific model version."""
        return self.models_path / name / f"version_{version}"
    
    def _save_model(self, model_version: ModelVersion, path: Path):
        """Save a model version to disk."""
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = path / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model_version.model, f)
        
        # Save metadata
        metadata_path = path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(model_version.metadata.to_dict(), f, indent=2)
        
        logger.info(f"Saved model {model_version.metadata.name} v{model_version.metadata.version} to {path}")
    
    def _load_model(self, path: Path) -> Optional[Union[Learner, GraphLearner]]:
        """Load a model from disk."""
        model_path = path / "model.pkl"
        if model_path.exists():
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def _generate_version(self, name: str) -> str:
        """Generate a new version number for a model."""
        if name not in self._models or not self._models[name]:
            return "1.0.0"
        
        # Get latest version and increment
        versions = [v.metadata.version for v in self._models[name]]
        latest = sorted(versions)[-1]
        
        # Simple version incrementing (assumes semantic versioning)
        parts = latest.split('.')
        if len(parts) == 3:
            major, minor, patch = parts
            return f"{major}.{minor}.{int(patch) + 1}"
        else:
            # Fallback to timestamp-based version
            return datetime.now().strftime("%Y%m%d.%H%M%S")
    
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
        # Generate version if not provided
        if version is None:
            version = self._generate_version(name)
        
        # Create metadata
        metadata = ModelMetadata(
            name=name,
            version=version,
            **metadata_kwargs
        )
        
        # Infer task type from model if possible
        if hasattr(model, 'task_type'):
            metadata.task_type = model.task_type
        
        # Create model version
        model_version = ModelVersion(
            model=model,
            metadata=metadata
        )
        
        # Add to registry
        if name not in self._models:
            self._models[name] = []
        self._models[name].append(model_version)
        
        # Save to disk
        path = self._get_model_path(name, version)
        self._save_model(model_version, path)
        
        # Update index
        self._save_index()
        
        logger.info(f"Registered model {name} v{version} with ID {model_version.model_id}")
        return model_version
    
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
        if name not in self._models:
            return None
        
        versions = self._models[name]
        if not versions:
            return None
        
        # Filter by stage if specified
        if stage is not None:
            versions = [v for v in versions if v.metadata.stage == stage]
            if not versions:
                return None
        
        # Get specific version or latest
        if version is not None:
            for v in versions:
                if v.metadata.version == version:
                    # Load model if not already loaded
                    if v.model is None:
                        path = self._get_model_path(name, version)
                        v.model = self._load_model(path)
                    return v
            return None
        else:
            # Get latest version
            latest = sorted(versions, key=lambda x: x.metadata.created_at)[-1]
            # Load model if not already loaded
            if latest.model is None:
                path = self._get_model_path(name, latest.metadata.version)
                latest.model = self._load_model(path)
            return latest
    
    def list_models(self) -> List[str]:
        """List all model names in the registry.
        
        Returns
        -------
        List[str]
            List of model names.
        """
        return list(self._models.keys())
    
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
        if name not in self._models:
            return []
        return [v.metadata.version for v in self._models[name]]
    
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
        if name not in self._models:
            return False
        
        if version is not None:
            # Delete specific version
            path = self._get_model_path(name, version)
            if path.exists():
                shutil.rmtree(path)
            
            # Remove from index
            self._models[name] = [
                v for v in self._models[name]
                if v.metadata.version != version
            ]
            
            # Remove model entry if no versions left
            if not self._models[name]:
                del self._models[name]
        else:
            # Delete all versions
            model_dir = self.models_path / name
            if model_dir.exists():
                shutil.rmtree(model_dir)
            del self._models[name]
        
        # Update index
        self._save_index()
        logger.info(f"Deleted model {name}" + (f" v{version}" if version else " (all versions)"))
        return True
    
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
        model = self.get_model(name, version)
        if model is None:
            return False
        
        model.set_stage(stage)
        
        # Update metadata on disk
        path = self._get_model_path(name, version)
        metadata_path = path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(model.metadata.to_dict(), f, indent=2)
        
        # Update index
        self._save_index()
        
        logger.info(f"Updated model {name} v{version} stage to {stage.value}")
        return True
    
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
        results = []
        
        for versions in self._models.values():
            for version in versions:
                # Check stage
                if stage is not None and version.metadata.stage != stage:
                    continue
                
                # Check task type
                if task_type is not None and version.metadata.task_type != task_type:
                    continue
                
                # Check tags
                if tags is not None:
                    if not all(
                        k in version.metadata.tags and version.metadata.tags[k] == v
                        for k, v in tags.items()
                    ):
                        continue
                
                results.append(version)
        
        return results
    
    def get_model_history(self, name: str) -> List[Dict[str, Any]]:
        """Get the history of a model.
        
        Parameters
        ----------
        name : str
            Name of the model.
            
        Returns
        -------
        List[Dict[str, Any]]
            List of version information dictionaries.
        """
        if name not in self._models:
            return []
        
        history = []
        for version in self._models[name]:
            history.append({
                'version': version.metadata.version,
                'created_at': version.metadata.created_at.isoformat(),
                'author': version.metadata.author,
                'stage': version.metadata.stage.value,
                'metrics': version.metadata.metrics,
                'tags': version.metadata.tags
            })
        
        return sorted(history, key=lambda x: x['created_at'])
    
    def cleanup_old_versions(self, name: str, keep_latest: int = 3):
        """Clean up old versions of a model, keeping only the latest ones.
        
        Parameters
        ----------
        name : str
            Name of the model.
        keep_latest : int
            Number of latest versions to keep.
        """
        if name not in self._models:
            return
        
        versions = self._models[name]
        if len(versions) <= keep_latest:
            return
        
        # Sort by creation date and keep only latest
        sorted_versions = sorted(versions, key=lambda x: x.metadata.created_at)
        to_delete = sorted_versions[:-keep_latest]
        
        for version in to_delete:
            # Don't delete production or staging models
            if version.metadata.stage in [ModelStage.PRODUCTION, ModelStage.STAGING]:
                continue
            self.delete_model(name, version.metadata.version)
        
        logger.info(f"Cleaned up old versions of {name}, kept {keep_latest} latest")