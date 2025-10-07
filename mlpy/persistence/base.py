"""Base classes and utilities for model persistence."""

import os
import json
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, Type, BinaryIO
from pathlib import Path
from datetime import datetime
import hashlib

from ..base import MLPYObject
from ..learners import Learner
from ..pipelines import Graph, GraphLearner
from ..utils.registry import Registry


# Registry for serializers
SERIALIZERS = Registry("serializers")


class ModelSerializer(ABC):
    """Abstract base class for model serializers.
    
    A serializer handles the conversion of MLPY objects to/from
    a specific format for persistence.
    
    Parameters
    ----------
    compression : str or None, default=None
        Compression method to use. Options depend on the serializer.
    """
    
    def __init__(self, compression: Optional[str] = None):
        self.compression = compression
        
    @abstractmethod
    def can_serialize(self, obj: Any) -> bool:
        """Check if this serializer can handle the given object.
        
        Parameters
        ----------
        obj : Any
            The object to check.
            
        Returns
        -------
        bool
            True if this serializer can handle the object.
        """
        pass
        
    @abstractmethod
    def serialize(self, obj: Any, path: Union[str, Path, BinaryIO]) -> Dict[str, Any]:
        """Serialize an object to file.
        
        Parameters
        ----------
        obj : Any
            The object to serialize.
        path : str, Path, or file-like
            Where to save the serialized object.
            
        Returns
        -------
        dict
            Metadata about the serialization.
        """
        pass
        
    @abstractmethod
    def deserialize(self, path: Union[str, Path, BinaryIO]) -> Any:
        """Load an object from file.
        
        Parameters
        ----------
        path : str, Path, or file-like
            Where to load from.
            
        Returns
        -------
        Any
            The deserialized object.
        """
        pass
        
    @property
    @abstractmethod
    def file_extension(self) -> str:
        """Default file extension for this format."""
        pass
        
    def get_metadata(self, obj: Any) -> Dict[str, Any]:
        """Extract metadata from an object.
        
        Parameters
        ----------
        obj : Any
            The object to extract metadata from.
            
        Returns
        -------
        dict
            Metadata dictionary.
        """
        metadata = {
            "serializer": self.__class__.__name__,
            "timestamp": datetime.now().isoformat(),
            "mlpy_version": self._get_mlpy_version(),
            "object_type": type(obj).__name__,
            "object_module": type(obj).__module__
        }
        
        # Add object-specific metadata
        if hasattr(obj, 'id'):
            metadata['object_id'] = obj.id
            
        if isinstance(obj, Learner):
            metadata['learner_class'] = obj.__class__.__name__
            metadata['is_trained'] = obj.is_trained
            metadata['properties'] = list(obj.properties)
            metadata['packages'] = list(obj.packages)
            
        elif isinstance(obj, (Graph, GraphLearner)):
            metadata['n_pipeops'] = len(obj.pipeops)
            metadata['input_ids'] = list(obj.input)
            metadata['output_ids'] = list(obj.output)
            
        return metadata
        
    def _get_mlpy_version(self) -> str:
        """Get MLPY version."""
        try:
            import mlpy
            return getattr(mlpy, '__version__', 'unknown')
        except:
            return 'unknown'


class ModelBundle:
    """Container for a model and its metadata.
    
    This class wraps a model with additional information needed
    for proper serialization and deserialization.
    
    Parameters
    ----------
    model : Any
        The model object to bundle.
    metadata : dict, optional
        Additional metadata to store.
    """
    
    def __init__(self, model: Any, metadata: Optional[Dict[str, Any]] = None):
        self.model = model
        self.metadata = metadata or {}
        self.metadata['bundle_version'] = '1.0'
        
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata entry."""
        self.metadata[key] = value
        
    def get_checksum(self) -> str:
        """Calculate checksum of the model."""
        # This is a simplified version - in practice would hash the serialized model
        model_str = str(type(self.model)) + str(id(self.model))
        return hashlib.md5(model_str.encode()).hexdigest()


def save_model(
    model: Any,
    path: Union[str, Path],
    serializer: Union[str, ModelSerializer] = "auto",
    metadata: Optional[Dict[str, Any]] = None,
    create_bundle: bool = True,
    **kwargs
) -> Path:
    """Save a model to disk.
    
    Parameters
    ----------
    model : Any
        The model to save. Can be a Learner, Pipeline, or any
        object supported by the serializers.
    path : str or Path
        Where to save the model. The file extension may be
        adjusted based on the serializer.
    serializer : str or ModelSerializer, default="auto"
        The serializer to use. Can be:
        - "auto": Automatically select based on model type
        - "pickle": Use pickle serialization
        - "joblib": Use joblib serialization
        - "json": Use JSON for simple models
        - "onnx": Use ONNX format (if available)
        - A ModelSerializer instance
    metadata : dict, optional
        Additional metadata to save with the model.
    create_bundle : bool, default=True
        Whether to wrap the model in a ModelBundle with metadata.
    **kwargs
        Additional arguments passed to the serializer.
        
    Returns
    -------
    Path
        The path where the model was saved.
        
    Examples
    --------
    >>> from mlpy.persistence import save_model
    >>> from mlpy.learners import LearnerRegrRF
    >>> 
    >>> # Train a model
    >>> learner = LearnerRegrRF()
    >>> learner.train(task)
    >>> 
    >>> # Save with auto-detection
    >>> save_model(learner, "my_model.pkl")
    >>> 
    >>> # Save with specific serializer
    >>> save_model(learner, "my_model.joblib", serializer="joblib")
    >>> 
    >>> # Save with metadata
    >>> save_model(learner, "my_model.pkl", metadata={
    ...     "experiment": "exp_001",
    ...     "accuracy": 0.95
    ... })
    """
    path = Path(path)
    
    # Get serializer
    if isinstance(serializer, str):
        if serializer == "auto":
            serializer_obj = _auto_select_serializer(model)
        else:
            serializer_class = SERIALIZERS.get(serializer)
            if serializer_class is None:
                raise ValueError(f"Unknown serializer: {serializer}")
            serializer_obj = serializer_class(**kwargs)
    else:
        serializer_obj = serializer
        
    # Check if serializer can handle the model
    if not serializer_obj.can_serialize(model):
        raise ValueError(
            f"Serializer {serializer_obj.__class__.__name__} "
            f"cannot handle {type(model).__name__} objects"
        )
        
    # Adjust file extension if needed
    if not path.suffix:
        path = path.with_suffix(serializer_obj.file_extension)
    elif path.suffix != serializer_obj.file_extension:
        warnings.warn(
            f"File extension {path.suffix} does not match "
            f"serializer extension {serializer_obj.file_extension}"
        )
        
    # Create parent directory if needed
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create bundle if requested
    if create_bundle:
        bundle = ModelBundle(model, metadata)
        bundle.add_metadata("serialization_metadata", serializer_obj.get_metadata(model))
        bundle.add_metadata("checksum", bundle.get_checksum())
        obj_to_save = bundle
    else:
        obj_to_save = model
        
    # Serialize
    serializer_obj.serialize(obj_to_save, path)
    
    # Save separate metadata file if not bundled
    if not create_bundle and metadata:
        meta_path = path.with_suffix('.meta.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
    return path


def load_model(
    path: Union[str, Path],
    serializer: Union[str, ModelSerializer] = "auto",
    return_bundle: bool = False,
    **kwargs
) -> Any:
    """Load a model from disk.
    
    Parameters
    ----------
    path : str or Path
        Path to the saved model.
    serializer : str or ModelSerializer, default="auto"
        The serializer to use. Can be:
        - "auto": Detect based on file extension
        - "pickle": Use pickle
        - "joblib": Use joblib
        - "json": Use JSON
        - "onnx": Use ONNX (if available)
        - A ModelSerializer instance
    return_bundle : bool, default=False
        If True and the file contains a ModelBundle, return
        the bundle instead of just the model.
    **kwargs
        Additional arguments passed to the serializer.
        
    Returns
    -------
    Any
        The loaded model, or ModelBundle if return_bundle=True.
        
    Examples
    --------
    >>> from mlpy.persistence import load_model
    >>> 
    >>> # Load model
    >>> learner = load_model("my_model.pkl")
    >>> 
    >>> # Load with bundle to access metadata
    >>> bundle = load_model("my_model.pkl", return_bundle=True)
    >>> print(bundle.metadata)
    >>> 
    >>> # Use specific serializer
    >>> learner = load_model("my_model.joblib", serializer="joblib")
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
        
    # Get serializer
    if isinstance(serializer, str):
        if serializer == "auto":
            serializer_obj = _auto_detect_serializer(path)
        else:
            serializer_class = SERIALIZERS.get(serializer)
            if serializer_class is None:
                raise ValueError(f"Unknown serializer: {serializer}")
            serializer_obj = serializer_class(**kwargs)
    else:
        serializer_obj = serializer
        
    # Deserialize
    obj = serializer_obj.deserialize(path)
    
    # Handle bundle
    if isinstance(obj, ModelBundle):
        if return_bundle:
            return obj
        else:
            # Verify checksum if available
            if 'checksum' in obj.metadata:
                current_checksum = obj.get_checksum()
                if current_checksum != obj.metadata['checksum']:
                    warnings.warn(
                        "Model checksum mismatch. The model may have been modified."
                    )
            return obj.model
    else:
        # Check for separate metadata file
        meta_path = path.with_suffix('.meta.json')
        if meta_path.exists() and return_bundle:
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            return ModelBundle(obj, metadata)
        return obj


def _auto_select_serializer(obj: Any) -> ModelSerializer:
    """Automatically select appropriate serializer for an object."""
    # Try serializers in order of preference
    preference_order = ['joblib', 'pickle', 'json', 'onnx']
    
    for name in preference_order:
        if name in SERIALIZERS:
            serializer_class = SERIALIZERS.get(name)
            serializer = serializer_class()
            if serializer.can_serialize(obj):
                return serializer
                
    raise ValueError(
        f"No suitable serializer found for {type(obj).__name__} object"
    )


def _auto_detect_serializer(path: Path) -> ModelSerializer:
    """Detect serializer based on file extension."""
    extension_map = {
        '.pkl': 'pickle',
        '.pickle': 'pickle',
        '.joblib': 'joblib',
        '.json': 'json',
        '.onnx': 'onnx'
    }
    
    ext = path.suffix.lower()
    if ext in extension_map:
        serializer_name = extension_map[ext]
        if serializer_name in SERIALIZERS:
            return SERIALIZERS.get(serializer_name)()
            
    # Default to pickle
    warnings.warn(
        f"Could not detect serializer from extension {ext}, using pickle"
    )
    return SERIALIZERS.get('pickle')()


__all__ = [
    "ModelSerializer",
    "ModelBundle",
    "save_model",
    "load_model",
    "SERIALIZERS"
]