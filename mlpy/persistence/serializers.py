"""Concrete serializer implementations."""

import pickle
import json
import warnings
from typing import Any, Dict, Union, BinaryIO, Optional
from pathlib import Path

from .base import ModelSerializer, SERIALIZERS, ModelBundle
from ..learners import Learner
from ..pipelines import Graph, GraphLearner

# Try to import optional dependencies
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


@SERIALIZERS.register("pickle")
class PickleSerializer(ModelSerializer):
    """Serializer using Python's pickle module.
    
    This is the most general serializer that can handle almost
    any Python object, but files are not portable across Python
    versions or architectures.
    
    Parameters
    ----------
    protocol : int, optional
        Pickle protocol version. If None, uses pickle.HIGHEST_PROTOCOL.
    compression : str or None, default=None
        Not supported for pickle. Use joblib for compression.
    """
    
    def __init__(self, protocol: Optional[int] = None, compression: Optional[str] = None):
        super().__init__(compression)
        self.protocol = protocol or pickle.HIGHEST_PROTOCOL
        
        if compression is not None:
            warnings.warn(
                "PickleSerializer does not support compression. "
                "Use JoblibSerializer for compressed serialization."
            )
            
    def can_serialize(self, obj: Any) -> bool:
        """Pickle can serialize most Python objects."""
        # Check if object is pickleable
        try:
            pickle.dumps(obj, protocol=self.protocol)
            return True
        except (pickle.PicklingError, TypeError):
            return False
            
    def serialize(self, obj: Any, path: Union[str, Path, BinaryIO]) -> Dict[str, Any]:
        """Serialize using pickle."""
        if hasattr(path, 'write'):
            # File-like object
            pickle.dump(obj, path, protocol=self.protocol)
        else:
            # Path
            with open(path, 'wb') as f:
                pickle.dump(obj, f, protocol=self.protocol)
                
        return {
            "protocol": self.protocol,
            "serializer": "pickle"
        }
        
    def deserialize(self, path: Union[str, Path, BinaryIO]) -> Any:
        """Deserialize using pickle."""
        if hasattr(path, 'read'):
            # File-like object
            return pickle.load(path)
        else:
            # Path
            with open(path, 'rb') as f:
                return pickle.load(f)
                
    @property
    def file_extension(self) -> str:
        """Default extension for pickle files."""
        return ".pkl"


@SERIALIZERS.register("joblib")
class JoblibSerializer(ModelSerializer):
    """Serializer using joblib.
    
    Joblib is optimized for scientific data (numpy arrays) and
    supports compression. It's the recommended serializer for
    scikit-learn models and large numerical data.
    
    Parameters
    ----------
    compression : int, str, or None, default=3
        Compression level (0-9) or method ('zlib', 'gzip', 'bz2', 'lzma').
    """
    
    def __init__(self, compression: Union[int, str, None] = 3):
        if not JOBLIB_AVAILABLE:
            raise ImportError(
                "joblib is required for JoblibSerializer. "
                "Install it with: pip install joblib"
            )
        super().__init__(compression)
        
    def can_serialize(self, obj: Any) -> bool:
        """Joblib can serialize most objects that pickle can."""
        try:
            # Quick check without actual serialization
            return hasattr(obj, '__dict__') or isinstance(obj, (list, dict, tuple))
        except:
            return False
            
    def serialize(self, obj: Any, path: Union[str, Path, BinaryIO]) -> Dict[str, Any]:
        """Serialize using joblib."""
        import joblib
        
        if hasattr(path, 'write'):
            # Joblib doesn't support file-like objects directly
            # Fall back to pickle for file-like objects
            warnings.warn(
                "JoblibSerializer does not support file-like objects. "
                "Using pickle instead."
            )
            pickle.dump(obj, path)
        else:
            joblib.dump(obj, path, compress=self.compression)
            
        return {
            "compression": self.compression,
            "serializer": "joblib"
        }
        
    def deserialize(self, path: Union[str, Path, BinaryIO]) -> Any:
        """Deserialize using joblib."""
        import joblib
        
        if hasattr(path, 'read'):
            # Fall back to pickle for file-like objects
            return pickle.load(path)
        else:
            return joblib.load(path)
            
    @property
    def file_extension(self) -> str:
        """Default extension for joblib files."""
        return ".joblib"


@SERIALIZERS.register("json")
class JSONSerializer(ModelSerializer):
    """Serializer using JSON format.
    
    This serializer is limited to simple objects that can be
    represented in JSON. It's mainly useful for model metadata,
    hyperparameters, and simple custom models.
    
    Parameters
    ----------
    indent : int, optional
        JSON indentation for pretty printing.
    compression : str or None, default=None
        Not supported for JSON.
    """
    
    def __init__(self, indent: Optional[int] = 2, compression: Optional[str] = None):
        super().__init__(compression)
        self.indent = indent
        
        if compression is not None:
            warnings.warn("JSONSerializer does not support compression.")
            
    def can_serialize(self, obj: Any) -> bool:
        """Check if object can be JSON serialized."""
        # For now, only support metadata and simple structures
        if isinstance(obj, ModelBundle):
            # Can save metadata but not complex models
            return isinstance(obj.metadata, dict)
        elif isinstance(obj, dict):
            try:
                json.dumps(obj)
                return True
            except (TypeError, ValueError):
                return False
        else:
            # Could extend to support custom model classes
            # that implement to_dict() and from_dict()
            return hasattr(obj, 'to_dict') and hasattr(type(obj), 'from_dict')
            
    def serialize(self, obj: Any, path: Union[str, Path, BinaryIO]) -> Dict[str, Any]:
        """Serialize to JSON."""
        # Prepare data for JSON
        if isinstance(obj, ModelBundle):
            # For bundles, we can only save metadata
            data = {
                "metadata": obj.metadata,
                "model_type": type(obj.model).__name__,
                "model_module": type(obj.model).__module__,
                "_is_bundle": True
            }
        elif hasattr(obj, 'to_dict'):
            data = {
                "model_data": obj.to_dict(),
                "model_type": type(obj).__name__,
                "model_module": type(obj).__module__,
                "_is_model": True
            }
        else:
            data = obj
            
        # Write JSON
        if hasattr(path, 'write'):
            # File-like object  
            if hasattr(path, 'mode') and 'b' in path.mode:
                # Binary mode
                json_str = json.dumps(data, indent=self.indent)
                path.write(json_str.encode('utf-8'))
            else:
                # Text mode
                json.dump(data, path, indent=self.indent)
        else:
            # Path
            with open(path, 'w') as f:
                json.dump(data, f, indent=self.indent)
                
        return {
            "serializer": "json",
            "has_model": "_is_model" in data
        }
        
    def deserialize(self, path: Union[str, Path, BinaryIO]) -> Any:
        """Deserialize from JSON."""
        # Read JSON
        if hasattr(path, 'read'):
            # File-like object
            if hasattr(path, 'mode') and 'b' in path.mode:
                # Binary mode
                data = json.loads(path.read().decode('utf-8'))
            else:
                # Text mode
                data = json.load(path)
        else:
            # Path
            with open(path, 'r') as f:
                data = json.load(f)
                
        # Reconstruct object
        if isinstance(data, dict):
            if data.get('_is_bundle'):
                # Can't reconstruct the model from JSON
                warnings.warn(
                    "JSON deserializer cannot reconstruct model objects, "
                    "only metadata is available."
                )
                return data['metadata']
            elif data.get('_is_model'):
                # Try to reconstruct model if class has from_dict
                model_type = data['model_type']
                model_module = data['model_module']
                
                try:
                    # Import the model class
                    import importlib
                    module = importlib.import_module(model_module)
                    model_class = getattr(module, model_type)
                    
                    if hasattr(model_class, 'from_dict'):
                        return model_class.from_dict(data['model_data'])
                    else:
                        warnings.warn(
                            f"Model class {model_type} does not have from_dict method"
                        )
                        return data['model_data']
                except Exception as e:
                    warnings.warn(f"Could not reconstruct model: {e}")
                    return data['model_data']
                    
        return data
        
    @property
    def file_extension(self) -> str:
        """Default extension for JSON files."""
        return ".json"


# Helper function to create serializer
def get_serializer(name: str, **kwargs) -> ModelSerializer:
    """Get a serializer by name.
    
    Parameters
    ----------
    name : str
        Name of the serializer: 'pickle', 'joblib', 'json', 'onnx'.
    **kwargs
        Arguments passed to the serializer constructor.
        
    Returns
    -------
    ModelSerializer
        The serializer instance.
        
    Raises
    ------
    ValueError
        If serializer name is not recognized.
    """
    if name not in SERIALIZERS:
        available = list(SERIALIZERS.keys())
        raise ValueError(
            f"Unknown serializer: {name}. "
            f"Available serializers: {available}"
        )
        
    serializer_class = SERIALIZERS.get(name)
    return serializer_class(**kwargs)


__all__ = [
    "PickleSerializer",
    "JoblibSerializer",
    "JSONSerializer",
    "get_serializer"
]