"""
Robust serialization for MLPY pipelines and models.
Supports multiple formats and ensures reproducibility.
"""

import pickle
import json
from pathlib import Path

# Importaciones opcionales para máxima compatibilidad
try:
    import cloudpickle
    HAS_CLOUDPICKLE = True
except ImportError:
    HAS_CLOUDPICKLE = False
    
try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
from typing import Any, Optional, Dict, Union
import hashlib
import warnings
from datetime import datetime
import numpy as np
import pandas as pd


class SerializationError(Exception):
    """Error durante la serialización."""
    pass


class ChecksumMismatchError(Exception):
    """Error cuando el checksum no coincide."""
    pass


class RobustSerializer:
    """
    Robust serialization with multiple backends and validation.
    
    Features:
    - Multiple serialization formats (pickle, cloudpickle, joblib, onnx)
    - Automatic versioning and checksums
    - Metadata tracking
    - Fallback mechanisms
    """
    
    # Formatos disponibles dinámicamente basados en instalaciones
    SUPPORTED_FORMATS = ['pickle', 'json']  # Siempre disponibles
    if HAS_CLOUDPICKLE:
        SUPPORTED_FORMATS.append('cloudpickle')
    if HAS_JOBLIB:
        SUPPORTED_FORMATS.append('joblib')
    # ONNX requiere instalación especial, se agrega dinámicamente
    
    def __init__(self, 
                 default_format: Optional[str] = None,
                 compression: Optional[str] = 'zlib',
                 track_metadata: bool = True):
        """
        Initialize serializer.
        
        Parameters
        ----------
        default_format : str
            Default serialization format
        compression : str or None
            Compression method ('zlib', 'gzip', 'bz2', 'xz', 'lzma')
        track_metadata : bool
            Whether to track metadata (version, timestamp, etc.)
        """
        # Seleccionar el mejor formato disponible por defecto
        if default_format is None:
            if HAS_CLOUDPICKLE:
                self.default_format = 'cloudpickle'
            elif HAS_JOBLIB:
                self.default_format = 'joblib'
            else:
                self.default_format = 'pickle'
        else:
            self.default_format = default_format
            
        self.compression = compression
        self.track_metadata = track_metadata
        
    def save(self, 
             obj: Any, 
             path: Union[str, Path],
             format: Optional[str] = None,
             metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Save object with robust serialization.
        
        Parameters
        ----------
        obj : Any
            Object to serialize
        path : str or Path
            Save path
        format : str, optional
            Serialization format (uses default if None)
        metadata : dict, optional
            Additional metadata to store
            
        Returns
        -------
        dict
            Serialization report with checksums and metadata
        """
        path = Path(path)
        format = format or self.default_format
        
        if format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format '{format}'. "
                f"Choose from {self.SUPPORTED_FORMATS}"
            )
        
        # Prepare metadata
        full_metadata = self._prepare_metadata(obj, metadata)
        
        # Serialize based on format
        try:
            if format == 'cloudpickle':
                self._save_cloudpickle(obj, path, full_metadata)
            elif format == 'joblib':
                self._save_joblib(obj, path, full_metadata)
            elif format == 'pickle':
                self._save_pickle(obj, path, full_metadata)
            elif format == 'onnx':
                self._save_onnx(obj, path, full_metadata)
            elif format == 'json':
                self._save_json(obj, path, full_metadata)
                
        except Exception as e:
            # Try fallback format
            warnings.warn(
                f"Failed to save with {format}: {e}. "
                f"Trying cloudpickle as fallback..."
            )
            self._save_cloudpickle(obj, path, full_metadata)
            full_metadata['format_used'] = 'cloudpickle (fallback)'
        
        # Calculate checksum
        full_metadata['checksum'] = self._calculate_checksum(path)
        
        # Save metadata separately
        if self.track_metadata:
            metadata_path = path.with_suffix('.meta.json')
            with open(metadata_path, 'w') as f:
                json.dump(full_metadata, f, indent=2, default=str)
        
        return full_metadata
    
    def load(self, 
             path: Union[str, Path],
             validate_checksum: bool = True) -> Any:
        """
        Load object with validation.
        
        Parameters
        ----------
        path : str or Path
            Load path
        validate_checksum : bool
            Whether to validate checksum
            
        Returns
        -------
        Any
            Loaded object
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        # Load metadata if available
        metadata_path = path.with_suffix('.meta.json')
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        # Validate checksum
        if validate_checksum and 'checksum' in metadata:
            actual_checksum = self._calculate_checksum(path)
            if actual_checksum != metadata['checksum']:
                if validate_checksum:
                    raise ChecksumMismatchError(
                        f"Checksum mismatch! File may have been modified. "
                        f"Expected: {metadata['checksum']}, "
                        f"Got: {actual_checksum}"
                    )
                else:
                    warnings.warn(
                        f"Checksum mismatch! File may have been modified. "
                        f"Expected: {metadata['checksum']}, "
                        f"Got: {actual_checksum}"
                    )
        
        # Determine format
        format = metadata.get('format', self._detect_format(path))
        
        # Load based on format
        try:
            if format in ['cloudpickle', 'cloudpickle (fallback)']:
                return self._load_cloudpickle(path)
            elif format == 'joblib':
                return self._load_joblib(path)
            elif format == 'pickle':
                return self._load_pickle(path)
            elif format == 'onnx':
                return self._load_onnx(path)
            elif format == 'json':
                return self._load_json(path)
            else:
                # Try to detect and load
                return self._auto_load(path)
                
        except Exception as e:
            raise RuntimeError(
                f"Failed to load file: {e}\n"
                f"Format: {format}\n"
                f"Path: {path}"
            )
    
    def _save_cloudpickle(self, obj: Any, path: Path, metadata: Dict):
        """Save using cloudpickle (most robust)."""
        if not HAS_CLOUDPICKLE:
            raise ImportError(
                "cloudpickle no está instalado. "
                "Instalar con: pip install cloudpickle"
            )
        with open(path, 'wb') as f:
            cloudpickle.dump({'obj': obj, 'metadata': metadata}, f)
        metadata['format'] = 'cloudpickle'
    
    def _load_cloudpickle(self, path: Path) -> Any:
        """Load cloudpickle file."""
        if not HAS_CLOUDPICKLE:
            # Intentar con pickle estándar como fallback
            return self._load_pickle(path)
        with open(path, 'rb') as f:
            data = cloudpickle.load(f)
        return data.get('obj', data)
    
    def _save_joblib(self, obj: Any, path: Path, metadata: Dict):
        """Save using joblib with compression."""
        if not HAS_JOBLIB:
            raise ImportError(
                "joblib no está instalado. "
                "Instalar con: pip install joblib"
            )
        joblib.dump(
            {'obj': obj, 'metadata': metadata},
            path,
            compress=self.compression
        )
        metadata['format'] = 'joblib'
    
    def _load_joblib(self, path: Path) -> Any:
        """Load joblib file."""
        if not HAS_JOBLIB:
            # Intentar con pickle estándar como fallback
            return self._load_pickle(path)
        data = joblib.load(path)
        return data.get('obj', data)
    
    def _save_pickle(self, obj: Any, path: Path, metadata: Dict):
        """Save using standard pickle."""
        with open(path, 'wb') as f:
            pickle.dump({'obj': obj, 'metadata': metadata}, f)
        metadata['format'] = 'pickle'
    
    def _load_pickle(self, path: Path) -> Any:
        """Load standard pickle file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data.get('obj', data)
    
    def _save_onnx(self, obj: Any, path: Path, metadata: Dict):
        """Save model in ONNX format for deployment."""
        try:
            import onnx
            from skl2onnx import to_onnx
            
            # Convert sklearn model to ONNX
            if hasattr(obj, 'predict'):
                # Assume sklearn-compatible model
                initial_types = [('input', FloatTensorType([None, obj.n_features_in_]))]
                onnx_model = to_onnx(obj, initial_types=initial_types)
                onnx.save_model(onnx_model, str(path))
                metadata['format'] = 'onnx'
            else:
                raise ValueError("Object doesn't support ONNX conversion")
                
        except ImportError:
            raise ImportError(
                "ONNX support requires: pip install onnx skl2onnx"
            )
    
    def _load_onnx(self, path: Path) -> Any:
        """Load ONNX model."""
        try:
            import onnx
            import onnxruntime as rt
            
            sess = rt.InferenceSession(str(path))
            return sess
            
        except ImportError:
            raise ImportError(
                "ONNX loading requires: pip install onnx onnxruntime"
            )
    
    def _save_json(self, obj: Any, path: Path, metadata: Dict):
        """Save JSON-serializable objects."""
        # Convert numpy/pandas to JSON-serializable format
        def convert(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            elif isinstance(o, pd.DataFrame):
                return o.to_dict('records')
            elif isinstance(o, pd.Series):
                return o.to_dict()
            elif hasattr(o, '__dict__'):
                return o.__dict__
            return o
        
        with open(path, 'w') as f:
            json.dump(
                {'obj': convert(obj), 'metadata': metadata},
                f,
                indent=2,
                default=str
            )
        metadata['format'] = 'json'
    
    def _load_json(self, path: Path) -> Any:
        """Load JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return data.get('obj', data)
    
    def _prepare_metadata(self, obj: Any, user_metadata: Optional[Dict]) -> Dict:
        """Prepare metadata for serialization."""
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'mlpy_version': self._get_mlpy_version(),
            'object_type': type(obj).__name__,
            'object_module': type(obj).__module__,
        }
        
        # Add object-specific metadata
        if hasattr(obj, 'get_params'):
            metadata['params'] = obj.get_params()
        if hasattr(obj, 'feature_names_in_'):
            metadata['feature_names'] = list(obj.feature_names_in_)
        if hasattr(obj, 'n_features_in_'):
            metadata['n_features'] = obj.n_features_in_
            
        # Add user metadata
        if user_metadata:
            metadata['user_metadata'] = user_metadata
            
        return metadata
    
    def _calculate_checksum(self, path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        sha256_hash = hashlib.sha256()
        with open(path, 'rb') as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _detect_format(self, path: Path) -> str:
        """Try to detect format from file."""
        suffix = path.suffix.lower()
        
        if suffix in ['.pkl', '.pickle']:
            return 'pickle'
        elif suffix == '.joblib':
            return 'joblib'
        elif suffix == '.onnx':
            return 'onnx'
        elif suffix == '.json':
            return 'json'
        else:
            # Try to detect by attempting to load
            return 'cloudpickle'  # Default assumption
    
    def _auto_load(self, path: Path) -> Any:
        """Try multiple formats to load file."""
        errors = []
        
        for format in ['cloudpickle', 'joblib', 'pickle', 'json']:
            try:
                if format == 'cloudpickle':
                    return self._load_cloudpickle(path)
                elif format == 'joblib':
                    return self._load_joblib(path)
                elif format == 'pickle':
                    return self._load_pickle(path)
                elif format == 'json':
                    return self._load_json(path)
            except Exception as e:
                errors.append(f"{format}: {e}")
                continue
        
        raise RuntimeError(
            f"Failed to load file with any format.\n"
            f"Errors:\n" + "\n".join(errors)
        )
    
    def _get_mlpy_version(self) -> str:
        """Get MLPY version."""
        try:
            import mlpy
            return getattr(mlpy, '__version__', 'unknown')
        except:
            return 'unknown'


# Convenience functions
serializer = RobustSerializer()

def save_pipeline(pipeline: Any, path: Union[str, Path], **kwargs) -> Dict:
    """Save MLPY pipeline with robust serialization."""
    return serializer.save(pipeline, path, **kwargs)

def load_pipeline(path: Union[str, Path], **kwargs) -> Any:
    """Load MLPY pipeline with validation."""
    return serializer.load(path, **kwargs)

def export_to_onnx(model: Any, path: Union[str, Path], **kwargs) -> Dict:
    """Export model to ONNX format for deployment."""
    return serializer.save(model, path, format='onnx', **kwargs)


# Alias functions for API compatibility
def save_model(obj: Any, path: Union[str, Path], **kwargs) -> Dict:
    """Save model with robust serialization."""
    return serializer.save(obj, path, **kwargs)


def load_model(path: Union[str, Path], **kwargs) -> Any:
    """Load model with validation."""
    return serializer.load(path, **kwargs)


def compute_checksum(path: Union[str, Path]) -> str:
    """Compute checksum of a file."""
    return serializer._calculate_checksum(Path(path))


def verify_checksum(path: Union[str, Path], expected_checksum: str) -> bool:
    """Verify checksum of a file."""
    actual = compute_checksum(path)
    return actual == expected_checksum