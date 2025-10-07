"""ONNX serializer for compatible models."""

import warnings
from typing import Any, Dict, Union, BinaryIO, Optional
from pathlib import Path
import numpy as np

from .base import ModelSerializer, SERIALIZERS
from ..learners import Learner, LearnerClassif, LearnerRegr

# Try to import ONNX dependencies
try:
    import onnx
    import skl2onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    
try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False


@SERIALIZERS.register("onnx")
class ONNXSerializer(ModelSerializer):
    """Serializer for ONNX format.
    
    ONNX (Open Neural Network Exchange) is an open format to
    represent machine learning models. It allows models to be
    transferred between different frameworks and deployed
    efficiently.
    
    This serializer supports scikit-learn models and other
    models that can be converted to ONNX format.
    
    Parameters
    ----------
    opset_version : int, optional
        ONNX opset version to use. If None, uses default.
    compression : str or None, default=None
        Not directly supported. ONNX files can be compressed
        separately if needed.
    initial_types : list, optional
        Initial types for ONNX conversion. If None, will be
        inferred from the model.
    """
    
    def __init__(
        self,
        opset_version: Optional[int] = None,
        compression: Optional[str] = None,
        initial_types: Optional[list] = None
    ):
        if not ONNX_AVAILABLE:
            raise ImportError(
                "onnx and skl2onnx are required for ONNXSerializer. "
                "Install them with: pip install onnx skl2onnx"
            )
            
        super().__init__(compression)
        self.opset_version = opset_version
        self.initial_types = initial_types
        
        if compression is not None:
            warnings.warn(
                "ONNXSerializer does not directly support compression. "
                "You can compress ONNX files separately if needed."
            )
            
    def can_serialize(self, obj: Any) -> bool:
        """Check if object can be serialized to ONNX.
        
        Currently supports:
        - Scikit-learn models wrapped in MLPY learners
        - Models that have been registered with skl2onnx
        """
        if not isinstance(obj, Learner):
            return False
            
        # Check if it's a sklearn wrapper
        if hasattr(obj, '_model') and hasattr(obj._model, 'fit'):
            # Check if sklearn model is supported by skl2onnx
            model = obj._model
            model_type = type(model)
            
            # Get list of supported models
            try:
                from skl2onnx import supported_converters
                supported_models = list(supported_converters.keys())
                return model_type in supported_models
            except:
                # Conservative approach - only support common models
                supported_names = [
                    'LinearRegression', 'LogisticRegression',
                    'DecisionTreeClassifier', 'DecisionTreeRegressor',
                    'RandomForestClassifier', 'RandomForestRegressor',
                    'GradientBoostingClassifier', 'GradientBoostingRegressor',
                    'SVC', 'SVR', 'KNeighborsClassifier', 'KNeighborsRegressor'
                ]
                return model_type.__name__ in supported_names
                
        return False
        
    def serialize(self, obj: Any, path: Union[str, Path, BinaryIO]) -> Dict[str, Any]:
        """Convert model to ONNX and save."""
        if not isinstance(obj, Learner) or not hasattr(obj, '_model'):
            raise ValueError("ONNXSerializer requires a trained MLPY Learner with sklearn model")
            
        if not obj.is_trained:
            raise ValueError("Model must be trained before serialization")
            
        model = obj._model
        
        # Determine initial types if not provided
        if self.initial_types is None:
            # Try to infer from the model
            if hasattr(model, 'n_features_in_'):
                n_features = model.n_features_in_
            elif hasattr(model, 'n_features_'):
                n_features = model.n_features_
            else:
                raise ValueError(
                    "Cannot determine number of input features. "
                    "Please provide initial_types parameter."
                )
                
            # Default to float inputs
            self.initial_types = [('input', FloatTensorType([None, n_features]))]
            
        # Set conversion options
        options = {}
        if isinstance(obj, LearnerClassif):
            # For classifiers, include probability outputs
            options['zipmap'] = False
            
        # Convert to ONNX
        try:
            if self.opset_version:
                onnx_model = convert_sklearn(
                    model,
                    initial_types=self.initial_types,
                    target_opset=self.opset_version,
                    options=options
                )
            else:
                onnx_model = convert_sklearn(
                    model,
                    initial_types=self.initial_types,
                    options=options
                )
        except Exception as e:
            raise RuntimeError(f"Failed to convert model to ONNX: {e}")
            
        # Save ONNX model
        if hasattr(path, 'write'):
            # File-like object
            path.write(onnx_model.SerializeToString())
        else:
            # Path
            onnx.save_model(onnx_model, str(path))
            
        # Return metadata
        metadata = {
            "serializer": "onnx",
            "mlpy_class": type(obj).__name__,
            "sklearn_class": type(model).__name__,
            "opset_version": onnx_model.opset_import[0].version,
            "input_name": self.initial_types[0][0],
            "input_shape": self.initial_types[0][1].shape,
            "n_outputs": len(onnx_model.graph.output)
        }
        
        if isinstance(obj, LearnerClassif) and hasattr(model, 'classes_'):
            metadata['classes'] = model.classes_.tolist()
            
        return metadata
        
    def deserialize(self, path: Union[str, Path, BinaryIO]) -> Any:
        """Load ONNX model.
        
        Note: This returns an ONNXRuntime session, not the original
        MLPY Learner. You'll need to wrap it appropriately.
        """
        if not ONNXRUNTIME_AVAILABLE:
            raise ImportError(
                "onnxruntime is required to load ONNX models. "
                "Install it with: pip install onnxruntime"
            )
            
        # Load ONNX model
        if hasattr(path, 'read'):
            # File-like object
            onnx_bytes = path.read()
            session = ort.InferenceSession(onnx_bytes)
        else:
            # Path
            session = ort.InferenceSession(str(path))
            
        # We return the ONNX session wrapped in a simple predictor
        return ONNXPredictor(session)
        
    @property
    def file_extension(self) -> str:
        """Default extension for ONNX files."""
        return ".onnx"


class ONNXPredictor:
    """Simple wrapper for ONNX runtime session.
    
    This provides a scikit-learn like interface for ONNX models.
    """
    
    def __init__(self, session):
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.output_names = [o.name for o in session.get_outputs()]
        
    def predict(self, X):
        """Make predictions."""
        if not isinstance(X, np.ndarray):
            X = np.array(X)
            
        # Ensure float32 as ONNX typically expects this
        if X.dtype != np.float32:
            X = X.astype(np.float32)
            
        # Run inference
        outputs = self.session.run(
            self.output_names,
            {self.input_name: X}
        )
        
        # Return first output (predictions)
        return outputs[0]
        
    def predict_proba(self, X):
        """Predict probabilities (for classifiers)."""
        if len(self.output_names) < 2:
            raise ValueError("Model does not support probability predictions")
            
        if not isinstance(X, np.ndarray):
            X = np.array(X)
            
        if X.dtype != np.float32:
            X = X.astype(np.float32)
            
        # Run inference
        outputs = self.session.run(
            self.output_names,
            {self.input_name: X}
        )
        
        # Return probabilities (usually second output)
        return outputs[1]


def convert_to_onnx(
    learner: Learner,
    output_path: Optional[Union[str, Path]] = None,
    initial_types: Optional[list] = None,
    opset_version: Optional[int] = None,
    **kwargs
) -> Union[bytes, Path]:
    """Convert an MLPY learner to ONNX format.
    
    Parameters
    ----------
    learner : Learner
        Trained MLPY learner to convert.
    output_path : str or Path, optional
        Where to save the ONNX model. If None, returns bytes.
    initial_types : list, optional
        Initial types for ONNX conversion.
    opset_version : int, optional
        ONNX opset version to use.
    **kwargs
        Additional arguments for conversion.
        
    Returns
    -------
    bytes or Path
        ONNX model bytes if output_path is None, otherwise the path.
        
    Examples
    --------
    >>> from mlpy.persistence.onnx_serializer import convert_to_onnx
    >>> 
    >>> # Convert to file
    >>> convert_to_onnx(learner, "model.onnx")
    >>> 
    >>> # Get ONNX bytes
    >>> onnx_bytes = convert_to_onnx(learner)
    """
    serializer = ONNXSerializer(
        opset_version=opset_version,
        initial_types=initial_types
    )
    
    if not serializer.can_serialize(learner):
        raise ValueError(f"Cannot convert {type(learner).__name__} to ONNX")
        
    if output_path is None:
        # Return bytes
        import io
        buffer = io.BytesIO()
        serializer.serialize(learner, buffer)
        return buffer.getvalue()
    else:
        # Save to file
        serializer.serialize(learner, output_path)
        return Path(output_path)


__all__ = [
    "ONNXSerializer",
    "ONNXPredictor", 
    "convert_to_onnx"
]