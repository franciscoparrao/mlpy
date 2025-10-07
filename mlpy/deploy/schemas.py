"""
Esquemas de datos para la API de deployment.

Define los modelos de datos para requests y responses de la API.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator
import numpy as np


class PredictionRequest(BaseModel):
    """Request para predicción.
    
    Attributes
    ----------
    data : Union[List[List[float]], Dict[str, List[Any]]]
        Datos para predicción. Puede ser una lista de listas (array)
        o un diccionario con columnas (como DataFrame).
    model_name : Optional[str]
        Nombre del modelo a usar. Si None, usa el modelo por defecto.
    model_version : Optional[str]
        Versión específica del modelo. Si None, usa la última.
    return_probabilities : bool
        Si retornar probabilidades para clasificación.
    """
    data: Union[List[List[float]], Dict[str, List[Any]]]
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    return_probabilities: bool = False
    
    @validator('data')
    def validate_data(cls, v):
        """Valida que los datos tengan formato correcto."""
        if isinstance(v, list):
            if not v:
                raise ValueError("Data cannot be empty")
            if not all(isinstance(row, list) for row in v):
                raise ValueError("Data must be list of lists")
        elif isinstance(v, dict):
            if not v:
                raise ValueError("Data cannot be empty")
            # Verificar que todas las columnas tengan la misma longitud
            lengths = [len(col) for col in v.values()]
            if len(set(lengths)) > 1:
                raise ValueError("All columns must have same length")
        else:
            raise ValueError("Data must be list or dict")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "data": [[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2]],
                "model_name": "iris_classifier",
                "model_version": "1.0.0",
                "return_probabilities": False
            }
        }


class PredictionResponse(BaseModel):
    """Response de predicción.
    
    Attributes
    ----------
    predictions : List[Union[int, float, str]]
        Predicciones del modelo.
    probabilities : Optional[List[List[float]]]
        Probabilidades por clase (solo para clasificación).
    model_name : str
        Nombre del modelo usado.
    model_version : str
        Versión del modelo usado.
    prediction_time : float
        Tiempo de predicción en segundos.
    timestamp : datetime
        Timestamp de la predicción.
    """
    predictions: List[Union[int, float, str]]
    probabilities: Optional[List[List[float]]] = None
    model_name: str
    model_version: str
    prediction_time: float
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [0, 0],
                "probabilities": [[0.95, 0.03, 0.02], [0.94, 0.04, 0.02]],
                "model_name": "iris_classifier",
                "model_version": "1.0.0",
                "prediction_time": 0.023,
                "timestamp": "2024-12-10T10:30:00"
            }
        }


class ModelInfo(BaseModel):
    """Información sobre un modelo.
    
    Attributes
    ----------
    name : str
        Nombre del modelo.
    version : str
        Versión del modelo.
    task_type : str
        Tipo de tarea (classification, regression, etc).
    features : List[str]
        Lista de features esperados.
    target : Optional[str]
        Nombre de la variable target.
    metrics : Optional[Dict[str, float]]
        Métricas del modelo.
    created_at : datetime
        Fecha de creación.
    stage : str
        Stage del modelo (development, staging, production).
    """
    name: str
    version: str
    task_type: str
    features: List[str]
    target: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    created_at: datetime
    stage: str = "development"
    
    class Config:
        schema_extra = {
            "example": {
                "name": "iris_classifier",
                "version": "1.0.0",
                "task_type": "classification",
                "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
                "target": "species",
                "metrics": {"accuracy": 0.95, "f1_score": 0.94},
                "created_at": "2024-12-10T09:00:00",
                "stage": "production"
            }
        }


class HealthCheck(BaseModel):
    """Health check response.
    
    Attributes
    ----------
    status : str
        Estado del servicio (healthy, unhealthy).
    version : str
        Versión de MLPY.
    models_loaded : int
        Número de modelos cargados.
    uptime : float
        Tiempo de uptime en segundos.
    """
    status: str
    version: str
    models_loaded: int
    uptime: float
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "version": "0.1.0",
                "models_loaded": 3,
                "uptime": 3600.5
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request para predicción en batch.
    
    Attributes
    ----------
    batch_id : str
        ID único del batch.
    data : Union[List[List[float]], Dict[str, List[Any]]]
        Datos para predicción.
    model_name : Optional[str]
        Nombre del modelo.
    model_version : Optional[str]
        Versión del modelo.
    async_mode : bool
        Si procesar de forma asíncrona.
    callback_url : Optional[str]
        URL para callback cuando termine (si async).
    """
    batch_id: str
    data: Union[List[List[float]], Dict[str, List[Any]]]
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    async_mode: bool = False
    callback_url: Optional[str] = None
    
    @validator('callback_url')
    def validate_callback(cls, v, values):
        """Valida que callback_url se provea si async_mode es True."""
        if values.get('async_mode') and not v:
            raise ValueError("callback_url required when async_mode is True")
        return v


class BatchPredictionResponse(BaseModel):
    """Response de predicción en batch.
    
    Attributes
    ----------
    batch_id : str
        ID del batch.
    status : str
        Estado del batch (pending, processing, completed, failed).
    predictions : Optional[List[Union[int, float, str]]]
        Predicciones (si completado).
    error : Optional[str]
        Mensaje de error (si falló).
    progress : float
        Progreso (0-100).
    """
    batch_id: str
    status: str
    predictions: Optional[List[Union[int, float, str]]] = None
    error: Optional[str] = None
    progress: float = 0.0


class ModelMetrics(BaseModel):
    """Métricas del modelo.
    
    Attributes
    ----------
    model_name : str
        Nombre del modelo.
    model_version : str
        Versión del modelo.
    total_predictions : int
        Total de predicciones realizadas.
    avg_prediction_time : float
        Tiempo promedio de predicción.
    error_rate : float
        Tasa de errores.
    last_prediction : Optional[datetime]
        Última predicción realizada.
    """
    model_name: str
    model_version: str
    total_predictions: int
    avg_prediction_time: float
    error_rate: float
    last_prediction: Optional[datetime] = None


class ErrorResponse(BaseModel):
    """Response de error.
    
    Attributes
    ----------
    error : str
        Tipo de error.
    message : str
        Mensaje descriptivo.
    details : Optional[Dict[str, Any]]
        Detalles adicionales del error.
    timestamp : datetime
        Cuando ocurrió el error.
    """
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        schema_extra = {
            "example": {
                "error": "ModelNotFound",
                "message": "Model 'unknown_model' not found in registry",
                "details": {"available_models": ["model1", "model2"]},
                "timestamp": "2024-12-10T10:30:00"
            }
        }