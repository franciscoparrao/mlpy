"""
Servidor API para modelos MLPY usando FastAPI.

Este módulo implementa un servidor REST API para servir modelos MLPY
con endpoints para predicción, información del modelo y monitoreo.
"""

import os
import time
import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path
import logging
import pandas as pd
import numpy as np

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ..registry import FileSystemRegistry, ModelVersion
from ..learners import Learner
from ..tasks import TaskClassif, TaskRegr
from ..monitoring import PerformanceMonitor, DataQualityMonitor
from .schemas import (
    PredictionRequest, PredictionResponse,
    ModelInfo, HealthCheck, ErrorResponse,
    BatchPredictionRequest, BatchPredictionResponse,
    ModelMetrics
)

logger = logging.getLogger(__name__)


class MLPYModelServer:
    """Servidor de modelos MLPY.
    
    Esta clase gestiona el ciclo de vida de los modelos y proporciona
    funcionalidad para servir predicciones a través de una API REST.
    
    Parameters
    ----------
    registry_path : str
        Ruta al registry de modelos.
    default_model : Optional[str]
        Nombre del modelo por defecto.
    enable_monitoring : bool
        Si habilitar monitoreo de modelos.
    max_models_in_memory : int
        Máximo número de modelos a mantener en memoria.
    """
    
    def __init__(
        self,
        registry_path: str = "./mlpy_models",
        default_model: Optional[str] = None,
        enable_monitoring: bool = True,
        max_models_in_memory: int = 10
    ):
        self.registry = FileSystemRegistry(registry_path)
        self.default_model = default_model
        self.enable_monitoring = enable_monitoring
        self.max_models_in_memory = max_models_in_memory
        
        # Cache de modelos cargados
        self.loaded_models: Dict[str, ModelVersion] = {}
        self.model_metrics: Dict[str, Dict] = {}
        
        # Monitoreo
        self.monitors: Dict[str, Any] = {}
        
        # Tiempo de inicio
        self.start_time = datetime.now()
        
        # Cargar modelo por defecto si se especifica
        if default_model:
            self._load_model(default_model)
    
    def _load_model(self, name: str, version: Optional[str] = None) -> ModelVersion:
        """Carga un modelo del registry.
        
        Parameters
        ----------
        name : str
            Nombre del modelo.
        version : Optional[str]
            Versión específica. Si None, carga la última.
            
        Returns
        -------
        ModelVersion
            El modelo cargado.
        """
        cache_key = f"{name}:{version or 'latest'}"
        
        # Verificar cache
        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]
        
        # Cargar del registry
        model = self.registry.get_model(name, version)
        if model is None:
            raise ValueError(f"Model {name} (version {version}) not found")
        
        # Gestionar cache (LRU simple)
        if len(self.loaded_models) >= self.max_models_in_memory:
            # Eliminar el modelo menos usado (simple FIFO por ahora)
            oldest_key = next(iter(self.loaded_models))
            del self.loaded_models[oldest_key]
        
        self.loaded_models[cache_key] = model
        
        # Inicializar métricas
        if cache_key not in self.model_metrics:
            self.model_metrics[cache_key] = {
                "total_predictions": 0,
                "total_time": 0.0,
                "errors": 0,
                "last_prediction": None
            }
        
        logger.info(f"Loaded model {name} version {model.metadata.version}")
        return model
    
    def predict(
        self,
        data: Union[List[List[float]], Dict[str, List[Any]], pd.DataFrame],
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        return_probabilities: bool = False
    ) -> Dict[str, Any]:
        """Realiza predicción con un modelo.
        
        Parameters
        ----------
        data : Union[List[List[float]], Dict[str, List[Any]], pd.DataFrame]
            Datos para predicción.
        model_name : Optional[str]
            Nombre del modelo. Si None, usa el por defecto.
        model_version : Optional[str]
            Versión del modelo.
        return_probabilities : bool
            Si retornar probabilidades (solo clasificación).
            
        Returns
        -------
        Dict[str, Any]
            Resultado de la predicción.
        """
        # Seleccionar modelo
        name = model_name or self.default_model
        if not name:
            raise ValueError("No model specified and no default model set")
        
        # Cargar modelo
        model = self._load_model(name, model_version)
        cache_key = f"{name}:{model_version or 'latest'}"
        
        # Preparar datos
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data
        
        # Crear tarea temporal
        # Asumimos que las columnas coinciden con las features del modelo
        if model.metadata.task_type == "classification":
            # Agregar columna target dummy
            df["_target"] = 0
            task = TaskClassif(df, target="_target")
        else:
            df["_target"] = 0.0
            task = TaskRegr(df, target="_target")
        
        # Medir tiempo
        start_time = time.time()
        
        try:
            # Realizar predicción
            prediction = model.model.predict(task)
            
            # Extraer resultados
            if hasattr(prediction, 'response'):
                predictions = prediction.response.tolist()
            else:
                predictions = prediction.predictions.tolist()
            
            # Obtener probabilidades si es clasificación
            probabilities = None
            if return_probabilities and model.metadata.task_type == "classification":
                if hasattr(prediction, 'prob'):
                    probabilities = prediction.prob.tolist()
            
            # Actualizar métricas
            elapsed_time = time.time() - start_time
            metrics = self.model_metrics[cache_key]
            metrics["total_predictions"] += len(predictions)
            metrics["total_time"] += elapsed_time
            metrics["last_prediction"] = datetime.now()
            
            return {
                "predictions": predictions,
                "probabilities": probabilities,
                "model_name": name,
                "model_version": model.metadata.version,
                "prediction_time": elapsed_time
            }
            
        except Exception as e:
            # Actualizar contador de errores
            self.model_metrics[cache_key]["errors"] += 1
            logger.error(f"Prediction error: {str(e)}")
            raise
    
    def get_model_info(self, name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """Obtiene información sobre un modelo.
        
        Parameters
        ----------
        name : str
            Nombre del modelo.
        version : Optional[str]
            Versión del modelo.
            
        Returns
        -------
        Dict[str, Any]
            Información del modelo.
        """
        model = self.registry.get_model(name, version)
        if model is None:
            raise ValueError(f"Model {name} not found")
        
        # Extraer información de features si es posible
        features = []
        if hasattr(model.model, 'feature_names'):
            features = model.model.feature_names
        
        return {
            "name": model.metadata.name,
            "version": model.metadata.version,
            "task_type": model.metadata.task_type,
            "features": features,
            "metrics": model.metadata.metrics,
            "created_at": model.metadata.created_at,
            "stage": model.metadata.stage.value
        }
    
    def list_models(self) -> List[str]:
        """Lista todos los modelos disponibles.
        
        Returns
        -------
        List[str]
            Lista de nombres de modelos.
        """
        return self.registry.list_models()
    
    def get_metrics(self, model_name: str, model_version: Optional[str] = None) -> Dict[str, Any]:
        """Obtiene métricas de un modelo.
        
        Parameters
        ----------
        model_name : str
            Nombre del modelo.
        model_version : Optional[str]
            Versión del modelo.
            
        Returns
        -------
        Dict[str, Any]
            Métricas del modelo.
        """
        cache_key = f"{model_name}:{model_version or 'latest'}"
        
        if cache_key not in self.model_metrics:
            return {
                "model_name": model_name,
                "model_version": model_version or "latest",
                "total_predictions": 0,
                "avg_prediction_time": 0.0,
                "error_rate": 0.0,
                "last_prediction": None
            }
        
        metrics = self.model_metrics[cache_key]
        total_preds = metrics["total_predictions"]
        
        return {
            "model_name": model_name,
            "model_version": model_version or "latest",
            "total_predictions": total_preds,
            "avg_prediction_time": metrics["total_time"] / max(total_preds, 1),
            "error_rate": metrics["errors"] / max(total_preds + metrics["errors"], 1),
            "last_prediction": metrics["last_prediction"]
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Verifica el estado del servidor.
        
        Returns
        -------
        Dict[str, Any]
            Estado del servidor.
        """
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "status": "healthy",
            "version": "0.1.0",
            "models_loaded": len(self.loaded_models),
            "uptime": uptime
        }


def create_app(
    registry_path: str = "./mlpy_models",
    default_model: Optional[str] = None,
    enable_auth: bool = False,
    api_key: Optional[str] = None,
    enable_cors: bool = True
) -> FastAPI:
    """Crea una aplicación FastAPI para servir modelos.
    
    Parameters
    ----------
    registry_path : str
        Ruta al registry de modelos.
    default_model : Optional[str]
        Modelo por defecto.
    enable_auth : bool
        Si habilitar autenticación.
    api_key : Optional[str]
        API key para autenticación.
    enable_cors : bool
        Si habilitar CORS.
        
    Returns
    -------
    FastAPI
        Aplicación FastAPI configurada.
    """
    # Crear servidor
    server = MLPYModelServer(registry_path, default_model)
    
    # Crear app FastAPI
    app = FastAPI(
        title="MLPY Model Server",
        description="API REST para servir modelos MLPY",
        version="0.1.0"
    )
    
    # Configurar CORS si está habilitado
    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Configurar autenticación si está habilitada
    security = HTTPBearer() if enable_auth else None
    
    def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
        """Verifica el token de autenticación."""
        if enable_auth:
            if not api_key or credentials.credentials != api_key:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )
        return credentials
    
    # Dependency para autenticación opcional
    auth_dep = Depends(verify_token) if enable_auth else None
    
    # Endpoints
    
    @app.get("/", response_model=HealthCheck)
    async def root():
        """Health check endpoint."""
        return server.health_check()
    
    @app.get("/health", response_model=HealthCheck)
    async def health():
        """Health check endpoint."""
        return server.health_check()
    
    @app.post("/predict", response_model=PredictionResponse)
    async def predict(
        request: PredictionRequest,
        auth: Optional[Any] = auth_dep
    ):
        """Realiza predicción con un modelo."""
        try:
            result = server.predict(
                data=request.data,
                model_name=request.model_name,
                model_version=request.model_version,
                return_probabilities=request.return_probabilities
            )
            
            return PredictionResponse(
                predictions=result["predictions"],
                probabilities=result.get("probabilities"),
                model_name=result["model_name"],
                model_version=result["model_version"],
                prediction_time=result["prediction_time"]
            )
            
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    @app.get("/models", response_model=List[str])
    async def list_models(auth: Optional[Any] = auth_dep):
        """Lista todos los modelos disponibles."""
        return server.list_models()
    
    @app.get("/models/{model_name}", response_model=ModelInfo)
    async def get_model_info(
        model_name: str,
        version: Optional[str] = None,
        auth: Optional[Any] = auth_dep
    ):
        """Obtiene información sobre un modelo."""
        try:
            info = server.get_model_info(model_name, version)
            return ModelInfo(**info)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
    
    @app.get("/models/{model_name}/metrics", response_model=ModelMetrics)
    async def get_model_metrics(
        model_name: str,
        version: Optional[str] = None,
        auth: Optional[Any] = auth_dep
    ):
        """Obtiene métricas de uso de un modelo."""
        metrics = server.get_metrics(model_name, version)
        return ModelMetrics(**metrics)
    
    @app.post("/batch/predict", response_model=BatchPredictionResponse)
    async def batch_predict(
        request: BatchPredictionRequest,
        auth: Optional[Any] = auth_dep
    ):
        """Realiza predicción en batch."""
        # Por ahora, implementación síncrona simple
        try:
            result = server.predict(
                data=request.data,
                model_name=request.model_name,
                model_version=request.model_version
            )
            
            return BatchPredictionResponse(
                batch_id=request.batch_id,
                status="completed",
                predictions=result["predictions"],
                progress=100.0
            )
            
        except Exception as e:
            return BatchPredictionResponse(
                batch_id=request.batch_id,
                status="failed",
                error=str(e),
                progress=0.0
            )
    
    # Manejo de errores global
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Manejador de excepciones HTTP."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "HTTPException",
                "message": exc.detail,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Manejador de excepciones generales."""
        logger.error(f"Unhandled exception: {str(exc)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred",
                "timestamp": datetime.now().isoformat()
            }
        )
    
    return app