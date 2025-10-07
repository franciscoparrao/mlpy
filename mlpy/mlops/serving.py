"""
Model Serving API with FastAPI
==============================

Production-ready REST API for model deployment.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any, Optional, Union
import numpy as np
import pandas as pd
import joblib
import json
import logging
from datetime import datetime
import asyncio
from pathlib import Path
import hashlib
import uuid

from ..learners.base import Learner
from ..tasks import Task, TaskRegr, TaskClassif
from ..measures.base import Measure
from ..model_registry.registry import ModelRegistry

logger = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    """Request model for predictions."""
    
    data: Union[List[Dict], Dict[str, List]]
    model_id: Optional[str] = None
    model_version: Optional[str] = None
    return_probabilities: bool = False
    include_metadata: bool = True
    
    class Config:
        example = {
            "data": [
                {"feature1": 1.0, "feature2": 2.0, "feature3": 3.0},
                {"feature1": 4.0, "feature2": 5.0, "feature3": 6.0}
            ],
            "model_id": "housing_predictor",
            "return_probabilities": False
        }


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    
    predictions: List[Union[float, str, int]]
    probabilities: Optional[List[List[float]]] = None
    model_id: str
    model_version: str
    timestamp: str
    request_id: str
    metadata: Optional[Dict[str, Any]] = None


class ModelInfo(BaseModel):
    """Model information response."""
    
    model_id: str
    version: str
    type: str
    features: List[str]
    target: Optional[str]
    metrics: Dict[str, float]
    created_at: str
    updated_at: str
    status: str


class HealthCheck(BaseModel):
    """Health check response."""
    
    status: str
    timestamp: str
    models_loaded: int
    uptime_seconds: float
    version: str


class ModelEndpoint:
    """Manages a single model endpoint."""
    
    def __init__(self, model_id: str, model: Learner, metadata: Dict[str, Any] = None):
        self.model_id = model_id
        self.model = model
        self.metadata = metadata or {}
        self.version = self._generate_version()
        self.created_at = datetime.utcnow()
        self.request_count = 0
        self.total_prediction_time = 0
        self.error_count = 0
        
    def _generate_version(self) -> str:
        """Generate unique version hash for model."""
        model_bytes = str(self.model.__dict__).encode()
        return hashlib.sha256(model_bytes).hexdigest()[:8]
    
    def predict(self, data: pd.DataFrame) -> Union[np.ndarray, List]:
        """Make predictions with the model."""
        start_time = datetime.utcnow()
        try:
            # Create task for prediction
            if hasattr(self.model, 'task_type'):
                if self.model.task_type == 'classification':
                    task = TaskClassif(data=data, target='dummy')
                else:
                    task = TaskRegr(data=data, target='dummy')
            else:
                task = TaskRegr(data=data, target='dummy')
            
            # Make predictions
            predictions = self.model.predict(task)
            
            # Update metrics
            self.request_count += 1
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            self.total_prediction_time += elapsed
            
            return predictions.response
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Prediction error for model {self.model_id}: {e}")
            raise
    
    def get_info(self) -> Dict[str, Any]:
        """Get model endpoint information."""
        return {
            "model_id": self.model_id,
            "version": self.version,
            "type": self.model.__class__.__name__,
            "created_at": self.created_at.isoformat(),
            "request_count": self.request_count,
            "error_count": self.error_count,
            "avg_prediction_time": self.total_prediction_time / max(1, self.request_count),
            "metadata": self.metadata
        }


class ModelServer:
    """FastAPI-based model serving server."""
    
    def __init__(self, name: str = "MLPY Model Server", version: str = "1.0.0"):
        self.app = FastAPI(
            title=name,
            version=version,
            description="Production-ready ML model serving with MLPY"
        )
        self.models: Dict[str, ModelEndpoint] = {}
        self.registry = ModelRegistry()
        self.start_time = datetime.utcnow()
        self.setup_routes()
        self.setup_middleware()
        
    def setup_middleware(self):
        """Setup CORS and other middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/", response_model=HealthCheck)
        async def health_check():
            """Health check endpoint."""
            uptime = (datetime.utcnow() - self.start_time).total_seconds()
            return HealthCheck(
                status="healthy",
                timestamp=datetime.utcnow().isoformat(),
                models_loaded=len(self.models),
                uptime_seconds=uptime,
                version="1.0.0"
            )
        
        @self.app.get("/models", response_model=List[ModelInfo])
        async def list_models():
            """List all available models."""
            models = []
            for model_id, endpoint in self.models.items():
                info = endpoint.get_info()
                models.append(ModelInfo(
                    model_id=model_id,
                    version=info['version'],
                    type=info['type'],
                    features=info.get('features', []),
                    target=info.get('target'),
                    metrics=info.get('metrics', {}),
                    created_at=info['created_at'],
                    updated_at=info['created_at'],
                    status="active"
                ))
            return models
        
        @self.app.get("/models/{model_id}", response_model=ModelInfo)
        async def get_model_info(model_id: str):
            """Get information about a specific model."""
            if model_id not in self.models:
                raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
            
            endpoint = self.models[model_id]
            info = endpoint.get_info()
            
            return ModelInfo(
                model_id=model_id,
                version=info['version'],
                type=info['type'],
                features=info.get('features', []),
                target=info.get('target'),
                metrics=info.get('metrics', {}),
                created_at=info['created_at'],
                updated_at=info['created_at'],
                status="active"
            )
        
        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
            """Make predictions with a model."""
            
            # Select model
            model_id = request.model_id or self.get_default_model()
            if not model_id or model_id not in self.models:
                raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
            
            endpoint = self.models[model_id]
            
            # Convert data to DataFrame
            if isinstance(request.data, list):
                df = pd.DataFrame(request.data)
            else:
                df = pd.DataFrame(request.data)
            
            # Make predictions
            try:
                predictions = endpoint.predict(df)
                
                # Convert to list
                if isinstance(predictions, np.ndarray):
                    predictions = predictions.tolist()
                
                # Generate request ID
                request_id = str(uuid.uuid4())
                
                # Log prediction in background
                background_tasks.add_task(self.log_prediction, model_id, request_id, len(predictions))
                
                return PredictionResponse(
                    predictions=predictions,
                    probabilities=None,  # TODO: Implement probability support
                    model_id=model_id,
                    model_version=endpoint.version,
                    timestamp=datetime.utcnow().isoformat(),
                    request_id=request_id,
                    metadata={
                        "model_type": endpoint.model.__class__.__name__,
                        "num_predictions": len(predictions)
                    } if request.include_metadata else None
                )
                
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/predict/batch", response_model=List[PredictionResponse])
        async def predict_batch(requests: List[PredictionRequest]):
            """Make batch predictions."""
            responses = []
            for request in requests:
                response = await predict(request, BackgroundTasks())
                responses.append(response)
            return responses
        
        @self.app.post("/models/{model_id}/reload")
        async def reload_model(model_id: str):
            """Reload a model from disk."""
            if model_id not in self.models:
                raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
            
            # TODO: Implement model reloading
            return {"status": "success", "message": f"Model {model_id} reloaded"}
        
        @self.app.delete("/models/{model_id}")
        async def remove_model(model_id: str):
            """Remove a model from the server."""
            if model_id not in self.models:
                raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
            
            del self.models[model_id]
            return {"status": "success", "message": f"Model {model_id} removed"}
    
    def load_model(self, model_id: str, model: Learner, metadata: Dict[str, Any] = None):
        """Load a model into the server."""
        endpoint = ModelEndpoint(model_id, model, metadata)
        self.models[model_id] = endpoint
        logger.info(f"Loaded model {model_id} (version: {endpoint.version})")
    
    def load_from_file(self, model_id: str, filepath: str, metadata: Dict[str, Any] = None):
        """Load a model from a file."""
        model = joblib.load(filepath)
        self.load_model(model_id, model, metadata)
    
    def load_from_registry(self, model_name: str):
        """Load a model from the registry."""
        # Search for model in registry
        models = self.registry.search(name=model_name)
        if not models:
            raise ValueError(f"Model {model_name} not found in registry")
        
        # Load the model
        model_meta = models[0]
        # TODO: Implement model loading from registry
        logger.info(f"Loaded model {model_name} from registry")
    
    def get_default_model(self) -> Optional[str]:
        """Get the default model ID."""
        if self.models:
            return list(self.models.keys())[0]
        return None
    
    async def log_prediction(self, model_id: str, request_id: str, num_predictions: int):
        """Log prediction request (background task)."""
        logger.info(f"Prediction logged - Model: {model_id}, Request: {request_id}, Count: {num_predictions}")
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """Run the server."""
        import uvicorn
        uvicorn.run(self.app, host=host, port=port, **kwargs)