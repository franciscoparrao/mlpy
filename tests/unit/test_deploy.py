"""
Tests para el módulo de deployment de MLPY.
"""

import pytest
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import json

from mlpy.deploy import (
    MLPYModelServer,
    MLPYClient,
    PredictionRequest,
    PredictionResponse,
    ModelInfo,
    HealthCheck
)
from mlpy.deploy.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelMetrics,
    ErrorResponse
)

from mlpy.registry import FileSystemRegistry, ModelStage
from mlpy.learners import LearnerClassifSklearn
from mlpy.tasks import TaskClassif

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification


class TestSchemas:
    """Test para los esquemas de datos."""
    
    def test_prediction_request_validation(self):
        """Test validación de PredictionRequest."""
        # Request válido con lista
        req = PredictionRequest(
            data=[[1, 2, 3], [4, 5, 6]],
            model_name="test_model",
            return_probabilities=True
        )
        assert req.data == [[1, 2, 3], [4, 5, 6]]
        assert req.model_name == "test_model"
        
        # Request válido con diccionario
        req = PredictionRequest(
            data={"col1": [1, 2], "col2": [3, 4]}
        )
        assert isinstance(req.data, dict)
        
        # Request inválido - data vacío
        with pytest.raises(ValueError, match="cannot be empty"):
            PredictionRequest(data=[])
        
        # Request inválido - columnas de diferente longitud
        with pytest.raises(ValueError, match="same length"):
            PredictionRequest(data={"col1": [1, 2], "col2": [3, 4, 5]})
    
    def test_prediction_response(self):
        """Test PredictionResponse."""
        resp = PredictionResponse(
            predictions=[0, 1, 0],
            probabilities=[[0.9, 0.1], [0.2, 0.8], [0.95, 0.05]],
            model_name="test_model",
            model_version="1.0.0",
            prediction_time=0.023
        )
        
        assert len(resp.predictions) == 3
        assert len(resp.probabilities) == 3
        assert resp.model_name == "test_model"
        assert resp.prediction_time == 0.023
    
    def test_model_info(self):
        """Test ModelInfo."""
        info = ModelInfo(
            name="test_model",
            version="1.0.0",
            task_type="classification",
            features=["feat1", "feat2", "feat3"],
            metrics={"accuracy": 0.95},
            created_at="2024-01-01T12:00:00",
            stage="production"
        )
        
        assert info.name == "test_model"
        assert info.task_type == "classification"
        assert len(info.features) == 3
        assert info.metrics["accuracy"] == 0.95
    
    def test_batch_prediction_request(self):
        """Test BatchPredictionRequest."""
        # Sin modo async
        req = BatchPredictionRequest(
            batch_id="batch_001",
            data=[[1, 2], [3, 4]],
            async_mode=False
        )
        assert req.batch_id == "batch_001"
        
        # Con modo async requiere callback_url
        with pytest.raises(ValueError, match="callback_url required"):
            BatchPredictionRequest(
                batch_id="batch_002",
                data=[[1, 2]],
                async_mode=True
            )


class TestMLPYModelServer:
    """Test para el servidor de modelos."""
    
    @pytest.fixture
    def temp_registry(self):
        """Crea un registry temporal con un modelo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = FileSystemRegistry(tmpdir)
            
            # Crear modelo de prueba
            X, y = make_classification(n_samples=100, n_features=4, random_state=42)
            df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(4)])
            df["target"] = y
            
            task = TaskClassif(df, target="target")
            learner = LearnerClassifSklearn(LogisticRegression())
            learner.train(task)
            learner.task_type = "classification"
            
            # Registrar modelo
            registry.register_model(
                model=learner,
                name="test_model",
                metrics={"accuracy": 0.9}
            )
            
            yield tmpdir
    
    def test_server_initialization(self, temp_registry):
        """Test inicialización del servidor."""
        server = MLPYModelServer(
            registry_path=temp_registry,
            default_model="test_model"
        )
        
        assert server.registry is not None
        assert server.default_model == "test_model"
        assert server.enable_monitoring is True
        assert len(server.loaded_models) == 1  # Modelo por defecto cargado
    
    def test_load_model(self, temp_registry):
        """Test carga de modelo."""
        server = MLPYModelServer(registry_path=temp_registry)
        
        model = server._load_model("test_model")
        assert model is not None
        assert model.metadata.name == "test_model"
        
        # Verificar cache
        cache_key = "test_model:latest"
        assert cache_key in server.loaded_models
    
    def test_predict(self, temp_registry):
        """Test predicción."""
        server = MLPYModelServer(
            registry_path=temp_registry,
            default_model="test_model"
        )
        
        # Datos de prueba
        data = [[1, 2, 3, 4], [5, 6, 7, 8]]
        
        result = server.predict(data)
        
        assert "predictions" in result
        assert "model_name" in result
        assert "model_version" in result
        assert "prediction_time" in result
        assert len(result["predictions"]) == 2
    
    def test_predict_with_probabilities(self, temp_registry):
        """Test predicción con probabilidades."""
        server = MLPYModelServer(
            registry_path=temp_registry,
            default_model="test_model"
        )
        
        data = [[1, 2, 3, 4]]
        
        result = server.predict(data, return_probabilities=True)
        
        assert "predictions" in result
        # Las probabilidades podrían o no estar disponibles según el modelo
    
    def test_get_model_info(self, temp_registry):
        """Test obtener información del modelo."""
        server = MLPYModelServer(registry_path=temp_registry)
        
        info = server.get_model_info("test_model")
        
        assert info["name"] == "test_model"
        assert "version" in info
        assert info["task_type"] == "classification"
        assert info["metrics"]["accuracy"] == 0.9
    
    def test_list_models(self, temp_registry):
        """Test listar modelos."""
        server = MLPYModelServer(registry_path=temp_registry)
        
        models = server.list_models()
        
        assert isinstance(models, list)
        assert "test_model" in models
    
    def test_get_metrics(self, temp_registry):
        """Test obtener métricas."""
        server = MLPYModelServer(
            registry_path=temp_registry,
            default_model="test_model"
        )
        
        # Hacer algunas predicciones
        data = [[1, 2, 3, 4]]
        server.predict(data)
        server.predict(data)
        
        metrics = server.get_metrics("test_model")
        
        assert metrics["model_name"] == "test_model"
        assert metrics["total_predictions"] == 2
        assert metrics["avg_prediction_time"] > 0
        assert metrics["error_rate"] == 0.0
    
    def test_health_check(self, temp_registry):
        """Test health check."""
        server = MLPYModelServer(registry_path=temp_registry)
        
        health = server.health_check()
        
        assert health["status"] == "healthy"
        assert health["version"] == "0.1.0"
        assert health["models_loaded"] >= 0
        assert health["uptime"] >= 0


class TestMLPYClient:
    """Test para el cliente de la API."""
    
    @pytest.fixture
    def mock_server(self):
        """Mock del servidor."""
        with patch('requests.get') as mock_get, \
             patch('requests.post') as mock_post:
            
            # Mock health check
            mock_get.return_value.json.return_value = {
                "status": "healthy",
                "version": "0.1.0",
                "models_loaded": 1,
                "uptime": 100.0
            }
            mock_get.return_value.raise_for_status = Mock()
            
            # Mock predict
            mock_post.return_value.json.return_value = {
                "predictions": [0, 1],
                "model_name": "test_model",
                "model_version": "1.0.0",
                "prediction_time": 0.02
            }
            mock_post.return_value.raise_for_status = Mock()
            
            yield mock_get, mock_post
    
    def test_client_initialization(self, mock_server):
        """Test inicialización del cliente."""
        mock_get, _ = mock_server
        
        client = MLPYClient(
            base_url="http://localhost:8000",
            api_key="test_key"
        )
        
        assert client.base_url == "http://localhost:8000"
        assert client.api_key == "test_key"
        assert "Authorization" in client.headers
        
        # Verificar que se llamó health check
        mock_get.assert_called()
    
    def test_predict(self, mock_server):
        """Test predicción con el cliente."""
        _, mock_post = mock_server
        
        client = MLPYClient()
        
        # Predicción con lista
        result = client.predict(
            data=[[1, 2, 3], [4, 5, 6]],
            model_name="test_model"
        )
        
        assert result["predictions"] == [0, 1]
        assert result["model_name"] == "test_model"
        
        # Verificar request
        mock_post.assert_called()
        call_args = mock_post.call_args
        assert call_args[1]["json"]["data"] == [[1, 2, 3], [4, 5, 6]]
    
    def test_predict_with_dataframe(self, mock_server):
        """Test predicción con DataFrame."""
        _, mock_post = mock_server
        
        client = MLPYClient()
        
        # Predicción con DataFrame
        df = pd.DataFrame({
            "col1": [1, 2],
            "col2": [3, 4],
            "col3": [5, 6]
        })
        
        result = client.predict(data=df)
        
        assert result["predictions"] == [0, 1]
        
        # Verificar que se convirtió correctamente
        call_args = mock_post.call_args
        data_sent = call_args[1]["json"]["data"]
        assert "col1" in data_sent
        assert data_sent["col1"] == [1, 2]
    
    def test_list_models(self, mock_server):
        """Test listar modelos."""
        mock_get, _ = mock_server
        mock_get.return_value.json.return_value = ["model1", "model2"]
        
        client = MLPYClient()
        models = client.list_models()
        
        assert models == ["model1", "model2"]
    
    def test_get_model_info(self, mock_server):
        """Test obtener información del modelo."""
        mock_get, _ = mock_server
        mock_get.return_value.json.return_value = {
            "name": "test_model",
            "version": "1.0.0",
            "task_type": "classification",
            "features": ["f1", "f2"]
        }
        
        client = MLPYClient()
        info = client.get_model_info("test_model")
        
        assert info["name"] == "test_model"
        assert info["task_type"] == "classification"
    
    def test_error_handling(self, mock_server):
        """Test manejo de errores."""
        mock_get, mock_post = mock_server
        
        # Simular error HTTP
        mock_post.return_value.raise_for_status.side_effect = \
            Exception("Server error")
        
        client = MLPYClient()
        
        with pytest.raises(Exception, match="Server error"):
            client.predict(data=[[1, 2, 3]])