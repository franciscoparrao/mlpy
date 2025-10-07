"""
Cliente Python para la API de MLPY.

Este módulo proporciona un cliente Python para interactuar con
el servidor de modelos MLPY de forma programática.
"""

import json
import time
from typing import Dict, List, Optional, Union, Any
import requests
from urllib.parse import urljoin
import pandas as pd
import numpy as np


class MLPYClient:
    """Cliente para el servidor de modelos MLPY.
    
    Este cliente facilita la interacción con el servidor API de MLPY,
    proporcionando métodos convenientes para predicción y gestión de modelos.
    
    Parameters
    ----------
    base_url : str
        URL base del servidor (ej: "http://localhost:8000").
    api_key : Optional[str]
        API key para autenticación.
    timeout : int
        Timeout para requests en segundos.
    verify_ssl : bool
        Si verificar certificados SSL.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 30,
        verify_ssl: bool = True
    ):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        
        # Configurar headers
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        
        # Verificar conexión
        self._verify_connection()
    
    def _verify_connection(self):
        """Verifica que el servidor esté disponible."""
        try:
            response = self.health_check()
            if response["status"] != "healthy":
                raise ConnectionError(f"Server is not healthy: {response}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Cannot connect to server at {self.base_url}: {str(e)}")
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Realiza una request HTTP al servidor.
        
        Parameters
        ----------
        method : str
            Método HTTP (GET, POST, etc).
        endpoint : str
            Endpoint de la API.
        data : Optional[Dict]
            Datos para enviar en el body.
        params : Optional[Dict]
            Parámetros de query.
            
        Returns
        -------
        Dict[str, Any]
            Respuesta del servidor.
        """
        url = urljoin(self.base_url, endpoint)
        
        try:
            if method.upper() == "GET":
                response = requests.get(
                    url,
                    headers=self.headers,
                    params=params,
                    timeout=self.timeout,
                    verify=self.verify_ssl
                )
            elif method.upper() == "POST":
                response = requests.post(
                    url,
                    headers=self.headers,
                    json=data,
                    params=params,
                    timeout=self.timeout,
                    verify=self.verify_ssl
                )
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            if e.response is not None:
                try:
                    error_detail = e.response.json()
                    raise Exception(f"API Error: {error_detail.get('message', str(e))}")
                except json.JSONDecodeError:
                    raise Exception(f"API Error: {e.response.text}")
            raise
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {str(e)}")
    
    def predict(
        self,
        data: Union[List[List[float]], Dict[str, List[Any]], pd.DataFrame, np.ndarray],
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        return_probabilities: bool = False
    ) -> Dict[str, Any]:
        """Realiza predicción con un modelo.
        
        Parameters
        ----------
        data : Union[List[List[float]], Dict[str, List[Any]], pd.DataFrame, np.ndarray]
            Datos para predicción.
        model_name : Optional[str]
            Nombre del modelo.
        model_version : Optional[str]
            Versión del modelo.
        return_probabilities : bool
            Si retornar probabilidades.
            
        Returns
        -------
        Dict[str, Any]
            Resultado de la predicción con campos:
            - predictions: Lista de predicciones
            - probabilities: Probabilidades (si solicitadas)
            - model_name: Nombre del modelo usado
            - model_version: Versión del modelo
            - prediction_time: Tiempo de predicción
        """
        # Convertir datos al formato correcto
        if isinstance(data, pd.DataFrame):
            # Convertir DataFrame a diccionario de columnas
            request_data = {col: data[col].tolist() for col in data.columns}
        elif isinstance(data, np.ndarray):
            # Convertir numpy array a lista
            request_data = data.tolist()
        else:
            request_data = data
        
        # Crear request
        request_body = {
            "data": request_data,
            "return_probabilities": return_probabilities
        }
        
        if model_name:
            request_body["model_name"] = model_name
        if model_version:
            request_body["model_version"] = model_version
        
        # Hacer predicción
        response = self._make_request("POST", "/predict", data=request_body)
        
        return response
    
    def predict_batch(
        self,
        batch_id: str,
        data: Union[List[List[float]], Dict[str, List[Any]], pd.DataFrame],
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        async_mode: bool = False,
        callback_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Realiza predicción en batch.
        
        Parameters
        ----------
        batch_id : str
            ID único del batch.
        data : Union[List[List[float]], Dict[str, List[Any]], pd.DataFrame]
            Datos para predicción.
        model_name : Optional[str]
            Nombre del modelo.
        model_version : Optional[str]
            Versión del modelo.
        async_mode : bool
            Si procesar de forma asíncrona.
        callback_url : Optional[str]
            URL para callback (si async).
            
        Returns
        -------
        Dict[str, Any]
            Estado del batch.
        """
        # Convertir datos
        if isinstance(data, pd.DataFrame):
            request_data = {col: data[col].tolist() for col in data.columns}
        elif isinstance(data, np.ndarray):
            request_data = data.tolist()
        else:
            request_data = data
        
        # Crear request
        request_body = {
            "batch_id": batch_id,
            "data": request_data,
            "async_mode": async_mode
        }
        
        if model_name:
            request_body["model_name"] = model_name
        if model_version:
            request_body["model_version"] = model_version
        if callback_url:
            request_body["callback_url"] = callback_url
        
        response = self._make_request("POST", "/batch/predict", data=request_body)
        
        return response
    
    def list_models(self) -> List[str]:
        """Lista todos los modelos disponibles.
        
        Returns
        -------
        List[str]
            Lista de nombres de modelos.
        """
        response = self._make_request("GET", "/models")
        return response
    
    def get_model_info(
        self,
        model_name: str,
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """Obtiene información sobre un modelo.
        
        Parameters
        ----------
        model_name : str
            Nombre del modelo.
        version : Optional[str]
            Versión del modelo.
            
        Returns
        -------
        Dict[str, Any]
            Información del modelo.
        """
        endpoint = f"/models/{model_name}"
        params = {"version": version} if version else None
        
        response = self._make_request("GET", endpoint, params=params)
        return response
    
    def get_model_metrics(
        self,
        model_name: str,
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """Obtiene métricas de uso de un modelo.
        
        Parameters
        ----------
        model_name : str
            Nombre del modelo.
        version : Optional[str]
            Versión del modelo.
            
        Returns
        -------
        Dict[str, Any]
            Métricas del modelo.
        """
        endpoint = f"/models/{model_name}/metrics"
        params = {"version": version} if version else None
        
        response = self._make_request("GET", endpoint, params=params)
        return response
    
    def health_check(self) -> Dict[str, Any]:
        """Verifica el estado del servidor.
        
        Returns
        -------
        Dict[str, Any]
            Estado del servidor con campos:
            - status: Estado (healthy/unhealthy)
            - version: Versión de MLPY
            - models_loaded: Número de modelos cargados
            - uptime: Tiempo de uptime
        """
        response = self._make_request("GET", "/health")
        return response
    
    def wait_for_batch(
        self,
        batch_id: str,
        timeout: int = 300,
        poll_interval: int = 2
    ) -> Dict[str, Any]:
        """Espera a que un batch termine.
        
        Parameters
        ----------
        batch_id : str
            ID del batch.
        timeout : int
            Timeout máximo en segundos.
        poll_interval : int
            Intervalo de polling en segundos.
            
        Returns
        -------
        Dict[str, Any]
            Resultado del batch.
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Obtener estado del batch
            # Nota: Este endpoint necesitaría ser implementado en el servidor
            response = self._make_request("GET", f"/batch/{batch_id}/status")
            
            if response["status"] in ["completed", "failed"]:
                return response
            
            time.sleep(poll_interval)
        
        raise TimeoutError(f"Batch {batch_id} did not complete within {timeout} seconds")


class AsyncMLPYClient:
    """Cliente asíncrono para el servidor MLPY.
    
    Versión asíncrona del cliente usando aiohttp para operaciones no bloqueantes.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None
    ):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    async def predict(
        self,
        data: Union[List[List[float]], Dict[str, List[Any]]],
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        return_probabilities: bool = False
    ) -> Dict[str, Any]:
        """Realiza predicción asíncrona."""
        import aiohttp
        
        url = f"{self.base_url}/predict"
        
        request_body = {
            "data": data,
            "return_probabilities": return_probabilities
        }
        
        if model_name:
            request_body["model_name"] = model_name
        if model_version:
            request_body["model_version"] = model_version
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=request_body,
                headers=self.headers
            ) as response:
                response.raise_for_status()
                return await response.json()
    
    async def batch_predict_async(
        self,
        batch_data: List[Dict[str, Any]],
        model_name: Optional[str] = None,
        model_version: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Realiza múltiples predicciones en paralelo."""
        import asyncio
        
        tasks = []
        for data in batch_data:
            task = self.predict(
                data=data,
                model_name=model_name,
                model_version=model_version
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results