"""
Integración con Google Cloud Platform (GCP) para MLPY.

Soporte para Google Cloud Storage, Vertex AI y otros servicios GCP.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
import tempfile
import os

from .base import CloudProvider, CloudStorage, CloudCompute, CloudConfig

logger = logging.getLogger(__name__)


@dataclass
class GCPConfig(CloudConfig):
    """Configuración específica para GCP.
    
    Attributes
    ----------
    credentials_path : Optional[str]
        Ruta al archivo de credenciales JSON.
    service_account_email : Optional[str]
        Email de la cuenta de servicio.
    """
    credentials_path: Optional[str] = None
    service_account_email: Optional[str] = None


class GCPProvider(CloudProvider):
    """Proveedor para Google Cloud Platform."""
    
    def _setup(self):
        """Configura los clientes de GCP."""
        try:
            from google.cloud import storage
            from google.cloud import aiplatform
            from google.oauth2 import service_account
            
            self.storage = storage
            self.aiplatform = aiplatform
            
            # Configurar credenciales
            if hasattr(self.config, 'credentials_path') and self.config.credentials_path:
                credentials = service_account.Credentials.from_service_account_file(
                    self.config.credentials_path
                )
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.config.credentials_path
            else:
                credentials = None  # Usar credenciales por defecto
            
            # Cliente de Storage
            self.storage_client = storage.Client(
                project=self.config.project_id,
                credentials=credentials
            )
            
            # Inicializar Vertex AI
            if self.config.project_id and self.config.region:
                aiplatform.init(
                    project=self.config.project_id,
                    location=self.config.region,
                    credentials=credentials
                )
            
            self._client = self.storage_client
            
            logger.info("GCP provider initialized")
            
        except ImportError:
            raise ImportError(
                "Google Cloud libraries not installed. Install with: "
                "pip install google-cloud-storage google-cloud-aiplatform"
            )
    
    def authenticate(self) -> bool:
        """Verifica autenticación con GCP."""
        try:
            # Intentar listar buckets para verificar credenciales
            list(self.storage_client.list_buckets(max_results=1))
            return True
        except Exception as e:
            logger.error(f"GCP authentication failed: {e}")
            return False
    
    def list_resources(self, resource_type: str) -> List[Dict[str, Any]]:
        """Lista recursos de GCP."""
        resources = []
        
        try:
            if resource_type == 'storage_buckets':
                buckets = self.storage_client.list_buckets()
                resources = [{'name': b.name, 'created': b.time_created} for b in buckets]
            
            elif resource_type == 'vertex_endpoints':
                endpoints = self.aiplatform.Endpoint.list()
                resources = [{'name': e.display_name, 'resource_name': e.resource_name} 
                           for e in endpoints]
            
            elif resource_type == 'vertex_models':
                models = self.aiplatform.Model.list()
                resources = [{'name': m.display_name, 'resource_name': m.resource_name}
                           for m in models]
            
            elif resource_type == 'vertex_jobs':
                jobs = self.aiplatform.CustomJob.list()
                resources = [{'name': j.display_name, 'state': j.state} for j in jobs]
            
        except Exception as e:
            logger.error(f"Error listing {resource_type}: {e}")
        
        return resources
    
    def get_resource(self, resource_id: str, resource_type: str) -> Optional[Dict[str, Any]]:
        """Obtiene información de un recurso GCP."""
        try:
            if resource_type == 'storage_bucket':
                bucket = self.storage_client.bucket(resource_id)
                return {
                    'name': bucket.name,
                    'location': bucket.location,
                    'storage_class': bucket.storage_class,
                    'created': bucket.time_created
                }
            
            elif resource_type == 'vertex_endpoint':
                endpoint = self.aiplatform.Endpoint(resource_id)
                return {
                    'name': endpoint.display_name,
                    'resource_name': endpoint.resource_name,
                    'deployed_models': endpoint.deployed_models
                }
            
            elif resource_type == 'vertex_model':
                model = self.aiplatform.Model(resource_id)
                return {
                    'name': model.display_name,
                    'resource_name': model.resource_name,
                    'artifact_uri': model.artifact_uri
                }
            
        except Exception as e:
            logger.error(f"Error getting {resource_type} {resource_id}: {e}")
        
        return None


class GCSStorage(CloudStorage):
    """Almacenamiento en Google Cloud Storage."""
    
    def __init__(self, provider: GCPProvider, bucket_name: str, create_if_not_exists: bool = False):
        """Inicializa GCS storage.
        
        Parameters
        ----------
        provider : GCPProvider
            Proveedor GCP.
        bucket_name : str
            Nombre del bucket.
        create_if_not_exists : bool
            Si crear el bucket si no existe.
        """
        super().__init__(provider, bucket_name)
        self.bucket = provider.storage_client.bucket(bucket_name)
        
        if create_if_not_exists:
            self._create_bucket_if_not_exists()
    
    def _create_bucket_if_not_exists(self):
        """Crea el bucket si no existe."""
        if not self.bucket.exists():
            try:
                self.bucket.location = self.provider.config.region or 'US'
                self.bucket.create()
                logger.info(f"Created GCS bucket: {self.bucket_name}")
            except Exception as e:
                logger.error(f"Could not create bucket: {e}")
    
    def upload_file(
        self,
        local_path: Union[str, Path],
        remote_path: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> bool:
        """Sube un archivo a GCS."""
        try:
            local_path = Path(local_path)
            blob = self.bucket.blob(remote_path)
            
            if metadata:
                blob.metadata = metadata
            
            blob.upload_from_filename(str(local_path))
            
            logger.debug(f"Uploaded {local_path} to gs://{self.bucket_name}/{remote_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            return False
    
    def download_file(
        self,
        remote_path: str,
        local_path: Union[str, Path]
    ) -> bool:
        """Descarga un archivo de GCS."""
        try:
            local_path = Path(local_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            blob = self.bucket.blob(remote_path)
            blob.download_to_filename(str(local_path))
            
            logger.debug(f"Downloaded gs://{self.bucket_name}/{remote_path} to {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            return False
    
    def list_files(
        self,
        prefix: Optional[str] = None,
        max_results: int = 1000
    ) -> List[str]:
        """Lista archivos en GCS."""
        files = []
        
        try:
            blobs = self.bucket.list_blobs(prefix=prefix, max_results=max_results)
            files = [blob.name for blob in blobs]
        except Exception as e:
            logger.error(f"Error listing files: {e}")
        
        return files
    
    def delete_file(self, remote_path: str) -> bool:
        """Elimina un archivo de GCS."""
        try:
            blob = self.bucket.blob(remote_path)
            blob.delete()
            logger.debug(f"Deleted gs://{self.bucket_name}/{remote_path}")
            return True
        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            return False
    
    def file_exists(self, remote_path: str) -> bool:
        """Verifica si un archivo existe en GCS."""
        blob = self.bucket.blob(remote_path)
        return blob.exists()
    
    def get_signed_url(
        self,
        remote_path: str,
        expiration: int = 3600,
        method: str = 'GET'
    ) -> Optional[str]:
        """Genera URL firmada para acceso temporal.
        
        Parameters
        ----------
        remote_path : str
            Ruta del archivo en GCS.
        expiration : int
            Tiempo de expiración en segundos.
        method : str
            Método HTTP ('GET' o 'PUT').
            
        Returns
        -------
        Optional[str]
            URL firmada o None.
        """
        try:
            from datetime import timedelta
            
            blob = self.bucket.blob(remote_path)
            url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(seconds=expiration),
                method=method
            )
            return url
        except Exception as e:
            logger.error(f"Error generating signed URL: {e}")
            return None


class VertexAICompute(CloudCompute):
    """Cómputo con Google Vertex AI."""
    
    def __init__(self, provider: GCPProvider, staging_bucket: str):
        """Inicializa Vertex AI compute.
        
        Parameters
        ----------
        provider : GCPProvider
            Proveedor GCP.
        staging_bucket : str
            Bucket para staging de datos.
        """
        super().__init__(provider)
        self.staging_bucket = staging_bucket
        self.aiplatform = provider.aiplatform
    
    def submit_training_job(
        self,
        job_name: str,
        script_path: str,
        instance_type: str = 'n1-standard-4',
        accelerator_type: Optional[str] = None,
        accelerator_count: int = 0,
        hyperparameters: Optional[Dict[str, Any]] = None,
        input_data: Optional[Dict[str, str]] = None,
        output_path: Optional[str] = None,
        container_uri: Optional[str] = None,
        **kwargs
    ) -> str:
        """Envía trabajo de entrenamiento a Vertex AI."""
        try:
            # Configurar contenedor
            if not container_uri:
                # Usar contenedor pre-construido
                container_uri = "gcr.io/cloud-aiplatform/training/scikit-learn-cpu.0-23:latest"
            
            # Configurar especificaciones de máquina
            machine_spec = {
                "machine_type": instance_type,
            }
            
            if accelerator_type and accelerator_count > 0:
                machine_spec["accelerator_type"] = accelerator_type
                machine_spec["accelerator_count"] = accelerator_count
            
            # Crear trabajo personalizado
            job = self.aiplatform.CustomJob(
                display_name=job_name,
                worker_pool_specs=[{
                    "machine_spec": machine_spec,
                    "replica_count": 1,
                    "container_spec": {
                        "image_uri": container_uri,
                        "command": ["python"],
                        "args": [script_path] + self._format_args(hyperparameters),
                    },
                }],
                staging_bucket=f"gs://{self.staging_bucket}",
            )
            
            # Ejecutar trabajo
            job.run(sync=False)
            
            logger.info(f"Submitted Vertex AI training job: {job_name}")
            return job.resource_name
            
        except Exception as e:
            logger.error(f"Error submitting training job: {e}")
            raise
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Obtiene estado del trabajo de Vertex AI."""
        try:
            job = self.aiplatform.CustomJob(job_id)
            
            return {
                'status': job.state.name,
                'display_name': job.display_name,
                'create_time': job.create_time,
                'update_time': job.update_time,
                'error': job.error if hasattr(job, 'error') else None
            }
            
        except Exception as e:
            logger.error(f"Error getting job status: {e}")
            return {'status': 'UNKNOWN', 'error': str(e)}
    
    def stop_job(self, job_id: str) -> bool:
        """Detiene trabajo de Vertex AI."""
        try:
            job = self.aiplatform.CustomJob(job_id)
            job.cancel()
            logger.info(f"Cancelled training job: {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping job: {e}")
            return False
    
    def deploy_model(
        self,
        model_path: str,
        endpoint_name: str,
        instance_type: str = 'n1-standard-4',
        min_replica_count: int = 1,
        max_replica_count: int = 1,
        serving_container_uri: Optional[str] = None,
        **kwargs
    ) -> str:
        """Despliega modelo en Vertex AI."""
        try:
            # Cargar o crear modelo
            model = self.aiplatform.Model.upload(
                display_name=f"{endpoint_name}-model",
                artifact_uri=model_path,
                serving_container_image_uri=serving_container_uri or 
                    "gcr.io/cloud-aiplatform/prediction/sklearn-cpu.0-23:latest"
            )
            
            # Crear o obtener endpoint
            endpoints = self.aiplatform.Endpoint.list(
                filter=f'display_name="{endpoint_name}"'
            )
            
            if endpoints:
                endpoint = endpoints[0]
            else:
                endpoint = self.aiplatform.Endpoint.create(
                    display_name=endpoint_name
                )
            
            # Desplegar modelo
            endpoint.deploy(
                model=model,
                deployed_model_display_name=model.display_name,
                machine_type=instance_type,
                min_replica_count=min_replica_count,
                max_replica_count=max_replica_count
            )
            
            logger.info(f"Deployed model to endpoint: {endpoint_name}")
            return endpoint.resource_name
            
        except Exception as e:
            logger.error(f"Error deploying model: {e}")
            raise
    
    def predict(
        self,
        endpoint_name: str,
        data: Any,
        **kwargs
    ) -> Any:
        """Realiza predicciones usando endpoint de Vertex AI."""
        try:
            # Obtener endpoint
            endpoints = self.aiplatform.Endpoint.list(
                filter=f'display_name="{endpoint_name}"'
            )
            
            if not endpoints:
                raise ValueError(f"Endpoint {endpoint_name} not found")
            
            endpoint = endpoints[0]
            
            # Hacer predicción
            predictions = endpoint.predict(instances=data)
            
            return predictions.predictions
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    def delete_endpoint(self, endpoint_name: str) -> bool:
        """Elimina endpoint de Vertex AI."""
        try:
            # Obtener endpoint
            endpoints = self.aiplatform.Endpoint.list(
                filter=f'display_name="{endpoint_name}"'
            )
            
            if endpoints:
                endpoint = endpoints[0]
                
                # Deshacer despliegue de modelos
                endpoint.undeploy_all()
                
                # Eliminar endpoint
                endpoint.delete()
                
                logger.info(f"Deleted endpoint: {endpoint_name}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting endpoint: {e}")
            return False
    
    def _format_args(self, hyperparameters: Optional[Dict[str, Any]]) -> List[str]:
        """Formatea hiperparámetros como argumentos de línea de comandos."""
        if not hyperparameters:
            return []
        
        args = []
        for key, value in hyperparameters.items():
            args.extend([f"--{key}", str(value)])
        
        return args