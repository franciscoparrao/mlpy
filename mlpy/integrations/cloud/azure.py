"""
Integración con Microsoft Azure para MLPY.

Soporte para Azure Blob Storage, Azure Machine Learning y otros servicios Azure.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
import tempfile

from .base import CloudProvider, CloudStorage, CloudCompute, CloudConfig

logger = logging.getLogger(__name__)


@dataclass
class AzureConfig(CloudConfig):
    """Configuración específica para Azure.
    
    Attributes
    ----------
    subscription_id : Optional[str]
        ID de suscripción de Azure.
    resource_group : Optional[str]
        Grupo de recursos.
    workspace_name : Optional[str]
        Nombre del workspace de Azure ML.
    connection_string : Optional[str]
        String de conexión para storage.
    tenant_id : Optional[str]
        ID del tenant de Azure AD.
    client_id : Optional[str]
        ID del cliente (service principal).
    client_secret : Optional[str]
        Secreto del cliente.
    """
    subscription_id: Optional[str] = None
    resource_group: Optional[str] = None
    workspace_name: Optional[str] = None
    connection_string: Optional[str] = None
    tenant_id: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None


class AzureProvider(CloudProvider):
    """Proveedor para Microsoft Azure."""
    
    def _setup(self):
        """Configura los clientes de Azure."""
        try:
            from azure.storage.blob import BlobServiceClient
            from azure.identity import DefaultAzureCredential, ClientSecretCredential
            from azureml.core import Workspace
            
            self.BlobServiceClient = BlobServiceClient
            self.Workspace = Workspace
            
            # Configurar credenciales
            if (hasattr(self.config, 'tenant_id') and self.config.tenant_id and
                hasattr(self.config, 'client_id') and self.config.client_id and
                hasattr(self.config, 'client_secret') and self.config.client_secret):
                
                self.credential = ClientSecretCredential(
                    tenant_id=self.config.tenant_id,
                    client_id=self.config.client_id,
                    client_secret=self.config.client_secret
                )
            else:
                self.credential = DefaultAzureCredential()
            
            # Cliente de Blob Storage
            if hasattr(self.config, 'connection_string') and self.config.connection_string:
                self.blob_service_client = BlobServiceClient.from_connection_string(
                    self.config.connection_string
                )
            else:
                # Usar cuenta de storage por defecto
                account_url = f"https://{self.config.project_id or 'storage'}.blob.core.windows.net"
                self.blob_service_client = BlobServiceClient(
                    account_url=account_url,
                    credential=self.credential
                )
            
            # Workspace de Azure ML
            self.ml_workspace = None
            if (hasattr(self.config, 'subscription_id') and self.config.subscription_id and
                hasattr(self.config, 'resource_group') and self.config.resource_group and
                hasattr(self.config, 'workspace_name') and self.config.workspace_name):
                
                try:
                    self.ml_workspace = Workspace(
                        subscription_id=self.config.subscription_id,
                        resource_group=self.config.resource_group,
                        workspace_name=self.config.workspace_name,
                        auth=self.credential
                    )
                except:
                    logger.warning("Could not initialize Azure ML workspace")
            
            self._client = self.blob_service_client
            
            logger.info("Azure provider initialized")
            
        except ImportError:
            raise ImportError(
                "Azure libraries not installed. Install with: "
                "pip install azure-storage-blob azure-identity azureml-core"
            )
    
    def authenticate(self) -> bool:
        """Verifica autenticación con Azure."""
        try:
            # Intentar listar containers para verificar credenciales
            containers = self.blob_service_client.list_containers()
            next(containers, None)  # Intentar obtener al menos uno
            return True
        except Exception as e:
            logger.error(f"Azure authentication failed: {e}")
            return False
    
    def list_resources(self, resource_type: str) -> List[Dict[str, Any]]:
        """Lista recursos de Azure."""
        resources = []
        
        try:
            if resource_type == 'storage_containers':
                containers = self.blob_service_client.list_containers()
                resources = [{'name': c.name, 'last_modified': c.last_modified} 
                           for c in containers]
            
            elif resource_type == 'ml_models' and self.ml_workspace:
                from azureml.core import Model
                models = Model.list(self.ml_workspace)
                resources = [{'name': m.name, 'version': m.version, 'id': m.id} 
                           for m in models]
            
            elif resource_type == 'ml_endpoints' and self.ml_workspace:
                from azureml.core.webservice import Webservice
                services = Webservice.list(self.ml_workspace)
                resources = [{'name': s.name, 'state': s.state, 'url': s.scoring_uri} 
                           for s in services]
            
            elif resource_type == 'ml_experiments' and self.ml_workspace:
                from azureml.core import Experiment
                experiments = Experiment.list(self.ml_workspace)
                resources = [{'name': e.name, 'id': e.id} for e in experiments]
            
        except Exception as e:
            logger.error(f"Error listing {resource_type}: {e}")
        
        return resources
    
    def get_resource(self, resource_id: str, resource_type: str) -> Optional[Dict[str, Any]]:
        """Obtiene información de un recurso Azure."""
        try:
            if resource_type == 'storage_container':
                container = self.blob_service_client.get_container_client(resource_id)
                properties = container.get_container_properties()
                return {
                    'name': properties['name'],
                    'last_modified': properties['last_modified'],
                    'etag': properties['etag']
                }
            
            elif resource_type == 'ml_model' and self.ml_workspace:
                from azureml.core import Model
                model = Model(self.ml_workspace, name=resource_id)
                return {
                    'name': model.name,
                    'version': model.version,
                    'id': model.id,
                    'created_time': model.created_time
                }
            
            elif resource_type == 'ml_endpoint' and self.ml_workspace:
                from azureml.core.webservice import Webservice
                service = Webservice(self.ml_workspace, name=resource_id)
                return {
                    'name': service.name,
                    'state': service.state,
                    'scoring_uri': service.scoring_uri,
                    'swagger_uri': service.swagger_uri
                }
            
        except Exception as e:
            logger.error(f"Error getting {resource_type} {resource_id}: {e}")
        
        return None


class BlobStorage(CloudStorage):
    """Almacenamiento en Azure Blob Storage."""
    
    def __init__(self, provider: AzureProvider, container_name: str, create_if_not_exists: bool = False):
        """Inicializa Blob storage.
        
        Parameters
        ----------
        provider : AzureProvider
            Proveedor Azure.
        container_name : str
            Nombre del container.
        create_if_not_exists : bool
            Si crear el container si no existe.
        """
        super().__init__(provider, container_name)
        self.container_client = provider.blob_service_client.get_container_client(container_name)
        
        if create_if_not_exists:
            self._create_container_if_not_exists()
    
    def _create_container_if_not_exists(self):
        """Crea el container si no existe."""
        try:
            self.container_client.get_container_properties()
        except:
            try:
                self.container_client.create_container()
                logger.info(f"Created Azure container: {self.bucket_name}")
            except Exception as e:
                logger.error(f"Could not create container: {e}")
    
    def upload_file(
        self,
        local_path: Union[str, Path],
        remote_path: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> bool:
        """Sube un archivo a Blob Storage."""
        try:
            local_path = Path(local_path)
            blob_client = self.container_client.get_blob_client(remote_path)
            
            with open(local_path, 'rb') as data:
                blob_client.upload_blob(data, metadata=metadata, overwrite=True)
            
            logger.debug(f"Uploaded {local_path} to {self.bucket_name}/{remote_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            return False
    
    def download_file(
        self,
        remote_path: str,
        local_path: Union[str, Path]
    ) -> bool:
        """Descarga un archivo de Blob Storage."""
        try:
            local_path = Path(local_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            blob_client = self.container_client.get_blob_client(remote_path)
            
            with open(local_path, 'wb') as file:
                download_stream = blob_client.download_blob()
                file.write(download_stream.readall())
            
            logger.debug(f"Downloaded {self.bucket_name}/{remote_path} to {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            return False
    
    def list_files(
        self,
        prefix: Optional[str] = None,
        max_results: int = 1000
    ) -> List[str]:
        """Lista archivos en Blob Storage."""
        files = []
        
        try:
            blobs = self.container_client.list_blobs(name_starts_with=prefix)
            
            for i, blob in enumerate(blobs):
                if i >= max_results:
                    break
                files.append(blob.name)
                
        except Exception as e:
            logger.error(f"Error listing files: {e}")
        
        return files
    
    def delete_file(self, remote_path: str) -> bool:
        """Elimina un archivo de Blob Storage."""
        try:
            blob_client = self.container_client.get_blob_client(remote_path)
            blob_client.delete_blob()
            logger.debug(f"Deleted {self.bucket_name}/{remote_path}")
            return True
        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            return False
    
    def file_exists(self, remote_path: str) -> bool:
        """Verifica si un archivo existe en Blob Storage."""
        try:
            blob_client = self.container_client.get_blob_client(remote_path)
            blob_client.get_blob_properties()
            return True
        except:
            return False
    
    def get_sas_url(
        self,
        remote_path: str,
        expiration: int = 3600,
        permission: str = 'r'
    ) -> Optional[str]:
        """Genera URL SAS para acceso temporal.
        
        Parameters
        ----------
        remote_path : str
            Ruta del archivo.
        expiration : int
            Tiempo de expiración en segundos.
        permission : str
            Permisos ('r' para lectura, 'w' para escritura).
            
        Returns
        -------
        Optional[str]
            URL SAS o None.
        """
        try:
            from datetime import datetime, timedelta
            from azure.storage.blob import generate_blob_sas, BlobSasPermissions
            
            blob_client = self.container_client.get_blob_client(remote_path)
            
            sas_token = generate_blob_sas(
                account_name=blob_client.account_name,
                container_name=self.bucket_name,
                blob_name=remote_path,
                permission=BlobSasPermissions(read='r' in permission, write='w' in permission),
                expiry=datetime.utcnow() + timedelta(seconds=expiration)
            )
            
            return f"{blob_client.url}?{sas_token}"
            
        except Exception as e:
            logger.error(f"Error generating SAS URL: {e}")
            return None


class AzureMLCompute(CloudCompute):
    """Cómputo con Azure Machine Learning."""
    
    def __init__(self, provider: AzureProvider):
        """Inicializa Azure ML compute.
        
        Parameters
        ----------
        provider : AzureProvider
            Proveedor Azure.
        """
        super().__init__(provider)
        self.workspace = provider.ml_workspace
        
        if not self.workspace:
            raise ValueError("Azure ML workspace not configured")
    
    def submit_training_job(
        self,
        job_name: str,
        script_path: str,
        compute_target: str = 'cpu-cluster',
        environment_name: str = 'AzureML-sklearn-0.24-ubuntu18.04-py37-cpu',
        hyperparameters: Optional[Dict[str, Any]] = None,
        input_data: Optional[Dict[str, str]] = None,
        output_path: Optional[str] = None,
        **kwargs
    ) -> str:
        """Envía trabajo de entrenamiento a Azure ML."""
        try:
            from azureml.core import Experiment, ScriptRunConfig, Environment
            from azureml.core.compute import ComputeTarget
            
            # Obtener o crear experimento
            experiment = Experiment(self.workspace, job_name)
            
            # Configurar entorno
            if environment_name.startswith('AzureML-'):
                # Usar entorno curado
                env = Environment.get(self.workspace, environment_name)
            else:
                # Crear entorno personalizado
                env = Environment(environment_name)
                env.python.conda_dependencies.add_pip_package('scikit-learn')
                env.python.conda_dependencies.add_pip_package('pandas')
                env.python.conda_dependencies.add_pip_package('numpy')
            
            # Configurar compute target
            compute = ComputeTarget(workspace=self.workspace, name=compute_target)
            
            # Configurar script
            config = ScriptRunConfig(
                source_directory=str(Path(script_path).parent),
                script=Path(script_path).name,
                compute_target=compute,
                environment=env,
                arguments=self._format_arguments(hyperparameters)
            )
            
            # Enviar experimento
            run = experiment.submit(config)
            
            logger.info(f"Submitted Azure ML training job: {job_name}")
            return run.id
            
        except Exception as e:
            logger.error(f"Error submitting training job: {e}")
            raise
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Obtiene estado del trabajo de Azure ML."""
        try:
            from azureml.core import Run
            
            run = Run.get(self.workspace, job_id)
            
            return {
                'status': run.status,
                'start_time': run.start_time,
                'end_time': run.end_time,
                'metrics': run.get_metrics(),
                'properties': run.properties
            }
            
        except Exception as e:
            logger.error(f"Error getting job status: {e}")
            return {'status': 'UNKNOWN', 'error': str(e)}
    
    def stop_job(self, job_id: str) -> bool:
        """Detiene trabajo de Azure ML."""
        try:
            from azureml.core import Run
            
            run = Run.get(self.workspace, job_id)
            run.cancel()
            
            logger.info(f"Cancelled training job: {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping job: {e}")
            return False
    
    def deploy_model(
        self,
        model_path: str,
        endpoint_name: str,
        compute_type: str = 'ACI',  # ACI o AKS
        cpu_cores: int = 1,
        memory_gb: int = 1,
        **kwargs
    ) -> str:
        """Despliega modelo en Azure ML."""
        try:
            from azureml.core import Model
            from azureml.core.webservice import AciWebservice, AksWebservice, Webservice
            from azureml.core.model import InferenceConfig
            from azureml.core.environment import Environment
            
            # Registrar modelo
            model = Model.register(
                workspace=self.workspace,
                model_path=model_path,
                model_name=f"{endpoint_name}-model"
            )
            
            # Configurar inferencia
            env = Environment.get(self.workspace, 'AzureML-sklearn-0.24-ubuntu18.04-py37-cpu')
            
            inference_config = InferenceConfig(
                environment=env,
                entry_script=kwargs.get('entry_script', 'score.py')
            )
            
            # Configurar despliegue
            if compute_type == 'ACI':
                deployment_config = AciWebservice.deploy_configuration(
                    cpu_cores=cpu_cores,
                    memory_gb=memory_gb
                )
            else:  # AKS
                deployment_config = AksWebservice.deploy_configuration(
                    cpu_cores=cpu_cores,
                    memory_gb=memory_gb
                )
            
            # Desplegar
            service = Model.deploy(
                workspace=self.workspace,
                name=endpoint_name,
                models=[model],
                inference_config=inference_config,
                deployment_config=deployment_config
            )
            
            service.wait_for_deployment(show_output=True)
            
            logger.info(f"Deployed model to endpoint: {endpoint_name}")
            return service.scoring_uri
            
        except Exception as e:
            logger.error(f"Error deploying model: {e}")
            raise
    
    def predict(
        self,
        endpoint_name: str,
        data: Any,
        **kwargs
    ) -> Any:
        """Realiza predicciones usando endpoint de Azure ML."""
        try:
            from azureml.core.webservice import Webservice
            import json
            
            service = Webservice(self.workspace, endpoint_name)
            
            # Serializar datos
            input_data = json.dumps({"data": data})
            
            # Hacer predicción
            predictions = service.run(input_data)
            
            return json.loads(predictions)
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    def delete_endpoint(self, endpoint_name: str) -> bool:
        """Elimina endpoint de Azure ML."""
        try:
            from azureml.core.webservice import Webservice
            
            service = Webservice(self.workspace, endpoint_name)
            service.delete()
            
            logger.info(f"Deleted endpoint: {endpoint_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting endpoint: {e}")
            return False
    
    def _format_arguments(self, hyperparameters: Optional[Dict[str, Any]]) -> List[str]:
        """Formatea hiperparámetros como argumentos."""
        if not hyperparameters:
            return []
        
        args = []
        for key, value in hyperparameters.items():
            args.extend([f"--{key}", str(value)])
        
        return args