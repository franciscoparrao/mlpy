"""
Clases base para integración con cloud providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class CloudConfig:
    """Configuración base para cloud providers.
    
    Attributes
    ----------
    region : Optional[str]
        Región del servicio cloud.
    credentials : Optional[Dict[str, str]]
        Credenciales de acceso.
    project_id : Optional[str]
        ID del proyecto/suscripción.
    timeout : int
        Timeout para operaciones en segundos.
    retry_count : int
        Número de reintentos.
    """
    region: Optional[str] = None
    credentials: Optional[Dict[str, str]] = None
    project_id: Optional[str] = None
    timeout: int = 300
    retry_count: int = 3
    verify_ssl: bool = True


class CloudProvider(ABC):
    """Clase base para proveedores de cloud."""
    
    def __init__(self, config: CloudConfig):
        """Inicializa el proveedor.
        
        Parameters
        ----------
        config : CloudConfig
            Configuración del proveedor.
        """
        self.config = config
        self._client = None
        self._setup()
    
    @abstractmethod
    def _setup(self):
        """Configura el cliente del proveedor."""
        pass
    
    @abstractmethod
    def authenticate(self) -> bool:
        """Autentica con el proveedor.
        
        Returns
        -------
        bool
            True si la autenticación fue exitosa.
        """
        pass
    
    @abstractmethod
    def list_resources(self, resource_type: str) -> List[Dict[str, Any]]:
        """Lista recursos del proveedor.
        
        Parameters
        ----------
        resource_type : str
            Tipo de recurso a listar.
            
        Returns
        -------
        List[Dict[str, Any]]
            Lista de recursos.
        """
        pass
    
    @abstractmethod
    def get_resource(self, resource_id: str, resource_type: str) -> Optional[Dict[str, Any]]:
        """Obtiene información de un recurso.
        
        Parameters
        ----------
        resource_id : str
            ID del recurso.
        resource_type : str
            Tipo de recurso.
            
        Returns
        -------
        Optional[Dict[str, Any]]
            Información del recurso o None.
        """
        pass
    
    @property
    def is_authenticated(self) -> bool:
        """Verifica si está autenticado."""
        return self._client is not None


class CloudStorage(ABC):
    """Clase base para almacenamiento en cloud."""
    
    def __init__(self, provider: CloudProvider, bucket_name: str):
        """Inicializa el almacenamiento.
        
        Parameters
        ----------
        provider : CloudProvider
            Proveedor de cloud.
        bucket_name : str
            Nombre del bucket/container.
        """
        self.provider = provider
        self.bucket_name = bucket_name
    
    @abstractmethod
    def upload_file(
        self,
        local_path: Union[str, Path],
        remote_path: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> bool:
        """Sube un archivo al storage.
        
        Parameters
        ----------
        local_path : Union[str, Path]
            Ruta local del archivo.
        remote_path : str
            Ruta remota en el storage.
        metadata : Optional[Dict[str, str]]
            Metadata del archivo.
            
        Returns
        -------
        bool
            True si se subió exitosamente.
        """
        pass
    
    @abstractmethod
    def download_file(
        self,
        remote_path: str,
        local_path: Union[str, Path]
    ) -> bool:
        """Descarga un archivo del storage.
        
        Parameters
        ----------
        remote_path : str
            Ruta remota en el storage.
        local_path : Union[str, Path]
            Ruta local donde guardar.
            
        Returns
        -------
        bool
            True si se descargó exitosamente.
        """
        pass
    
    @abstractmethod
    def list_files(
        self,
        prefix: Optional[str] = None,
        max_results: int = 1000
    ) -> List[str]:
        """Lista archivos en el storage.
        
        Parameters
        ----------
        prefix : Optional[str]
            Prefijo para filtrar archivos.
        max_results : int
            Número máximo de resultados.
            
        Returns
        -------
        List[str]
            Lista de rutas de archivos.
        """
        pass
    
    @abstractmethod
    def delete_file(self, remote_path: str) -> bool:
        """Elimina un archivo del storage.
        
        Parameters
        ----------
        remote_path : str
            Ruta remota del archivo.
            
        Returns
        -------
        bool
            True si se eliminó exitosamente.
        """
        pass
    
    @abstractmethod
    def file_exists(self, remote_path: str) -> bool:
        """Verifica si un archivo existe.
        
        Parameters
        ----------
        remote_path : str
            Ruta remota del archivo.
            
        Returns
        -------
        bool
            True si el archivo existe.
        """
        pass
    
    def upload_directory(
        self,
        local_dir: Union[str, Path],
        remote_prefix: str,
        recursive: bool = True
    ) -> int:
        """Sube un directorio completo.
        
        Parameters
        ----------
        local_dir : Union[str, Path]
            Directorio local.
        remote_prefix : str
            Prefijo remoto.
        recursive : bool
            Si subir recursivamente.
            
        Returns
        -------
        int
            Número de archivos subidos.
        """
        local_dir = Path(local_dir)
        uploaded = 0
        
        pattern = "**/*" if recursive else "*"
        for file_path in local_dir.glob(pattern):
            if file_path.is_file():
                relative_path = file_path.relative_to(local_dir)
                remote_path = f"{remote_prefix}/{relative_path}".replace("\\", "/")
                
                if self.upload_file(file_path, remote_path):
                    uploaded += 1
                    logger.debug(f"Uploaded {file_path} to {remote_path}")
        
        return uploaded
    
    def download_directory(
        self,
        remote_prefix: str,
        local_dir: Union[str, Path]
    ) -> int:
        """Descarga un directorio completo.
        
        Parameters
        ----------
        remote_prefix : str
            Prefijo remoto.
        local_dir : Union[str, Path]
            Directorio local.
            
        Returns
        -------
        int
            Número de archivos descargados.
        """
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        
        files = self.list_files(prefix=remote_prefix)
        downloaded = 0
        
        for remote_path in files:
            # Obtener ruta relativa
            relative_path = remote_path.replace(remote_prefix, "").lstrip("/")
            local_path = local_dir / relative_path
            
            # Crear directorio si es necesario
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            if self.download_file(remote_path, local_path):
                downloaded += 1
                logger.debug(f"Downloaded {remote_path} to {local_path}")
        
        return downloaded


class CloudCompute(ABC):
    """Clase base para cómputo en cloud."""
    
    def __init__(self, provider: CloudProvider):
        """Inicializa el servicio de cómputo.
        
        Parameters
        ----------
        provider : CloudProvider
            Proveedor de cloud.
        """
        self.provider = provider
    
    @abstractmethod
    def submit_training_job(
        self,
        job_name: str,
        script_path: str,
        instance_type: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
        input_data: Optional[Dict[str, str]] = None,
        output_path: Optional[str] = None,
        **kwargs
    ) -> str:
        """Envía un trabajo de entrenamiento.
        
        Parameters
        ----------
        job_name : str
            Nombre del trabajo.
        script_path : str
            Ruta del script de entrenamiento.
        instance_type : str
            Tipo de instancia.
        hyperparameters : Optional[Dict[str, Any]]
            Hiperparámetros del modelo.
        input_data : Optional[Dict[str, str]]
            Rutas de datos de entrada.
        output_path : Optional[str]
            Ruta de salida.
        **kwargs
            Argumentos adicionales.
            
        Returns
        -------
        str
            ID del trabajo.
        """
        pass
    
    @abstractmethod
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Obtiene el estado de un trabajo.
        
        Parameters
        ----------
        job_id : str
            ID del trabajo.
            
        Returns
        -------
        Dict[str, Any]
            Estado del trabajo.
        """
        pass
    
    @abstractmethod
    def stop_job(self, job_id: str) -> bool:
        """Detiene un trabajo.
        
        Parameters
        ----------
        job_id : str
            ID del trabajo.
            
        Returns
        -------
        bool
            True si se detuvo exitosamente.
        """
        pass
    
    @abstractmethod
    def deploy_model(
        self,
        model_path: str,
        endpoint_name: str,
        instance_type: str,
        **kwargs
    ) -> str:
        """Despliega un modelo.
        
        Parameters
        ----------
        model_path : str
            Ruta del modelo.
        endpoint_name : str
            Nombre del endpoint.
        instance_type : str
            Tipo de instancia.
        **kwargs
            Argumentos adicionales.
            
        Returns
        -------
        str
            URL del endpoint.
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        endpoint_name: str,
        data: Any,
        **kwargs
    ) -> Any:
        """Realiza predicciones usando un endpoint.
        
        Parameters
        ----------
        endpoint_name : str
            Nombre del endpoint.
        data : Any
            Datos para predicción.
        **kwargs
            Argumentos adicionales.
            
        Returns
        -------
        Any
            Predicciones.
        """
        pass
    
    @abstractmethod
    def delete_endpoint(self, endpoint_name: str) -> bool:
        """Elimina un endpoint.
        
        Parameters
        ----------
        endpoint_name : str
            Nombre del endpoint.
            
        Returns
        -------
        bool
            True si se eliminó exitosamente.
        """
        pass
    
    def list_jobs(
        self,
        status_filter: Optional[str] = None,
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """Lista trabajos de entrenamiento.
        
        Parameters
        ----------
        status_filter : Optional[str]
            Filtro por estado.
        max_results : int
            Número máximo de resultados.
            
        Returns
        -------
        List[Dict[str, Any]]
            Lista de trabajos.
        """
        return []
    
    def list_endpoints(self) -> List[Dict[str, Any]]:
        """Lista endpoints desplegados.
        
        Returns
        -------
        List[Dict[str, Any]]
            Lista de endpoints.
        """
        return []