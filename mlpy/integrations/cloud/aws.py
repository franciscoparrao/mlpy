"""
Integración con Amazon Web Services (AWS) para MLPY.

Soporte para S3, SageMaker y otros servicios AWS.
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
class AWSConfig(CloudConfig):
    """Configuración específica para AWS.
    
    Attributes
    ----------
    aws_access_key_id : Optional[str]
        Access key ID de AWS.
    aws_secret_access_key : Optional[str]
        Secret access key de AWS.
    aws_session_token : Optional[str]
        Session token (para credenciales temporales).
    profile_name : Optional[str]
        Nombre del perfil AWS.
    endpoint_url : Optional[str]
        URL del endpoint (para S3 compatible).
    """
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_session_token: Optional[str] = None
    profile_name: Optional[str] = None
    endpoint_url: Optional[str] = None


class AWSProvider(CloudProvider):
    """Proveedor para Amazon Web Services."""
    
    def _setup(self):
        """Configura los clientes de AWS."""
        try:
            import boto3
            from botocore.config import Config
            
            self.boto3 = boto3
            
            # Configuración de boto3
            boto_config = Config(
                region_name=self.config.region,
                retries={'max_attempts': self.config.retry_count}
            )
            
            # Credenciales
            session_kwargs = {}
            if hasattr(self.config, 'profile_name') and self.config.profile_name:
                session_kwargs['profile_name'] = self.config.profile_name
            elif hasattr(self.config, 'aws_access_key_id'):
                session_kwargs.update({
                    'aws_access_key_id': self.config.aws_access_key_id,
                    'aws_secret_access_key': self.config.aws_secret_access_key,
                    'aws_session_token': self.config.aws_session_token
                })
            
            # Crear sesión
            self.session = boto3.Session(**session_kwargs)
            
            # Clientes
            self.s3_client = self.session.client('s3', config=boto_config)
            self.sagemaker_client = self.session.client('sagemaker', config=boto_config)
            
            self._client = self.s3_client  # Cliente principal
            
            logger.info("AWS provider initialized")
            
        except ImportError:
            raise ImportError("boto3 not installed. Install with: pip install boto3")
    
    def authenticate(self) -> bool:
        """Verifica autenticación con AWS."""
        try:
            # Intentar listar buckets para verificar credenciales
            self.s3_client.list_buckets()
            return True
        except Exception as e:
            logger.error(f"AWS authentication failed: {e}")
            return False
    
    def list_resources(self, resource_type: str) -> List[Dict[str, Any]]:
        """Lista recursos de AWS."""
        resources = []
        
        try:
            if resource_type == 's3_buckets':
                response = self.s3_client.list_buckets()
                resources = response.get('Buckets', [])
            
            elif resource_type == 'sagemaker_endpoints':
                response = self.sagemaker_client.list_endpoints()
                resources = response.get('Endpoints', [])
            
            elif resource_type == 'sagemaker_training_jobs':
                response = self.sagemaker_client.list_training_jobs()
                resources = response.get('TrainingJobSummaries', [])
            
            elif resource_type == 'sagemaker_models':
                response = self.sagemaker_client.list_models()
                resources = response.get('Models', [])
            
        except Exception as e:
            logger.error(f"Error listing {resource_type}: {e}")
        
        return resources
    
    def get_resource(self, resource_id: str, resource_type: str) -> Optional[Dict[str, Any]]:
        """Obtiene información de un recurso AWS."""
        try:
            if resource_type == 's3_bucket':
                response = self.s3_client.get_bucket_location(Bucket=resource_id)
                return {'Bucket': resource_id, 'Location': response}
            
            elif resource_type == 'sagemaker_endpoint':
                return self.sagemaker_client.describe_endpoint(EndpointName=resource_id)
            
            elif resource_type == 'sagemaker_training_job':
                return self.sagemaker_client.describe_training_job(TrainingJobName=resource_id)
            
            elif resource_type == 'sagemaker_model':
                return self.sagemaker_client.describe_model(ModelName=resource_id)
            
        except Exception as e:
            logger.error(f"Error getting {resource_type} {resource_id}: {e}")
        
        return None


class S3Storage(CloudStorage):
    """Almacenamiento en Amazon S3."""
    
    def __init__(self, provider: AWSProvider, bucket_name: str, create_if_not_exists: bool = False):
        """Inicializa S3 storage.
        
        Parameters
        ----------
        provider : AWSProvider
            Proveedor AWS.
        bucket_name : str
            Nombre del bucket.
        create_if_not_exists : bool
            Si crear el bucket si no existe.
        """
        super().__init__(provider, bucket_name)
        
        if create_if_not_exists:
            self._create_bucket_if_not_exists()
    
    def _create_bucket_if_not_exists(self):
        """Crea el bucket si no existe."""
        try:
            self.provider.s3_client.head_bucket(Bucket=self.bucket_name)
        except:
            try:
                if self.provider.config.region and self.provider.config.region != 'us-east-1':
                    self.provider.s3_client.create_bucket(
                        Bucket=self.bucket_name,
                        CreateBucketConfiguration={'LocationConstraint': self.provider.config.region}
                    )
                else:
                    self.provider.s3_client.create_bucket(Bucket=self.bucket_name)
                logger.info(f"Created S3 bucket: {self.bucket_name}")
            except Exception as e:
                logger.error(f"Could not create bucket: {e}")
    
    def upload_file(
        self,
        local_path: Union[str, Path],
        remote_path: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> bool:
        """Sube un archivo a S3."""
        try:
            local_path = Path(local_path)
            
            extra_args = {}
            if metadata:
                extra_args['Metadata'] = metadata
            
            self.provider.s3_client.upload_file(
                str(local_path),
                self.bucket_name,
                remote_path,
                ExtraArgs=extra_args if extra_args else None
            )
            
            logger.debug(f"Uploaded {local_path} to s3://{self.bucket_name}/{remote_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            return False
    
    def download_file(
        self,
        remote_path: str,
        local_path: Union[str, Path]
    ) -> bool:
        """Descarga un archivo de S3."""
        try:
            local_path = Path(local_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.provider.s3_client.download_file(
                self.bucket_name,
                remote_path,
                str(local_path)
            )
            
            logger.debug(f"Downloaded s3://{self.bucket_name}/{remote_path} to {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            return False
    
    def list_files(
        self,
        prefix: Optional[str] = None,
        max_results: int = 1000
    ) -> List[str]:
        """Lista archivos en S3."""
        files = []
        
        try:
            paginator = self.provider.s3_client.get_paginator('list_objects_v2')
            
            kwargs = {'Bucket': self.bucket_name, 'MaxKeys': max_results}
            if prefix:
                kwargs['Prefix'] = prefix
            
            for page in paginator.paginate(**kwargs):
                if 'Contents' in page:
                    files.extend([obj['Key'] for obj in page['Contents']])
                
                if len(files) >= max_results:
                    break
            
        except Exception as e:
            logger.error(f"Error listing files: {e}")
        
        return files[:max_results]
    
    def delete_file(self, remote_path: str) -> bool:
        """Elimina un archivo de S3."""
        try:
            self.provider.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=remote_path
            )
            logger.debug(f"Deleted s3://{self.bucket_name}/{remote_path}")
            return True
        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            return False
    
    def file_exists(self, remote_path: str) -> bool:
        """Verifica si un archivo existe en S3."""
        try:
            self.provider.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=remote_path
            )
            return True
        except:
            return False
    
    def get_presigned_url(
        self,
        remote_path: str,
        expiration: int = 3600,
        operation: str = 'get_object'
    ) -> Optional[str]:
        """Genera URL pre-firmada para acceso temporal.
        
        Parameters
        ----------
        remote_path : str
            Ruta del archivo en S3.
        expiration : int
            Tiempo de expiración en segundos.
        operation : str
            Operación ('get_object' o 'put_object').
            
        Returns
        -------
        Optional[str]
            URL pre-firmada o None.
        """
        try:
            url = self.provider.s3_client.generate_presigned_url(
                operation,
                Params={'Bucket': self.bucket_name, 'Key': remote_path},
                ExpiresIn=expiration
            )
            return url
        except Exception as e:
            logger.error(f"Error generating presigned URL: {e}")
            return None


class SageMakerCompute(CloudCompute):
    """Cómputo con Amazon SageMaker."""
    
    def __init__(self, provider: AWSProvider, role_arn: str):
        """Inicializa SageMaker compute.
        
        Parameters
        ----------
        provider : AWSProvider
            Proveedor AWS.
        role_arn : str
            ARN del rol IAM para SageMaker.
        """
        super().__init__(provider)
        self.role_arn = role_arn
        self.client = provider.sagemaker_client
    
    def submit_training_job(
        self,
        job_name: str,
        script_path: str,
        instance_type: str = 'ml.m5.xlarge',
        instance_count: int = 1,
        hyperparameters: Optional[Dict[str, Any]] = None,
        input_data: Optional[Dict[str, str]] = None,
        output_path: Optional[str] = None,
        framework: str = 'sklearn',
        framework_version: str = '1.0-1',
        **kwargs
    ) -> str:
        """Envía trabajo de entrenamiento a SageMaker."""
        try:
            # Configurar imagen del contenedor según framework
            image_uri = self._get_image_uri(framework, framework_version)
            
            # Configuración del trabajo
            training_config = {
                'TrainingJobName': job_name,
                'RoleArn': self.role_arn,
                'AlgorithmSpecification': {
                    'TrainingImage': image_uri,
                    'TrainingInputMode': 'File'
                },
                'ResourceConfig': {
                    'InstanceType': instance_type,
                    'InstanceCount': instance_count,
                    'VolumeSizeInGB': 30
                },
                'StoppingCondition': {
                    'MaxRuntimeInSeconds': 86400  # 24 horas
                }
            }
            
            # Hiperparámetros
            if hyperparameters:
                training_config['HyperParameters'] = {
                    k: str(v) for k, v in hyperparameters.items()
                }
            
            # Datos de entrada
            if input_data:
                training_config['InputDataConfig'] = [
                    {
                        'ChannelName': name,
                        'DataSource': {
                            'S3DataSource': {
                                'S3DataType': 'S3Prefix',
                                'S3Uri': uri,
                                'S3DataDistributionType': 'FullyReplicated'
                            }
                        }
                    }
                    for name, uri in input_data.items()
                ]
            
            # Salida
            if output_path:
                training_config['OutputDataConfig'] = {
                    'S3OutputPath': output_path
                }
            
            # Crear trabajo
            response = self.client.create_training_job(**training_config)
            
            logger.info(f"Submitted SageMaker training job: {job_name}")
            return response['TrainingJobArn']
            
        except Exception as e:
            logger.error(f"Error submitting training job: {e}")
            raise
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Obtiene estado del trabajo de SageMaker."""
        try:
            # job_id puede ser el nombre o ARN
            job_name = job_id.split('/')[-1] if '/' in job_id else job_id
            
            response = self.client.describe_training_job(TrainingJobName=job_name)
            
            return {
                'status': response['TrainingJobStatus'],
                'secondary_status': response.get('SecondaryStatus'),
                'start_time': response.get('TrainingStartTime'),
                'end_time': response.get('TrainingEndTime'),
                'metrics': response.get('FinalMetricDataList', [])
            }
            
        except Exception as e:
            logger.error(f"Error getting job status: {e}")
            return {'status': 'UNKNOWN', 'error': str(e)}
    
    def stop_job(self, job_id: str) -> bool:
        """Detiene trabajo de SageMaker."""
        try:
            job_name = job_id.split('/')[-1] if '/' in job_id else job_id
            
            self.client.stop_training_job(TrainingJobName=job_name)
            logger.info(f"Stopped training job: {job_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping job: {e}")
            return False
    
    def deploy_model(
        self,
        model_path: str,
        endpoint_name: str,
        instance_type: str = 'ml.t2.medium',
        initial_instance_count: int = 1,
        framework: str = 'sklearn',
        framework_version: str = '1.0-1',
        **kwargs
    ) -> str:
        """Despliega modelo en SageMaker."""
        try:
            # Crear modelo
            model_name = f"{endpoint_name}-model"
            image_uri = self._get_image_uri(framework, framework_version)
            
            self.client.create_model(
                ModelName=model_name,
                PrimaryContainer={
                    'Image': image_uri,
                    'ModelDataUrl': model_path
                },
                ExecutionRoleArn=self.role_arn
            )
            
            # Crear configuración del endpoint
            config_name = f"{endpoint_name}-config"
            
            self.client.create_endpoint_config(
                EndpointConfigName=config_name,
                ProductionVariants=[{
                    'VariantName': 'AllTraffic',
                    'ModelName': model_name,
                    'InitialInstanceCount': initial_instance_count,
                    'InstanceType': instance_type
                }]
            )
            
            # Crear endpoint
            response = self.client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=config_name
            )
            
            logger.info(f"Deploying model to endpoint: {endpoint_name}")
            return response['EndpointArn']
            
        except Exception as e:
            logger.error(f"Error deploying model: {e}")
            raise
    
    def predict(
        self,
        endpoint_name: str,
        data: Any,
        content_type: str = 'application/json',
        **kwargs
    ) -> Any:
        """Realiza predicciones usando endpoint de SageMaker."""
        try:
            runtime_client = self.provider.session.client('sagemaker-runtime')
            
            # Serializar datos
            if content_type == 'application/json':
                import json
                body = json.dumps(data)
            else:
                body = data
            
            # Invocar endpoint
            response = runtime_client.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType=content_type,
                Body=body
            )
            
            # Deserializar respuesta
            result = response['Body'].read()
            
            if content_type == 'application/json':
                return json.loads(result)
            else:
                return result
                
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    def delete_endpoint(self, endpoint_name: str) -> bool:
        """Elimina endpoint de SageMaker."""
        try:
            # Eliminar endpoint
            self.client.delete_endpoint(EndpointName=endpoint_name)
            
            # También eliminar config y modelo
            try:
                self.client.delete_endpoint_config(
                    EndpointConfigName=f"{endpoint_name}-config"
                )
                self.client.delete_model(
                    ModelName=f"{endpoint_name}-model"
                )
            except:
                pass
            
            logger.info(f"Deleted endpoint: {endpoint_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting endpoint: {e}")
            return False
    
    def _get_image_uri(self, framework: str, framework_version: str) -> str:
        """Obtiene URI de imagen de contenedor para framework."""
        region = self.provider.config.region or 'us-east-1'
        
        # URIs base para diferentes frameworks
        # Estos son ejemplos, deberían actualizarse según región y versión
        image_uris = {
            'sklearn': f'683313688378.dkr.ecr.{region}.amazonaws.com/sagemaker-scikit-learn:{framework_version}-cpu-py3',
            'pytorch': f'763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:{framework_version}-cpu-py38',
            'tensorflow': f'763104351884.dkr.ecr.{region}.amazonaws.com/tensorflow-training:{framework_version}-cpu-py38',
            'xgboost': f'683313688378.dkr.ecr.{region}.amazonaws.com/sagemaker-xgboost:{framework_version}'
        }
        
        return image_uris.get(framework, image_uris['sklearn'])